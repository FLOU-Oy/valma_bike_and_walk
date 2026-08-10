"""Origin-destination travel-time matrices.

The whole point of this module is that an OD matrix over many origins is a
sequence of single-source shortest-path runs, and the only two things that
matter at scale are how fast one run is and how much memory a batch of them
holds.

SciPy's Dijkstra returns a dense ``(n_sources, n_nodes)`` block, which for a
country-sized network is gigabytes if you ask for too many sources at once. So
sources are processed in chunks sized from a memory budget, and each chunk is
immediately narrowed to the destination columns we actually want.
"""

from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from valma_bike_and_walk.network import RoutableNetwork

logger = logging.getLogger(__name__)

#: How much memory one chunk's dense distance block may occupy.
DEFAULT_CHUNK_BYTES = 256 * 1024**2

# Set once per worker process by _init_worker.
_WORKER_STATE: dict[str, object] = {}


def choose_chunk_size(n_nodes: int, budget_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """Largest number of sources whose dense distance block fits the budget."""
    per_source = max(1, n_nodes * 8)  # SciPy always returns float64
    return int(max(1, min(n_nodes, budget_bytes // per_source)))


def _chunks(values: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, values.shape[0], size):
        yield values[start : start + size]


def _solve_chunk(
    graph: csr_matrix,
    sources: np.ndarray,
    targets: np.ndarray,
    max_seconds: float | None,
    dtype: np.dtype,
) -> np.ndarray:
    distances = dijkstra(
        graph,
        directed=True,
        indices=sources,
        limit=np.inf if max_seconds is None else max_seconds,
    )
    return np.ascontiguousarray(distances[:, targets], dtype=dtype)


def _init_worker(cache_dir: str, shape: tuple[int, int]) -> None:
    """
    Rebuild the CSR in a worker from memory-mapped arrays.

    Memory-mapping rather than pickling matters on Windows, where every worker
    is a fresh process: the OS pages one shared read-only copy of the network in
    instead of each worker inflating its own.
    """
    base = Path(cache_dir)
    indptr = np.load(base / "indptr.npy", mmap_mode="r")
    indices = np.load(base / "indices.npy", mmap_mode="r")
    data = np.load(base / "data.npy", mmap_mode="r")
    _WORKER_STATE["graph"] = csr_matrix(
        (data, indices, indptr), shape=shape, copy=False
    )


def _worker_solve(
    sources: np.ndarray,
    targets: np.ndarray,
    max_seconds: float | None,
    dtype_name: str,
) -> np.ndarray:
    graph = _WORKER_STATE["graph"]
    return _solve_chunk(graph, sources, targets, max_seconds, np.dtype(dtype_name))


def travel_time_matrix(
    network: RoutableNetwork,
    sources: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray | None = None,
    max_seconds: float | None = None,
    workers: int = 1,
    chunk_size: int | None = None,
    dtype: npt.DTypeLike = np.float32,
    progress_every: int = 10,
) -> np.ndarray:
    """
    Travel time in seconds from every source node to every target node.

    Parameters:
    - network: the routable network to search.
    - sources / targets: node *indices* into the network (see
      ``RoutableNetwork.nearest_nodes``). ``targets`` defaults to ``sources``.
    - max_seconds: stop expanding beyond this travel time. Anything further away
      comes back as infinity. On a big network this is by far the cheapest way
      to speed the whole thing up, if your analysis has a cutoff anyway.
    - workers: processes to spread source chunks over. 1 keeps it in-process.
    - chunk_size: sources per Dijkstra call; defaults to a 256 MB block.

    Returns an ``(len(sources), len(targets))`` array; unreachable pairs are inf.
    """
    sources = np.asarray(sources, dtype=np.int32)
    targets = sources if targets is None else np.asarray(targets, dtype=np.int32)
    dtype = np.dtype(dtype)

    if sources.size == 0 or targets.size == 0:
        return np.empty((sources.size, targets.size), dtype=dtype)
    if sources.min() < 0 or targets.min() < 0:
        raise ValueError(
            "Source and target indices must be non-negative; unsnapped points must be filtered out first."
        )
    if sources.max() >= network.n_nodes or targets.max() >= network.n_nodes:
        raise ValueError("Node index out of range for this network.")

    if chunk_size is None:
        chunk_size = choose_chunk_size(network.n_nodes)
    n_chunks = int(np.ceil(sources.size / chunk_size))

    logger.info(
        "OD matrix %d x %d over %d nodes: %d chunks of <=%d sources, %d worker(s)",
        sources.size,
        targets.size,
        network.n_nodes,
        n_chunks,
        chunk_size,
        workers,
    )

    result = np.empty((sources.size, targets.size), dtype=dtype)

    if workers <= 1:
        graph = network.csr()
        for i, chunk in enumerate(_chunks(sources, chunk_size)):
            start = i * chunk_size
            result[start : start + chunk.size] = _solve_chunk(
                graph, chunk, targets, max_seconds, dtype
            )
            if progress_every and (i + 1) % progress_every == 0:
                logger.info("  %d/%d chunks", i + 1, n_chunks)
        return result

    with tempfile.TemporaryDirectory(prefix="valma-od-") as tmp:
        base = Path(tmp)
        np.save(base / "indptr.npy", network.indptr)
        np.save(base / "indices.npy", network.indices)
        np.save(base / "data.npy", network.travel_time)
        shape = (network.n_nodes, network.n_nodes)

        offsets = list(range(0, sources.size, chunk_size))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(base), shape),
        ) as pool:
            futures = {
                pool.submit(
                    _worker_solve,
                    sources[start : start + chunk_size],
                    targets,
                    max_seconds,
                    dtype.name,
                ): start
                for start in offsets
            }
            for done, (future, start) in enumerate(futures.items(), start=1):
                block = future.result()
                result[start : start + block.shape[0]] = block
                if progress_every and done % progress_every == 0:
                    logger.info("  %d/%d chunks", done, n_chunks)

    return result


def default_workers() -> int:
    """A sensible worker count that leaves the machine usable."""
    return max(1, (os.cpu_count() or 2) - 2)


def summarise(matrix: np.ndarray) -> dict[str, float]:
    """Quick sanity statistics for an OD matrix, ignoring unreachable pairs."""
    finite = np.isfinite(matrix)
    reachable = matrix[finite]
    off_diagonal = reachable[reachable > 0]
    return {
        "pairs": float(matrix.size),
        "reachable_fraction": float(finite.mean()),
        "min_minutes": (
            float(off_diagonal.min() / 60) if off_diagonal.size else float("nan")
        ),
        "median_minutes": (
            float(np.median(off_diagonal) / 60) if off_diagonal.size else float("nan")
        ),
        "max_minutes": (
            float(off_diagonal.max() / 60) if off_diagonal.size else float("nan")
        ),
    }
