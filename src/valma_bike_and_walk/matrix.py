"""Origin-destination travel-time matrices.

The whole point of this module is that an OD matrix over many origins is a
sequence of single-source shortest-path runs, and the only two things that
matter at scale are how fast one run is and how much memory a batch of them
holds.

SciPy's Dijkstra returns a dense ``(n_sources, n_nodes)`` block, which for a
country-sized network is gigabytes if you ask for too many sources at once. So
sources are processed in chunks sized from a memory budget, and each chunk is
immediately narrowed to the destination columns we actually want.

Two entry points, differing in what a row of the result means:

- :func:`travel_time_matrix` -- one point per row, the plain node-to-node
  matrix. This is the single-centroid method, unchanged.
- :func:`zone_travel_time_matrix` -- one *zone* per row, built by routing
  between the weighted access points of
  :class:`valma_bike_and_walk.zones.Zones` and aggregating point-to-point times
  back to zone level with those weights. See that module for why one point per
  zone gets short trips wrong, and :func:`zone_travel_time_matrix` for how the
  two-tier scheme keeps the cost of fixing it close to the single-point one.
"""

from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from valma_bike_and_walk.network import RoutableNetwork
from valma_bike_and_walk.speeds import SECONDS_PER_HOUR, profile_for
from valma_bike_and_walk.zones import Zones

logger = logging.getLogger(__name__)

#: How much memory one chunk's dense distance block may occupy.
DEFAULT_CHUNK_BYTES = 256 * 1024**2

#: How much of a zone pair's weight has to be reachable for the near tier of a
#: two-tier run to be trusted. A cell where only the closest access-point pairs
#: came back inside the cutoff would average over exactly those, biasing it low;
#: at 1.0 such a cell falls through to the far tier instead, which has no cutoff
#: problem because it has no cutoff at that range.
DEFAULT_NEAR_REACHABLE_FRACTION = 1.0

#: Slack when comparing a summed weight fraction against a threshold.
_REACHABLE_TOLERANCE = 1e-9

#: Mean distance between two points drawn uniformly from a disc of radius R,
#: as a multiple of R: 128 / (45 * pi). The classic closed form behind the
#: "equal-area circle" intrazonal estimate.
_DISC_MEAN_DISTANCE = 128.0 / (45.0 * np.pi)

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
            # Pop each future as it's consumed (and iterate as-completed
            # rather than submission order) so a finished chunk's block is
            # freed once it's copied into `result`, instead of every
            # completed future -- and the block inside it -- staying alive
            # in `futures` until the very last chunk finishes.
            done = 0
            for future in as_completed(futures):
                start = futures.pop(future)
                block = future.result()
                result[start : start + block.shape[0]] = block
                done += 1
                if progress_every and done % progress_every == 0:
                    logger.info("  %d/%d chunks", done, n_chunks)

    return result


def _solve_zone_chunk(
    graph: csr_matrix,
    source_nodes: np.ndarray,
    source_access: np.ndarray,
    source_weight: np.ndarray,
    source_starts: np.ndarray,
    target_nodes: np.ndarray,
    target_access: np.ndarray,
    target_weight: np.ndarray,
    target_starts: np.ndarray,
    max_seconds: float | None,
    decay_mu: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One chunk of whole zones: Dijkstra, then reduce point pairs to zone pairs.

    Returns the numerator and denominator of the weighted mean, both
    ``(zones_in_chunk, n_zones)``, rather than the mean itself -- the two have
    to be summed across chunks separately, and keeping them apart is also what
    makes the reachability bookkeeping work: the denominator *is* the weight of
    the access-point pairs that came back inside the cutoff.

    The reduction is two ``reduceat`` passes, not a matmul, because every access
    point belongs to exactly one zone and the points are stored sorted by zone.
    That makes the zone-weight "matrix" a segment sum: contiguous column groups
    collapse to one column each, then contiguous row groups to one row each.
    """
    distance = dijkstra(
        graph,
        directed=True,
        indices=source_nodes,
        limit=np.inf if max_seconds is None else max_seconds,
    )
    block = distance[:, target_nodes]
    block += source_access[:, None]
    block += target_access[None, :]

    # inf marks "not reached within the cutoff". exp(-mu * inf) is 0 rather than
    # a warning, but mask explicitly anyway so both branches read the same.
    finite = np.isfinite(block)
    value = block if decay_mu is None else np.exp(-decay_mu * block)
    value = np.where(finite, value, 0.0)

    weighted = value * target_weight[None, :]
    reached = np.where(finite, target_weight[None, :], 0.0)

    numerator = np.add.reduceat(weighted, target_starts, axis=1)
    denominator = np.add.reduceat(reached, target_starts, axis=1)
    numerator *= source_weight[:, None]
    denominator *= source_weight[:, None]

    return (
        np.add.reduceat(numerator, source_starts, axis=0),
        np.add.reduceat(denominator, source_starts, axis=0),
    )


def _init_zone_worker(cache_dir: str, shape: tuple[int, int]) -> None:
    """Like :func:`_init_worker`, plus the destination side, which every chunk shares."""
    _init_worker(cache_dir, shape)
    base = Path(cache_dir)
    for name in ("target_nodes", "target_access", "target_weight", "target_starts"):
        _WORKER_STATE[name] = np.load(base / f"{name}.npy", mmap_mode="r")


def _worker_zone_solve(
    source_nodes: np.ndarray,
    source_access: np.ndarray,
    source_weight: np.ndarray,
    source_starts: np.ndarray,
    max_seconds: float | None,
    decay_mu: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    return _solve_zone_chunk(
        _WORKER_STATE["graph"],  # type: ignore[arg-type]
        source_nodes,
        source_access,
        source_weight,
        source_starts,
        _WORKER_STATE["target_nodes"],  # type: ignore[arg-type]
        _WORKER_STATE["target_access"],  # type: ignore[arg-type]
        _WORKER_STATE["target_weight"],  # type: ignore[arg-type]
        _WORKER_STATE["target_starts"],  # type: ignore[arg-type]
        max_seconds,
        decay_mu,
    )


def _accumulate_zone_matrix(
    network: RoutableNetwork,
    zones: Zones,
    max_seconds: float | None,
    decay_mu: float | None,
    workers: int,
    chunk_size: int | None,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run every zone's access points as sources and sum the chunk reductions."""
    n = zones.n_zones
    numerator = np.zeros((n, n), dtype=np.float64)
    denominator = np.zeros((n, n), dtype=np.float64)

    if chunk_size is None:
        chunk_size = choose_chunk_size(network.n_nodes)
    ranges = list(zones.chunk_ranges(chunk_size))
    logger.info(
        "Zone OD matrix %d x %d from %d access point(s) over %d nodes: "
        "%d chunks of <=%d points, %d worker(s)",
        n,
        n,
        zones.n_points,
        network.n_nodes,
        len(ranges),
        chunk_size,
        workers,
    )

    indptr = zones.indptr
    target_starts = indptr[:-1]

    def chunk_args(
        lo: int, hi: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """This chunk's source nodes, access times, weights and row offsets."""
        first, last = int(indptr[lo]), int(indptr[hi])
        return (
            zones.node_index[first:last],
            zones.access_seconds[first:last],
            zones.weight[first:last],
            indptr[lo:hi] - first,
        )

    if workers <= 1:
        graph = network.csr()
        for done, (lo, hi) in enumerate(ranges, start=1):
            num, den = _solve_zone_chunk(
                graph,
                *chunk_args(lo, hi),
                zones.node_index,
                zones.access_seconds,
                zones.weight,
                target_starts,
                max_seconds,
                decay_mu,
            )
            numerator[lo:hi] += num
            denominator[lo:hi] += den
            if progress_every and done % progress_every == 0:
                logger.info("  %d/%d chunks", done, len(ranges))
        return numerator, denominator

    with tempfile.TemporaryDirectory(prefix="valma-zone-od-") as tmp:
        base = Path(tmp)
        np.save(base / "indptr.npy", network.indptr)
        np.save(base / "indices.npy", network.indices)
        np.save(base / "data.npy", network.travel_time)
        np.save(base / "target_nodes.npy", zones.node_index)
        np.save(base / "target_access.npy", zones.access_seconds)
        np.save(base / "target_weight.npy", zones.weight)
        np.save(base / "target_starts.npy", target_starts)
        shape = (network.n_nodes, network.n_nodes)

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_zone_worker,
            initargs=(str(base), shape),
        ) as pool:
            futures = {
                pool.submit(
                    _worker_zone_solve, *chunk_args(lo, hi), max_seconds, decay_mu
                ): (lo, hi)
                for lo, hi in ranges
            }
            done = 0
            for future in as_completed(futures):
                lo, hi = futures.pop(future)
                num, den = future.result()
                numerator[lo:hi] += num
                denominator[lo:hi] += den
                done += 1
                if progress_every and done % progress_every == 0:
                    logger.info("  %d/%d chunks", done, len(ranges))

    return numerator, denominator


def _drop_self_pairs(
    numerator: np.ndarray,
    denominator: np.ndarray,
    zones: Zones,
    decay_mu: float | None,
) -> np.ndarray:
    """
    Remove the a==b terms from the intrazonal cells, and say what full weight is.

    A zone's own diagonal picks up one "trip" per access point that never leaves
    it. Those are an artefact of discretising a continuous zone -- two people
    drawn independently from the same zone coincide with probability zero in
    reality -- and they drag the intrazonal time down by a factor that depends
    on nothing but K. Subtracting them from both numerator and denominator
    leaves the weighted mean over genuinely distinct point pairs.

    Returns the full (unreachable-free) denominator each cell is measured
    against: 1 everywhere, less the self weight on the diagonal.
    """
    full = np.ones(numerator.shape, dtype=np.float64)
    self_weight = zones.self_weight

    # A self-pair's travel time is not zero once access time is charged: it is
    # the leg from the demand point to its node and back again.
    self_time = 2.0 * zones.access_seconds
    self_value = self_time if decay_mu is None else np.exp(-decay_mu * self_time)
    self_numerator = np.add.reduceat(zones.weight**2 * self_value, zones.indptr[:-1])

    diagonal = np.arange(zones.n_zones)
    numerator[diagonal, diagonal] -= self_numerator
    denominator[diagonal, diagonal] -= self_weight
    full[diagonal, diagonal] -= self_weight
    return full


def _intrazonal_fallback(network: RoutableNetwork, zones: Zones) -> np.ndarray:
    """
    Crow-flies intrazonal time from the zone's area, for when routing gave nothing.

    Two points drawn uniformly from a disc of radius R are 128R/(45*pi) apart on
    average. Crude -- real paths bend around things, and real demand is not
    uniform over a zone -- but finite and roughly right, which beats the
    alternatives when there is no routed answer at all: a zone with one access
    point has no distinct pair to measure, and a two-tier run whose near tier
    did not reach would otherwise report twice the representative point's
    access time.

    Used *only* where the diagonal came back unreachable, never to clamp a
    routed value. A zone whose demand really is concentrated in one corner has a
    genuinely shorter intrazonal trip than a uniform disc of the same area, and
    finding that out is the whole reason for placing points by weight.
    """
    speed_ms = profile_for(network.mode).base_kph * (1000.0 / SECONDS_PER_HOUR)
    return _DISC_MEAN_DISTANCE * zones.equivalent_radius_m() / speed_ms


def multipoint_travel_time_matrix(
    network: RoutableNetwork,
    zones: Zones,
    max_seconds: float | None = None,
    min_reachable_fraction: float = 0.0,
    decay_mu: float | None = None,
    exclude_self_pairs: bool = True,
    workers: int = 1,
    chunk_size: int | None = None,
    dtype: npt.DTypeLike = np.float32,
    progress_every: int = 10,
) -> np.ndarray:
    """
    Zone-to-zone travel time, averaged over every pair of weighted access points.

    Every access point of every zone is routed as a source, which is what makes
    this ``K`` times the work of :func:`travel_time_matrix`. Use
    :func:`zone_travel_time_matrix` instead unless you specifically want the
    unstitched multi-point matrix -- it wraps this with a cutoff and a
    single-point far tier that between them bring the cost back down.

    - min_reachable_fraction: report a cell only if at least this share of its
      access-point pairs came back inside ``max_seconds``. 0 averages over
      whatever was reachable; 1 demands all of them. Immaterial without a cutoff.
    - decay_mu: aggregate on ``exp(-mu * t)`` and take the logsum, rather than
      averaging time arithmetically. The right choice when the matrix feeds a
      gravity or logit model, where demand responds to ``exp(-mu * t)`` and
      Jensen's inequality makes the arithmetic mean the wrong summary; leave it
      None for a matrix meant to be read as minutes.
    - exclude_self_pairs: drop the "trip" from an access point to itself when
      averaging the diagonal. See :func:`_drop_self_pairs`.

    Returns an ``(n_zones, n_zones)`` array in ``zones.ids`` order; cells with
    too little reachable weight are inf, exactly as unreachable pairs are in
    :func:`travel_time_matrix`.
    """
    if zones.n_zones == 0:
        return np.empty((0, 0), dtype=np.dtype(dtype))

    numerator, denominator = _accumulate_zone_matrix(
        network, zones, max_seconds, decay_mu, workers, chunk_size, progress_every
    )

    if exclude_self_pairs:
        full = _drop_self_pairs(numerator, denominator, zones, decay_mu)
    else:
        full = np.ones(numerator.shape, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(full > 0, denominator / full, 0.0)
        usable = (denominator > 0) & (
            fraction >= min_reachable_fraction - _REACHABLE_TOLERANCE
        )
        mean = np.where(usable, numerator / np.where(usable, denominator, 1.0), np.inf)
        if decay_mu is not None:
            mean = np.where(
                usable, -np.log(np.maximum(mean, 1e-300)) / decay_mu, np.inf
            )

    return np.ascontiguousarray(mean, dtype=np.dtype(dtype))


def zone_travel_time_matrix(
    network: RoutableNetwork,
    zones: Zones,
    max_seconds: float | None = None,
    near_seconds: float | None = None,
    near_reachable_fraction: float = DEFAULT_NEAR_REACHABLE_FRACTION,
    decay_mu: float | None = None,
    exclude_self_pairs: bool = True,
    intrazonal_fallback: bool = True,
    workers: int = 1,
    chunk_size: int | None = None,
    dtype: npt.DTypeLike = np.float32,
    progress_every: int = 10,
) -> np.ndarray:
    """
    Zone-to-zone travel time, multi-point where it matters and single-point where it does not.

    Without ``near_seconds`` this is just :func:`multipoint_travel_time_matrix`:
    every access point routed against every other, ``K`` times the Dijkstra runs
    of a single-centroid matrix.

    With ``near_seconds`` it becomes two tiers, which is the point. The
    single-point approximation's relative error falls as ``R^2/2d^2`` (see
    :mod:`valma_bike_and_walk.zones`), so it is only worth paying for multiple
    points over short distances -- and short distances are cheap to search,
    because a Dijkstra tree's settled node count grows with the *square* of its
    cutoff. A 15-minute tree is roughly a sixteenth of a 60-minute one, so eight
    access points explored to 15 minutes cost about half of one unbounded run:

    - **near tier** -- every access point, bounded at ``near_seconds``. Accurate
      exactly where the single-point error lives, including the whole diagonal.
    - **far tier** -- one representative point per zone, bounded at
      ``max_seconds``. This is the classic single-centroid matrix, and it is what
      ``--zones`` degrades to for pairs far enough apart that it makes no odds.

    Near values win wherever they exist; the rest fall through to the far tier.
    Set ``near_seconds`` from the zone system rather than by feel: at ``d = 5R``
    the single-point error is already down to about 2 %, so a cutoff covering
    five times a typical zone radius is usually plenty. Note that with
    ``near_reachable_fraction`` at its default of 1, a zone pair is only
    resolved when its *furthest* access-point pair is inside the cutoff, so the
    effective reach is roughly ``near_seconds`` less the time to cross two
    zones. Budget for that rather than being surprised by it.

    ``intrazonal_fallback`` fills any diagonal cell the multi-point pass could
    not produce with the equal-area-circle estimate; see
    :func:`_intrazonal_fallback`.
    """
    dtype = np.dtype(dtype)
    if zones.n_zones == 0:
        return np.empty((0, 0), dtype=dtype)

    near = multipoint_travel_time_matrix(
        network,
        zones,
        max_seconds=max_seconds if near_seconds is None else near_seconds,
        min_reachable_fraction=(
            0.0 if near_seconds is None else near_reachable_fraction
        ),
        decay_mu=decay_mu,
        exclude_self_pairs=exclude_self_pairs,
        workers=workers,
        chunk_size=chunk_size,
        dtype=dtype,
        progress_every=progress_every,
    )
    resolved = np.isfinite(near)

    # Whether the *diagonal* is trustworthy is a separate question from whether
    # the stitched matrix has a finite number in it: an unresolved diagonal
    # falls through to the far tier, where a zone's representative point routes
    # to itself in no time at all. Only the multi-point pass can measure how far
    # apart two people in one zone are, so its verdict is what decides.
    diagonal = np.arange(zones.n_zones)
    routed_diagonal = resolved[diagonal, diagonal]

    if near_seconds is None:
        result = near
    else:
        logger.info(
            "Near tier: multi-point within %.0f s; far tier: one point per zone",
            near_seconds,
        )
        far = travel_time_matrix(
            network,
            zones.representative.node_index,
            max_seconds=max_seconds,
            workers=workers,
            chunk_size=chunk_size,
            dtype=dtype,
            progress_every=progress_every,
        )
        access = zones.representative_access_seconds.astype(dtype)
        far = far + access[:, None] + access[None, :]

        logger.info(
            "Near tier resolved %.1f %% of zone pairs; the rest fall back to the "
            "single representative point",
            100.0 * float(resolved.mean()),
        )
        result = np.where(resolved, near, far).astype(dtype)

    if intrazonal_fallback and not routed_diagonal.all():
        logger.info(
            "Intrazonal time for %d of %d zone(s) came from the equal-area "
            "circle rather than from routing",
            int((~routed_diagonal).sum()),
            zones.n_zones,
        )
        result[diagonal, diagonal] = np.where(
            routed_diagonal,
            result[diagonal, diagonal],
            _intrazonal_fallback(network, zones).astype(dtype),
        )

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
