"""All-or-nothing (AON) traffic assignment: load an OD demand matrix onto link volumes.

Given a demand matrix and a routable network, this loads every OD pair's
demand onto every link of that pair's shortest (least travel-time) path, and
sums the result per link. That is "all-or-nothing" assignment -- the classic
first assignment method taught in transport modelling, and the simplest: one
shortest-path tree per origin, no rebalancing.

It is also a good match for walking and cycling specifically. AON's usual
weakness is that it ignores congestion -- real traffic reroutes around a busy
link, so a capacity-constrained or equilibrium method is normally needed to
avoid dumping implausible volumes onto one "best" road. Footpaths and bike
lanes essentially never reach that kind of capacity, so the effect AON misses
does not really apply here; volumes come out close to a capacity-aware method
without needing one. If that stops being true for a given network (e.g. a
single bridge or ferry link genuinely bottlenecks everyone), an iterative
capacitated method (Frank-Wolfe / MSA over a volume-delay function) or a
multi-path (stochastic user equilibrium) assignment is the standard fix --
neither is implemented here, since it needs a volume-delay function this
project's networks don't otherwise carry.

Scale: demand is read and kept as a SciPy sparse matrix throughout, never
densified. That matters once you're at zone counts like Finland's own
municipalities or postcode areas (~1e4): a dense (1e4 x 1e4) float64 matrix is
already 800 MB before the network is even considered, and most of it would be
structural zero anyway -- OD demand between distant zones is normally either
absent or negligible. If your demand source only produces a dense matrix,
threshold away near-zero entries before assigning (`demand.demand_matrix`
does this once for you if it's built from a long/sparse-format file, which
avoids the dense step entirely).

The network side reuses the same chunked-Dijkstra approach as
:func:`valma_bike_and_walk.matrix.travel_time_matrix`, for the same reason:
SciPy's Dijkstra returns a dense ``(sources_in_chunk, n_nodes)`` block, so
memory is bounded by chunk size rather than by the number of origins. Runtime
scales with the network's edge count (each Dijkstra relaxes every edge from
an expanded node) as well as its node count -- for a country-sized edge set,
narrowing the network to the relevant area/mode, or setting ``max_seconds``,
is the biggest lever available.

Each chunk's own *result*, on the other hand, is kept sparse (see
``_solve_chunk``) and merged into the running total as soon as it's ready
(in completion order, not submission order) rather than being collected as a
full network-sized array per chunk and held until the whole run finishes --
important once there are enough sources that "one array per chunk" stops
being a rounding error next to the network itself.

Three things keep the parallel path's memory flat in the worker count, all of
which matter far more once zones spread demand over access points (see
:func:`assign_zone_traffic`), because that multiplies both the size of the
demand and the number of chunks:

- The demand, the targets and the graph reach workers by **memory mapping**,
  not as ``submit`` arguments. A task argument is pickled and unpickled per
  task per worker, so passing the demand that way gave every worker its own
  full copy of it -- the one thing in the whole run that scales with zones
  squared. See :func:`_init_worker`.
- Chunks are **submitted in a bounded window** rather than all at once, so
  the pool holds a couple of rounds of work items rather than one per chunk.
- What a chunk hands *back* is sparse both ways: touched edges only, and any
  demand it could not place aggregated to zone level rather than listed pair
  by pair (see :class:`_Unplaced`).
"""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterator, Sequence, cast

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, issparse
from scipy.sparse.csgraph import dijkstra

from valma_bike_and_walk.matrix import DEFAULT_CHUNK_BYTES
from valma_bike_and_walk.network import RoutableNetwork
from valma_bike_and_walk.zones import Zones

logger = logging.getLogger(__name__)

#: SciPy's sentinel for "this node has no predecessor" -- either it's the
#: source itself, or Dijkstra never reached it.
_NO_PREDECESSOR = -9999

# Set once per worker process by _init_worker.
_WORKER_STATE: dict[str, object] = {}


def choose_chunk_size(n_nodes: int, budget_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """
    Largest number of sources whose dense distance+predecessor block fits the budget.

    Like :func:`valma_bike_and_walk.matrix.choose_chunk_size`, but assignment
    needs the predecessor block too (int32, on top of Dijkstra's float64
    distances) to be able to walk paths back to each source.
    """
    per_source = max(1, n_nodes * (8 + 4))
    return int(max(1, min(n_nodes, budget_bytes // per_source)))


def _chunks(values: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, values.shape[0], size):
        yield values[start : start + size]


def _edge_keys(network: RoutableNetwork) -> np.ndarray:
    """
    A sortable id for every directed edge: ``u * n_nodes + v``.

    CSR rows are non-decreasing and each row's columns are already sorted (a
    side effect of how ``network._build_csr`` builds them, to dedupe parallel
    edges), so this key array comes out strictly increasing -- meaning its
    position *is* the edge index. That turns an (u, v) pair straight into an
    edge index with ``np.searchsorted``, without building or maintaining a
    separate lookup structure.
    """
    rows = np.repeat(
        np.arange(network.n_nodes, dtype=np.int64), np.diff(network.indptr)
    )
    cols = network.indices.astype(np.int64)
    return rows * network.n_nodes + cols


def _load_source(
    source: int,
    predecessors: np.ndarray,
    node: np.ndarray,
    load: np.ndarray,
    edge_keys: np.ndarray,
    n_nodes: int,
    link_volume: np.ndarray,
) -> np.ndarray:
    """
    All-or-nothing load of one source's nonzero demand onto ``link_volume``.

    Walks every destination back toward the source one hop at a time, all
    destinations in lockstep, adding that destination's (fixed) demand onto
    every link crossed. Destinations that share the tail of their shortest
    path -- the usual case near the source, where many trees converge -- walk
    that shared tail together in the same vectorised step, rather than each
    retracing it independently.

    ``node``/``load`` are this source's nonzero demand: destination node
    indices and the (positive) demand to each.

    Returns a boolean mask over them marking the ones with no path from
    ``source`` -- which is more than a count, because a caller running a
    *bounded* search needs to know exactly which demand went unplaced so it can
    route it another way. See :func:`assign_zone_traffic`'s two tiers.
    Destinations that *are* the source are not marked: their trip is
    zero-length, so there is nothing to place and nothing missing.
    """
    if node.size == 0:
        return np.zeros(0, dtype=bool)

    node = node.astype(np.int64).copy()
    load = load.astype(np.float64)

    at_source = node == source
    unreachable = (predecessors[node] == _NO_PREDECESSOR) & ~at_source
    keep = ~at_source & ~unreachable

    node, load = node[keep], load[keep]
    if node.size == 0:
        return unreachable

    still = np.ones(node.shape[0], dtype=bool)
    hops = 0
    max_hops = n_nodes + 1  # a shortest-path tree has no cycles; this is a safety valve
    while still.any():
        hops += 1
        if hops > max_hops:
            raise RuntimeError(
                "Shortest-path walk from source did not terminate within "
                f"{max_hops} hops; the predecessor tree may be corrupt."
            )
        idx = np.flatnonzero(still)
        cur = node[idx]
        pred = predecessors[cur]
        edge_idx = np.searchsorted(edge_keys, pred.astype(np.int64) * n_nodes + cur)
        np.add.at(link_volume, edge_idx, load[idx])
        node[idx] = pred
        still[idx] = pred != source

    return unreachable


def _as_demand_csr(
    demand: np.ndarray | csr_matrix, n_sources: int, n_targets: int
) -> csr_matrix:
    """
    Normalise demand to CSR, and drop non-positive entries.

    A dense array is accepted for convenience (and for parity with
    :func:`valma_bike_and_walk.matrix.travel_time_matrix`'s API), but this is
    the only place it's touched as dense -- everything downstream works off
    the sparse form so memory tracks the number of *nonzero* OD pairs, not
    ``n_sources * n_targets``.
    """
    if issparse(demand):
        matrix = demand.tocsr()  # type: ignore[union-attr]
    else:
        matrix = csr_matrix(np.asarray(demand, dtype=np.float64))
    if matrix.shape != (n_sources, n_targets):
        raise ValueError(
            f"demand has shape {matrix.shape}, expected {(n_sources, n_targets)} "
            "(len(sources) x len(targets))."
        )
    matrix = matrix.astype(np.float64)
    negative = int((matrix.data < 0).sum())
    if negative:
        logger.warning("Ignoring %d negative demand entr(y/ies).", negative)
    matrix.data[matrix.data <= 0] = 0
    matrix.eliminate_zeros()
    return matrix


class _Unplaced:
    """
    Demand a bounded search could not place, accumulated at *group* level.

    Chunks report what they failed to place so a caller can route it some other
    way, but reporting it pair by pair would hand back an array the size of the
    demand itself. Groups collapse that: with one group per zone, a chunk's
    report is at most (zones in chunk) x (zones), summed as it goes, and it
    comes back in exactly the shape the fallback pass wants as input.
    """

    def __init__(self, row_group: np.ndarray, col_group: np.ndarray, n_groups: int):
        self.row_group = row_group
        self.col_group = col_group
        self.n_groups = n_groups
        self._rows: list[np.ndarray] = []
        self._cols: list[np.ndarray] = []
        self._values: list[np.ndarray] = []

    def add(self, row: int, columns: np.ndarray, values: np.ndarray) -> None:
        if columns.size == 0:
            return
        self._rows.append(
            np.full(columns.shape[0], self.row_group[row], dtype=np.int64)
        )
        self._cols.append(self.col_group[columns])
        self._values.append(values)

    def collect(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sum duplicates once per chunk, so the payload is groups, not pairs."""
        if not self._rows:
            empty_index = np.empty(0, dtype=np.int64)
            return empty_index, empty_index, np.empty(0, dtype=np.float64)
        matrix = coo_matrix(
            (
                np.concatenate(self._values),
                (np.concatenate(self._rows), np.concatenate(self._cols)),
            ),
            shape=(self.n_groups, self.n_groups),
        ).tocoo()
        matrix.sum_duplicates()
        return matrix.row, matrix.col, matrix.data


#: What a chunk hands back: touched edge indices, their volumes, how many OD
#: pairs went unplaced, and (row, col, value) of that demand at group level.
_ChunkResult = tuple[np.ndarray, np.ndarray, int, tuple[np.ndarray, ...]]


def _solve_chunk(
    graph: csr_matrix,
    sources: np.ndarray,
    targets: np.ndarray,
    demand_data: np.ndarray,
    demand_indices: np.ndarray,
    demand_indptr: np.ndarray,
    row_offset: int,
    edge_keys: np.ndarray,
    n_nodes: int,
    max_seconds: float | None,
    n_edges: int,
    groups: tuple[np.ndarray, np.ndarray, int] | None = None,
) -> _ChunkResult:
    """
    One chunk of sources: Dijkstra, then load each row's demand.

    Returns a *sparse* update -- ``(edge_index, volume, ...)`` for only the
    edges this chunk actually touched -- rather than a full ``n_edges``-long
    array. A chunk of even a few dozen sources only ever touches a small
    fraction of a country-sized network's edges, so handing back (and, across
    the process-pool path, pickling) a whole network-sized array per chunk
    wastes memory and IPC bandwidth in direct proportion to the number of
    chunks -- which is exactly backwards, since chunking exists to keep memory
    *down*. See :func:`assign_traffic`'s merge step.

    Demand arrives as its three CSR arrays rather than as a ``csr_matrix``, so
    that the parallel path can hand workers a memory *mapping* of it instead of
    a value to pickle. That distinction is the difference between one copy of
    the demand and one per worker -- see :func:`_init_worker`.
    """
    _dist, pred = dijkstra(
        graph,
        directed=True,
        indices=sources,
        limit=np.inf if max_seconds is None else max_seconds,
        return_predecessors=True,
    )
    link_volume = np.zeros(n_edges, dtype=np.float64)
    dropped = 0
    unplaced = _Unplaced(*groups) if groups is not None else None

    for row, source in enumerate(sources):
        lo = int(demand_indptr[row_offset + row])
        hi = int(demand_indptr[row_offset + row + 1])
        columns = np.asarray(demand_indices[lo:hi])
        load = np.asarray(demand_data[lo:hi])
        missed = _load_source(
            int(source),
            pred[row],
            targets[columns],
            load,
            edge_keys,
            n_nodes,
            link_volume,
        )
        dropped += int(missed.sum())
        if unplaced is not None:
            unplaced.add(row_offset + row, columns[missed], load[missed])

    touched = np.flatnonzero(link_volume)
    report = unplaced.collect() if unplaced is not None else ()
    return touched, link_volume[touched], dropped, report


def _init_worker(cache_dir: str, shape: tuple[int, int]) -> None:
    """
    Rebuild the graph, edge keys, targets and demand in a worker, memory-mapped.

    Everything a chunk needs beyond its own source indices lives here rather
    than in the arguments to each task, and that is load-bearing rather than
    tidiness. A ``pool.submit`` argument is pickled and unpickled *per task, per
    worker*: passing the demand matrix that way gave every one of N workers its
    own full copy, so a zone run -- where the demand is already K^2 times bigger
    for having been spread over access points -- multiplied that by the worker
    count and ran the machine out of memory. Memory-mapping it instead means one
    read-only copy the OS pages in and shares between every worker.
    """
    base = Path(cache_dir)
    indptr = np.load(base / "indptr.npy", mmap_mode="r")
    indices = np.load(base / "indices.npy", mmap_mode="r")
    data = np.load(base / "data.npy", mmap_mode="r")
    _WORKER_STATE["graph"] = csr_matrix(
        (data, indices, indptr), shape=shape, copy=False
    )
    _WORKER_STATE["edge_keys"] = np.load(base / "edge_keys.npy", mmap_mode="r")
    _WORKER_STATE["n_nodes"] = shape[0]
    _WORKER_STATE["n_edges"] = int(indices.shape[0])

    for name in (
        "targets",
        "demand_data",
        "demand_indices",
        "demand_indptr",
        "row_group",
        "col_group",
    ):
        path = base / f"{name}.npy"
        if path.exists():
            _WORKER_STATE[name] = np.load(path, mmap_mode="r")


def _worker_solve(
    sources: np.ndarray,
    row_offset: int,
    max_seconds: float | None,
    n_groups: int | None,
) -> _ChunkResult:
    groups = (
        None
        if n_groups is None
        else (
            cast(np.ndarray, _WORKER_STATE["row_group"]),
            cast(np.ndarray, _WORKER_STATE["col_group"]),
            n_groups,
        )
    )
    return _solve_chunk(
        cast(csr_matrix, _WORKER_STATE["graph"]),
        sources,
        cast(np.ndarray, _WORKER_STATE["targets"]),
        cast(np.ndarray, _WORKER_STATE["demand_data"]),
        cast(np.ndarray, _WORKER_STATE["demand_indices"]),
        cast(np.ndarray, _WORKER_STATE["demand_indptr"]),
        row_offset,
        cast(np.ndarray, _WORKER_STATE["edge_keys"]),
        cast(int, _WORKER_STATE["n_nodes"]),
        max_seconds,
        cast(int, _WORKER_STATE["n_edges"]),
        groups,
    )


def assign_traffic(
    network: RoutableNetwork,
    sources: Sequence[int] | np.ndarray,
    demand: np.ndarray | csr_matrix,
    targets: Sequence[int] | np.ndarray | None = None,
    max_seconds: float | None = None,
    workers: int = 1,
    chunk_size: int | None = None,
    progress_every: int = 10,
) -> np.ndarray:
    """
    All-or-nothing traffic assignment.

    For every OD pair with positive demand, the full demand travels the
    pair's shortest path (by travel time) and is added to every link on it.
    See the module docstring for what that does and doesn't capture.

    Parameters mirror :func:`valma_bike_and_walk.matrix.travel_time_matrix`:
    - sources / targets: node *indices* into ``network`` (see
      ``RoutableNetwork.nearest_nodes``). ``targets`` defaults to ``sources``.
    - demand: dense array or SciPy sparse matrix, shape
      ``(len(sources), len(targets))``. Non-positive entries (including the
      diagonal -- intrazonal trips never touch the network) are ignored.
      Prefer a sparse matrix (see :mod:`valma_bike_and_walk.demand`) once
      ``sources``/``targets`` run into the thousands.
    - max_seconds: stop expanding a source's search beyond this travel time.
      OD pairs further apart come back unreached (and get dropped, like any
      other unreachable pair) rather than assigned. Bounds both time and the
      per-chunk Dijkstra memory, and is the cheapest lever for a
      country-sized network.
    - workers: processes to spread source chunks over. 1 keeps it in-process.
    - chunk_size: sources per Dijkstra call; defaults to a 256 MB block
      (see :func:`choose_chunk_size`).

    Returns a float64 array of shape ``(network.n_edges,)``: total volume on
    every directed edge, positionally aligned with ``network.travel_time``,
    ``network.length`` and ``network.indices`` -- the same alignment
    :func:`valma_bike_and_walk.gpkg.edges_to_geodataframe` uses, so a volume
    column drops straight in (see :func:`link_volume_frame`).
    """
    volume, _ = _assign(
        network,
        sources,
        demand,
        targets,
        max_seconds,
        workers,
        chunk_size,
        progress_every,
        groups=None,
    )
    return volume


def _assign(
    network: RoutableNetwork,
    sources: Sequence[int] | np.ndarray,
    demand: np.ndarray | csr_matrix,
    targets: Sequence[int] | np.ndarray | None,
    max_seconds: float | None,
    workers: int,
    chunk_size: int | None,
    progress_every: int,
    groups: tuple[np.ndarray, np.ndarray, int] | None,
) -> tuple[np.ndarray, csr_matrix | None]:
    """
    The engine behind :func:`assign_traffic`, plus a report of what it could not place.

    ``groups`` is ``(row_group, col_group, n_groups)``: give it a source-point to
    zone map and a target-point to zone map, and the demand that no chunk could
    place -- everything beyond ``max_seconds`` -- comes back as an
    ``(n_groups, n_groups)`` sparse matrix ready to be assigned some other way.
    Without it the second return value is None and nothing extra is tracked.
    """
    sources = np.asarray(sources, dtype=np.int64)
    targets = sources if targets is None else np.asarray(targets, dtype=np.int64)
    empty = np.zeros(network.n_edges, dtype=np.float64)

    if sources.size == 0 or targets.size == 0:
        return empty, None
    if sources.min() < 0 or targets.min() < 0:
        raise ValueError(
            "Source and target indices must be non-negative; unsnapped points must be filtered out first."
        )
    if sources.max() >= network.n_nodes or targets.max() >= network.n_nodes:
        raise ValueError("Node index out of range for this network.")

    demand = _as_demand_csr(demand, sources.size, targets.size)
    if demand.nnz == 0:
        logger.warning("Demand matrix has no positive entries; nothing to assign.")
        return empty, None

    edge_keys = _edge_keys(network)

    if chunk_size is None:
        chunk_size = choose_chunk_size(network.n_nodes)
    offsets = list(range(0, sources.size, chunk_size))
    logger.info(
        "Assigning %d nonzero OD pair(s) from %d source(s) over %d nodes / %d edges: "
        "%d chunks of <=%d sources, %d worker(s)",
        demand.nnz,
        sources.size,
        network.n_nodes,
        network.n_edges,
        len(offsets),
        chunk_size,
        workers,
    )

    link_volume = np.zeros(network.n_edges, dtype=np.float64)
    dropped_total = 0
    n_groups = None if groups is None else groups[2]
    unplaced_parts: list[coo_matrix] = []

    def merge(result: _ChunkResult) -> None:
        nonlocal dropped_total
        idx, values, dropped, report = result
        link_volume[idx] += values
        dropped_total += dropped
        if n_groups is not None and report and report[0].size:
            unplaced_parts.append(
                coo_matrix(
                    (report[2], (report[0], report[1])), shape=(n_groups, n_groups)
                )
            )

    if workers <= 1:
        graph = network.csr()
        for done, start in enumerate(offsets, start=1):
            merge(
                _solve_chunk(
                    graph,
                    sources[start : start + chunk_size],
                    targets,
                    demand.data,
                    demand.indices,
                    demand.indptr,
                    start,
                    edge_keys,
                    network.n_nodes,
                    max_seconds,
                    network.n_edges,
                    groups,
                )
            )
            if progress_every and done % progress_every == 0:
                logger.info("  %d/%d chunks", done, len(offsets))
    else:
        with tempfile.TemporaryDirectory(prefix="valma-assign-") as tmp:
            base = Path(tmp)
            np.save(base / "indptr.npy", network.indptr)
            np.save(base / "indices.npy", network.indices)
            np.save(base / "data.npy", network.travel_time)
            np.save(base / "edge_keys.npy", edge_keys)
            np.save(base / "targets.npy", targets)
            np.save(base / "demand_data.npy", demand.data)
            np.save(base / "demand_indices.npy", demand.indices)
            np.save(base / "demand_indptr.npy", demand.indptr)
            if groups is not None:
                np.save(base / "row_group.npy", groups[0])
                np.save(base / "col_group.npy", groups[1])
            shape = (network.n_nodes, network.n_nodes)

            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(str(base), shape),
            ) as pool:
                # Submit a bounded window rather than every chunk at once. An
                # eager `for start in offsets` comprehension queues one work
                # item per chunk, each holding its arguments and later its
                # result alive until the whole run drains -- which on a zone
                # run, where chunks number in the thousands, is memory spent in
                # proportion to how finely the work was divided. Refilling as
                # results land keeps at most a couple of rounds in flight.
                pending: dict[Future[_ChunkResult], int] = {}
                queue = iter(offsets)

                def submit_next() -> bool:
                    start = next(queue, None)
                    if start is None:
                        return False
                    pending[
                        pool.submit(
                            _worker_solve,
                            sources[start : start + chunk_size],
                            start,
                            max_seconds,
                            n_groups,
                        )
                    ] = start
                    return True

                for _ in range(min(len(offsets), 2 * max(workers, 1))):
                    submit_next()

                done = 0
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        del pending[future]
                        merge(future.result())
                        submit_next()
                        done += 1
                        if progress_every and done % progress_every == 0:
                            logger.info("  %d/%d chunks", done, len(offsets))

    if dropped_total and n_groups is None:
        logger.warning(
            "%d OD pair(s) had demand but no path within reach; dropped from "
            "the assignment.",
            dropped_total,
        )
    elif dropped_total:
        # Not a warning: a caller that asked for the unplaced demand back has
        # somewhere else to send it, so an unreached pair here is the cutoff
        # doing its job, not demand going missing.
        logger.info(
            "%d OD pair(s) were beyond the cutoff; handing them back to the caller.",
            dropped_total,
        )

    if n_groups is None:
        return link_volume, None
    if not unplaced_parts:
        return link_volume, csr_matrix((n_groups, n_groups), dtype=np.float64)

    unplaced = sum(unplaced_parts[1:], unplaced_parts[0]).tocsr()
    unplaced.sum_duplicates()
    return link_volume, unplaced


def _intrazonal_pairs(zones: Zones, intrazonal: np.ndarray) -> coo_matrix:
    """
    Spread each zone's own demand over the ordered pairs of its access points.

    A zone's intrazonal demand is the one part of an OD matrix a single-centroid
    model cannot place at all: origin and destination are the same node, the
    path is empty, and the trips vanish. For walking and cycling that is a large
    share of all travel and precisely the share that belongs on local streets,
    so it is worth getting right.

    Pair ``(a, b)`` takes ``D_ii * w_a * w_b``, restricted to ``a != b`` and
    renormalised by ``1 - sum(w^2)`` so the zone's total demand is preserved
    rather than quietly losing the self-pair share. A zone with a single access
    point has no distinct pair to carry its intrazonal demand; those zones are
    reported and skipped.
    """
    indptr = zones.indptr
    self_weight = zones.self_weight

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    unplaced = 0.0

    for zone in np.flatnonzero(intrazonal > 0):
        lo, hi = int(indptr[zone]), int(indptr[zone + 1])
        scale = 1.0 - float(self_weight[zone])
        if hi - lo < 2 or scale <= 0:
            unplaced += float(intrazonal[zone])
            continue

        points = np.arange(lo, hi)
        weight = zones.weight[lo:hi]
        origin = np.repeat(points, points.shape[0])
        destination = np.tile(points, points.shape[0])
        value = (
            float(intrazonal[zone])
            * np.repeat(weight, weight.shape[0])
            * np.tile(weight, weight.shape[0])
            / scale
        )
        distinct = origin != destination
        rows.append(origin[distinct])
        cols.append(destination[distinct])
        values.append(value[distinct])

    if unplaced > 0:
        logger.warning(
            "%.6g intrazonal demand could not be placed: those zones have only "
            "one access point, so there is no path inside them to load.",
            unplaced,
        )

    n = zones.n_points
    if not rows:
        return coo_matrix((n, n), dtype=np.float64)
    return coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )


def expand_demand(
    demand: np.ndarray | csr_matrix,
    zones: Zones,
    exclude_self_pairs: bool = True,
) -> csr_matrix:
    """
    Split a zone-level OD matrix over the zones' weighted access points.

    Zone pair ``(i, j)`` becomes the ``K_i * K_j`` point pairs between their
    access points, each taking ``D_ij * w_a * w_b`` -- so total demand is
    conserved and every zone's share leaves and arrives spread across its own
    area instead of piling onto one node. That is the whole fix for volumes
    stacking up around a centroid: nothing about the assignment changes, only
    where the trips start and end.

    The interzonal part is two sparse products, ``W @ D @ W.T``. The intrazonal
    part is built separately by :func:`_intrazonal_pairs`, because it needs the
    self-pairs excluded and the rest renormalised.

    Cost is the honest one: nonzeros grow by roughly ``K^2``. At the default K
    of 8 that is 64x more OD pairs, which is why they stay sparse -- and it
    costs no extra shortest-path search, since :func:`assign_traffic` runs one
    Dijkstra per *source*, of which there are ``K`` times as many either way.

    Returns an ``(n_points, n_points)`` CSR matrix ready for
    :func:`assign_traffic` with ``zones.node_index`` as its sources.
    """
    matrix = _as_demand_csr(demand, zones.n_zones, zones.n_zones)
    weights = zones.weight_matrix()

    intrazonal = matrix.diagonal()
    interzonal = (matrix - diags(intrazonal, dtype=np.float64)).tocsr()
    interzonal.eliminate_zeros()

    expanded = (weights @ interzonal @ weights.T).tocsr()
    if intrazonal.any():
        if exclude_self_pairs:
            expanded = (expanded + _intrazonal_pairs(zones, intrazonal)).tocsr()
        else:
            expanded = (
                expanded + weights @ diags(intrazonal, dtype=np.float64) @ weights.T
            ).tocsr()
    expanded.sum_duplicates()
    expanded.eliminate_zeros()

    logger.info(
        "Expanded %d zone OD pair(s) over %d access point(s): %d point pair(s), "
        "%.6g total demand (was %.6g)",
        matrix.nnz,
        zones.n_points,
        expanded.nnz,
        float(expanded.sum()),
        float(matrix.sum()),
    )
    return expanded


def assign_zone_traffic(
    network: RoutableNetwork,
    zones: Zones,
    demand: np.ndarray | csr_matrix,
    max_seconds: float | None = None,
    near_seconds: float | None = None,
    exclude_self_pairs: bool = True,
    workers: int = 1,
    chunk_size: int | None = None,
    progress_every: int = 10,
) -> np.ndarray:
    """
    All-or-nothing assignment of a zone-level demand matrix, distributed over access points.

    ``demand`` is ``(n_zones, n_zones)`` in ``zones.ids`` order; the result is
    the same per-directed-edge volume array :func:`assign_traffic` returns, so
    everything downstream -- :func:`link_volume_frame`, the GeoPackage writer --
    is unchanged.

    Two things this does that a single-centroid assignment cannot. Trips enter
    and leave the network across the whole zone, so volumes near a centroid stop
    being an artefact of where the centroid was put. And intrazonal demand
    actually loads onto links, instead of being dropped for having no path.

    **Two tiers.** Without ``near_seconds`` every access point is routed with an
    unbounded search, which is ``K`` times the work of a single-centroid run.
    With it, the run splits the way
    :func:`valma_bike_and_walk.matrix.zone_travel_time_matrix` does, and for the
    same reason -- spreading trip ends over a zone only changes the route
    materially at short range, and a bounded search is cheap because a
    Dijkstra tree's settled node count grows with the square of its cutoff:

    1. Every access point, bounded at ``near_seconds``. This places all the
       demand whose trips are short enough for the distribution to matter,
       including everything intrazonal.
    2. Whatever tier 1 could not reach -- reported back per zone pair, not
       guessed at from a distance threshold -- assigned from one representative
       point per zone, bounded at ``max_seconds``.

    Because tier 2 works from what tier 1 actually failed to place, no demand
    falls between the two: a pair is either loaded distributed, loaded from the
    representative points, or genuinely unreachable and dropped with a warning,
    exactly as in a single-tier run.

    What neither tier models is the access leg itself: the walk from a building
    to the nearest network node is charged as *time* in the travel-time matrix
    but has no link to be loaded onto, so it contributes no volume. That is the
    right answer for a leg that is off-network by construction.
    """
    expanded = expand_demand(demand, zones, exclude_self_pairs=exclude_self_pairs)

    if near_seconds is None:
        volume, _ = _assign(
            network,
            zones.node_index,
            expanded,
            None,
            max_seconds,
            workers,
            chunk_size,
            progress_every,
            groups=None,
        )
        return volume

    zone_of_point = zones.zone_of_point
    logger.info(
        "Near tier: every access point within %.0f s; far tier: whatever is left, "
        "from one point per zone",
        near_seconds,
    )
    volume, unplaced = _assign(
        network,
        zones.node_index,
        expanded,
        None,
        near_seconds,
        workers,
        chunk_size,
        progress_every,
        groups=(zone_of_point, zone_of_point, zones.n_zones),
    )
    del expanded

    assert unplaced is not None

    # Intrazonal demand cannot fall through to the far tier: both ends would be
    # the zone's one representative node, so there is no path and the trips
    # would vanish without so much as a count. If a zone is wider than the near
    # cutoff, the cutoff is the thing to fix.
    stranded = unplaced.diagonal()
    if stranded.any():
        logger.warning(
            "%.6g intrazonal demand across %d zone(s) did not fit inside the "
            "near cutoff of %.0f s. Those zones are wider than the cutoff, and "
            "the far tier cannot place intrazonal trips (both ends are the same "
            "representative node), so this demand is unassigned. Raise the near "
            "cutoff to cover them.",
            float(stranded.sum()),
            int((stranded > 0).sum()),
            near_seconds,
        )
        unplaced = (unplaced - diags(stranded, dtype=np.float64)).tocsr()
        unplaced.eliminate_zeros()

    if unplaced.nnz == 0:
        logger.info("Near tier placed every interzonal OD pair; no far tier needed.")
        return volume

    logger.info(
        "Far tier: %d zone pair(s) carrying %.6g demand were beyond the near "
        "cutoff; assigning them between representative points",
        unplaced.nnz,
        float(unplaced.sum()),
    )
    far, _ = _assign(
        network,
        zones.representative.node_index,
        unplaced,
        None,
        max_seconds,
        workers,
        chunk_size,
        progress_every,
        groups=None,
    )
    return volume + far


def link_volume_frame(
    network: RoutableNetwork, link_volume: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Per-edge arrays -- OSM endpoint ids, length, time, assigned volume.

    Aligned the same way as :func:`valma_bike_and_walk.gpkg.edges_to_geodataframe`,
    so this dict's ``"volume"`` array can be passed straight to that
    function's ``extra_columns`` to draw volumes on the network's geometry.
    """
    rows = np.repeat(np.arange(network.n_nodes), np.diff(network.indptr))
    cols = network.indices
    return {
        "u": network.node_ids[rows],
        "v": network.node_ids[cols],
        "length_m": network.length.astype(np.float64),
        "travel_time_s": network.travel_time,
        "volume": link_volume,
    }


def summarise(link_volume: np.ndarray) -> dict[str, float]:
    """Quick sanity statistics for an assignment result."""
    used = link_volume[link_volume > 0]
    return {
        "edges": float(link_volume.size),
        "loaded_edges": float(used.size),
        "loaded_fraction": (
            float(used.size / link_volume.size) if link_volume.size else float("nan")
        ),
        "total_volume": float(link_volume.sum()),
        "max_volume": float(used.max()) if used.size else float("nan"),
        "median_loaded_volume": float(np.median(used)) if used.size else float("nan"),
    }
