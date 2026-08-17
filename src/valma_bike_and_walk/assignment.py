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
(``as_completed``, not submission order) rather than being collected as a
full network-sized array per chunk and held until the whole run finishes --
important once there are enough sources that "one array per chunk" stops
being a rounding error next to the network itself.
"""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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
) -> int:
    """
    All-or-nothing load of one source's nonzero demand onto ``link_volume``.

    Walks every destination back toward the source one hop at a time, all
    destinations in lockstep, adding that destination's (fixed) demand onto
    every link crossed. Destinations that share the tail of their shortest
    path -- the usual case near the source, where many trees converge -- walk
    that shared tail together in the same vectorised step, rather than each
    retracing it independently.

    ``node``/``load`` are this source's nonzero demand: destination node
    indices and the (positive) demand to each. Returns how many of them had
    no path from ``source`` and were dropped.
    """
    if node.size == 0:
        return 0

    node = node.astype(np.int64).copy()
    load = load.astype(np.float64)

    at_source = node == source
    unreachable = (predecessors[node] == _NO_PREDECESSOR) & ~at_source
    keep = ~at_source & ~unreachable
    dropped = int(unreachable.sum())

    node, load = node[keep], load[keep]
    if node.size == 0:
        return dropped

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

    return dropped


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


def _solve_chunk(
    graph: csr_matrix,
    sources: np.ndarray,
    targets: np.ndarray,
    demand: csr_matrix,
    row_offset: int,
    edge_keys: np.ndarray,
    n_nodes: int,
    max_seconds: float | None,
    n_edges: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    One chunk of sources: Dijkstra, then load each row's demand.

    Returns a *sparse* update -- ``(edge_index, volume, dropped)`` for only
    the edges this chunk actually touched -- rather than a full
    ``n_edges``-long array. A chunk of even a few dozen sources only ever
    touches a small fraction of a country-sized network's edges, so handing
    back (and, across the process-pool path, pickling) a whole network-sized
    array per chunk wastes memory and IPC bandwidth in direct proportion to
    the number of chunks -- which is exactly backwards, since chunking
    exists to keep memory *down*. See :func:`assign_traffic`'s merge step.
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
    for row, source in enumerate(sources):
        lo, hi = demand.indptr[row_offset + row], demand.indptr[row_offset + row + 1]
        node = targets[demand.indices[lo:hi]]
        load = demand.data[lo:hi]
        dropped += _load_source(
            int(source), pred[row], node, load, edge_keys, n_nodes, link_volume
        )
    touched = np.flatnonzero(link_volume)
    return touched, link_volume[touched], dropped


def _init_worker(cache_dir: str, shape: tuple[int, int]) -> None:
    """Rebuild the CSR graph and edge-key lookup in a worker, memory-mapped (see matrix.py)."""
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


def _worker_solve(
    sources: np.ndarray,
    targets: np.ndarray,
    demand: csr_matrix,
    row_offset: int,
    max_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    return _solve_chunk(
        _WORKER_STATE["graph"],
        sources,
        targets,
        demand,
        row_offset,
        cast(np.ndarray, _WORKER_STATE["edge_keys"]),
        cast(int, _WORKER_STATE["n_nodes"]),
        max_seconds,
        cast(int, _WORKER_STATE["n_edges"]),
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
    sources = np.asarray(sources, dtype=np.int64)
    targets = sources if targets is None else np.asarray(targets, dtype=np.int64)

    if sources.size == 0 or targets.size == 0:
        return np.zeros(network.n_edges, dtype=np.float64)
    if sources.min() < 0 or targets.min() < 0:
        raise ValueError(
            "Source and target indices must be non-negative; unsnapped points must be filtered out first."
        )
    if sources.max() >= network.n_nodes or targets.max() >= network.n_nodes:
        raise ValueError("Node index out of range for this network.")

    demand = _as_demand_csr(demand, sources.size, targets.size)
    if demand.nnz == 0:
        logger.warning("Demand matrix has no positive entries; nothing to assign.")
        return np.zeros(network.n_edges, dtype=np.float64)

    edge_keys = _edge_keys(network)

    if chunk_size is None:
        chunk_size = choose_chunk_size(network.n_nodes)
    n_chunks = int(np.ceil(sources.size / chunk_size))
    logger.info(
        "Assigning %d nonzero OD pair(s) from %d source(s) over %d nodes / %d edges: "
        "%d chunks of <=%d sources, %d worker(s)",
        demand.nnz,
        sources.size,
        network.n_nodes,
        network.n_edges,
        n_chunks,
        chunk_size,
        workers,
    )

    link_volume = np.zeros(network.n_edges, dtype=np.float64)
    dropped_total = 0

    if workers <= 1:
        graph = network.csr()
        for i, chunk in enumerate(_chunks(sources, chunk_size)):
            start = i * chunk_size
            idx, vals, dropped = _solve_chunk(
                graph,
                chunk,
                targets,
                demand,
                start,
                edge_keys,
                network.n_nodes,
                max_seconds,
                network.n_edges,
            )
            link_volume[idx] += vals
            dropped_total += dropped
            if progress_every and (i + 1) % progress_every == 0:
                logger.info("  %d/%d chunks", i + 1, n_chunks)
    else:
        with tempfile.TemporaryDirectory(prefix="valma-assign-") as tmp:
            base = Path(tmp)
            np.save(base / "indptr.npy", network.indptr)
            np.save(base / "indices.npy", network.indices)
            np.save(base / "data.npy", network.travel_time)
            np.save(base / "edge_keys.npy", edge_keys)
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
                        demand,
                        start,
                        max_seconds,
                    ): start
                    for start in offsets
                }
                # as_completed + popping each future as it's consumed matters
                # as much as the sparse return above: `futures` would
                # otherwise hold every finished future -- and the (large,
                # even if now sparse) result inside it -- alive until the
                # very last one completes, for no reason once we've already
                # merged it into link_volume.
                done = 0
                for future in as_completed(futures):
                    idx, vals, dropped = future.result()
                    link_volume[idx] += vals
                    dropped_total += dropped
                    del futures[future]
                    done += 1
                    if progress_every and done % progress_every == 0:
                        logger.info("  %d/%d chunks", done, n_chunks)

    if dropped_total:
        logger.warning(
            "%d OD pair(s) had demand but no path within reach; dropped from the assignment.",
            dropped_total,
        )
    return link_volume


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
    exclude_self_pairs: bool = True,
    workers: int = 1,
    chunk_size: int | None = None,
    progress_every: int = 10,
) -> np.ndarray:
    """
    All-or-nothing assignment of a zone-level demand matrix, distributed over access points.

    :func:`expand_demand` followed by :func:`assign_traffic`. ``demand`` is
    ``(n_zones, n_zones)`` in ``zones.ids`` order; the result is the same
    per-directed-edge volume array :func:`assign_traffic` returns, so everything
    downstream -- :func:`link_volume_frame`, the GeoPackage writer -- is
    unchanged.

    Two things this does that a single-centroid assignment cannot. Trips enter
    and leave the network across the whole zone, so volumes near a centroid stop
    being an artefact of where the centroid was put. And intrazonal demand
    actually loads onto links, instead of being dropped for having no path.

    What it still does not model is the access leg itself: the walk from a
    building to the nearest network node is charged as *time* in the travel-time
    matrix but has no link to be loaded onto, so it contributes no volume. That
    is the right answer for a leg that is off-network by construction.
    """
    return assign_traffic(
        network,
        zones.node_index,
        expand_demand(demand, zones, exclude_self_pairs=exclude_self_pairs),
        max_seconds=max_seconds,
        workers=workers,
        chunk_size=chunk_size,
        progress_every=progress_every,
    )


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
