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

The network side is :func:`valma_bike_and_walk.routing.aon`: one bounded
search per source, which stops once every destination with demand from that
source has been settled, followed by a single pass over the settled nodes,
farthest first, that pushes each node's accumulated demand onto the edge it was
reached by. Every path from a source is loaded in that one pass, so the cost is
proportional to the size of the tree rather than to the number of destinations
times the length of their paths.

Workers are threads sharing one copy of the graph and of the demand; each keeps
its own volume array, summed once at the end. Any demand a bounded search could
not place comes back as a mask over the demand's nonzeros, which
:func:`_assign` turns into a zone-level matrix for the caller to route another
way (see :func:`assign_zone_traffic`'s two tiers).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, issparse

from valma_bike_and_walk import routing
from valma_bike_and_walk.network import RoutableNetwork
from valma_bike_and_walk.zones import Zones

logger = logging.getLogger(__name__)


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
      other unreachable pair) rather than assigned. The cheapest lever for a
      country-sized network.
    - workers: threads to spread the sources over.
    - chunk_size: sources per task handed to a thread; sized automatically.

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
    zone map and a target-point to zone map, and the demand that could not be
    placed -- everything beyond ``max_seconds`` -- comes back as an
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

    logger.info(
        "Assigning %d nonzero OD pair(s) from %d source(s) over %d nodes / %d edges, "
        "%d worker(s)",
        demand.nnz,
        sources.size,
        network.n_nodes,
        network.n_edges,
        workers,
    )
    per_thread = routing.workspace_bytes(network.n_nodes, network.n_edges, flow=True)
    logger.info(
        "Routing scratch space: %.0f MB per worker, %.1f GB for %d worker(s)",
        per_thread / 1e6,
        per_thread * max(1, workers) / 1e9,
        max(1, workers),
    )

    link_volume, missed = routing.aon(
        network.routing_graph(),
        sources,
        targets,
        demand.indptr,
        demand.indices,
        demand.data,
        limit=max_seconds,
        workers=workers,
        chunk_size=chunk_size,
        progress_every=progress_every,
    )
    dropped = int(missed.sum())

    if dropped and groups is None:
        logger.warning(
            "%d OD pair(s) had demand but no path within reach; dropped from "
            "the assignment.",
            dropped,
        )
    elif dropped:
        # Not a warning: a caller that asked for the unplaced demand back has
        # somewhere else to send it, so an unreached pair here is the cutoff
        # doing its job, not demand going missing.
        logger.info(
            "%d OD pair(s) were beyond the cutoff; handing them back to the caller.",
            dropped,
        )

    if groups is None:
        return link_volume, None

    row_group, col_group, n_groups = groups
    position = np.flatnonzero(missed)
    rows = np.repeat(np.arange(sources.size), np.diff(demand.indptr))[position]
    unplaced = coo_matrix(
        (
            demand.data[position],
            (row_group[rows], col_group[demand.indices[position]]),
        ),
        shape=(n_groups, n_groups),
    ).tocsr()
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
