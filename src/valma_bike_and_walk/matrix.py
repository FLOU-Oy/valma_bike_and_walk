"""Origin-destination travel-time matrices.

An OD matrix over many origins is a sequence of single-source shortest-path
searches, one per origin. The searches themselves are the compiled kernels in
:mod:`valma_bike_and_walk.routing`; this module decides what to search from,
how far, and what a row of the result means.

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
from typing import Sequence

import numpy as np
import numpy.typing as npt

from valma_bike_and_walk import routing
from valma_bike_and_walk.network import RoutableNetwork
from valma_bike_and_walk.speeds import SECONDS_PER_HOUR, profile_for
from valma_bike_and_walk.zones import Zones

logger = logging.getLogger(__name__)

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


def _log_workspace(network: RoutableNetwork, workers: int, flow: bool = False) -> None:
    per_thread = routing.workspace_bytes(network.n_nodes, network.n_edges, flow=flow)
    logger.info(
        "Routing scratch space: %.0f MB per worker, %.1f GB for %d worker(s)",
        per_thread / 1e6,
        per_thread * max(1, workers) / 1e9,
        max(1, workers),
    )


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
    - workers: threads to spread the sources over.
    - chunk_size: sources per task handed to a thread; sized automatically.

    Returns an ``(len(sources), len(targets))`` array; unreachable pairs are inf.
    """
    sources = np.asarray(sources, dtype=np.int64)
    targets = sources if targets is None else np.asarray(targets, dtype=np.int64)
    dtype = np.dtype(dtype)

    if sources.size == 0 or targets.size == 0:
        return np.empty((sources.size, targets.size), dtype=dtype)
    if sources.min() < 0 or targets.min() < 0:
        raise ValueError(
            "Source and target indices must be non-negative; unsnapped points must be filtered out first."
        )
    if sources.max() >= network.n_nodes or targets.max() >= network.n_nodes:
        raise ValueError("Node index out of range for this network.")

    logger.info(
        "OD matrix %d x %d over %d nodes, %d worker(s)",
        sources.size,
        targets.size,
        network.n_nodes,
        workers,
    )
    _log_workspace(network, workers)

    result = np.empty((sources.size, targets.size), dtype=dtype)
    routing.point_matrix(
        network.routing_graph(),
        sources,
        targets,
        result,
        limit=max_seconds,
        workers=workers,
        chunk_size=chunk_size,
        progress_every=progress_every,
    )
    return result


def _accumulate_zone_matrix(
    network: RoutableNetwork,
    zones: Zones,
    max_seconds: float | None,
    decay_mu: float | None,
    workers: int,
    chunk_size: int | None,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run every zone's access points as sources and sum their weighted terms.

    Returns the numerator and denominator of the weighted mean, both
    ``(n_zones, n_zones)``, rather than the mean itself: the denominator *is*
    the weight of the access-point pairs that came back inside the cutoff,
    which is what the reachability bookkeeping needs.
    """
    n = zones.n_zones
    numerator = np.zeros((n, n), dtype=np.float64)
    denominator = np.zeros((n, n), dtype=np.float64)

    points_per_task = routing.task_size(zones.n_points, workers, chunk_size)
    ranges = list(zones.chunk_ranges(points_per_task))
    logger.info(
        "Zone OD matrix %d x %d from %d access point(s) over %d nodes: "
        "%d chunks of <=%d points, %d worker(s)",
        n,
        n,
        zones.n_points,
        network.n_nodes,
        len(ranges),
        points_per_task,
        workers,
    )
    _log_workspace(network, workers)

    routing.zone_sums(
        network.routing_graph(),
        routing.AccessPoints(
            indptr=zones.indptr,
            node=zones.node_index,
            access=zones.access_seconds,
            weight=zones.weight,
        ),
        ranges,
        numerator,
        denominator,
        limit=max_seconds,
        decay_mu=decay_mu,
        workers=workers,
        progress_every=progress_every,
    )
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
