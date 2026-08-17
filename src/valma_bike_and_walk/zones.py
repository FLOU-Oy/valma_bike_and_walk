"""Zones as polygons, represented by several weighted access points each.

Routing every trip from one centroid node per zone is the classic simplification,
and it has two well-known costs. Volumes pile up implausibly on whatever links
happen to meet the centroid node, and short trips -- within a zone, or between
neighbouring ones -- come out wrong by however far the centroid happens to sit
from where people actually are. Intrazonal trips come out as exactly zero, which
for walking and cycling throws away a large share of all travel.

The fix is standard practice in the big modelling packages under the name
*multiple centroid connectors*: represent a zone by K weighted points instead of
one, route between the points, and aggregate back to zone level with those
weights. This module builds that representation;
:func:`valma_bike_and_walk.matrix.zone_travel_time_matrix` and
:func:`valma_bike_and_walk.assignment.assign_zone_traffic` consume it.

**How wrong is one point, and where.** Write a trip as running from ``c_i + u``
to ``c_j + v``, where ``c`` are the zone centroids and ``u``/``v`` are the
within-zone offsets of the real origin and destination. Expanding
``|d + v - u|`` around ``d = |c_i - c_j|``, the linear term averages to zero by
symmetry and what survives is

    E|a - b| ~ d + E[perpendicular offset^2] / 2d = d + O(R^2 / d)

for a zone of characteristic radius ``R``. So the centroid approximation always
*underestimates* mean travel time, and its relative error falls as ``R^2/2d^2``
-- second order, not first. At ``d = 5R`` it is about 2 %; at ``d = 2R`` it is
12 %; at ``d = 0`` it is unbounded. That is the whole story in one line: the
error is concentrated in short trips, which is exactly where a walking or
cycling model spends its attention, and it is why the two-tier scheme in
:func:`valma_bike_and_walk.matrix.zone_travel_time_matrix` can afford to fall
back to a single point beyond a few zone radii.

**Where the points come from.** Two sources, both handled here:

- ``weights_path``: any point (or polygon, reduced to representative points)
  layer -- building centroids, a population grid converted to points, workplace
  registers -- with an optional column giving each point's weight. This is the
  accurate option when the data exists.
- the **network itself**, the default. Street density is a serviceable proxy for
  where people are and needs no extra data source at all: every network node in
  the zone is a candidate, weighted by the length of link attached to it. Length
  rather than node count matters here, because this project deliberately splits
  links at tag changes as well as at junctions (see the README), so a node count
  measures how finely OSM is tagged as much as how dense the streets are.

**Reducing to K.** However many candidates a zone has, they are reduced to at
most ``points_per_zone`` by laying a grid over the zone, coarsening it until at
most K cells are occupied, and taking each occupied cell's weighted centroid.
That is stratified sampling rather than random sampling: it is deterministic
(no seed to record, no run-to-run drift), it spreads points over the zone by
construction, and it preserves the weighted mean position exactly -- so ``K=1``
falls out of the same code as the weighted centroid, and the representative
point of a zone is the same whatever ``K`` is.

**Access time.** Snapping teleports a point to its nearest node for free, which
at walking pace can quietly gift a trip twelve minutes over the 1000 m default
snapping limit. Each access point therefore carries the time to cover its own
snap distance, added at both ends of every trip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix

from valma_bike_and_walk.centroids import (
    DEFAULT_MAX_SNAP_M,
    Centroids,
    normalise_ids,
)
from valma_bike_and_walk.config import PROJECTED_CRS, WGS84
from valma_bike_and_walk.network import RoutableNetwork
from valma_bike_and_walk.speeds import SECONDS_PER_HOUR, profile_for

logger = logging.getLogger(__name__)

#: Access points per zone. Error falls off fast in K -- as 1/sqrt(K) for random
#: placement, faster for the stratified placement used here -- so this is
#: normally indistinguishable from a much larger number, at a fraction of the
#: routing cost (which is linear in K on the origin side).
DEFAULT_POINTS_PER_ZONE = 8

#: Points per batch when testing which zone each candidate falls in. Building
#: shapely geometries for a country's three million network nodes at once costs
#: more memory than the network does; in batches it costs nothing.
_JOIN_CHUNK = 200_000

#: Safety valve on the grid-coarsening loop in :func:`_reduce_points`.
_MAX_COARSEN_STEPS = 64


def _group_starts(counts: np.ndarray) -> np.ndarray:
    """Group start offsets for ``np.add.reduceat``, from per-group counts."""
    starts = np.zeros(counts.shape[0], dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    return starts


@dataclass
class Zones:
    """
    Zones, each represented by one or more weighted access points.

    Points are stored flat and sorted by zone, with ``indptr`` delimiting each
    zone's block exactly the way a CSR matrix delimits its rows -- so a zone's
    points are ``node_index[indptr[i]:indptr[i + 1]]``, always contiguous, and
    aggregating a per-point quantity up to zone level is one ``reduceat``.

    ``weight`` sums to 1 within every zone. ``representative`` holds the
    single-point view of the same zones (the weighted mean position, snapped),
    positionally aligned with ``ids`` -- that is what the far tier of a two-tier
    matrix routes from, and what keeps "one point per zone" available as a
    method rather than as a separate code path.
    """

    ids: np.ndarray  # (n_zones,)
    indptr: np.ndarray  # (n_zones + 1,) into the per-point arrays
    node_index: np.ndarray  # (n_points,) network node index, always >= 0
    weight: np.ndarray  # (n_points,) sums to 1 within each zone
    access_seconds: np.ndarray  # (n_points,) time to cover the snap distance
    x: np.ndarray  # (n_points,) PROJECTED_CRS metres
    y: np.ndarray
    area_m2: np.ndarray  # (n_zones,) 0 for zones given as points
    representative: Centroids  # one point per zone, aligned with ids
    representative_access_seconds: np.ndarray  # (n_zones,)

    @property
    def n_zones(self) -> int:
        return int(self.ids.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.node_index.shape[0])

    def __len__(self) -> int:
        return self.n_zones

    @property
    def points_per_zone(self) -> np.ndarray:
        return np.diff(self.indptr)

    @property
    def zone_of_point(self) -> np.ndarray:
        """(n_points,) the zone each point belongs to, non-decreasing."""
        return np.repeat(np.arange(self.n_zones), self.points_per_zone)

    @property
    def self_weight(self) -> np.ndarray:
        """
        (n_zones,) sum of squared point weights -- the weight of the a==b pairs.

        Those pairs are an artefact of representing a continuous zone by a finite
        set of points: two people drawn independently from the same zone land on
        the same point with probability zero in reality, but with probability
        ``sum(w^2)`` here, and the "trip" between them is not a trip. Both the
        intrazonal travel time and the intrazonal assignment discount them by
        exactly this much.
        """
        return np.add.reduceat(self.weight**2, self.indptr[:-1])

    def weight_matrix(self) -> csr_matrix:
        """
        ``(n_points, n_zones)`` sparse map from zone quantities to point ones.

        One nonzero per row -- each point belongs to exactly one zone -- so
        ``W @ demand @ W.T`` splits a zone-level OD matrix over access points in
        two sparse products. See
        :func:`valma_bike_and_walk.assignment.expand_demand`.
        """
        return csr_matrix(
            (
                self.weight,
                self.zone_of_point,
                np.arange(self.n_points + 1, dtype=np.int64),
            ),
            shape=(self.n_points, self.n_zones),
        )

    def chunk_ranges(self, max_points: int) -> Iterator[tuple[int, int]]:
        """
        Split the zones into contiguous ``[lo, hi)`` runs of at most ``max_points``.

        Routing chunks have to hold whole zones: a zone's rows are aggregated
        together, so splitting one across two Dijkstra chunks would mean carrying
        a partial numerator and denominator between them for no gain. A zone with
        more points than the budget still gets its own chunk rather than being
        split -- the budget is a memory target, not a hard limit.
        """
        counts = self.points_per_zone
        lo = 0
        while lo < self.n_zones:
            hi, total = lo + 1, int(counts[lo])
            while hi < self.n_zones and total + int(counts[hi]) <= max_points:
                total += int(counts[hi])
                hi += 1
            yield lo, hi
            lo = hi

    def equivalent_radius_m(self) -> np.ndarray:
        """
        (n_zones,) radius of the circle with the same area as each zone.

        The scale ``R`` the module docstring's ``R^2/2d^2`` error bound is
        written in, and so the natural unit for choosing how far out the
        multi-point tier needs to reach.
        """
        return np.sqrt(np.maximum(self.area_m2, 0.0) / np.pi)

    def summarise(self) -> dict[str, float]:
        """Quick sanity statistics, for the CLI to print after loading."""
        counts = self.points_per_zone
        access = self.access_seconds
        return {
            "zones": float(self.n_zones),
            "access_points": float(self.n_points),
            "points_per_zone_mean": float(counts.mean()) if counts.size else 0.0,
            "points_per_zone_max": float(counts.max()) if counts.size else 0.0,
            "access_seconds_median": float(np.median(access)) if access.size else 0.0,
            "access_seconds_max": float(access.max()) if access.size else 0.0,
        }


def read_zones(
    path: Path,
    id_column: str | None = None,
    crs: str = WGS84,
) -> tuple[np.ndarray, gpd.GeoSeries]:
    """
    Read a zone layer, projected to :data:`PROJECTED_CRS`.

    Any vector format GeoPandas reads works. Polygons are the point of the
    exercise, but points are accepted too and behave as one-point zones -- so an
    existing centroid file can be handed to ``--zones`` unchanged, and differs
    from a ``--centroids`` run only in that its snap distance is now charged for.
    """
    frame = gpd.read_file(Path(path))
    if frame.crs is None:
        frame = frame.set_crs(crs)
    frame = frame.to_crs(PROJECTED_CRS)

    if id_column is not None:
        if id_column not in frame.columns:
            raise ValueError(f"{path} has no id column {id_column!r}.")
        ids = normalise_ids(frame[id_column].to_numpy())
    else:
        ids = np.arange(len(frame))

    return ids, frame.geometry


def read_weight_points(
    path: Path,
    weight_column: str | None = None,
    x_column: str = "lon",
    y_column: str = "lat",
    crs: str = WGS84,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the point layer that says where demand actually sits inside a zone.

    Buildings, address points, a population or workplace grid converted to
    points -- anything with a position and, optionally, a size. Polygons are
    reduced to representative points, so a building footprint layer can be handed
    over as-is. Without ``weight_column`` every point counts equally, which is
    right for address points and wrong for a population grid.

    Returns (x, y, weight) with coordinates in :data:`PROJECTED_CRS` metres.
    """
    path = Path(path)

    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        frame = pd.read_csv(path, sep=None, engine="python")
        missing = [c for c in (x_column, y_column) if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path} has no column(s) {missing}; available: {list(frame.columns)}"
            )
        points = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame[x_column], frame[y_column]),
            crs=crs,
        )
    else:
        points = gpd.read_file(path)
        if points.crs is None:
            points = points.set_crs(crs)

    points = points.to_crs(PROJECTED_CRS)

    if weight_column is not None:
        if weight_column not in points.columns:
            raise ValueError(
                f"{path} has no weight column {weight_column!r}; "
                f"available: {list(points.columns)}"
            )
        weight = points[weight_column].to_numpy(dtype=float)
        weight = np.clip(np.nan_to_num(weight, nan=0.0, posinf=0.0), 0.0, None)
    else:
        weight = np.ones(len(points), dtype=float)

    geometry = points.geometry
    if not (geometry.geom_type == "Point").all():
        geometry = geometry.representative_point()

    logger.info(
        "Read %d weight point(s) from %s (total weight %.6g)",
        len(points),
        path,
        float(weight.sum()),
    )
    return geometry.x.to_numpy(dtype=float), geometry.y.to_numpy(dtype=float), weight


def network_node_weights(
    network: RoutableNetwork,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Every network node as a candidate demand point, weighted by street length.

    The zero-extra-data option: street density stands in for population density.
    A node's weight is half the length of link meeting it, counting both the
    edges leaving it and the edges arriving, so a one-way link still contributes
    at both of its ends.

    Weighting by length rather than by node count is deliberate. This project
    splits a way wherever its tags change as well as at junctions, so node counts
    partly measure how finely OSM has been tagged -- a stretch of road tagged in
    five pieces would otherwise count five times as much as the same road tagged
    in one. Length is invariant to that. Only relative weights within a zone
    matter, since they are normalised per zone afterwards.
    """
    rows = np.repeat(np.arange(network.n_nodes), np.diff(network.indptr))
    length = network.length.astype(np.float64)
    weight = np.bincount(rows, weights=length, minlength=network.n_nodes)
    weight += np.bincount(network.indices, weights=length, minlength=network.n_nodes)
    return network.x, network.y, 0.5 * weight


def _zone_of_points(
    x: np.ndarray, y: np.ndarray, geometry: gpd.GeoSeries
) -> np.ndarray:
    """
    Which zone each candidate point falls in; -1 for none.

    ``intersects`` rather than ``within``, so a node sitting exactly on a shared
    boundary -- the normal case for a network node on a zone edge -- lands in a
    zone instead of nowhere. Where zones overlap, the lowest-numbered one wins;
    that is arbitrary, but a point genuinely in two zones is a problem for the
    zone layer to fix rather than something to model here.
    """
    tree = shapely.STRtree(geometry.to_numpy())
    out = np.full(x.shape[0], -1, dtype=np.int64)

    for start in range(0, x.shape[0], _JOIN_CHUNK):
        stop = min(start + _JOIN_CHUNK, x.shape[0])
        points = shapely.points(x[start:stop], y[start:stop])
        point_pos, zone_pos = tree.query(points, predicate="intersects")
        if point_pos.size == 0:
            continue
        order = np.lexsort((zone_pos, point_pos))
        point_pos, zone_pos = point_pos[order], zone_pos[order]
        first = np.ones(point_pos.shape[0], dtype=bool)
        first[1:] = point_pos[1:] != point_pos[:-1]
        out[start + point_pos[first]] = zone_pos[first]

    return out


def _reduce_points(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce one zone's candidates to at most ``k`` weighted points.

    Lay a square grid over the candidates, coarsen it until at most ``k`` cells
    are occupied, then collapse each occupied cell to its weighted centroid.
    Deterministic, and it preserves the weighted mean position exactly: summing
    ``w_c * centroid_c`` over cells gives back ``sum(w_i * x_i)``.
    """
    if x.shape[0] <= k:
        return x, y, weight

    if weight.sum() <= 0:  # no usable weights: count every candidate equally
        weight = np.ones_like(weight)

    x0, y0 = float(x.min()), float(y.min())
    span = max(float(x.max()) - x0, float(y.max()) - y0)
    if span <= 0:  # every candidate at the same spot
        return x[:1].copy(), y[:1].copy(), np.array([float(weight.sum())])

    cell = span / np.sqrt(k)
    inverse = np.zeros(x.shape[0], dtype=np.int64)
    n_cells = x.shape[0]
    for _ in range(_MAX_COARSEN_STEPS):
        columns = np.floor((x - x0) / cell).astype(np.int64)
        rows = np.floor((y - y0) / cell).astype(np.int64)
        _, inverse = np.unique(
            rows * (int(columns.max()) + 2) + columns, return_inverse=True
        )
        inverse = np.ravel(inverse)
        n_cells = int(inverse.max()) + 1
        if n_cells <= k:
            break
        cell *= 1.3

    cell_weight = np.bincount(inverse, weights=weight, minlength=n_cells)
    safe = np.where(cell_weight > 0, cell_weight, 1.0)
    cell_x = np.bincount(inverse, weights=weight * x, minlength=n_cells) / safe
    cell_y = np.bincount(inverse, weights=weight * y, minlength=n_cells) / safe

    if n_cells > k:
        # Coarsening ran out of steps (pathological spacing). Keep the heaviest
        # cells; weights are renormalised per zone afterwards regardless.
        keep = np.sort(np.argsort(cell_weight)[::-1][:k])
        cell_x, cell_y, cell_weight = cell_x[keep], cell_y[keep], cell_weight[keep]

    return cell_x, cell_y, cell_weight


def _place_access_points(
    geometry: gpd.GeoSeries,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    candidate_weight: np.ndarray,
    points_per_zone: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Place each zone's access points. Returns (zone_index, x, y, weight).

    Weights come back normalised to sum to 1 within every zone. A zone with no
    candidate inside it falls back to its own representative point, so no zone is
    lost for want of a building or a network node.
    """
    n_zones = len(geometry)
    zone_of = _zone_of_points(candidate_x, candidate_y, geometry)
    inside = np.flatnonzero(zone_of >= 0)
    logger.info(
        "%d of %d candidate point(s) fall inside a zone",
        inside.shape[0],
        zone_of.shape[0],
    )

    # Group candidates by zone once, so the per-zone loop below is pure slicing.
    inside = inside[np.argsort(zone_of[inside], kind="stable")]
    counts = np.bincount(zone_of[inside], minlength=n_zones)
    starts = np.zeros(n_zones + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    fallback = geometry.representative_point()
    fallback_x = fallback.x.to_numpy(dtype=float)
    fallback_y = fallback.y.to_numpy(dtype=float)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    per_zone = np.zeros(n_zones, dtype=np.int64)
    empty = 0

    for zone in range(n_zones):
        block = inside[starts[zone] : starts[zone + 1]]
        if block.size == 0:
            zx = fallback_x[zone : zone + 1]
            zy = fallback_y[zone : zone + 1]
            zw = np.ones(1)
            empty += 1
        else:
            zx, zy, zw = _reduce_points(
                candidate_x[block],
                candidate_y[block],
                candidate_weight[block],
                points_per_zone,
            )
            if zw.sum() <= 0:
                zw = np.ones_like(zw)
        xs.append(np.asarray(zx, dtype=float))
        ys.append(np.asarray(zy, dtype=float))
        ws.append(np.asarray(zw, dtype=float) / float(np.sum(zw)))
        per_zone[zone] = len(zx)

        if n_zones >= 2000 and (zone + 1) % 2000 == 0:
            logger.info("  placed access points for %d/%d zones", zone + 1, n_zones)

    if empty:
        logger.warning(
            "%d zone(s) had no candidate demand point inside them; using their own "
            "representative point instead",
            empty,
        )

    return (
        np.repeat(np.arange(n_zones), per_zone),
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ws),
    )


def _merge_shared_nodes(
    zone_index: np.ndarray,
    node_index: np.ndarray,
    weight: np.ndarray,
    access: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Collapse access points of one zone that snapped to the same network node.

    Common in a small zone, or wherever the reduction grid came out finer than
    the network. Two rows on the same node in the same zone route identically, so
    merging them saves a whole Dijkstra each and changes no result: the merged
    point carries the summed weight and the weighted mean of everything else.
    """
    if node_index.size == 0:
        return zone_index, node_index, weight, access, x, y

    order = np.lexsort((node_index, zone_index))
    zone_index, node_index = zone_index[order], node_index[order]
    weight, access = weight[order], access[order]
    x, y = x[order], y[order]

    key = zone_index * (int(node_index.max()) + 1) + node_index
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    inverse = np.ravel(inverse)
    n_points = first.shape[0]
    if n_points == key.shape[0]:
        return zone_index, node_index, weight, access, x, y

    logger.info(
        "Merged %d access point(s) sharing a network node with another in the "
        "same zone",
        key.shape[0] - n_points,
    )
    merged_weight = np.bincount(inverse, weights=weight, minlength=n_points)
    safe = np.where(merged_weight > 0, merged_weight, 1.0)
    return (
        zone_index[first],
        node_index[first],
        merged_weight,
        np.bincount(inverse, weights=weight * access, minlength=n_points) / safe,
        np.bincount(inverse, weights=weight * x, minlength=n_points) / safe,
        np.bincount(inverse, weights=weight * y, minlength=n_points) / safe,
    )


def load_zones(
    network: RoutableNetwork,
    path: Path,
    id_column: str | None = None,
    crs: str = WGS84,
    weights_path: Path | None = None,
    weight_column: str | None = None,
    weight_x_column: str = "lon",
    weight_y_column: str = "lat",
    weight_crs: str = WGS84,
    points_per_zone: int = DEFAULT_POINTS_PER_ZONE,
    access_speed_kmh: float | None = None,
    charge_access_time: bool = True,
    max_snap_distance: float = DEFAULT_MAX_SNAP_M,
) -> Zones:
    """
    Read a zone layer and attach ``points_per_zone`` weighted access points to each.

    ``weights_path`` names where demand actually sits inside a zone (buildings, a
    population grid as points); without it the network's own nodes are used,
    weighted by street length -- see :func:`network_node_weights`.

    Zones whose access points all fail to snap within ``max_snap_distance`` are
    dropped with a warning, exactly as ``Centroids.drop_unsnapped`` does; nothing
    is silently routed from somewhere kilometres away.
    """
    if points_per_zone < 1:
        raise ValueError(f"points_per_zone must be at least 1, got {points_per_zone}.")

    ids, geometry = read_zones(path, id_column=id_column, crs=crs)
    n_zones = len(ids)
    if n_zones == 0:
        raise ValueError(f"{path} holds no zones.")
    logger.info("Read %d zone(s) from %s", n_zones, path)

    if weights_path is not None:
        candidates = read_weight_points(
            weights_path,
            weight_column=weight_column,
            x_column=weight_x_column,
            y_column=weight_y_column,
            crs=weight_crs,
        )
        logger.info("Distributing zone demand by %s", Path(weights_path).name)
    else:
        candidates = network_node_weights(network)
        logger.info("Distributing zone demand by network node density")

    zone_index, x, y, weight = _place_access_points(
        geometry, *candidates, points_per_zone=points_per_zone
    )

    speed_kmh = (
        access_speed_kmh
        if access_speed_kmh is not None
        else profile_for(network.mode).base_kph
    )
    if speed_kmh <= 0:
        raise ValueError(f"access_speed_kmh must be positive, got {speed_kmh}.")
    metres_per_second = speed_kmh * (1000.0 / SECONDS_PER_HOUR)

    node_index, snap_distance = network.nearest_nodes(
        x, y, max_distance=max_snap_distance
    )

    # The representative point is the weighted mean of the access points, which
    # -- because the grid reduction preserves the weighted centroid -- is also
    # the weighted mean of every candidate that went into them. Computed before
    # any unsnappable point is dropped, so it does not move when they are.
    starts = _group_starts(np.bincount(zone_index, minlength=n_zones))
    rep_x = np.add.reduceat(weight * x, starts)
    rep_y = np.add.reduceat(weight * y, starts)
    rep_node, rep_distance = network.nearest_nodes(
        rep_x, rep_y, max_distance=max_snap_distance
    )

    keep_point = node_index >= 0
    if not keep_point.all():
        logger.warning(
            "%d of %d access point(s) are further than %.0f m from any %s node "
            "and were dropped",
            int((~keep_point).sum()),
            keep_point.shape[0],
            max_snap_distance,
            network.mode,
        )

    keep_zone = (np.bincount(zone_index[keep_point], minlength=n_zones) > 0) & (
        rep_node >= 0
    )
    if not keep_zone.all():
        logger.warning(
            "Dropping %d zone(s) with no access point that snaps to the network",
            int((~keep_zone).sum()),
        )
    keep_point &= keep_zone[zone_index]

    remap = np.full(n_zones, -1, dtype=np.int64)
    remap[keep_zone] = np.arange(int(keep_zone.sum()))

    access = snap_distance[keep_point] / metres_per_second
    zone_index, node_index, weight, access, x, y = _merge_shared_nodes(
        remap[zone_index[keep_point]],
        node_index[keep_point],
        weight[keep_point],
        access if charge_access_time else np.zeros(int(keep_point.sum())),
        x[keep_point],
        y[keep_point],
    )

    counts = np.bincount(zone_index, minlength=int(keep_zone.sum()))
    indptr = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    totals = np.add.reduceat(weight, indptr[:-1])
    weight = weight / np.repeat(np.where(totals > 0, totals, 1.0), counts)

    rep_access = rep_distance[keep_zone] / metres_per_second
    zones = Zones(
        ids=np.asarray(ids)[keep_zone],
        indptr=indptr,
        node_index=node_index.astype(np.int64),
        weight=weight,
        access_seconds=access,
        x=x,
        y=y,
        area_m2=geometry.area.to_numpy(dtype=float)[keep_zone],
        representative=Centroids(
            ids=np.asarray(ids)[keep_zone],
            x=rep_x[keep_zone],
            y=rep_y[keep_zone],
            node_index=rep_node[keep_zone],
            snap_distance=rep_distance[keep_zone],
        ),
        representative_access_seconds=(
            rep_access if charge_access_time else np.zeros(int(keep_zone.sum()))
        ),
    )

    stats = zones.summarise()
    logger.info(
        "Zones ready: %d zone(s), %d access point(s), %.1f per zone on average; "
        "access time median %.0f s, max %.0f s",
        int(stats["zones"]),
        int(stats["access_points"]),
        stats["points_per_zone_mean"],
        stats["access_seconds_median"],
        stats["access_seconds_max"],
    )
    return zones


def zone_points_frame(zones: Zones) -> gpd.GeoDataFrame:
    """
    The access points as a GeoDataFrame, for looking at in QGIS.

    Placement is the part of this whole scheme most worth eyeballing: a
    population grid read with the wrong column, or a zone whose network is barely
    mapped, shows up here immediately and nowhere else.
    """
    return gpd.GeoDataFrame(
        {
            "zone_id": np.repeat(zones.ids, zones.points_per_zone),
            "weight": zones.weight,
            "access_seconds": zones.access_seconds,
            "node_index": zones.node_index,
        },
        geometry=gpd.points_from_xy(zones.x, zones.y),
        crs=PROJECTED_CRS,
    )
