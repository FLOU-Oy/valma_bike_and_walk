"""The link layer: the editable GeoPackage that sits between OSM and the graph.

Stage one (:mod:`valma_bike_and_walk.osm`) writes it, you may open it in QGIS and
change it, and stage two (:mod:`valma_bike_and_walk.network`) turns it into a
routable graph. Everything the graph needs is a column here, so an edit in QGIS
is the whole configuration surface -- there is no second place to change.

What the columns mean
---------------------
``link_id``        this layer's identity. Results join back on it.
``u``, ``v``       OSM node ids of the two ends; the graph's topology.
``highway``        way type. Sets the speed, via :mod:`valma_bike_and_walk.speeds`.
``surface``        what it is made of. Scales that speed.
``oneway``,        direction. ``oneway_bicycle`` overrides ``oneway`` for bikes,
``oneway_bicycle`` so a contraflow cycle lane is modelled.
``junction``       ``roundabout`` implies oneway even without the tag.
``length_m``       metres. **Recomputed from the geometry on every read**, so
                   moving a vertex in QGIS is enough to change the cost.
``speed_kmh``      derived, informational -- for styling the layer.
``travel_time_s``  derived forward-direction travel time, including the
                   bicycle traffic-light penalty when applicable.
``travel_time_reverse_s``  derived reverse-direction travel time.
``speed_override_kmh``  empty for you to fill in. Where it is set, it wins over
                   ``highway``/``surface`` entirely.

Editing rules
-------------
* Change ``highway`` or ``surface`` and the speed follows on the next build.
* Set ``speed_override_kmh`` to force a speed regardless of the tags.
* Move or reshape a geometry and ``length_m`` follows.
* Delete a row and the link is gone.
* Draw a new row and leave ``u``/``v`` empty: the build snaps each end to the
  nearest existing link end within :data:`SNAP_TOLERANCE_M` and joins there, or
  mints a new node (a negative id, as JOSM does) if nothing is near.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from valma_bike_and_walk.config import PROJECTED_CRS, validate_mode
from valma_bike_and_walk.speeds import profile_for

logger = logging.getLogger(__name__)

#: Layer name inside the GeoPackage.
LINKS_LAYER = "links"

#: How close a hand-drawn link end has to be to an existing one to join it.
#: Generous enough to forgive a shaky mouse, tight enough not to weld together
#: two genuinely separate ends of a street.
SNAP_TOLERANCE_M = 2.0

#: Tag values that make a way one-way. "-1" and "T" mean one-way *against* the
#: digitised direction.
ONEWAY_VALUES = frozenset({"yes", "true", "1", "-1", "T", "F"})
ONEWAY_REVERSED_VALUES = frozenset({"-1", "T"})

#: Columns a link layer must carry for the build to be able to read it.
REQUIRED_COLUMNS = ("link_id", "u", "v", "highway", "geometry")

#: Columns the build derives itself; present in a written layer for styling.
DERIVED_COLUMNS = ("length_m", "speed_kmh", "travel_time_s", "travel_time_reverse_s")

OVERRIDE_COLUMN = "speed_override_kmh"

TRAFFIC_LIGHT_PENALTY_SECONDS = 15.0

GRADE_MODERATE_THRESHOLD = 0.025
GRADE_STEEP_THRESHOLD = 0.05
GRADE_MAX_VALID = 0.20
GRADE_MODERATE_FACTOR = 2.32
GRADE_STEEP_FACTOR = 2.50

_BICYCLE_CROSSING_HIGHWAYS = frozenset(
    {
        "cycleway",
        "path",
        "track",
        "bridleway",
        "footway",
        "pedestrian",
        "steps",
        "elevator",
    }
)

_CAR_LANE_CROSSING_HIGHWAYS = frozenset(
    {
        "living_street",
        "residential",
        "service",
        "unclassified",
        "tertiary",
        "tertiary_link",
        "secondary",
        "secondary_link",
        "primary",
        "primary_link",
        "trunk",
        "trunk_link",
    }
)


# --------------------------------------------------------------------------
# Reading and writing the GeoPackage
# --------------------------------------------------------------------------


def write_links(links: gpd.GeoDataFrame, path: Path, layer: str = LINKS_LAYER) -> Path:
    """Write the link layer, replacing any layer of the same name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    links.to_file(path, driver="GPKG", layer=layer)
    logger.info(
        "Wrote %d links to %s (%.1f MB)", len(links), path, path.stat().st_size / 1e6
    )
    return path


def read_links(path: Path, layer: str = LINKS_LAYER) -> gpd.GeoDataFrame:
    """Read a link layer back, tolerating whatever QGIS did to it."""
    path = Path(path)
    links = gpd.read_file(path, layer=layer)
    missing = [
        c for c in REQUIRED_COLUMNS if c not in links.columns and c != "geometry"
    ]
    if missing:
        raise ValueError(
            f"{path} (layer {layer!r}) is missing required column(s) {missing}. "
            "Was it written by `valma extract`?"
        )
    if links.crs is None:
        raise ValueError(f"{path} (layer {layer!r}) has no CRS.")
    logger.info("Read %d links from %s", len(links), path)
    return links


# --------------------------------------------------------------------------
# Repairing an edited layer
# --------------------------------------------------------------------------


def _endpoint_coordinates(links: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """First and last vertex of every link, in projected metres."""
    geometry = links.geometry.to_numpy()
    first = shapely.get_point(geometry, 0)
    last = shapely.get_point(geometry, -1)

    transformer = Transformer.from_crs(links.crs, PROJECTED_CRS, always_xy=True)
    fx, fy = transformer.transform(shapely.get_x(first), shapely.get_y(first))
    lx, ly = transformer.transform(shapely.get_x(last), shapely.get_y(last))
    return np.column_stack([fx, fy]), np.column_stack([lx, ly])


def _as_node_ids(values: pd.Series) -> np.ndarray:
    """Node id column as int64, with anything unset marked 0.

    OSM never issues node id 0, so it is free to mean "this end has not been
    identified yet" -- which is what a link drawn in QGIS looks like.
    """
    numbers = pd.to_numeric(values, errors="coerce")
    return numbers.fillna(0).to_numpy(dtype=np.int64)


def resolve_endpoints(links: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (u, v) node ids, inventing ids for ends that do not have one.

    Rows that came from OSM keep their node ids exactly, which is what lets two
    separately extracted areas be concatenated and still join up. Rows drawn by
    hand have no ids, so each of their ends is snapped to the nearest *known*
    end within :data:`SNAP_TOLERANCE_M` and adopts its id; an end near nothing
    gets a fresh negative id of its own.
    """
    u = _as_node_ids(links["u"])
    v = _as_node_ids(links["v"])
    if not (u == 0).any() and not (v == 0).any():
        return u, v

    head, tail = _endpoint_coordinates(links)
    points = np.vstack([head, tail])
    ids = np.concatenate([u, v])
    known = ids != 0

    n_unknown = int((~known).sum())
    logger.info("Snapping %d link end(s) with no OSM node id", n_unknown)

    if known.any():
        tree = cKDTree(points[known])
        distance, nearest = tree.query(points[~known], k=1)
        adopted = np.where(distance <= SNAP_TOLERANCE_M, ids[known][nearest], 0)
        ids[~known] = adopted
        logger.info("  %d snapped onto an existing node", int((adopted != 0).sum()))

    # Ends still without an id: cluster the coincident ones together so that two
    # new links drawn end to end connect, then hand each cluster a negative id.
    still_unknown = np.flatnonzero(ids == 0)
    if still_unknown.size:
        remaining = points[still_unknown]
        clusters = cKDTree(remaining).query_ball_tree(
            cKDTree(remaining), r=SNAP_TOLERANCE_M
        )
        label = np.full(still_unknown.size, -1, dtype=np.int64)
        next_label = 0
        for i, neighbours in enumerate(clusters):
            if label[i] != -1:
                continue
            for j in neighbours:
                if label[j] == -1:
                    label[j] = next_label
            next_label += 1
        ids[still_unknown] = -(label + 1)
        logger.info("  %d given a new node id", next_label)

    n = len(links)
    return ids[:n], ids[n:]


def normalise(links: gpd.GeoDataFrame, mode: str) -> gpd.GeoDataFrame:
    """
    Bring an edited link layer back to the state the graph builder expects.

    Lengths and speeds are recomputed rather than trusted, so an edit in QGIS
    takes effect without anyone having to remember to also update the derived
    columns. Missing ``link_id``s are filled in, and missing endpoints resolved
    by :func:`resolve_endpoints`.
    """
    validate_mode(mode)
    links = links.copy()

    empty = links.geometry.isna() | links.geometry.is_empty
    if empty.any():
        logger.warning("Dropping %d link(s) with no geometry", int(empty.sum()))
        links = links.loc[~empty].reset_index(drop=True)
    if len(links) == 0:
        raise ValueError("The link layer is empty.")

    links["link_id"] = _fill_link_ids(links["link_id"])
    links["length_m"] = geometry_lengths(links)

    u, v = resolve_endpoints(links)
    links["u"], links["v"] = u, v

    self_loop = u == v
    if self_loop.any():
        # A loop cannot shorten any path and only confuses the component search.
        logger.info("Dropping %d self-looping link(s)", int(self_loop.sum()))
        links = links.loc[~self_loop].reset_index(drop=True)

    links["speed_kmh"] = link_speeds(links, mode)
    _set_travel_times(links, mode)
    return links


def _set_travel_times(links: gpd.GeoDataFrame, mode: str) -> None:
    """Set forward and reverse travel times on a link layer."""
    base = links["length_m"] / (links["speed_kmh"] / 3.6)
    forward_penalty = np.zeros(len(links), dtype=float)
    reverse_penalty = np.zeros(len(links), dtype=float)
    if mode == "bike":
        signal_crossing = (
            links["crossing"].eq("traffic_signals")
            if "crossing" in links.columns
            else pd.Series(False, index=links.index)
        )
        adjacent_signal = _signal_flag(links, "traffic_signal_at_start") | _signal_flag(
            links, "traffic_signal_at_end"
        )
        # On a cycleway the crossing tag is the whole story; on a car lane the tag
        # may be a crossing the cyclist rides past, so only a signal node counts.
        bicycle_signal = signal_crossing & links["highway"].isin(
            _BICYCLE_CROSSING_HIGHWAYS
        )
        car_signal = adjacent_signal & links["highway"].isin(
            _CAR_LANE_CROSSING_HIGHWAYS
        )
        signal_penalty = (bicycle_signal | car_signal).to_numpy(
            dtype=bool
        ) * TRAFFIC_LIGHT_PENALTY_SECONDS
        forward_penalty += signal_penalty
        reverse_penalty += signal_penalty

    if "ascent_m" in links.columns:
        length = links["length_m"].to_numpy(dtype=float)
        ascent = _metres(links, "ascent_m")
        descent = _metres(links, "descent_m")
        # Reverse is the same link read backwards, so its climb is the descent.
        forward_penalty += _grade_penalty(ascent, length)
        reverse_penalty += _grade_penalty(descent, length)

    links["travel_time_s"] = base.to_numpy(dtype=float) + forward_penalty
    links["travel_time_reverse_s"] = base.to_numpy(dtype=float) + reverse_penalty


def _signal_flag(links: gpd.GeoDataFrame, column: str) -> pd.Series:
    """A boolean column, or all-False when the layer does not carry it."""
    if column not in links.columns:
        return pd.Series(False, index=links.index)
    return links[column].fillna(False).astype(bool)


def _metres(links: gpd.GeoDataFrame, column: str) -> np.ndarray:
    """A float column in metres, or all-zero when the layer does not carry it."""
    if column not in links.columns:
        return np.zeros(len(links), dtype=float)
    return (
        pd.to_numeric(links[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )


def _grade_penalty(ascent_m: np.ndarray, length_m: np.ndarray) -> np.ndarray:
    """Return an uphill time penalty from ascent and link length.

    A gradient at or above :data:`GRADE_MAX_VALID` is not a road a bicycle climbs
    -- it is a bad elevation sample -- so it is dropped rather than believed.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient = np.divide(
            np.maximum(ascent_m, 0.0),
            length_m,
            out=np.zeros_like(ascent_m, dtype=float),
            where=length_m > 0,
        )
    factor = np.where(
        gradient >= GRADE_MAX_VALID,
        0.0,
        np.where(
            gradient > GRADE_STEEP_THRESHOLD,
            GRADE_STEEP_FACTOR,
            np.where(gradient >= GRADE_MODERATE_THRESHOLD, GRADE_MODERATE_FACTOR, 0.0),
        ),
    )
    return factor * gradient * length_m


def _fill_link_ids(values: pd.Series) -> np.ndarray:
    ids = pd.to_numeric(values, errors="coerce")
    missing = ids.isna() | ids.duplicated(keep="first")
    if not missing.any():
        return ids.to_numpy(dtype=np.int64)

    filled = ids.to_numpy(dtype=float, copy=True)
    start = int(np.nanmax(filled)) + 1 if np.isfinite(filled).any() else 0
    fresh = np.arange(start, start + int(missing.sum()), dtype=np.int64)
    filled[missing.to_numpy()] = fresh
    logger.info("Assigned %d new link_id(s)", int(missing.sum()))
    return filled.astype(np.int64)


def geometry_lengths(links: gpd.GeoDataFrame) -> np.ndarray:
    """Length of every link in projected metres, straight from its geometry."""
    if links.crs is not None and str(links.crs).upper() == PROJECTED_CRS:
        return links.geometry.length.to_numpy(dtype=float)
    return links.geometry.to_crs(PROJECTED_CRS).length.to_numpy(dtype=float)


def link_speeds(links: gpd.GeoDataFrame, mode: str) -> np.ndarray:
    """Speed in km/h per link: the profile's answer, unless overridden."""
    profile = profile_for(mode)
    speed = profile.speeds_kph(
        links["highway"],
        links["surface"] if "surface" in links.columns else None,
        links["segregated"] if "segregated" in links.columns else None,
    )
    if OVERRIDE_COLUMN in links.columns:
        override = pd.to_numeric(links[OVERRIDE_COLUMN], errors="coerce").to_numpy(
            dtype=float
        )
        forced = np.isfinite(override) & (override > 0)
        if forced.any():
            logger.info("Using %s on %d link(s)", OVERRIDE_COLUMN, int(forced.sum()))
            speed = np.where(forced, override, speed)
    return speed


def annotate(links: gpd.GeoDataFrame, mode: str) -> gpd.GeoDataFrame:
    """Add the derived columns to a freshly extracted layer, ready to write."""
    links = links.copy()
    links["speed_kmh"] = link_speeds(links, mode)
    _set_travel_times(links, mode)
    # An empty column is an invitation: QGIS shows it, and filling one cell is
    # all it takes to pin a link's speed.
    links[OVERRIDE_COLUMN] = np.nan
    return links


# --------------------------------------------------------------------------
# Directed expansion
# --------------------------------------------------------------------------


def _effective_oneway(links: gpd.GeoDataFrame, mode: str) -> pd.Series:
    """The ``oneway`` value that applies to this mode, per link."""
    oneway = (
        links["oneway"].astype("object")
        if "oneway" in links.columns
        else pd.Series([None] * len(links), dtype="object")
    )
    oneway = oneway.reset_index(drop=True)
    if mode == "bike" and "oneway_bicycle" in links.columns:
        # A contraflow cycle lane is oneway=yes + oneway:bicycle=no. The
        # bicycle-specific tag wins wherever it is set.
        override = links["oneway_bicycle"].astype("object").reset_index(drop=True)
        oneway = override.where(override.notna(), oneway)
    return oneway


def directed_edges(links: gpd.GeoDataFrame, mode: str) -> pd.DataFrame:
    """
    Expand links into directed edges.

    Walking is bidirectional throughout: people ignore one-way signs, and
    modelling them would invent detours that nobody walks. Cycling honours
    ``oneway`` together with ``oneway:bicycle``, and treats a roundabout as
    one-way whether or not it is tagged so.

    Returns one row per direction with ``link_id``, ``u``, ``v``,
    ``travel_time_s``, ``length_m`` and ``direction`` (+1 along the link's
    digitised direction, -1 against it).
    """
    validate_mode(mode)

    base = pd.DataFrame(
        {
            "link_id": links["link_id"].to_numpy(dtype=np.int64),
            "u": links["u"].to_numpy(dtype=np.int64),
            "v": links["v"].to_numpy(dtype=np.int64),
            "travel_time_s": links["travel_time_s"].to_numpy(dtype=float),
            "length_m": links["length_m"].to_numpy(dtype=float),
        }
    )

    if mode == "walk":
        forward = base.assign(direction=np.int8(1))
        backward = base.rename(columns={"u": "v", "v": "u"}).assign(
            direction=np.int8(-1)
        )
        return pd.concat([forward, backward], ignore_index=True)

    oneway = _effective_oneway(links, mode)
    is_oneway = oneway.isin(ONEWAY_VALUES).to_numpy(dtype=bool, copy=True)
    if "junction" in links.columns:
        is_oneway |= (links["junction"] == "roundabout").to_numpy()
    against = oneway.isin(ONEWAY_REVERSED_VALUES).to_numpy()

    two_way = ~is_oneway
    along = is_oneway & ~against
    reverse_only = is_oneway & against

    forward = base.loc[along | two_way].assign(direction=np.int8(1))
    reverse_time = (
        links["travel_time_reverse_s"].to_numpy(dtype=float)
        if "travel_time_reverse_s" in links.columns
        else base["travel_time_s"].to_numpy(dtype=float)
    )
    backward = (
        base.assign(travel_time_s=reverse_time)
        .loc[reverse_only | two_way]
        .rename(columns={"u": "v", "v": "u"})
        .assign(direction=np.int8(-1))
    )

    logger.info(
        "Directed expansion: %d links -> %d edges (%d one-way)",
        len(links),
        len(forward) + len(backward),
        int(is_oneway.sum()),
    )
    return pd.concat([forward, backward], ignore_index=True)
