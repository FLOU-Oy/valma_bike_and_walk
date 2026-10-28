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
``under_construction``  true where the way is under roadworks in OSM and was
                   read as what it is being built into (``extract
                   --include-construction``, the default).
``length_m``       metres. **Recomputed from the geometry on every read**, so
                   moving a vertex in QGIS is enough to change the cost.
``speed_kmh``      derived, informational -- for styling the layer.
``speed_reverse_kmh``  the same against the digitised direction. Differs only
                   where a cycle lane or track serves one direction of a road.
``travel_time_s``  derived forward-direction travel time, including the
                   wait at traffic lights.
``travel_time_reverse_s``  derived reverse-direction travel time.
``signal_wait_s``, ``signal_wait_reverse_s``  derived, informational -- the
                   part of each travel time spent waiting at traffic lights.
``speed_override_kmh``  empty for you to fill in. Where it is set, it wins over
                   ``highway``/``surface`` entirely.
``bicycle``, ``bicycle_forward``, ``bicycle_backward``  the OSM tags. A road
                   with ``use_sidepath`` -- or ``cycleway*=separate`` -- in a
                   direction rides :data:`SIDEPATH_FACTOR` slower that way.
``ascent_override_m``, ``descent_override_m``  empty for you to fill in (added
                   by ``valma dem``). Where set, each wins over the DEM's
                   ``ascent_m``/``descent_m`` on that link.

Editing rules
-------------
* Change ``highway`` or ``surface`` and the speed follows on the next build.
* Set ``speed_override_kmh`` to force a speed regardless of the tags.
* Set ``ascent_override_m``/``descent_override_m`` where the DEM is wrong. A
  link with both set is left alone by ``valma dem``.
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
from valma_bike_and_walk.vector_io import read_vector, write_vector

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

#: Seconds of the forward travel time spent waiting at traffic lights.
SIGNAL_WAIT_COLUMN = "signal_wait_s"
SIGNAL_WAIT_REVERSE_COLUMN = "signal_wait_reverse_s"

OVERRIDE_COLUMN = "speed_override_kmh"

#: A road whose cyclists are told to use the path alongside -- ``bicycle=
#: use_sidepath`` (or ``:forward`` / ``:backward``), or a ``cycleway*=separate``
#: pointing at that path -- takes this many times as long to ride, in the
#: direction it applies to. A penalty, not a ban: an address on the road stays
#: reachable, but a long stretch loses to the path even with its crossings.
SIDEPATH_FACTOR = 2

#: Hand-set cars per hour (the unit of ``car_volume``, written by ``valma
#: traffic``) that wins over the matched volume wherever it is filled in.
CAR_VOLUME_COLUMN = "car_volume"
CAR_VOLUME_OVERRIDE_COLUMN = "car_volume_override"

#: On a path running beside a road, that road's cars per hour (``valma
#: traffic``): charged as if the path carried them itself.
ADJACENT_CAR_VOLUME_COLUMN = "adjacent_car_volume"

#: Hand-set climb in metres, along the digitised direction, that wins over the
#: DEM's ``ascent_m`` / ``descent_m`` wherever it is filled in.
ASCENT_OVERRIDE_COLUMN = "ascent_override_m"
DESCENT_OVERRIDE_COLUMN = "descent_override_m"

#: Wait at a light, per signal node passed, for a layer extracted before
#: signalled junctions (``signal_entry`` and friends) were worked out.
TRAFFIC_LIGHT_PENALTY_SECONDS = 10

#: The same wait on foot, on such a layer. The walk speeds are walking pace in
#: motion, so the time spent standing at a red light is not in them.
WALK_TRAFFIC_LIGHT_PENALTY_SECONDS = 10

# Signalled junctions (valma_bike_and_walk.signals): a road pays entering one, a
# path pays stepping onto a signalled crossing. Walking and cycling wait alike:
# a light does not care who is waiting at it.

#: Wait at a signalled junction, seconds. The average for someone arriving at a
#: random moment is red**2 / (2 * cycle): with a 90 s cycle and 40 % of it
#: green for the way you are going, 16 s. One number for every approach and
#: every crossing: if the road waited longer than a crossing, stepping onto the
#: crossing and turning onto the road there would be the cheap way in.
SIGNAL_WAIT_S = 15.0

#: Wait at a mid-block pedestrian light, seconds: it turns red for the road only
#: when someone has pushed the button, and the one crossing waits for its turn.
MIDBLOCK_SIGNAL_WAIT_S = 8.0

#: Slowing down and yielding at a crossing without lights, on a bike.
UNSIGNALLED_CROSSING_PENALTY_SECONDS = 2

#: Each link's share of its crossing way's charge, written at extract time
#: (``osm._crossing_way_share``): the way's length on this link over its whole
#: length, or 0 where a crossing node on the way is charged instead. A layer
#: without the column is charged a whole crossing per link, as before.
CROSSING_SHARE_COLUMN = "crossing_way_share"

#: ``crossing`` values that are not an unsignalled crossing: lights are charged
#: separately, and ``no`` says there is no crossing at all.
_NOT_UNSIGNALLED_CROSSING = frozenset({"", "traffic_signals", "no"})

GRADE_MODERATE_THRESHOLD = 0.025
GRADE_STEEP_THRESHOLD = 0.05
GRADE_MAX_VALID = 0.20
GRADE_LOW_FACTOR = 4.587*1.2
GRADE_MODERATE_FACTOR = 4.587*1.2
GRADE_STEEP_FACTOR = 3.744*1.2

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

#: Path-like classes: where a crossing is crossed rather than passed by.
PATH_HIGHWAYS = _BICYCLE_CROSSING_HIGHWAYS

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
    write_vector(links, path, driver="GPKG", layer=layer)
    logger.info(
        "Wrote %d links to %s (%.1f MB)", len(links), path, path.stat().st_size / 1e6
    )
    return path


def read_links(path: Path, layer: str = LINKS_LAYER) -> gpd.GeoDataFrame:
    """Read a link layer back, tolerating whatever QGIS did to it."""
    path = Path(path)
    links = read_vector(path, layer=layer)
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

    links["speed_kmh"], links["speed_reverse_kmh"] = link_speeds_by_direction(
        links, mode
    )
    _set_travel_times(links, mode)
    return links


def _set_travel_times(links: gpd.GeoDataFrame, mode: str) -> None:
    """Set forward and reverse travel times on a link layer."""
    base = links["length_m"] / (links["speed_kmh"] / 3.6)
    reverse_speed = (
        links["speed_reverse_kmh"]
        if "speed_reverse_kmh" in links.columns
        else links["speed_kmh"]
    )
    reverse_base = links["length_m"] / (reverse_speed / 3.6)
    forward_penalty, reverse_penalty = traffic_light_waits(links, mode)
    # Written out with the layer, so the wait each link is charged can be
    # styled and checked in QGIS. Informational: rebuilt on every read.
    links[SIGNAL_WAIT_COLUMN] = forward_penalty
    links[SIGNAL_WAIT_REVERSE_COLUMN] = reverse_penalty
    forward_penalty = forward_penalty.copy()
    reverse_penalty = reverse_penalty.copy()
    path = links["highway"].isin(_BICYCLE_CROSSING_HIGHWAYS).to_numpy(dtype=bool)
    crossing = _crossing_tags(links)
    has_share = CROSSING_SHARE_COLUMN in links.columns
    share = (
        pd.to_numeric(links[CROSSING_SHARE_COLUMN], errors="coerce").fillna(0.0).to_numpy()
        if has_share
        else None
    )

    if mode == "bike":
        # Yielding at a crossing without lights -- only for the cyclist crossing
        # the road on the path; riding along the road past a zebra costs nothing.
        crossing_ends = (
            _signal_flag(links, "crossing_node_at_start").astype(int)
            + _signal_flag(links, "crossing_node_at_end").astype(int)
        ).to_numpy()
        unsignalled_way = (~crossing.isin(_NOT_UNSIGNALLED_CROSSING)).to_numpy(
            dtype=bool
        ) & ~crossing.str.split(";").map(lambda v: "traffic_signals" in v).to_numpy(
            dtype=bool
        ) & path
        crossing_penalty = UNSIGNALLED_CROSSING_PENALTY_SECONDS * (
            0.5 * crossing_ends * path
            + (share * unsignalled_way if has_share else unsignalled_way)
        )
        forward_penalty += crossing_penalty
        reverse_penalty += crossing_penalty

    if {"ascent_m", ASCENT_OVERRIDE_COLUMN, DESCENT_OVERRIDE_COLUMN} & set(links.columns):
        length = links["length_m"].to_numpy(dtype=float)
        ascent, descent = link_climb(links)
        # Reverse is the same link read backwards, so its climb is the descent.
        forward_penalty += _grade_penalty(ascent, length)
        reverse_penalty += _grade_penalty(descent, length)

    links["travel_time_s"] = base.to_numpy(dtype=float) + forward_penalty
    links["travel_time_reverse_s"] = reverse_base.to_numpy(dtype=float) + reverse_penalty


def traffic_light_waits(links: gpd.GeoDataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Seconds of wait at traffic lights charged on each link: (forward, reverse).

    The part of ``travel_time_s`` / ``travel_time_reverse_s`` spent standing at
    red lights, worked out from the module constants as they are now.
    """
    forward = np.zeros(len(links), dtype=float)
    reverse = np.zeros(len(links), dtype=float)
    if "signal_entry" in links.columns:
        # Signalled junctions (signals.py): a road entering one, a path stepping
        # onto a signalled crossing, or either getting onto the junction's
        # roads inside it -- each in the direction of travel that does it.
        waits = signal_waits()
        forward += waits[_codes(links, "signal_entry")]
        reverse += waits[_codes(links, "signal_entry_reverse")]
        forward += waits[_codes(links, "signal_cross_end")]
        reverse += waits[_codes(links, "signal_cross_start")]
        return forward, reverse

    # A layer from before signalled junctions: a wait per signal node, charged
    # once per *passage*. A signal node is always a link end, and passing it
    # means riding one link in and another out, so each link touching it pays
    # half. A light stops whoever passes it, on a path or along the road.
    wait = {
        "bike": TRAFFIC_LIGHT_PENALTY_SECONDS,
        "walk": WALK_TRAFFIC_LIGHT_PENALTY_SECONDS,
    }.get(mode, 0.0)
    if not wait:
        return forward, reverse
    signal_ends = (
        _signal_flag(links, "traffic_signal_at_start").astype(int)
        + _signal_flag(links, "traffic_signal_at_end").astype(int)
    ).to_numpy()
    at_light = links["highway"].isin(
        _BICYCLE_CROSSING_HIGHWAYS | _CAR_LANE_CROSSING_HIGHWAYS
    ).to_numpy(dtype=bool)
    signal_penalty = wait * 0.5 * signal_ends * at_light
    # A crossing way tagged with signals the nodes do not carry. It is split
    # where it meets the road, so its pieces share one charge by length
    # (CROSSING_SHARE_COLUMN, worked out at extract time).
    path = links["highway"].isin(_BICYCLE_CROSSING_HIGHWAYS).to_numpy(dtype=bool)
    signal_way = _crossing_tags(links).str.split(";").map(
        lambda v: "traffic_signals" in v
    ).to_numpy(dtype=bool) & path
    if CROSSING_SHARE_COLUMN in links.columns:
        share = pd.to_numeric(links[CROSSING_SHARE_COLUMN], errors="coerce").fillna(0.0)
        signal_penalty = signal_penalty + wait * share.to_numpy() * signal_way
    else:
        # A layer from before the share column: per link, as it was charged,
        # where no signal node on it is charged already.
        signal_penalty = signal_penalty + wait * (signal_way & (signal_ends == 0))
    forward += signal_penalty
    reverse += signal_penalty
    return forward, reverse


def _crossing_tags(links: gpd.GeoDataFrame) -> pd.Series:
    """The ``crossing`` column as stripped strings, empty where absent."""
    if "crossing" not in links.columns:
        return pd.Series("", index=links.index)
    return links["crossing"].fillna("").astype(str).str.strip()


def signal_waits() -> np.ndarray:
    """
    Seconds of wait per signal code (signals.NONE, JUNCTION, MIDBLOCK).

    Indexed by code: 2 and 4, from an earlier scheme, wait as JUNCTION. Read
    from the module constants on every call, so a calibration that varies them
    sees the change.
    """
    return np.array(
        [0.0, SIGNAL_WAIT_S, SIGNAL_WAIT_S, MIDBLOCK_SIGNAL_WAIT_S, SIGNAL_WAIT_S]
    )


def _codes(links: gpd.GeoDataFrame, column: str) -> np.ndarray:
    """A signal-code column as indices into :func:`signal_waits`, 0 if absent."""
    if column not in links.columns:
        return np.zeros(len(links), dtype=np.int64)
    codes = pd.to_numeric(links[column], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return np.clip(codes, 0, 4)


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


def link_climb(links: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Metres climbed and dropped along each link's digitised direction.

    The DEM's ``ascent_m`` / ``descent_m``, except where ``ascent_override_m``
    or ``descent_override_m`` is filled in: each override wins on its own, so
    fixing only the climb leaves the DEM's descent in place.
    """
    climb = []
    for column, override_column in (
        ("ascent_m", ASCENT_OVERRIDE_COLUMN),
        ("descent_m", DESCENT_OVERRIDE_COLUMN),
    ):
        metres = _metres(links, column)
        if override_column in links.columns:
            override = pd.to_numeric(links[override_column], errors="coerce").to_numpy(
                dtype=float
            )
            forced = np.isfinite(override)
            if forced.any():
                logger.info("Using %s on %d link(s)", override_column, int(forced.sum()))
                metres = np.where(forced, np.maximum(override, 0.0), metres)
        climb.append(metres)
    return climb[0], climb[1]


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
            np.where(
                gradient >= GRADE_MODERATE_THRESHOLD,
                GRADE_MODERATE_FACTOR,
                GRADE_LOW_FACTOR,
            ),
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


def link_speeds_by_direction(
    links: gpd.GeoDataFrame, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Speed in km/h per link, along and against its digitised direction.

    The profile's answer, unless ``speed_override_kmh`` is set, which then
    holds both ways. The two directions differ only for bikes, where a cycle
    lane or track serves one side of a road; see :func:`_cycleway_by_direction`
    -- and where a road tells cyclists to use the path alongside in only one
    direction, see :func:`_sidepath_by_direction`.
    """
    profile = profile_for(mode)
    forward_cycleway = reverse_cycleway = None
    if mode == "bike":
        forward_cycleway, reverse_cycleway = _cycleway_by_direction(links)

    def speeds(cycleway: pd.Series | None) -> np.ndarray:
        return profile.speeds_kph(
            links["highway"],
            links["surface"] if "surface" in links.columns else None,
            links["segregated"] if "segregated" in links.columns else None,
            cycleway,
            car_volumes(links),
        )

    forward = speeds(forward_cycleway)
    reverse = (
        forward
        if forward_cycleway is None or forward_cycleway.equals(reverse_cycleway)
        else speeds(reverse_cycleway)
    )
    if mode == "bike":
        along, against = _sidepath_by_direction(links, forward_cycleway, reverse_cycleway)
        if along.any() or against.any():
            logger.info(
                "Riding %d link direction(s) %g times slower for a sidepath",
                int(along.sum() + against.sum()),
                SIDEPATH_FACTOR,
            )
            forward = np.where(along, forward / SIDEPATH_FACTOR, forward)
            reverse = np.where(against, reverse / SIDEPATH_FACTOR, reverse)
    if OVERRIDE_COLUMN in links.columns:
        override = pd.to_numeric(links[OVERRIDE_COLUMN], errors="coerce").to_numpy(
            dtype=float
        )
        forced = np.isfinite(override) & (override > 0)
        if forced.any():
            logger.info("Using %s on %d link(s)", OVERRIDE_COLUMN, int(forced.sum()))
            forward = np.where(forced, override, forward)
            reverse = np.where(forced, override, reverse)
    return forward, reverse


_CYCLEWAY_COLUMNS = ("cycleway", "cycleway_left", "cycleway_right", "cycleway_both")


def car_volumes(links: gpd.GeoDataFrame) -> pd.Series | None:
    """
    Cars per hour on each link: ``car_volume_override`` where it is filled in,
    else ``car_volume``, else -- on a path beside a road -- that road's
    ``adjacent_car_volume``. None when the layer carries none of them.
    """
    columns = [
        c
        for c in (CAR_VOLUME_COLUMN, CAR_VOLUME_OVERRIDE_COLUMN, ADJACENT_CAR_VOLUME_COLUMN)
        if c in links.columns
    ]
    if not columns:
        return None
    volume = (
        pd.to_numeric(links[CAR_VOLUME_COLUMN], errors="coerce")
        if CAR_VOLUME_COLUMN in links.columns
        else pd.Series(np.nan, index=links.index)
    )
    if ADJACENT_CAR_VOLUME_COLUMN in links.columns:
        adjacent = pd.to_numeric(links[ADJACENT_CAR_VOLUME_COLUMN], errors="coerce")
        volume = volume.where(volume.notna(), adjacent)
    if CAR_VOLUME_OVERRIDE_COLUMN in links.columns:
        override = pd.to_numeric(links[CAR_VOLUME_OVERRIDE_COLUMN], errors="coerce")
        volume = override.where(override.notna(), volume)
    return volume


def _sidepath_by_direction(
    links: gpd.GeoDataFrame,
    forward_cycleway: pd.Series | None,
    reverse_cycleway: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roads whose cyclists are told to use the path alongside: along, against.

    ``bicycle=use_sidepath`` holds both ways, ``bicycle:forward`` /
    ``bicycle:backward`` along / against the digitised direction. A
    ``separate`` among the cycleway values serving a direction (see
    :func:`_cycleway_by_direction`) says the same: that side's cycleway is
    mapped as its own way. Paths themselves are never penalised.
    """
    road = ~links["highway"].isin(PATH_HIGHWAYS).to_numpy(dtype=bool)

    def says(column: str) -> np.ndarray:
        if column not in links.columns:
            return np.zeros(len(links), dtype=bool)
        values = links[column].fillna("").astype(str).str.lower().str.split(";")
        return values.map(lambda parts: "use_sidepath" in [p.strip() for p in parts]).to_numpy(
            dtype=bool
        )

    def separate(cycleway: pd.Series | None) -> np.ndarray:
        if cycleway is None:
            return np.zeros(len(links), dtype=bool)
        return cycleway.str.contains(r"(?:^|;)\s*separate\s*(?:;|$)", regex=True).to_numpy(
            dtype=bool
        )

    both = says("bicycle")
    along = road & (both | says("bicycle_forward") | separate(forward_cycleway))
    against = road & (both | says("bicycle_backward") | separate(reverse_cycleway))
    return along, against


def _cycleway_by_direction(
    links: gpd.GeoDataFrame,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    The ``cycleway`` values that apply to each direction of travel, per link.

    Returned as ``;``-joined strings, one series along the digitised direction
    and one against it, for :meth:`SpeedProfile.speeds_kph` to pick the best
    infrastructure out of. Following OSM, and with traffic on the right:

    * ``cycleway:right`` serves travel along the way, ``cycleway:left`` travel
      against it, and ``cycleway:both`` serves both.
    * Plain ``cycleway`` means both sides on a two-way road. On a one-way road
      it is the lane in the direction of traffic, so a contraflow
      (``oneway:bicycle=no``) rider does not get it.
    * Where bikes may only go one way, every lane on the road serves that one
      direction -- on a one-way street the lane may well be on the left.

    ``cycleway:<side>:oneway`` is not read, so a two-way track on one side of
    the road serves only the side's own direction here.
    """
    if not any(col in links.columns for col in _CYCLEWAY_COLUMNS):
        return None, None

    def column(name: str) -> pd.Series:
        if name not in links.columns:
            return pd.Series("", index=links.index, dtype=object)
        return links[name].fillna("").astype(str).str.lower()

    plain, left, right, both = (column(name) for name in _CYCLEWAY_COLUMNS)

    car_oneway = (
        links["oneway"].astype("object")
        if "oneway" in links.columns
        else pd.Series(None, index=links.index, dtype=object)
    )
    car_against = car_oneway.isin(ONEWAY_REVERSED_VALUES)
    car_along = car_oneway.isin(ONEWAY_VALUES) & ~car_against

    bike_oneway = pd.Series(
        _effective_oneway(links, "bike").isin(ONEWAY_VALUES).to_numpy(dtype=bool),
        index=links.index,
    )
    if "junction" in links.columns:
        bike_oneway |= links["junction"].eq("roundabout")

    forward = plain.where(~car_against, "").str.cat([both, right], sep=";")
    reverse = plain.where(~car_along, "").str.cat([both, left], sep=";")
    every_side = plain.str.cat([both, left, right], sep=";")
    forward = forward.where(~bike_oneway, every_side)
    reverse = reverse.where(~bike_oneway, every_side)
    return forward, reverse


def annotate(links: gpd.GeoDataFrame, mode: str) -> gpd.GeoDataFrame:
    """Add the derived columns to a freshly extracted layer, ready to write."""
    links = links.copy()
    links["speed_kmh"], links["speed_reverse_kmh"] = link_speeds_by_direction(
        links, mode
    )
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

    # The way back can cost more than the way there -- uphill is the
    # descent read backwards -- so it has its own time wherever there is one.
    reverse_time = (
        links["travel_time_reverse_s"].to_numpy(dtype=float)
        if "travel_time_reverse_s" in links.columns
        else base["travel_time_s"].to_numpy(dtype=float)
    )

    if mode == "walk":
        forward = base.assign(direction=np.int8(1))
        backward = (
            base.assign(travel_time_s=reverse_time)
            .rename(columns={"u": "v", "v": "u"})
            .assign(direction=np.int8(-1))
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
