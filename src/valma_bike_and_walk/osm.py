"""Everything that talks to pyosmium and the .osm.pbf file.

This is stage one of the pipeline: a PBF goes in, a table of *links* comes out.
A link is one stretch of way between two junctions -- so the interstitial nodes
that only carry a bend in the road never reach the graph at all, and every row
is something a person could sensibly select and edit in QGIS.

Keeping the osmium calls in one module means the rest of the package deals only
with plain GeoDataFrames and numpy arrays.

Why one streaming pass is enough
--------------------------------
osmium reads the file as a stream and holds only a node-coordinate index, so
peak memory tracks the *network* rather than the file. A country therefore fits
in one pass, and nothing here splits the extent up to get through it.
"""

from __future__ import annotations

import array
import logging
from typing import Iterator, Mapping, Protocol, Sequence, cast

import geopandas as gpd
import numpy as np
import osmium
import shapely
from pyproj import Transformer
from shapely.geometry import base as shapely_base
from shapely.geometry import box

from valma_bike_and_walk.config import (
    PROJECTED_CRS,
    WGS84,
    Settings,
    validate_mode,
)

logger = logging.getLogger(__name__)

#: Way tags carried through to the link layer. Everything else is dropped at
#: read time -- the tag dictionaries are most of a PBF's bulk, and the two that
#: set travel speed (highway, surface) plus the direction tags are all the
#: pipeline downstream ever reads.
WAY_TAGS: tuple[str, ...] = (
    "highway",
    "surface",
    "oneway",
    "oneway:bicycle",
    "junction",
    "name",
    "bridge",
    "tunnel",
)


def _multi(value: str) -> list[str]:
    """Split an OSM multi-value tag (``access=private;delivery``) into its parts."""
    return [part.strip() for part in value.split(";")] if ";" in value else [value]


class Tags(Protocol):
    """What both a plain dict and osmium's own ``TagList`` offer us."""

    def get(self, key: str, default: str | None = None) -> str | None: ...

    def __contains__(self, key: str, /) -> bool: ...


class WayFilter:
    """An "exclude" tag filter: a way is dropped if any listed value matches.

    The tables mirror OSMnx's ``walk`` and ``bike`` network filters, with its
    regular expressions written out as the values they match. Multi-value tags
    match if *any* of their parts is listed, again as OSMnx does.
    """

    def __init__(self, excluded: Mapping[str, Sequence[str]]) -> None:
        self.excluded = {key: frozenset(values) for key, values in excluded.items()}

    def keeps(self, tags: Tags) -> bool:
        if "highway" not in tags:
            return False
        for key, bad in self.excluded.items():
            value = tags.get(key)
            if value is not None and any(part in bad for part in _multi(value)):
                return False
        return True


# Not a physical, routable way -- either not built yet, not there any more, or
# not part of the street network at all.
_ALWAYS_EXCLUDED_HIGHWAY = [
    "abandoned",
    "construction",
    "no",
    "planned",
    "platform",
    "proposed",
    "raceway",
    "razed",
    "rest_area",
    "services",
]

_MOTOR_HIGHWAY = ["motor", "motorway", "motorway_link"]

#: Walkable ways. Service roads stay: a parking aisle is walkable even if it is
#: not a pleasant walk. Ways whose sidewalk is mapped separately are dropped, so
#: the sidewalk is not counted twice.
WALK_FILTER = WayFilter(
    {
        "area": ["yes"],
        "access": ["private"],
        "highway": _ALWAYS_EXCLUDED_HIGHWAY
        + ["bus_guideway", "cycleway"]
        + _MOTOR_HIGHWAY,
        "foot": ["no"],
        "service": ["private"],
        "sidewalk": ["separate"],
        "sidewalk:both": ["separate"],
        "sidewalk:left": ["separate"],
        "sidewalk:right": ["separate"],
    }
)

#: Cyclable ways. Footways, steps and elevators stay: shared-use paths are
#: often only tagged footway, and stairs or a lift can still be part of a
#: real route -- the speed profile prices the dismount-and-push instead.
BIKE_FILTER = WayFilter(
    {
        "area": ["yes"],
        "access": ["private"],
        "highway": _ALWAYS_EXCLUDED_HIGHWAY
        + ["bus_guideway", "corridor", "escalator"]
        + _MOTOR_HIGHWAY,
        "bicycle": ["no"],
        "service": ["private"],
    }
)

WAY_FILTERS: dict[str, WayFilter] = {"walk": WALK_FILTER, "bike": BIKE_FILTER}


def filter_for(mode: str) -> WayFilter:
    validate_mode(mode)
    return WAY_FILTERS[mode]


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


class _WayBuffer:
    """Node refs, coordinates and tags of the ways we decided to keep.

    Flat ``array.array``s rather than lists of tuples: a country's walking
    network runs to tens of millions of node references, and boxed Python ints
    would cost several times what the finished network does.
    """

    def __init__(self) -> None:
        self.way_ids = array.array("q")
        self.refs = array.array("q")
        self.x = array.array("i")  # osmium's native 1e-7 degree integers
        self.y = array.array("i")
        self.counts = array.array("q")  # nodes per way
        self.tags: list[list[str | None]] = [[] for _ in WAY_TAGS]

    def add(
        self, way: osmium.osm.Way, refs: list[int], xs: list[int], ys: list[int]
    ) -> None:
        self.way_ids.append(way.id)
        self.refs.extend(refs)
        self.x.extend(xs)
        self.y.extend(ys)
        self.counts.append(len(refs))
        for column, key in zip(self.tags, WAY_TAGS):
            column.append(way.tags.get(key))

    def __len__(self) -> int:
        return len(self.way_ids)


def _iter_ways(pbf_path, index_storage: str) -> Iterator[osmium.osm.Way]:
    """Stream the file's ways, each with its node coordinates filled in.

    Nodes have to be read for their coordinates, but only ways are yielded --
    the entity filter drops the nodes in C++ rather than crossing into Python
    tens of millions of times.
    """
    processor = (
        osmium.FileProcessor(str(pbf_path), osmium.osm.NODE | osmium.osm.WAY)
        .with_locations(index_storage)
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("highway"))
    )
    # The entity filter guarantees ways, but the processor is typed as yielding
    # any OSM object.
    return cast(Iterator[osmium.osm.Way], iter(processor))


def _read_ways(
    settings: Settings, mode: str, clip_bounds: Sequence[float] | None
) -> _WayBuffer:
    way_filter = filter_for(mode)
    buffer = _WayBuffer()
    seen = 0

    if clip_bounds is None:
        min_lon = min_lat = -np.inf
        max_lon = max_lat = np.inf
    else:
        min_lon, min_lat, max_lon, max_lat = clip_bounds

    for way in _iter_ways(settings.require_pbf(), settings.index_storage):
        seen += 1
        if not way_filter.keeps(way.tags):
            continue

        refs: list[int] = []
        xs: list[int] = []
        ys: list[int] = []
        inside = False
        for node in way.nodes:
            location = node.location
            if not location.valid():
                # A way clipped at the edge of a regional extract references
                # nodes the file does not carry. Dropping the whole way keeps
                # its geometry and its length honest.
                refs = []
                break
            refs.append(node.ref)
            xs.append(location.x)
            ys.append(location.y)
            if (
                not inside
                and min_lon <= location.lon <= max_lon
                and min_lat <= location.lat <= max_lat
            ):
                inside = True

        if len(refs) < 2 or not inside:
            continue
        buffer.add(way, refs, xs, ys)

    logger.info("Scanned %d highway ways, kept %d for mode %r", seen, len(buffer), mode)
    if len(buffer) == 0:
        raise ValueError(
            f"No {mode} network found in {settings.pbf_path} for this extent."
        )
    return buffer


def _junction_positions(refs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Boolean mask of the flat-ref positions where a link must start or end.

    A node splits the way if another way (or another part of the same way) also
    references it, and every way's own two ends split it as well. That is the
    exact topological simplification: what is left between two marks carries no
    junction, only shape.
    """
    boundary = np.zeros(refs.shape[0], dtype=bool)

    # Referenced more than once anywhere in the kept ways.
    order = np.argsort(refs, kind="stable")
    sorted_refs = refs[order]
    repeated = np.zeros(sorted_refs.shape[0], dtype=bool)
    same_as_next = sorted_refs[:-1] == sorted_refs[1:]
    repeated[:-1] |= same_as_next
    repeated[1:] |= same_as_next
    boundary[order] = repeated

    # Plus the first and last node of every way.
    ends = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=ends[1:])
    boundary[ends[:-1]] = True
    boundary[ends[1:] - 1] = True
    return boundary


def _ragged_positions(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Flat indices of ``concatenate([arange(s, s + n) for s, n in ...])``."""
    total = int(lengths.sum())
    offsets = np.zeros(lengths.shape[0], dtype=np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])
    return (
        np.arange(total, dtype=np.int64)
        - np.repeat(offsets, lengths)
        + np.repeat(starts, lengths)
    )


def read_links(
    settings: Settings,
    mode: str,
    clip: shapely_base.BaseGeometry | None = None,
) -> gpd.GeoDataFrame:
    """
    Parse the PBF and return one row per junction-to-junction link.

    Columns: ``link_id``, ``osm_way_id``, ``u``, ``v`` (OSM node ids of the two
    ends), the tags in :data:`WAY_TAGS`, ``length_m`` and a WGS84 ``geometry``.
    ``link_id`` is this layer's own identity and is what every later stage joins
    results back on.
    """
    validate_mode(mode)

    buffer = _read_ways(settings, mode, clip.bounds if clip is not None else None)

    refs = np.frombuffer(buffer.refs, dtype=np.int64)
    counts = np.frombuffer(buffer.counts, dtype=np.int64)
    # osmium stores coordinates as 1e-7 degree integers; that is the full
    # precision OSM has, so this conversion is exact rather than a rounding.
    lon = np.frombuffer(buffer.x, dtype=np.int32) * 1e-7
    lat = np.frombuffer(buffer.y, dtype=np.int32) * 1e-7

    boundary = _junction_positions(refs, counts)
    marks = np.flatnonzero(boundary)

    # Consecutive marks bound a link -- unless they straddle the end of one way
    # and the start of the next, which they do exactly when the owning way
    # changes.
    way_of_position = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    same_way = way_of_position[marks[:-1]] == way_of_position[marks[1:]]
    starts = marks[:-1][same_way]
    ends = marks[1:][same_way]
    if starts.shape[0] == 0:
        raise ValueError(
            f"No {mode} links found in {settings.pbf_path} for this extent."
        )

    lengths = ends - starts + 1
    positions = _ragged_positions(starts, lengths)
    link_of_vertex = np.repeat(np.arange(starts.shape[0], dtype=np.int64), lengths)

    vertex_lon = lon[positions]
    vertex_lat = lat[positions]
    geometry = shapely.linestrings(
        np.column_stack([vertex_lon, vertex_lat]), indices=link_of_vertex
    )

    # Length in the projected CRS: the same metres everything else in the
    # project measures in, and one vectorised transform for the whole file.
    transformer = Transformer.from_crs(WGS84, PROJECTED_CRS, always_xy=True)
    px, py = transformer.transform(vertex_lon, vertex_lat)
    step = np.hypot(np.diff(px), np.diff(py))
    within = link_of_vertex[1:] == link_of_vertex[:-1]
    length_m = np.bincount(
        link_of_vertex[1:][within], weights=step[within], minlength=starts.shape[0]
    )

    links_per_way = np.bincount(way_of_position[starts], minlength=counts.shape[0])
    way_ids = np.frombuffer(buffer.way_ids, dtype=np.int64)

    columns: dict[str, np.ndarray] = {
        "link_id": np.arange(starts.shape[0], dtype=np.int64),
        "osm_way_id": np.repeat(way_ids, links_per_way),
        "u": refs[starts],
        "v": refs[ends],
    }
    for key, values in zip(WAY_TAGS, buffer.tags):
        columns[_column_name(key)] = np.repeat(
            np.asarray(values, dtype=object), links_per_way
        )
    columns["length_m"] = length_m

    links = gpd.GeoDataFrame(columns, geometry=geometry, crs=WGS84)

    if clip is not None and not _is_rectangle(clip):
        links = _clip_exactly(links, clip)

    logger.info(
        "Built %d links from %d ways (%d junction nodes)",
        len(links),
        len(buffer),
        np.unique(np.concatenate([columns["u"], columns["v"]])).shape[0],
    )
    return links


def _column_name(tag: str) -> str:
    """``oneway:bicycle`` -> ``oneway_bicycle``: a colon is awkward in SQL."""
    return tag.replace(":", "_")


def _is_rectangle(geometry: shapely_base.BaseGeometry) -> bool:
    return bool(shapely.equals(geometry, shapely.box(*geometry.bounds)))


def _clip_exactly(
    links: gpd.GeoDataFrame, clip: shapely_base.BaseGeometry
) -> gpd.GeoDataFrame:
    """Keep links that touch the clip polygon, whole -- never cut their geometry.

    Cutting would leave a link end that is not an OSM node, and the ids in
    ``u``/``v`` are how tiles and edits are stitched back together.
    """
    tree = shapely.STRtree(links.geometry.to_numpy())
    keep = np.zeros(len(links), dtype=bool)
    keep[tree.query(clip, predicate="intersects")] = True
    kept = links.loc[keep].reset_index(drop=True)
    kept["link_id"] = np.arange(len(kept), dtype=np.int64)
    logger.info(
        "Clipped to the area polygon: %d of %d links kept", len(kept), len(links)
    )
    return kept


# --------------------------------------------------------------------------
# Administrative boundaries
# --------------------------------------------------------------------------


def find_boundary(
    settings: Settings,
    area: str,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """
    Look up an administrative boundary inside the PBF, with no network access.

    osmium has to assemble every candidate multipolygon relation in the extract
    to do this, which on a whole-country file is minutes rather than seconds, so
    the answer is cached as GeoJSON and paid only once per area.
    """
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    name = area.split(",")[0].strip()
    cache_file = settings.boundary_cache_path(area)

    if cache_file.exists() and not force_reload:
        logger.info("Loading boundary from cache: %s", cache_file)
        return gpd.read_file(cache_file)

    logger.info("Scanning %s for the boundary of %r (slow)", settings.pbf_path, name)
    factory = osmium.geom.WKBFactory()
    processor = osmium.FileProcessor(str(settings.require_pbf())).with_areas(
        osmium.filter.TagFilter(("boundary", "administrative"))
    )

    names: list[str] = []
    geometries: list[shapely_base.BaseGeometry] = []
    for obj in processor:
        if obj.type_str() != "a" or obj.tags.get("boundary") != "administrative":
            continue
        obj_name = obj.tags.get("name")
        if obj_name is None or name not in obj_name:
            continue
        polygon = cast(osmium.osm.Area, obj)
        try:
            geometries.append(
                shapely.from_wkb(bytes.fromhex(factory.create_multipolygon(polygon)))
            )
        except (RuntimeError, ValueError):
            # A relation with a broken ring assembles to nothing; skip it rather
            # than fail the run, since another candidate usually matches.
            continue
        names.append(obj_name)

    if not geometries:
        raise ValueError(
            f"No administrative boundary named {name!r} in {settings.pbf_path}."
        )

    boundaries = gpd.GeoDataFrame({"name": names}, geometry=geometries, crs=WGS84)

    # The name match is a substring one, so "Espoo" also returns "Pohjois-Espoo"
    # and "Espoonlahti". Prefer an exact name, then the largest such area.
    exact = boundaries[boundaries["name"] == name]
    if len(exact) > 0:
        boundaries = exact
    if len(boundaries) > 1:
        boundaries = boundaries.loc[[boundaries.geometry.area.idxmax()]]

    boundaries = boundaries.reset_index(drop=True)
    logger.info(
        "Matched boundary %r, bounds %s",
        boundaries.iloc[0]["name"],
        boundaries.total_bounds,
    )
    boundaries.to_file(cache_file, driver="GeoJSON")
    return boundaries


def _bbox_geometry(bbox: Sequence[float]) -> shapely_base.BaseGeometry:
    if len(bbox) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat].")
    return box(bbox[0], bbox[1], bbox[2], bbox[3])


def resolve_clip(
    settings: Settings,
    area: str | None,
    bbox: Sequence[float] | None,
    force_reload: bool = False,
) -> tuple[shapely_base.BaseGeometry | None, str]:
    """Work out the clipping geometry and a stable cache key for it."""
    if bbox is not None:
        return _bbox_geometry(bbox), "bbox_" + "_".join(f"{c:.4f}" for c in bbox)
    if area is not None:
        boundary = find_boundary(settings, area, force_reload=force_reload)
        key = area.split(",")[0].strip().replace(" ", "_")
        return boundary.geometry.iloc[0], key
    return None, "full"


__all__ = [
    "WAY_TAGS",
    "WayFilter",
    "WALK_FILTER",
    "BIKE_FILTER",
    "filter_for",
    "find_boundary",
    "read_links",
    "resolve_clip",
]
