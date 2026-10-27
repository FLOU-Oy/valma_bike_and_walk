"""What the viewer serves, held in memory: links, run volumes, counters.

Geometry and values are kept apart on purpose. The link geometry is read once,
reprojected to Web Mercator and cached as GeoParquet; tiles are cut from it.
Every value the map can colour by is a plain array with one entry per link
row, so the browser fetches it once and recolours without touching geometry.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from valma_bike_and_walk import links as links_module
from valma_bike_and_walk.connectors import island_link_ids, merge_connectors
from valma_bike_and_walk.osm import CONSTRUCTION_COLUMN
from valma_bike_and_walk.speeds import (
    CYCLE_LANE_FACTOR,
    CYCLE_TRACK_FACTOR,
    MIXED_TRAFFIC_FACTOR,
    ROAD_CLASS_BETA,
    carriageway_divisor,
    link_disutility,
    profile_for,
    traffic_factor,
)
from valma_bike_and_walk.vector_io import read_vector

logger = logging.getLogger(__name__)

WEB_MERCATOR = "EPSG:3857"

#: Bumped whenever prepare_links changes what it writes, so old caches rebuild.
PREPARED_VERSION = 2

#: Low-zoom thinning: a link is drawn from the first zoom whose threshold its
#: volume reaches. Without volumes the road class decides instead.
VOLUME_ZOOMS = ((7, 1500.0), (8, 800.0), (9, 400.0), (10, 200.0), (11, 50.0), (12, 5.0))
DETAIL_ZOOM = 13
CLASS_ZOOMS = {
    "trunk": 7, "trunk_link": 9, "primary": 8, "primary_link": 10,
    "secondary": 9, "secondary_link": 11, "tertiary": 10, "tertiary_link": 11,
    "cycleway": 11, "residential": 12, "unclassified": 11, "living_street": 12,
}


def cache_key(path: Path) -> str:
    """A name for a cached derivative: the file's stem, size and mtime."""
    stat = Path(path).stat()
    return f"{Path(path).stem}-{stat.st_size}-{stat.st_mtime_ns}"


def prepare_links(
    links_path: Path, cache_dir: Path, connectors_path: Path | None = None
) -> gpd.GeoDataFrame:
    """
    The link layer in Web Mercator, read from the GeoParquet cache when the
    GeoPackage has not changed since it was written.

    Gap connectors stand in for real paths and are kept; island crossings carry
    no real traffic and are left off, as on the ``assign --gpkg`` map.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"v{PREPARED_VERSION}-" + cache_key(links_path)
    if connectors_path is not None:
        key += "+" + cache_key(connectors_path)
    cached = cache_dir / f"{key}.parquet"
    if cached.exists():
        logger.info("Loading prepared links from %s", cached)
        return _plain_strings(gpd.read_parquet(cached))

    logger.info("Preparing %s for the viewer (once per version of the file)", links_path)
    links = links_module.read_links(links_path)
    if connectors_path is not None:
        connectors = links_module.read_links(connectors_path)
        crossings = island_link_ids(connectors)
        links = merge_connectors(links, connectors)
        links = links[~links["link_id"].isin(crossings)]
    links = links.to_crs(WEB_MERCATOR).reset_index(drop=True)
    if "length_m" not in links.columns or links["length_m"].isna().any():
        links["length_m"] = links_module.geometry_lengths(links)
    # GeoParquet wants plain column types. Object columns are what a merge with
    # the connector layer (which lacks most tag columns) leaves behind, so they
    # are typed by their content: a flag must stay a flag -- written as text,
    # "False" would read back as a true value.
    for column in links.columns:
        if column != "geometry" and links[column].dtype == object:
            links[column] = _typed(links[column])
    links.to_parquet(cached)
    logger.info("Cached %d links to %s", len(links), cached)
    return _plain_strings(links)


def _typed(values: pd.Series) -> pd.Series:
    """An object column as booleans, numbers or text, by what it holds."""
    present = values.dropna()
    if len(present) and present.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        return values.fillna(False).astype(bool)
    if len(present) and present.map(lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, bool)).all():
        return pd.to_numeric(values, errors="coerce")
    return values.astype("string")


def _plain_strings(links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Text columns back as plain objects with None for missing.

    Pandas' ``string`` dtype keeps a missing tag as ``<NA>``, and ``<NA> ==
    "yes"`` is ``<NA>`` rather than False -- which turns a speed factor into
    NaN. The speed code is written for what a GeoPackage read gives, so the
    cache must give the same.
    """
    for column in links.columns:
        if isinstance(links[column].dtype, pd.StringDtype):
            values = links[column].astype(object)
            links[column] = values.where(links[column].notna(), None)
    return links


@dataclass
class Run:
    """One assignment result, as per-link volumes in each direction."""

    name: str
    path: Path
    forward: np.ndarray  # float32, one per link row
    reverse: np.ndarray
    #: The routed travel time each way, seconds, penalties included; None for
    #: files written before ``travel_time_s`` was saved. NaN where unrouted.
    forward_time: np.ndarray | None = None
    reverse_time: np.ndarray | None = None

    @property
    def total(self) -> np.ndarray:
        return self.forward + self.reverse


def load_run(path: Path, links: gpd.GeoDataFrame, name: str | None = None) -> Run:
    """
    Read ``link_volumes_<mode>.npz`` onto the link rows.

    Files written since ``link_id``/``direction`` were added join on those;
    older ones are matched on their end nodes, either way round.
    """
    data = np.load(path)
    volume = np.asarray(data["volume"], dtype=np.float64)
    n = len(links)
    forward = np.zeros(n, dtype=np.float64)
    reverse = np.zeros(n, dtype=np.float64)
    forward_time = reverse_time = None

    if "link_id" in data.files:
        position = pd.Index(links["link_id"].to_numpy()).get_indexer(data["link_id"])
        along = np.asarray(data["direction"]) >= 0
        ok = position >= 0
        np.add.at(forward, position[ok & along], volume[ok & along])
        np.add.at(reverse, position[ok & ~along], volume[ok & ~along])
        if "travel_time_s" in data.files:
            time = np.asarray(data["travel_time_s"], dtype=np.float32)
            forward_time = np.full(n, np.nan, dtype=np.float32)
            reverse_time = np.full(n, np.nan, dtype=np.float32)
            forward_time[position[ok & along]] = time[ok & along]
            reverse_time[position[ok & ~along]] = time[ok & ~along]
    else:
        ends = pd.MultiIndex.from_arrays(
            [links["u"].to_numpy(np.int64), links["v"].to_numpy(np.int64)]
        )
        first = ~ends.duplicated()
        lookup = pd.MultiIndex.from_arrays(
            [ends.get_level_values(0)[first], ends.get_level_values(1)[first]]
        )
        rows = np.flatnonzero(first)
        u, v = np.asarray(data["u"], np.int64), np.asarray(data["v"], np.int64)
        along = lookup.get_indexer(pd.MultiIndex.from_arrays([u, v]))
        back = lookup.get_indexer(pd.MultiIndex.from_arrays([v, u]))
        ok = along >= 0
        np.add.at(forward, rows[along[ok]], volume[ok])
        rest = ~ok & (back >= 0)
        np.add.at(reverse, rows[back[rest]], volume[rest])

    placed = forward.sum() + reverse.sum()
    if volume.sum() > 0 and placed < 0.99 * volume.sum():
        logger.warning(
            "%s: only %.1f%% of the volume matched a link in this layer -- is it "
            "the run of this link layer?",
            path,
            100 * placed / volume.sum(),
        )
    return Run(
        name=name or run_name(path),
        path=Path(path),
        forward=forward.astype(np.float32),
        reverse=reverse.astype(np.float32),
        forward_time=forward_time,
        reverse_time=reverse_time,
    )


def run_name(path: Path) -> str:
    path = Path(path)
    return f"{path.parent.name}/{path.stem}" if path.parent.name else path.stem


def discover_runs(paths: list[Path], mode: str) -> list[Path]:
    """Volume files named directly, or found inside named directories."""
    found: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            found.extend(sorted(path.rglob(f"link_volumes_{mode}.npz")))
        elif path.exists():
            found.append(path)
        else:
            logger.warning("No such run: %s", path)
    return found


def min_zooms(links: gpd.GeoDataFrame, importance: np.ndarray | None) -> np.ndarray:
    """The first zoom each link is drawn at, so low-zoom tiles stay small."""
    zoom = np.full(len(links), DETAIL_ZOOM, dtype=np.uint8)
    if importance is not None:
        for z, threshold in reversed(VOLUME_ZOOMS):
            zoom[importance >= threshold] = z
    highway = links["highway"].astype("string").fillna("")
    by_class = highway.map(CLASS_ZOOMS).fillna(DETAIL_ZOOM).to_numpy(np.uint8)
    return np.minimum(zoom, by_class) if importance is None else zoom


@dataclass
class ViewerData:
    """Everything the server answers from."""

    links: gpd.GeoDataFrame
    mode: str
    runs: dict[str, Run] = field(default_factory=dict)
    counters: dict | None = None
    #: Select-link service for counter clicks; set when demand was given.
    flows: object | None = None
    #: Point-to-point router (viewer.route.Router), when the graph is loaded.
    router: object | None = None
    #: Municipality of each link (viewer.municipalities.Municipalities).
    municipalities: object | None = None
    #: Trip length histograms (viewer.trip_lengths.TripLengthGroups).
    trip_lengths: object | None = None
    _speeds: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.geometry = self.links.geometry.to_numpy()
        logger.info("Indexing %d links", len(self.geometry))
        self.tree = shapely.STRtree(self.geometry)
        importance = None
        if self.runs:
            importance = np.maximum.reduce([run.total for run in self.runs.values()])
        self.min_zoom = min_zooms(self.links, importance)

    @property
    def bounds_wgs84(self) -> list[float]:
        box = gpd.GeoSeries([shapely.box(*self.links.total_bounds)], crs=WEB_MERCATOR)
        return [float(v) for v in box.to_crs("EPSG:4326").total_bounds]

    # -- measures --------------------------------------------------------

    def speeds(self) -> tuple[np.ndarray, np.ndarray]:
        """Current-code bike/walk speeds, both directions; computed on first use."""
        if self._speeds is None:
            logger.info("Computing speeds for %d links", len(self.links))
            forward, reverse = links_module.link_speeds_by_direction(self.links, self.mode)
            self._speeds = (forward.astype(np.float32), reverse.astype(np.float32))
        return self._speeds

    def measure(self, run: str, name: str) -> np.ndarray:
        """One float32 value per link row, for the map to colour by."""
        if run.startswith("counter/"):
            if self.flows is None:
                raise KeyError(run)
            result = self.flows.cached(counter_job(self, int(run.split("/", 1)[1])))
            if result is None or name not in ("volume", "volume_forward", "volume_reverse"):
                raise KeyError(name)
            which = {"volume": "total", "volume_forward": "forward", "volume_reverse": "reverse"}[name]
            return result.dense(len(self.links), which)
        if run == "network":
            forward, reverse = self.speeds()
            arrays = {
                "speed": np.minimum(forward, reverse),
                "speed_forward": forward,
                "speed_reverse": reverse,
                "under_construction": _flag(self.links, CONSTRUCTION_COLUMN),
            }
        else:
            if run not in self.runs:
                raise KeyError(run)
            r = self.runs[run]
            arrays = {"volume": r.total, "volume_forward": r.forward, "volume_reverse": r.reverse}
        if name not in arrays:
            raise KeyError(name)
        return np.ascontiguousarray(arrays[name], dtype=np.float32)

    # -- inspector -------------------------------------------------------

    def explain(self, index: int) -> dict:
        """A link's tags, and how its speed and travel time are built up."""
        row = self.links.iloc[[index]].copy()
        tags = {
            k: _plain(v)
            for k, v in row.drop(columns="geometry").iloc[0].items()
            if _plain(v) is not None
        }
        result: dict = {"index": index, "tags": tags}
        result["speed"] = _speed_breakdown(row, self.mode)
        result["volumes"] = {
            name: {
                "forward": float(r.forward[index]),
                "reverse": float(r.reverse[index]),
                "total": float(r.forward[index] + r.reverse[index]),
            }
            for name, r in self.runs.items()
        }
        return result


def _flag(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=np.float32)
    return frame[column].fillna(False).astype(bool).to_numpy(np.float32)


def _plain(value):
    """A JSON-friendly scalar, or None for missing values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value if isinstance(value, (int, float, bool, str)) else str(value)


def _segregation_of(row: gpd.GeoDataFrame) -> float:
    """The shared/segregated path factor of a one-row frame."""
    from valma_bike_and_walk.speeds import segregation_factor

    return segregation_factor(row["highway"], row.get("segregated"))[0]


def _speed_breakdown(row: gpd.GeoDataFrame, mode: str) -> dict:
    """The factors behind one link's speed and travel time, per direction."""
    profile = profile_for(mode)
    r = row.iloc[0]
    highway = str(r.get("highway"))
    nominal = float(profile.highway_kph.get(highway, profile.base_kph))
    surface = r.get("surface")
    surface_factor = float(profile.surface_factor.get(surface, 1.0)) if isinstance(surface, str) else 1.0

    steps: dict = {"highway": highway, "nominal_kmh": nominal, "surface_factor": surface_factor}
    directions: dict = {}
    if mode == "bike":
        segregated = float(_segregation_of(row))
        volumes = links_module.car_volumes(row)
        volume = np.nan if volumes is None else volumes.iloc[0]
        steps.update(segregated_factor=segregated, car_volume=_plain(volume))
        forward_cw, reverse_cw = links_module._cycleway_by_direction(row)
        for label, cw in (("forward", forward_cw), ("reverse", reverse_cw)):
            values = "" if cw is None else str(cw.iloc[0])
            parts = {p.strip() for p in values.split(";")}
            facility, factor = (
                ("track", CYCLE_TRACK_FACTOR) if "track" in parts
                else ("lane", CYCLE_LANE_FACTOR) if "lane" in parts
                else ("mixed traffic", MIXED_TRAFFIC_FACTOR)
            )
            on_road = highway in ROAD_CLASS_BETA
            disutility = link_disutility(row["highway"], np.array([factor]))
            directions[label] = {
                "facility": facility if on_road else "not a carriageway",
                "carriageway_divisor": carriageway_divisor(highway, factor) if on_road else 1.0,
                "traffic_factor": float(traffic_factor(disutility, [volume])[0]),
            }

    forward, reverse = links_module.link_speeds_by_direction(row, mode)
    timed = row.copy()
    timed["speed_kmh"], timed["speed_reverse_kmh"] = forward, reverse
    links_module._set_travel_times(timed, mode)
    length = float(timed["length_m"].iloc[0])
    for label, speed, time in (
        ("forward", forward[0], timed["travel_time_s"].iloc[0]),
        ("reverse", reverse[0], timed["travel_time_reverse_s"].iloc[0]),
    ):
        moving = length / (float(speed) / 3.6)
        directions.setdefault(label, {}).update(
            speed_kmh=float(speed),
            moving_time_s=moving,
            penalties_s=float(time) - moving,
            travel_time_s=float(time),
        )
    steps["length_m"] = length
    steps["directions"] = directions
    return steps


# -- counters ------------------------------------------------------------


def _match_name(name: str) -> str:
    return re.sub(r"\s*PP$", "", str(name).strip(), flags=re.IGNORECASE).casefold()


def load_counters(
    data: ViewerData, count_lines_path: Path, observed_path: Path, layer: str = "count_lines"
) -> dict:
    """
    Count lines and observed averages, ready to be scored against any run.

    Matching follows ``scripts/compare_counter_volumes.py``: the observed
    counter with the same name (minus a "PP" suffix), nearest the line.
    """
    lines = read_vector(count_lines_path, layer=layer).to_crs(WEB_MERCATOR)
    observed = pd.read_csv(observed_path, encoding="utf-8-sig")
    points = gpd.GeoDataFrame(
        observed,
        geometry=gpd.points_from_xy(observed["x"], observed["y"]),
        crs="EPSG:3879",
    ).to_crs(WEB_MERCATOR)
    points["match"] = points["counter_name"].map(_match_name)

    records = []
    for _, line in lines.iterrows():
        candidates = points[points["match"] == _match_name(line["counter"])]
        if candidates.empty:
            continue
        nearest = candidates.loc[candidates.geometry.distance(line.geometry).idxmin()]
        hits = data.tree.query(line.geometry, predicate="intersects")
        records.append(
            {
                "name": str(line["counter"]),
                "city": _plain(nearest.get("city")),
                "observed": float(nearest["average_bikes_per_year"]),
                "links": [int(i) for i in hits],
                "point": nearest.geometry,
                "line": line.geometry,
            }
        )
    logger.info("Matched %d count line(s) to observed counters", len(records))
    return {"records": records}


def counter_job(data: ViewerData, index: int) -> str:
    """The cache name of a counter's select-link map: its name and links."""
    record = data.counters["records"][index]
    ids = data.links["link_id"].to_numpy()[record["links"]]
    return f"{record['name']}|{','.join(str(int(i)) for i in sorted(ids))}"


def counters_geojson(data: ViewerData, run: str | None) -> dict:
    """Counters as GeoJSON points (WGS84), scored against ``run`` if given."""
    if not data.counters:
        return {"type": "FeatureCollection", "features": []}
    records = data.counters["records"]
    total = data.runs[run].total if run in data.runs else None
    pts = gpd.GeoSeries([r["point"] for r in records], crs=WEB_MERCATOR).to_crs("EPSG:4326")
    lns = gpd.GeoSeries([r["line"] for r in records], crs=WEB_MERCATOR).to_crs("EPSG:4326")
    features = []
    for index, (r, p, ln) in enumerate(zip(records, pts, lns)):
        modeled = float(total[r["links"]].sum()) if total is not None else None
        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(p),
                "properties": {
                    "index": index,
                    "name": r["name"],
                    "city": r["city"],
                    "observed": r["observed"],
                    "modeled": modeled,
                    "ratio": (modeled / r["observed"]) if modeled is not None and r["observed"] > 0 else None,
                    "links": r["links"],
                    "line": shapely.geometry.mapping(ln)["coordinates"],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}
