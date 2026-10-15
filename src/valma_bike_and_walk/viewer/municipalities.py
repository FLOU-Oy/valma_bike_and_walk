"""Municipalities in valma view: which links each holds, and how fast it rides.

A municipality's speed is what its cyclists average over the run:

    bicycle-km / bicycle-hours = sum(volume * length) / sum(volume * time)

where ``time`` is the run's own routed ``travel_time_s``, so signal, crossing
and gradient penalties count, and each direction counts with its own time --
the climb is slower than the descent. Weighting by volume * length rather than
averaging link speeds is what keeps a 2 m link from counting like a 2 km one.

A link belongs to the municipality its midpoint falls in. Connectors from
``valma connect`` (negative ``link_id``) are not ridden, so they are left out.

With a demand OMX holding a matrix per mode, each municipality also gets the
mode's share of the trips leaving it, and the table carries a straight-line fit
of that share on the speed, so the two can be plotted against each other.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from valma_bike_and_walk.viewer.data import WEB_MERCATOR, Run, cache_key
from valma_bike_and_walk.vector_io import read_vector

logger = logging.getLogger(__name__)

#: Outlines are only drawn, so they are simplified to this, in metres.
OUTLINE_TOLERANCE_M = 50.0

#: Bumped whenever what is cached changes.
CACHE_VERSION = 1


@dataclass
class Municipalities:
    names: list[str]
    #: Index into ``names`` for each link row; -1 outside every polygon.
    of_link: np.ndarray
    #: GeoJSON geometry of each, WGS84, simplified.
    outlines: list[dict]
    #: [west, south, east, north] of each, WGS84.
    bounds: list[list[float]]
    #: Trips leaving each by the viewer's mode, and by every mode; see
    #: :func:`load_trip_shares`. None without a demand file.
    mode_trips: np.ndarray | None = None
    all_trips: np.ndarray | None = None

    def stats(self, links: gpd.GeoDataFrame, run: Run) -> dict:
        """Bicycle-km, bicycle-hours and their ratio, per municipality."""
        if run.forward_time is None or run.reverse_time is None:
            raise ValueError(
                f"{run.name} was written without travel times; run assign again "
                "to get municipality speeds."
            )
        k = len(self.names)
        length = links["length_m"].to_numpy(dtype=np.float64)
        keep = (self.of_link >= 0) & (links["link_id"].to_numpy() >= 0)
        km = np.zeros(k)
        hours = np.zeros(k)
        used = np.zeros(len(links), dtype=bool)
        for volume, time in ((run.forward, run.forward_time), (run.reverse, run.reverse_time)):
            volume = volume.astype(np.float64)
            time = time.astype(np.float64)
            ok = keep & (volume > 0) & np.isfinite(time) & (time > 0)
            used |= ok
            m = self.of_link[ok]
            km += np.bincount(m, weights=volume[ok] * length[ok] / 1000.0, minlength=k)
            hours += np.bincount(m, weights=volume[ok] * time[ok] / 3600.0, minlength=k)
        count = np.bincount(self.of_link[used], minlength=k)

        def ratio(a: float, b: float) -> float | None:
            return float(a / b) if b > 0 else None

        rows = [
            {
                "index": i,
                "name": self.names[i],
                "speed_kmh": ratio(km[i], hours[i]),
                "bike_km": float(km[i]),
                "bike_hours": float(hours[i]),
                "links": int(count[i]),
            }
            for i in range(k)
        ]
        total = {
            "speed_kmh": ratio(km.sum(), hours.sum()),
            "bike_km": float(km.sum()),
            "bike_hours": float(hours.sum()),
            "links": int(count.sum()),
        }
        result = {"run": run.name, "rows": rows, "total": total, "fit": None}
        if self.mode_trips is not None and self.all_trips is not None:
            for row, trips, every in zip(rows, self.mode_trips, self.all_trips):
                row.update(
                    mode_trips=float(trips),
                    all_trips=float(every),
                    mode_share=ratio(trips, every),
                )
            total.update(
                mode_trips=float(self.mode_trips.sum()),
                all_trips=float(self.all_trips.sum()),
                mode_share=ratio(self.mode_trips.sum(), self.all_trips.sum()),
            )
            result["fit"] = share_fit(rows)
        return result

    def outline(self, index: int) -> dict:
        return {
            "type": "Feature",
            "geometry": self.outlines[index],
            "properties": {"name": self.names[index], "bounds": self.bounds[index]},
        }


def share_fit(rows: list[dict]) -> dict | None:
    """
    Least-squares line of mode share on speed, across municipalities.

    ``r2`` counts each municipality once; ``r2_weighted`` weights them by their
    trips, so a village of a few hundred trips cannot move it as much as a city.
    """
    usable = [
        r for r in rows
        if r["speed_kmh"] is not None and r.get("mode_share") is not None
    ]
    if len(usable) < 3:
        return None
    x = np.array([r["speed_kmh"] for r in usable])
    y = np.array([r["mode_share"] for r in usable])
    w = np.array([r["all_trips"] for r in usable])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)

    def r2(weights: np.ndarray) -> float:
        mx, my = np.average(x, weights=weights), np.average(y, weights=weights)
        cov = np.sum(weights * (x - mx) * (y - my))
        spread = np.sum(weights * (x - mx) ** 2) * np.sum(weights * (y - my) ** 2)
        return float(cov**2 / spread)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2(np.ones_like(x)),
        "r2_weighted": r2(w),
        "n": len(usable),
    }


def load_trip_shares(
    municipalities: Municipalities,
    demand_path: Path,
    matrix: str,
    mapping: str,
    zones_path: Path,
    zone_column: str,
    municipality_column: str,
    cache_dir: Path,
) -> None:
    """
    Attach each municipality's trips by ``matrix`` and by every matrix.

    Trips are counted where they start: the zone's row of each OMX matrix,
    put in a municipality through ``zone_column`` of ``zones_path`` (the same
    zone table the municipalities come from).
    """
    key = (
        f"trip-shares-v{CACHE_VERSION}-{cache_key(demand_path)}-{matrix}-{mapping}"
        f"-{cache_key(zones_path)}-{zone_column}-{municipality_column}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{key}.json"
    if cached.exists():
        with cached.open(encoding="utf-8") as f:
            saved = json.load(f)
    else:
        from valma_bike_and_walk.demand import omx_origin_totals

        logger.info("Summing trips by mode per municipality from %s", demand_path)
        ids, totals = omx_origin_totals(demand_path, mapping)
        if matrix not in totals:
            raise ValueError(
                f"{demand_path} has no matrix {matrix!r}; available: {sorted(totals)}"
            )
        table = read_vector(
            zones_path, columns=[zone_column, municipality_column], ignore_geometry=True
        )
        missing = {zone_column, municipality_column} - set(table.columns)
        if missing:
            raise ValueError(f"{zones_path} has no column(s) {sorted(missing)}.")
        table = table.dropna()
        lookup = dict(
            zip(
                np.asarray(table[zone_column], dtype=np.int64).tolist(),
                table[municipality_column].astype(str),
            )
        )
        names = [lookup.get(int(z)) for z in ids]
        unplaced = np.array([n is None for n in names])
        every = sum(totals.values())
        if unplaced.any():
            logger.warning(
                "%d of %d demand zones (%.1f%% of the trips) have no municipality in %s",
                int(unplaced.sum()),
                len(ids),
                100 * every[unplaced].sum() / max(every.sum(), 1e-12),
                zones_path,
            )
        frame = pd.DataFrame({"name": names, "mode": totals[matrix], "all": every})
        grouped = frame.dropna(subset=["name"]).groupby("name")[["mode", "all"]].sum()
        saved = {
            "mode": {n: float(v) for n, v in grouped["mode"].items()},
            "all": {n: float(v) for n, v in grouped["all"].items()},
        }
        with cached.open("w", encoding="utf-8") as f:
            json.dump(saved, f)
    municipalities.mode_trips = np.array(
        [saved["mode"].get(n, 0.0) for n in municipalities.names]
    )
    municipalities.all_trips = np.array(
        [saved["all"].get(n, 0.0) for n in municipalities.names]
    )


def load_municipalities(
    links: gpd.GeoDataFrame, path: Path, column: str, cache_dir: Path
) -> Municipalities:
    """
    Place every link in a municipality, cached per link layer and polygon file.

    ``path`` may hold the municipalities themselves or anything finer that
    names its municipality in ``column`` (the model's zones do): links are
    matched to the polygons as they are, and only the outlines are dissolved.
    """
    ids = links["link_id"].to_numpy(dtype=np.int64)
    fingerprint = hashlib.sha1(ids.tobytes()).hexdigest()[:12]
    key = f"municipalities-v{CACHE_VERSION}-{cache_key(path)}-{column}-{fingerprint}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays, meta = cache_dir / f"{key}.npy", cache_dir / f"{key}.json"
    if arrays.exists() and meta.exists():
        logger.info("Loading municipalities from %s", meta)
        with meta.open(encoding="utf-8") as f:
            saved = json.load(f)
        return Municipalities(
            saved["names"], np.load(arrays), saved["outlines"], saved["bounds"]
        )

    logger.info("Placing %d links in municipalities from %s", len(links), path)
    polygons = read_vector(path, columns=[column])
    if column not in polygons.columns:
        raise ValueError(f"{path} has no column {column!r}.")
    polygons = polygons[polygons[column].notna() & polygons.geometry.notna()]
    polygons = polygons.to_crs(WEB_MERCATOR).reset_index(drop=True)
    names = sorted(polygons[column].astype(str).unique())
    code = polygons[column].astype(str).map({n: i for i, n in enumerate(names)}).to_numpy()

    midpoints = shapely.line_interpolate_point(
        links.geometry.to_numpy(), 0.5, normalized=True
    )
    tree = shapely.STRtree(polygons.geometry.to_numpy())
    # Intersects, not within: a midpoint exactly on a zone edge -- roads often
    # are the edge -- is within neither zone, but it does touch both.
    point, polygon = tree.query(midpoints, predicate="intersects")
    of_link = np.full(len(links), -1, dtype=np.int32)
    # Reversed, so where polygons meet or overlap the first one listed wins.
    of_link[point[::-1]] = code[polygon[::-1]]
    logger.info(
        "%d of %d links fall outside every municipality",
        int((of_link < 0).sum()),
        len(links),
    )

    dissolved = polygons.assign(_m=code).dissolve("_m").sort_index()
    shapes = dissolved.geometry.simplify(OUTLINE_TOLERANCE_M).to_crs("EPSG:4326")
    outlines = [shapely.geometry.mapping(g) for g in shapes]
    bounds = [[round(float(v), 6) for v in g.bounds] for g in shapes]

    np.save(arrays, of_link)
    with meta.open("w", encoding="utf-8") as f:
        json.dump({"names": names, "outlines": outlines, "bounds": bounds}, f)
    return Municipalities(names, of_link, outlines, bounds)
