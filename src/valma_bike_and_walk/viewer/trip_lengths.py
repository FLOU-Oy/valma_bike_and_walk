"""Trip length distributions in valma view: by municipality, region or country.

``valma assign`` writes ``trip_lengths_<mode>.npz`` beside its volumes: every
OD pair's demand and the network length of the path it was loaded on. Here the
trips are binned by that length, weighted by their demand, and grouped by the
municipality they start in -- and from there by region (the HLT 2016 regional
reports, by default) and for the whole country.

Intrazonal trips start and end at the same centroid, so the network gives them
no length. They are given the mean distance between two random points of a disc
the size of their zone, ``128 / (45 pi) * r`` with ``r = sqrt(area / pi)`` --
about 0.91 r -- and binned with the rest. Their share is reported alongside,
since that length is an estimate rather than a route.

Trips with no path within the assignment's cutoff have no length either; they
are counted apart as unreached rather than put in the last bin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from valma_bike_and_walk.config import PROJECTED_CRS
from valma_bike_and_walk.vector_io import read_vector

logger = logging.getLogger(__name__)

#: Bin edges, km. The last bin is open.
EDGES_KM = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0)
LABELS = ("0–1", "1–2", "2–3", "3–5", "5–7", "7–10", "10–20", "20+")

#: Mean distance between two uniformly random points in a unit disc.
DISC_MEAN_DISTANCE = 128.0 / (45.0 * np.pi)


def trip_lengths_path(run_path: Path) -> Path:
    """Where ``assign`` puts the lengths that go with ``link_volumes_<mode>.npz``."""
    run_path = Path(run_path)
    mode = run_path.stem.removeprefix("link_volumes_")
    return run_path.with_name(f"trip_lengths_{mode}.npz")


def load_zone_table(path: Path, zone_column: str, municipality_column: str) -> pd.DataFrame:
    """Each zone's municipality and equal-area radius (m), indexed by zone id."""
    zones = read_vector(path, columns=[zone_column, municipality_column])
    missing = {zone_column, municipality_column} - set(zones.columns)
    if missing:
        raise ValueError(f"{path} has no column(s) {sorted(missing)}.")
    zones = zones.dropna(subset=[zone_column, municipality_column])
    zones = zones.drop_duplicates(subset=[zone_column])
    area = zones.geometry.to_crs(PROJECTED_CRS).area.to_numpy()
    return pd.DataFrame(
        {
            "municipality": zones[municipality_column].astype(str).to_numpy(),
            "radius_m": np.sqrt(area / np.pi),
        },
        index=pd.Index(np.asarray(zones[zone_column], dtype=np.int64), name="zone"),
    )


def load_regions(path: Path) -> dict[str, str]:
    """``municipality;region`` rows as a dict, in file order."""
    table = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    columns = {c.strip().lower(): c for c in table.columns}
    if not {"municipality", "region"} <= set(columns):
        raise ValueError(f"{path} needs columns 'municipality' and 'region'.")
    pairs = table[[columns["municipality"], columns["region"]]].dropna()
    return {m.strip(): r.strip() for m, r in pairs.itertuples(index=False)}


@dataclass
class TripLengthGroups:
    """What grouping a run's trips needs; histograms are computed once per run."""

    zones: pd.DataFrame
    municipalities: list[str]
    regions: dict[str, str] = field(default_factory=dict)
    _cache: dict[str, dict] = field(default_factory=dict)

    def histograms(self, name: str, path: Path) -> dict:
        key = f"{name}|{path}"
        if key not in self._cache:
            self._cache[key] = self._compute(path)
        return self._cache[key]

    def _compute(self, path: Path) -> dict:
        data = np.load(path)
        origin = np.asarray(data["origin"], dtype=np.int64)
        destination = np.asarray(data["destination"], dtype=np.int64)
        demand = np.asarray(data["demand"], dtype=np.float64)
        length = np.asarray(data["length_m"], dtype=np.float64)

        row = self.zones.index.get_indexer(origin)
        known = row >= 0
        intrazonal = origin == destination
        radius = np.where(known, self.zones["radius_m"].to_numpy()[np.maximum(row, 0)], 0.0)
        length = np.where(intrazonal, DISC_MEAN_DISTANCE * radius, length)
        reached = np.isfinite(length)

        k = len(self.municipalities)
        index = {n: i for i, n in enumerate(self.municipalities)}
        zone_muni = np.array(
            [index.get(n, -1) for n in self.zones["municipality"]], dtype=np.int64
        )
        # Trips from zones outside every municipality still count for the country.
        group = np.where(known, zone_muni[np.maximum(row, 0)], -1)
        group = np.where(group >= 0, group, k)

        nb = len(LABELS)
        bins = np.digitize(np.where(reached, length, 0.0) / 1000.0, EDGES_KM[1:])
        on = reached
        counts = np.bincount(
            group[on] * nb + bins[on], weights=demand[on], minlength=(k + 1) * nb
        ).reshape(k + 1, nb)

        def per_group(mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
            return np.bincount(group[mask], weights=weights[mask], minlength=k + 1)

        stats = {
            "counts": counts,
            "intrazonal": per_group(intrazonal & reached, demand),
            "unreached": per_group(~reached, demand),
            "length_sum": per_group(reached, demand * np.where(reached, length, 0.0)),
        }

        def summary(selector) -> dict:
            c = stats["counts"][selector]
            if c.ndim == 2:
                c = c.sum(axis=0)
            trips = float(c.sum())
            pick = lambda a: float(np.sum(a[selector]))  # noqa: E731
            return {
                "trips": c.tolist(),
                "total": trips,
                "shares": (c / trips).tolist() if trips > 0 else [0.0] * nb,
                "mean_km": pick(stats["length_sum"]) / trips / 1000.0 if trips > 0 else None,
                "intrazonal": pick(stats["intrazonal"]),
                "unreached": pick(stats["unreached"]),
            }

        municipalities = {
            name: summary(np.array([i])) for i, name in enumerate(self.municipalities)
        }
        regions: dict[str, dict] = {}
        for region in dict.fromkeys(self.regions.values()):
            members = [index[m] for m, r in self.regions.items() if r == region and m in index]
            if members:
                regions[region] = summary(np.array(members))
                regions[region]["municipalities"] = [self.municipalities[i] for i in members]
        missing = sorted(m for m in self.regions if m not in index)
        if missing:
            logger.warning("Region file names municipalities not in the zones: %s", missing)
        unplaced = float(demand[group == k].sum())
        if unplaced:
            logger.warning(
                "%.0f trip(s) start in zones without a municipality; counted for "
                "the country only",
                unplaced,
            )
        return {
            "labels": list(LABELS),
            "edges_km": list(EDGES_KM),
            "country": summary(slice(None)),
            "regions": regions,
            "municipalities": municipalities,
        }
