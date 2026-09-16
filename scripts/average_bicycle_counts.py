"""Create annual bicycle-count averages from counter Excel workbooks."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

LOGGER = logging.getLogger(__name__)
OUTPUT_COLUMNS = ["city", "x", "y", "counter_name", "average_bikes_per_year"]
COUNTER_MERGE_DISTANCE_M = 110.0


def _match_name(name: str) -> str:
    """Normalize harmless PP suffixes for matching tabs against each other."""
    return re.sub(r"\s*PP$", "", name.strip(), flags=re.IGNORECASE).casefold()


def _parse_coordinate_pair(text: str) -> tuple[float, float]:
    text = text.replace("\xa0", " ").strip()
    if ";" in text:
        parts = [part.strip() for part in text.split(";", 1)]
    else:
        parts = re.split(r",\s+(?=[+-]?\d)|,(?=[+-]?\d\.)", text, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate pair: {text!r}")
    first = _parse_number(parts[0])
    second = _parse_number(parts[1])
    return first, second


def _parse_number(text: str) -> float:
    return float(text.strip().replace("\xa0", "").replace(" ", "").replace(",", "."))


def _find_sheet(sheets: list[str], token: str) -> str:
    matches = [sheet for sheet in sheets if token.casefold() in sheet.casefold()]
    if not matches and token == "PP":
        matches = [
            sheet
            for sheet in sheets
            if "pyöräil" in sheet.casefold() or "pyorail" in sheet.casefold()
        ]
    if len(matches) != 1:
        raise ValueError(f"Expected one sheet containing {token!r}; found {matches}.")
    return matches[0]


def _city_name(path: Path) -> str:
    """Return the city prefix from a counter workbook filename."""
    return re.split(r"\s+(?:käpy|pyöräilijät)\b", path.stem, maxsplit=1, flags=re.IGNORECASE)[0]


def _read_locations(path: Path, sheet: str) -> dict[str, tuple[float, float]]:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    rows = []
    to_gk25 = Transformer.from_crs("EPSG:4326", "EPSG:3879", always_xy=True)
    to_gk25_from_tm35 = Transformer.from_crs("EPSG:3067", "EPSG:3879", always_xy=True)
    header_row = None
    latitude_column = longitude_column = None
    for row_index, values in enumerate(raw.itertuples(index=False, name=None)):
        labels = {
            str(value).strip().casefold(): column_index
            for column_index, value in enumerate(values)
            if pd.notna(value)
        }
        latitude_column = labels.get("lat", labels.get("latitude"))
        longitude_column = labels.get("lon", labels.get("longitude"))
        if latitude_column is not None and longitude_column is not None:
            header_row = row_index
            break

    for row_index, values in enumerate(raw.itertuples(index=False, name=None)):
        if header_row is not None and row_index <= header_row:
            continue
        name = str(values[0]).strip() if pd.notna(values[0]) else ""
        if not name:
            continue

        if header_row is not None:
            try:
                latitude = _parse_number(str(values[latitude_column]))
                longitude = _parse_number(str(values[longitude_column]))
            except (TypeError, ValueError):
                continue
            x, y = to_gk25.transform(longitude, latitude)
            rows.append({"counter_name": name, "x": x, "y": y})
            continue

        numeric = [pd.to_numeric(value, errors="coerce") for value in values[1:]]
        projected = [value for value in numeric if pd.notna(value)]
        if len(projected) >= 2:
            first, second = float(projected[0]), float(projected[1])
            if -90 <= first <= 90 and -180 <= second <= 180:
                x, y = to_gk25.transform(second, first)
            elif 6_000_000 <= first <= 8_000_000 and 20_000_000 <= second <= 30_000_000:
                x, y = second, first
            elif 20_000_000 <= first <= 30_000_000 and 6_000_000 <= second <= 8_000_000:
                x, y = first, second
            elif 6_000_000 <= first <= 8_000_000 and 200_000 <= second <= 500_000:
                x, y = to_gk25_from_tm35.transform(second, first)
            else:
                continue
        else:
            text_values = [
                value.strip()
                for value in values[1:]
                if isinstance(value, str) and value.strip()
            ]
            if len(text_values) >= 2:
                try:
                    first = _parse_number(text_values[0])
                    second = _parse_number(text_values[1])
                except ValueError:
                    first = second = float("nan")
                if first > 1_000_000:
                    x, y = to_gk25_from_tm35.transform(second, first)
                    rows.append({"counter_name": name, "x": x, "y": y})
                    continue
            coordinate_text = next(
                (value for value in text_values if "," in value),
                None,
            )
            if coordinate_text is None:
                continue
            first, second = _parse_coordinate_pair(coordinate_text)
            if first > 1_000_000:
                x, y = to_gk25_from_tm35.transform(second, first)
            else:
                x, y = to_gk25.transform(second, first)
        rows.append({"counter_name": name, "x": x, "y": y})

    rows = pd.DataFrame(rows, columns=["counter_name", "x", "y"])

    duplicate_names = rows.loc[
        rows["counter_name"].duplicated(keep=False), "counter_name"
    ].unique()
    for name in duplicate_names:
        LOGGER.warning("%s: ignoring later coordinate rows for %s", path.name, name)
    rows = rows.drop_duplicates(subset="counter_name", keep="first")

    locations = {
        row.counter_name: (float(row.x), float(row.y))
        for row in rows.itertuples(index=False)
    }
    return locations


def _read_observations(
    path: Path, sheet: str
) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    frame = pd.read_excel(path, sheet_name=sheet)
    date_column = frame.columns[0]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=[date_column])

    observations: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    for counter in frame.columns[1:]:
        counter_name = str(counter).strip()
        if not counter_name or counter_name.casefold().startswith("unnamed"):
            continue
        values = pd.to_numeric(frame[counter], errors="coerce")
        observations[counter_name].extend(
            (date, float(value))
            for date, value in zip(frame[date_column], values)
            if pd.notna(value)
        )
    return observations


def _annual_average(observations: list[tuple[pd.Timestamp, float]]) -> float:
    by_date = {date.date(): value for date, value in observations}
    selected: list[float] = []
    for date in pd.date_range("2023-01-01", "2023-12-31", freq="D"):
        current = by_date.get(date.date(), np.nan)
        previous = by_date.get(date.replace(year=2022).date(), np.nan)
        if pd.notna(current) and current > 0:
            selected.append(float(current))
        elif pd.notna(previous) and previous > 0:
            selected.append(float(previous))

    if not selected:
        return float("nan")
    return float(np.mean(selected))


def _merge_nearby_counters(
    counters: pd.DataFrame, distance_m: float = COUNTER_MERGE_DISTANCE_M
) -> pd.DataFrame:
    """Merge counters within ``distance_m`` metres, separately per city."""
    merged_rows = []
    for city, group in counters.groupby("city", sort=False):
        group = group.reset_index(drop=True)
        parent = list(range(len(group)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        coordinates = group[["x", "y"]].to_numpy(dtype=float)
        for left, right in cKDTree(coordinates).query_pairs(distance_m):
            union(left, right)

        group["merge_group"] = [find(index) for index in range(len(group))]
        for _, cluster in group.groupby("merge_group", sort=False):
            first = cluster.iloc[0].copy()
            first["average_bikes_per_year"] = cluster[
                "average_bikes_per_year"
            ].sum(min_count=1)
            merged_rows.append(first.drop(labels="merge_group"))

        merged_count = len(group) - group["merge_group"].nunique()
        if merged_count:
            LOGGER.info(
                "%s: merged %d nearby counter(s) within %.0f m",
                city,
                merged_count,
                distance_m,
            )

    return pd.DataFrame(merged_rows, columns=OUTPUT_COLUMNS)


def build_output(input_directory: Path) -> pd.DataFrame:
    observations: dict[tuple[str, str], list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    locations: dict[tuple[str, str], tuple[float, float]] = {}
    files = sorted((*input_directory.glob("*.xlsx"), *input_directory.glob("*.xls")))
    if not files:
        raise FileNotFoundError(f"No Excel files found in {input_directory}.")

    for path in files:
        city = _city_name(path)
        workbook = pd.ExcelFile(path)
        pp_sheet = _find_sheet(workbook.sheet_names, "PP")
        location_sheet = _find_sheet(workbook.sheet_names, "Sijaintitiedot")
        for name, values in _read_observations(path, pp_sheet).items():
            observations[(city, name)].extend(values)
        for name, coordinate in _read_locations(path, location_sheet).items():
            locations.setdefault((city, _match_name(name)), coordinate)

    missing_locations = sorted(
        (city, name)
        for city, name in observations
        if (city, _match_name(name)) not in locations
    )
    if missing_locations:
        raise ValueError(f"Counters missing coordinates: {missing_locations}")

    rows = []
    for (city, name), values in observations.items():
        x, y = locations[(city, _match_name(name))]
        rows.append(
            {
                "city": city,
                "x": x,
                "y": y,
                "counter_name": name,
                "average_bikes_per_year": _annual_average(values),
            }
        )
    counters = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    counters = _merge_nearby_counters(counters)
    return counters.sort_values(["city", "counter_name"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("data"),
        help="Directory containing the counter Excel files (default: data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/bicycle_counter_annual_averages.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output = build_output(args.input_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    LOGGER.info("Wrote %d counter averages to %s", len(output), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())