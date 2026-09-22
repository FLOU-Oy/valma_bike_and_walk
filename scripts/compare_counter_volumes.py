"""Compare modeled bike volumes with observed bicycle counter averages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

COUNTER_LAYER = "count_lines"
VOLUME_LAYER = "links"
EXPANSION_FACTOR = 1 #/ 0.547 * 2 / 0.407589
PROJECTED_CRS = "EPSG:3879"


def _match_name(name: str) -> str:
    return re.sub(r"\s*PP$", "", str(name).strip(), flags=re.IGNORECASE).casefold()


def _read_observed(path: Path) -> pd.DataFrame:
    observed = pd.read_csv(path, encoding="utf-8-sig")
    required = {"x", "y", "counter_name", "average_bikes_per_year"}
    missing = required - set(observed.columns)
    if missing:
        raise ValueError(f"Observed CSV is missing columns: {sorted(missing)}")
    observed["match_name"] = observed["counter_name"].map(_match_name)
    return observed


def _modeled_volumes(
    count_lines_path: Path, bike_volumes_path: Path
) -> pd.DataFrame:
    counters = gpd.read_file(count_lines_path, layer=COUNTER_LAYER)
    volumes = gpd.read_file(bike_volumes_path, layer=VOLUME_LAYER)
    if "counter" not in counters.columns:
        raise ValueError("count_lines.gpkg must contain a 'counter' column.")
    if "volume" not in volumes.columns:
        raise ValueError("bike_volumes.gpkg must contain a 'volume' column.")

    volumes["volume"] = pd.to_numeric(volumes["volume"], errors="coerce").fillna(0.0)
    volumes["modeled_bike_volume"] = volumes["volume"] * EXPANSION_FACTOR
    volumes.to_file(
        bike_volumes_path,
        layer=VOLUME_LAYER,
        driver="GPKG",
        mode="w",
    )

    counters = counters[["counter", "geometry"]].to_crs(PROJECTED_CRS)
    volumes = volumes[["volume", "geometry"]].to_crs(PROJECTED_CRS)

    joined = gpd.sjoin(
        volumes,
        counters.rename(columns={"counter": "counter_name"}),
        how="inner",
        predicate="intersects",
    )
    totals = joined.groupby("index_right", sort=False)["volume"].sum()
    counters["modeled_bike_volume"] = counters.index.map(totals).fillna(0.0)
    counters["modeled_bike_volume"] *= EXPANSION_FACTOR
    counters["match_name"] = counters["counter"].map(_match_name)
    return counters.rename(columns={"counter": "counter_name"})[
        ["counter_name", "match_name", "modeled_bike_volume", "geometry"]
    ]


def compare(
    count_lines_path: Path,
    bike_volumes_path: Path,
    observed_path: Path,
) -> gpd.GeoDataFrame:
    modeled = _modeled_volumes(count_lines_path, bike_volumes_path)
    observed = _read_observed(observed_path)

    # Geometry comes from x/y in bicycle_counter_annual_averages.csv
    observed_points = gpd.GeoDataFrame(
        observed,
        geometry=gpd.points_from_xy(observed["x"], observed["y"]),
        crs=PROJECTED_CRS,
    )

    rows = []
    for _, counter in modeled.iterrows():
        candidates = observed_points[
            observed_points["match_name"] == counter["match_name"]
        ].copy()

        if candidates.empty:
            raise ValueError(
                f"No observed counter matches {counter['counter_name']!r}."
            )

        # If multiple observed counters have the same name,
        # choose the one closest to the modeled count line.
        distances = candidates.geometry.distance(counter.geometry)
        observed_row = candidates.loc[distances.idxmin()]

        rows.append(
            {
                "city": observed_row.get("city", ""),
                "counter_name": counter["counter_name"],
                "modeled_bike_volume": counter["modeled_bike_volume"],
                "observed_bikes_per_year": observed_row["average_bikes_per_year"],
                "distance_to_observed_m": float(
                    distances.loc[observed_row.name]
                ),
                # IMPORTANT: use geometry from the observed CSV point
                "geometry": observed_row.geometry,
            }
        )

    return gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=PROJECTED_CRS,
    )


def plot_comparison(data: pd.DataFrame, path: Path, title: str | None = None) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    cities = sorted(data["city"].dropna().unique())
    colors = plt.get_cmap("tab20", len(cities))
    for color_index, city in enumerate(cities):
        city_data = data[data["city"] == city]
        axis.scatter(
            city_data["observed_bikes_per_year"],
            city_data["modeled_bike_volume"],
            label=city,
            color=colors(color_index),
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
        )
    for _, row in data.iterrows():
        axis.annotate(
            row["counter_name"],
            (row["observed_bikes_per_year"], row["modeled_bike_volume"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=4,
        )
    maximum = float(
        np.nanmax(
            [data["observed_bikes_per_year"].max(), data["modeled_bike_volume"].max()]
        )
    )
    axis.plot([0, maximum], [0, maximum], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Observed average bicycles per day", fontsize=8)
    axis.set_ylabel("Modeled bicycles per day (expanded)", fontsize=8)
    axis.set_title(
        title or "Modeled versus observed bicycle counter volumes", fontsize=10
    )
    axis.tick_params(axis="both", labelsize=7)
    if data["city"].nunique() > 1:
        axis.legend(title="City", fontsize=7, title_fontsize=8, loc="best")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-lines", type=Path, default=Path("data/count_lines.gpkg"))
    parser.add_argument("--bike-volumes", type=Path, default=Path("output/bike_volumes.gpkg"))
    parser.add_argument(
        "--observed",
        type=Path,
        default=Path("output/bicycle_counter_annual_averages.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/counter_volume_comparison.csv"),
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("output/counter_volume_scatter.png"),
    )
    parser.add_argument(
        "--city-plots-directory",
        type=Path,
        default=Path("output/counter_volume_scatter_by_city"),
        help="Directory for one scatterplot per city.",
    )
    parser.add_argument(
    "--output-gpkg",
    type=Path,
    default=Path("output/counter_volume_comparison.gpkg"),
    )
    args = parser.parse_args()
    comparison = compare(args.count_lines, args.bike_volumes, args.observed)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    # CSV without GeoPandas/Shapely geometry
    comparison.drop(columns="geometry").to_csv(
        args.output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # GeoPackage point layer
    args.output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_file(
        args.output_gpkg,
        layer="counter_volume_comparison",
        driver="GPKG",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    plot_comparison(comparison, args.output_plot)
    args.city_plots_directory.mkdir(parents=True, exist_ok=True)
    for city, city_data in comparison.groupby("city", sort=True):
        filename = re.sub(r"[^\w.-]+", "_", str(city), flags=re.UNICODE)
        plot_comparison(
            city_data,
            args.city_plots_directory / f"{filename}.png",
            title=f"{city}: modeled versus observed bicycle volumes",
        )
    print(f"Wrote {len(comparison):,} counter comparisons to {args.output_csv}")
    print(f"Wrote scatterplot to {args.output_plot}")
    print(f"Wrote city scatterplots to {args.city_plots_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())