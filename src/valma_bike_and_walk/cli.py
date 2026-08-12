"""Command line interface.

valma build  --pbf finland.osm.pbf --mode bike --area Espoo
valma matrix --pbf finland.osm.pbf --mode walk --centroids points.csv
valma assign --pbf finland.osm.pbf --mode bike --centroids points.csv --demand od.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from valma_bike_and_walk.assignment import (
    assign_traffic,
    link_volume_frame,
)
from valma_bike_and_walk.assignment import summarise as summarise_assignment
from valma_bike_and_walk.centroids import DEFAULT_MAX_SNAP_M, load_centroids
from valma_bike_and_walk.config import FINLAND_BOUNDS, MODES, Settings
from valma_bike_and_walk.demand import (
    align_to_ids,
    clip_minimum,
    demand_matrix,
    read_demand_long,
    read_demand_npz,
    read_demand_omx,
)
from valma_bike_and_walk.extract import resolve_clip
from valma_bike_and_walk.gpkg import write_edges_gpkg
from valma_bike_and_walk.matrix import default_workers, summarise, travel_time_matrix
from valma_bike_and_walk.network import (
    RoutableNetwork,
    build_network,
    build_network_tiled,
    load_network,
)

logger = logging.getLogger("valma_bike_and_walk")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pbf",
        type=Path,
        help="Path to the .osm.pbf extract. Not needed when --network is given.",
    )
    parser.add_argument(
        "--network",
        type=Path,
        help=(
            "Load an already-built network .npz (as written by 'valma build') "
            "instead of reading the PBF. Skips parsing entirely."
        ),
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--area", help="Administrative area to clip to, looked up in the PBF."
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Clip to this box. Much cheaper than --area on a country extract.",
    )
    parser.add_argument(
        "--search-bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Coarse box that speeds up the --area boundary lookup. Must fully contain the area.",
    )
    parser.add_argument(
        "--force-reload", action="store_true", help="Ignore cached results."
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help=(
            "Read the extent one tile at a time. Peak memory is then set by the "
            "busiest tile rather than by the whole area, which is what makes a "
            "country-wide build fit in RAM. Costs one PBF pass per tile."
        ),
    )
    parser.add_argument(
        "--tile-degrees",
        type=float,
        default=2.0,
        help="Tile size for --tiled. Use the largest your memory allows.",
    )
    parser.add_argument(
        "--tile-workers",
        type=int,
        default=1,
        help=(
            "Read this many tiles at once. Worth raising for a tiled build: a "
            "per-tile bounding box leaves most of the reader's own workers idle, "
            "so running whole tiles concurrently uses the machine far better. "
            "Peak memory is roughly this many times the busiest tile."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "out_of_core", "in_memory"),
        default="auto",
        help=(
            "PBF reader. 'out_of_core' uses every core but decodes the whole "
            "file to temporary shards on each read (~1.3 GB of disk writes per "
            "read of a 190 MB extract). 'in_memory' is single-core but never "
            "spills. 'auto' (default) picks in_memory for parallel tiled builds, "
            "where concurrent spilling would saturate the disk, and out_of_core "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--osm-workers",
        default="auto",
        help="Parse workers for --engine out_of_core: an integer or 'auto' (default).",
    )


def _osm_workers(value: str) -> int | str | None:
    """argparse hands us a string; pyrosm wants an int, "auto" or None."""
    if value in ("auto", "none", "None"):
        return None if value != "auto" else "auto"
    try:
        return int(value)
    except ValueError:
        raise SystemExit(
            f"--osm-workers must be an integer or 'auto', got {value!r}."
        ) from None


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(
        pbf_path=args.pbf,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        engine=args.engine,
        osm_workers=_osm_workers(args.osm_workers),
    )


def _tiled_bounds(args: argparse.Namespace) -> tuple[float, ...]:
    return tuple(args.bbox) if args.bbox else FINLAND_BOUNDS


def _tiled_cache_key(args: argparse.Namespace) -> str:
    if not args.bbox:
        return "tiled"
    return "tiled_" + "_".join(f"{c:.2f}" for c in _tiled_bounds(args))


def _built_path(args: argparse.Namespace):
    """Where a build with these arguments puts its network."""
    settings = _settings(args)
    geometry = getattr(args, "gpkg", False)
    if args.tiled:
        return settings.network_cache_path(
            args.mode, _tiled_cache_key(args), geometry=geometry
        )
    _, extent_key = resolve_clip(settings, args.area, args.bbox, args.search_bbox)
    return settings.network_cache_path(args.mode, extent_key, geometry=geometry)


def _network(args: argparse.Namespace) -> RoutableNetwork:
    """Load or build the network, whichever the arguments ask for."""
    keep_geometry = getattr(args, "gpkg", False)

    if args.network is not None:
        network = load_network(args.network)
        if network.mode != args.mode:
            raise SystemExit(
                f"{args.network} holds a {network.mode!r} network but --mode is {args.mode!r}."
            )
        if keep_geometry and network.geometry is None:
            raise SystemExit(
                f"{args.network} was built without geometry, so --gpkg has nothing to "
                "export. Rebuild it with --gpkg (from --pbf) instead of --network."
            )
        logger.info(
            "Loaded %s network: %d nodes, %d edges",
            network.mode,
            network.n_nodes,
            network.n_edges,
        )
        return network

    if args.pbf is None:
        raise SystemExit(
            "Give either --pbf (to build) or --network (to load a built one)."
        )

    settings = _settings(args)
    if args.tiled:
        return build_network_tiled(
            settings,
            mode=args.mode,
            bounds=_tiled_bounds(args),
            tile_degrees=args.tile_degrees,
            force_reload=args.force_reload,
            cache_key=_tiled_cache_key(args),
            tile_workers=args.tile_workers,
            keep_geometry=keep_geometry,
        )
    return build_network(
        settings,
        mode=args.mode,
        area=args.area,
        bbox=args.bbox,
        search_bbox=args.search_bbox,
        force_reload=args.force_reload,
        keep_geometry=keep_geometry,
    )


def cmd_build(args: argparse.Namespace) -> int:
    network = _network(args)
    print(f"{args.mode}: {network.n_nodes:,} nodes, {network.n_edges:,} edges")

    if args.pbf is not None:
        path = _built_path(args)
        print(f"\nSaved to {path}")
        print("Route with it directly, without re-reading the PBF:")
        print(
            f"  valma matrix --network {path} --mode {args.mode} "
            f"--centroids points.csv --id-column id"
        )

    if args.gpkg:
        gpkg_path = args.output_dir / f"{args.mode}_links.gpkg"
        write_edges_gpkg(network, gpkg_path)
        print(f"\nWrote links to {gpkg_path} ({network.n_edges:,} rows)")
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    network = _network(args)

    centroids = load_centroids(
        network,
        args.centroids,
        id_column=args.id_column,
        x_column=args.x_column,
        y_column=args.y_column,
        crs=args.centroid_crs,
        max_snap_distance=args.max_snap_distance,
    ).drop_unsnapped()

    if len(centroids) == 0:
        logger.error("No centroids could be snapped to the network.")
        return 1

    matrix = travel_time_matrix(
        network,
        centroids.node_index,
        max_seconds=args.max_minutes * 60 if args.max_minutes else None,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"travel_times_{args.mode}.npz"
    np.savez_compressed(out, ids=centroids.ids, seconds=matrix)
    print(
        f"Wrote {out} ({matrix.shape[0]}x{matrix.shape[1]}, {out.stat().st_size / 1e6:.1f} MB)"
    )

    for key, value in summarise(matrix).items():
        print(f"  {key}: {value:,.3f}")
    return 0


def cmd_assign(args: argparse.Namespace) -> int:
    network = _network(args)

    centroids = load_centroids(
        network,
        args.centroids,
        id_column=args.id_column,
        x_column=args.x_column,
        y_column=args.y_column,
        crs=args.centroid_crs,
        max_snap_distance=args.max_snap_distance,
    ).drop_unsnapped()

    if len(centroids) == 0:
        logger.error("No centroids could be snapped to the network.")
        return 1

    suffix = args.demand.suffix.lower()
    if suffix == ".omx":
        if not args.demand_matrix:
            raise SystemExit("--demand-matrix is required for an OMX --demand file.")
        ids, demand = read_demand_omx(
            args.demand,
            matrix_name=args.demand_matrix,
            mapping_name=args.demand_mapping,
            minimum_demand=args.min_demand,
        )
        demand = align_to_ids(ids, demand, centroids.ids)
    elif suffix == ".npz":
        ids, demand = read_demand_npz(args.demand)
        demand = align_to_ids(ids, demand, centroids.ids)
    else:
        long_demand = read_demand_long(
            args.demand,
            origin_column=args.origin_column,
            destination_column=args.destination_column,
            demand_column=args.demand_column,
        )
        demand = demand_matrix(long_demand, centroids.ids)

    demand = clip_minimum(demand, args.min_demand)

    link_volume = assign_traffic(
        network,
        centroids.node_index,
        demand,
        max_seconds=args.max_minutes * 60 if args.max_minutes else None,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = link_volume_frame(network, link_volume)
    out = args.output_dir / f"link_volumes_{args.mode}.npz"
    np.savez_compressed(out, **frame)  # type: ignore[arg-type]
    print(f"Wrote {out} ({network.n_edges:,} links, {out.stat().st_size / 1e6:.1f} MB)")

    if network.geometry is not None:
        gpkg_path = args.output_dir / f"{args.mode}_volumes.gpkg"
        write_edges_gpkg(network, gpkg_path, extra_columns={"volume": link_volume})
        print(f"Wrote {gpkg_path} ({network.n_edges:,} rows)")

    for key, value in summarise_assignment(link_volume).items():
        print(f"  {key}: {value:,.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="valma", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build and cache a routable network.")
    _add_common(build)
    build.add_argument(
        "--gpkg",
        action="store_true",
        help=(
            "Also write the network's links to <output-dir>/<mode>_links.gpkg, "
            "with geometry plus length, travel time and speed -- for "
            "visualizing the network and routing costs in GIS software. "
            "Carries road geometry through the whole build, so it costs extra "
            "memory and gets its own cache files (separate from a plain "
            "build's)."
        ),
    )
    build.set_defaults(func=cmd_build)

    matrix = sub.add_parser("matrix", help="Compute an OD travel-time matrix.")
    _add_common(matrix)
    matrix.add_argument(
        "--centroids", type=Path, required=True, help="CSV or vector file of points."
    )
    matrix.add_argument("--id-column")
    matrix.add_argument("--x-column", default="lon")
    matrix.add_argument("--y-column", default="lat")
    matrix.add_argument("--centroid-crs", default="EPSG:4326")
    matrix.add_argument("--max-snap-distance", type=float, default=DEFAULT_MAX_SNAP_M)
    matrix.add_argument(
        "--max-minutes",
        type=float,
        help="Cut off the search here. The single biggest speed-up available.",
    )
    matrix.add_argument("--workers", type=int, default=default_workers())
    matrix.add_argument("--chunk-size", type=int)
    matrix.set_defaults(func=cmd_matrix)

    assign = sub.add_parser(
        "assign",
        help="Assign an OD demand matrix onto the network's shortest paths (link volumes).",
    )
    _add_common(assign)
    assign.add_argument(
        "--gpkg",
        action="store_true",
        help=(
            "Also write link volumes to <output-dir>/<mode>_volumes.gpkg for "
            "mapping. Needs a network built with --gpkg (or --network "
            "pointing at one)."
        ),
    )
    assign.add_argument(
        "--centroids", type=Path, required=True, help="CSV or vector file of points."
    )
    assign.add_argument("--id-column")
    assign.add_argument("--x-column", default="lon")
    assign.add_argument("--y-column", default="lat")
    assign.add_argument("--centroid-crs", default="EPSG:4326")
    assign.add_argument("--max-snap-distance", type=float, default=DEFAULT_MAX_SNAP_M)
    assign.add_argument(
        "--demand",
        type=Path,
        required=True,
        help=(
            "OD demand: a CSV/TSV of (origin, destination, demand) rows -- "
            "only nonzero demand needs a row -- a .npz with 'ids' and a "
            "dense 'demand' matrix, or a .omx file (see --demand-matrix)."
        ),
    )
    assign.add_argument("--origin-column", default="origin_id")
    assign.add_argument("--destination-column", default="destination_id")
    assign.add_argument("--demand-column", default="demand")
    assign.add_argument(
        "--demand-matrix",
        help=(
            "Name of the matrix inside the OMX file, e.g. 'bike'. Required "
            "when --demand is a .omx file; ignored otherwise."
        ),
    )
    assign.add_argument(
        "--demand-mapping",
        default="zone_number",
        help=(
            "Name of the OMX lookup that maps matrix row/column position to "
            "zone/centroid id (default: 'zone_number')."
        ),
    )
    assign.add_argument(
        "--min-demand",
        type=float,
        default=0.0,
        help=(
            "Clip OD pairs with demand below this to zero before assigning, "
            "so they don't need a shortest path built at all. Useful for "
            "screening a large, mostly-noise demand matrix."
        ),
    )
    assign.add_argument(
        "--max-minutes",
        type=float,
        help="Don't search a source beyond this travel time. OD pairs further "
        "apart than this are dropped, unassigned.",
    )
    assign.add_argument("--workers", type=int, default=default_workers())
    assign.add_argument("--chunk-size", type=int)
    assign.set_defaults(func=cmd_assign)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
