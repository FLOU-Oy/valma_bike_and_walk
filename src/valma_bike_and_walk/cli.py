"""Command line interface.

The pipeline is two stages, and the GeoPackage between them is the point:

  valma extract --pbf finland.osm.pbf --mode bike --bbox 24.5 60.1 25.1 60.4
     -> output/bike_links.gpkg          <- open in QGIS, edit, save
  valma build   --links output/bike_links.gpkg --mode bike
     -> output/bike.npz                 <- routable graph

  valma matrix  --network output/bike.npz --mode bike --centroids points.csv
  valma assign  --network output/bike.npz --mode bike --centroids points.csv \\
                --demand od.csv --links output/bike_links.gpkg --gpkg

`build`, `matrix` and `assign` also take `--pbf` directly and run the earlier
stages themselves -- handy when there is nothing to edit. Those runs park their
intermediates under `--cache-dir`, keyed on the PBF and the extent; anything a
command names in its own output goes to `--output-dir`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from valma_bike_and_walk import links as links_module
from valma_bike_and_walk.assignment import (
    assign_traffic,
    link_volume_frame,
)
from valma_bike_and_walk.assignment import summarise as summarise_assignment
from valma_bike_and_walk.centroids import DEFAULT_MAX_SNAP_M, load_centroids
from valma_bike_and_walk.config import DEFAULT_INDEX_STORAGE, MODES, Settings
from valma_bike_and_walk.demand import (
    align_to_ids,
    clip_minimum,
    demand_matrix,
    read_demand_long,
    read_demand_npz,
    read_demand_omx,
)
from valma_bike_and_walk.gpkg import write_edges_gpkg
from valma_bike_and_walk.matrix import default_workers, summarise, travel_time_matrix
from valma_bike_and_walk.network import (
    RoutableNetwork,
    build_links,
    build_network,
    load_network,
    network_from_links,
)
from valma_bike_and_walk.osm import resolve_clip

logger = logging.getLogger("valma_bike_and_walk")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--force-reload", action="store_true", help="Ignore cached results."
    )


def _add_extent(parser: argparse.ArgumentParser) -> None:
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
        "--index-storage",
        default=DEFAULT_INDEX_STORAGE,
        help=(
            "osmium node-location index. 'flex_mem' (default) holds coordinates "
            "in memory and is right up to a country; for a planet file use a "
            "file-backed one, e.g. 'sparse_file_array,nodes.idx'."
        ),
    )


def _add_source(parser: argparse.ArgumentParser) -> None:
    """Where the network comes from: a built .npz, a link layer, or a PBF."""
    parser.add_argument(
        "--network",
        type=Path,
        help="Load an already-built network .npz (as written by 'valma build').",
    )
    parser.add_argument(
        "--links",
        type=Path,
        help=(
            "Build from this link GeoPackage (as written by 'valma extract', "
            "edits and all). Also names the layer that --gpkg draws results on."
        ),
    )
    parser.add_argument(
        "--pbf",
        type=Path,
        help="Path to the .osm.pbf extract. Runs the extract stage itself.",
    )
    _add_extent(parser)


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(
        pbf_path=getattr(args, "pbf", None),
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        index_storage=getattr(args, "index_storage", DEFAULT_INDEX_STORAGE),
    )


def _network_out(args: argparse.Namespace, settings: Settings) -> Path:
    """Where the graph built from ``--links`` is written, and looked for again."""
    out = getattr(args, "out", None)
    return Path(out) if out else settings.network_path(args.mode, args.links)


def _network_and_links(args: argparse.Namespace) -> tuple[RoutableNetwork, Path | None]:
    """Load or build the network, and say which link layer it came from."""
    if args.network is not None:
        network = load_network(args.network)
        if network.mode != args.mode:
            raise SystemExit(
                f"{args.network} holds a {network.mode!r} network but --mode is "
                f"{args.mode!r}."
            )
        logger.info(
            "Loaded %s network: %d nodes, %d edges",
            network.mode,
            network.n_nodes,
            network.n_edges,
        )
        return network, args.links

    settings = _settings(args)

    if args.links is not None:
        path = _network_out(args, settings)
        if path.exists() and not args.force_reload:
            logger.info("Loading network from %s", path)
            return load_network(path), args.links
        network = network_from_links(links_module.read_links(args.links), args.mode)
        network.save(path)
        return network, args.links

    if args.pbf is None:
        raise SystemExit(
            "Give one of --network (a built graph), --links (a link GeoPackage) "
            "or --pbf (to run both stages)."
        )

    network = build_network(
        settings,
        mode=args.mode,
        area=args.area,
        bbox=args.bbox,
        force_reload=args.force_reload,
    )
    _, extent_key = resolve_clip(settings, args.area, args.bbox)
    return network, settings.links_cache_path(args.mode, extent_key)


def cmd_extract(args: argparse.Namespace) -> int:
    settings = _settings(args)
    if args.pbf is None:
        raise SystemExit("valma extract needs --pbf.")

    out = args.out or args.output_dir / f"{args.mode}_links.gpkg"
    path, links = build_links(
        settings,
        mode=args.mode,
        area=args.area,
        bbox=args.bbox,
        force_reload=args.force_reload,
        links_path=out,
    )
    print(f"{args.mode}: {len(links):,} links -> {path}")
    print("\nEdit it in QGIS if you like, then build the routable graph:")
    print(f"  valma build --links {path} --mode {args.mode}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    network, links_path = _network_and_links(args)
    print(f"{args.mode}: {network.n_nodes:,} nodes, {network.n_edges:,} edges")

    if args.network is None:
        settings = _settings(args)
        if args.links is not None:
            path = _network_out(args, settings)
        else:
            _, extent_key = resolve_clip(settings, args.area, args.bbox)
            path = settings.network_cache_path(args.mode, extent_key)
        print(f"\nSaved to {path}")
        if links_path is not None:
            print(f"Built from {links_path}")
        print("Route with it directly, without re-reading the link layer:")
        print(
            f"  valma matrix --network {path} --mode {args.mode} "
            f"--centroids points.csv --id-column id"
        )
    return 0


def _centroids(args: argparse.Namespace, network: RoutableNetwork):
    return load_centroids(
        network,
        args.centroids,
        id_column=args.id_column,
        x_column=args.x_column,
        y_column=args.y_column,
        crs=args.centroid_crs,
        max_snap_distance=args.max_snap_distance,
    ).drop_unsnapped()


def cmd_matrix(args: argparse.Namespace) -> int:
    network, _ = _network_and_links(args)
    centroids = _centroids(args, network)

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
        f"Wrote {out} ({matrix.shape[0]}x{matrix.shape[1]}, "
        f"{out.stat().st_size / 1e6:.1f} MB)"
    )

    for key, value in summarise(matrix).items():
        print(f"  {key}: {value:,.3f}")
    return 0


def cmd_assign(args: argparse.Namespace) -> int:
    network, links_path = _network_and_links(args)
    centroids = _centroids(args, network)

    if len(centroids) == 0:
        logger.error("No centroids could be snapped to the network.")
        return 1

    if args.gpkg and links_path is None:
        raise SystemExit(
            "--gpkg draws volumes on the link layer's geometry, so it needs "
            "--links pointing at the GeoPackage this network was built from."
        )

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

    if args.gpkg:
        assert links_path is not None
        gpkg_path = args.output_dir / f"{args.mode}_volumes.gpkg"
        write_edges_gpkg(
            network, links_path, gpkg_path, extra_columns={"volume": link_volume}
        )
        print(f"Wrote {gpkg_path} ({network.n_edges:,} rows)")

    for key, value in summarise_assignment(link_volume).items():
        print(f"  {key}: {value:,.3f}")
    return 0


def _add_routing(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--centroids", type=Path, required=True, help="CSV or vector file of points."
    )
    parser.add_argument("--id-column")
    parser.add_argument("--x-column", default="lon")
    parser.add_argument("--y-column", default="lat")
    parser.add_argument("--centroid-crs", default="EPSG:4326")
    parser.add_argument("--max-snap-distance", type=float, default=DEFAULT_MAX_SNAP_M)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--chunk-size", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="valma",
        description=__doc__,
        # The description is a worked example with its own layout; argparse
        # would otherwise reflow it into a wall of text.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser(
        "extract",
        help="Stage 1: read the PBF into an editable link GeoPackage.",
    )
    _add_common(extract)
    _add_extent(extract)
    extract.add_argument(
        "--pbf", type=Path, required=True, help="Path to the .osm.pbf extract."
    )
    extract.add_argument(
        "--out",
        type=Path,
        help="Where to write the link layer (default <output-dir>/<mode>_links.gpkg).",
    )
    extract.set_defaults(func=cmd_extract)

    build = sub.add_parser(
        "build", help="Stage 2: turn a link GeoPackage into a routable graph."
    )
    _add_common(build)
    _add_source(build)
    build.add_argument(
        "--out",
        type=Path,
        help=(
            "Where to write the graph (default <output-dir>/<mode>.npz, named "
            "after the link layer). Only applies when building from --links."
        ),
    )
    build.set_defaults(func=cmd_build)

    matrix = sub.add_parser("matrix", help="Compute an OD travel-time matrix.")
    _add_common(matrix)
    _add_source(matrix)
    _add_routing(matrix)
    matrix.add_argument(
        "--max-minutes",
        type=float,
        help="Cut off the search here. The single biggest speed-up available.",
    )
    matrix.set_defaults(func=cmd_matrix)

    assign = sub.add_parser(
        "assign",
        help="Assign an OD demand matrix onto the network's shortest paths.",
    )
    _add_common(assign)
    _add_source(assign)
    _add_routing(assign)
    assign.add_argument(
        "--gpkg",
        action="store_true",
        help=(
            "Also write link volumes to <output-dir>/<mode>_volumes.gpkg, drawn "
            "on the link layer's own geometry. Needs --links."
        ),
    )
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
