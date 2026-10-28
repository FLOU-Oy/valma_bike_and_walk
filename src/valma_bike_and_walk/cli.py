"""Command line interface.

The pipeline is two stages, and the GeoPackage between them is the point:

  valma extract --pbf finland.osm.pbf --mode bike --bbox 24.5 60.1 25.1 60.4
     -> output/bike_links.gpkg          <- open in QGIS, edit, save

  valma dem     --links output/bike_links.gpkg
     -> the same file, + z_u/z_v/ascent_m/descent_m  (optional; needs a key)

  valma traffic --links output/bike_links.gpkg --volumes model_links.gpkg
     -> the same file, + car_volume on the roads that carry cars   (optional)

  valma connect --links output/bike_links.gpkg
     -> output/bike_connectors.gpkg: islands joined to the rest (optional;
        pass it on with --connectors, or islands are dropped from the graph)

  valma matrix  --links output/bike_links.gpkg --mode bike --centroids points.csv
  valma assign  --links output/bike_links.gpkg --mode bike --centroids points.csv \\
                --demand od.csv --gpkg

`matrix` and `assign` route between zones given either way round. `--centroids`
takes one point per zone, the classic method. `--zones` takes polygons instead
and spreads each zone's trip ends over several weighted points inside it, which
is what stops volumes piling onto one centroid node, gets short trips right and
lets intrazonal demand load onto the network at all:

  valma matrix  --links output/bike_links.gpkg --mode bike \\
                --zones taz.gpkg --id-column zone_id --near-minutes 15

`matrix` and `assign` build the routable graph from `--links` themselves and
cache it under `--cache-dir`, so there is normally no need to know it exists.
Running `valma build --links output/bike_links.gpkg --mode bike` explicitly
instead writes it to `--output-dir` as `bike.npz` -- a named, reusable graph
worth keeping around when many `matrix`/`assign` runs will share it. Either
way, point later runs at it directly with `--network output/bike.npz` to skip
re-reading the link layer.

`build`, `matrix` and `assign` also take `--pbf` directly and run the earlier
stages themselves -- handy when there is nothing to edit. Those runs park their
intermediates under `--cache-dir`, keyed on the PBF and the extent; anything a
command names in its own output goes to `--output-dir`.

Any option can also come from a TOML settings file, and several can be layered
-- a base, a subarea, a scenario -- with the command line on top:

  valma assign -c scenarios/base.toml -c scenarios/2040.toml --gpkg

`--show-config` prints what a command would run with, and where each value
came from. `extract`, `build`, `matrix`, `assign` and `overlay` leave the
settings they ran with in `--output-dir` as `valma_<command>_<mode>.toml`,
which `-c` reads back.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np

from valma_bike_and_walk import config_file
from valma_bike_and_walk import links as links_module
from valma_bike_and_walk.assignment import (
    select_link_assignment,
    assign_zone_traffic,
    link_volume_frame,
    trip_length_assignment,
)
from valma_bike_and_walk.assignment import summarise as summarise_assignment
from valma_bike_and_walk.cache import file_stamp, load_key
from valma_bike_and_walk.centroids import (
    DEFAULT_MAX_SNAP_M,
    Centroids,
    load_centroids,
)
from valma_bike_and_walk.connectors import (
    DEFAULT_CONNECTOR_SPEED_KMH,
    DEFAULT_GAP_M,
    DEFAULT_MIN_COMPONENT_NODES,
    GAP,
    KIND_COLUMN,
    island_connectors,
    island_link_ids,
    merge_connectors,
)
from valma_bike_and_walk.config import (
    DEFAULT_INDEX_STORAGE,
    MODES,
    PROJECTED_CRS,
    WGS84,
    Settings,
)
from valma_bike_and_walk.demand import (
    align_to_ids,
    clip_minimum,
    demand_matrix,
    read_demand_long,
    read_demand_npz,
    read_demand_omx,
    write_omx_matrix,
)
from valma_bike_and_walk.elevation import (
    API_KEY_ENV,
    DEFAULT_RESOLUTION_M,
    DEFAULT_TILE_M,
    INSECURE_ENV,
    RESOLUTIONS_M,
    DemCache,
    add_elevation,
    tiles_for_bounds,
)
from valma_bike_and_walk.gpkg import write_edges_gpkg
from valma_bike_and_walk.matrix import (
    default_workers,
    summarise,
    travel_time_matrix,
    zone_travel_time_matrix,
)
from valma_bike_and_walk.network import (
    NetworkFormatError,
    RoutableNetwork,
    build_links,
    build_network,
    load_network,
    network_for_links,
)
from valma_bike_and_walk.osm import resolve_clip
from valma_bike_and_walk.speeds import profile_for
from valma_bike_and_walk.overlay import overlay, read_model_minutes, read_omx_dense
from valma_bike_and_walk.traffic import (
    DEFAULT_ADJACENT_DISTANCE_M,
    DEFAULT_MAX_ANGLE_DEG,
    DEFAULT_MAX_DISTANCE_M,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_SPACING_M,
    DEFAULT_VOLUME_COLUMN,
    VOLUME_COLUMN,
    MatchSettings,
    add_volumes,
    read_volumes,
    summarise as summarise_volumes,
    unmatched_sources,
    volumes_from_assignment,
)
from valma_bike_and_walk.zones import (
    DEFAULT_POINTS_PER_ZONE,
    Zones,
    load_zones,
    zone_points_frame,
)
from valma_bike_and_walk.vector_io import write_vector

logger = logging.getLogger("valma_bike_and_walk")

#: Sentinel for `matrix --omx` given bare (no path): write the default name,
#: distinguishable from "not given at all" (args.omx is None) and from an
#: explicit path (a Path instance).
_OMX_DEFAULT_PATH = object()

#: Sentinel for `overlay --baseline-speed-kmh` given bare: use the mode's base
#: speed from its speed profile.
_MODE_BASE_SPEED = object()


#: Commands that leave the settings they ran with next to their results.
_RECORDED = ("extract", "build", "matrix", "assign", "overlay")


class _TakenEarly(argparse.Action):
    """
    ``-c`` and ``--show-config``, as far as argparse is concerned.

    ``main`` takes both off the command line before parsing it, since the files
    decide what the parser accepts. Reaching this means one was abbreviated
    past what that early pass recognises.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"write {option_string} in full: -c/--config or --show-config")


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        action=_TakenEarly,
        metavar="FILE",
        default=argparse.SUPPRESS,
        help=(
            "TOML settings file: any option, spelled without the dashes. Give "
            "several to layer them, later over earlier; options given here "
            "still win."
        ),
    )
    parser.add_argument(
        "--show-config",
        action=_TakenEarly,
        nargs=0,
        default=argparse.SUPPRESS,
        help="Print the settings this would run with, and where each came from; run nothing.",
    )


def _add_output_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--scenario",
        help=(
            "Name of the scenario. Results go to <output-dir>/<scenario>/, so "
            "scenarios do not overwrite one another. Usually set in a settings "
            "file."
        ),
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    _add_output_dir(parser)
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help=(
            "Rebuild cached results even if their inputs are unchanged. Not "
            "needed after editing a link layer: that is detected."
        ),
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
    parser.add_argument(
        "--include-construction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Read ways under roadworks (highway=construction) as what they are "
            "being built into, e.g. construction=cycleway as a cycleway, and "
            "mark them under_construction (default: on). Roadworks are "
            "temporary; dropping them leaves holes in the network. Note that "
            "the tag also covers new ways that have not opened yet."
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
            "edits and all). The graph is cached under --cache-dir, so there "
            "is no separate 'valma build' step to run first. Also names the "
            "layer that --gpkg draws results on."
        ),
    )
    parser.add_argument(
        "--connectors",
        type=Path,
        help=(
            "Connector link layer to add to --links, as written by 'valma "
            "connect'. Keeps islands in the graph instead of dropping them "
            "with the other disconnected parts."
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
        include_construction=getattr(args, "include_construction", True),
    )


def _network_out(args: argparse.Namespace, settings: Settings) -> Path:
    """Where `valma build` writes the graph, and looks for it again."""
    out = getattr(args, "out", None)
    if out:
        return Path(out)
    return settings.network_path(args.mode, args.links, args.connectors)


def _link_layer(args: argparse.Namespace) -> gpd.GeoDataFrame:
    """``--links``, with ``--connectors`` appended when given."""
    links = links_module.read_links(args.links)
    if args.connectors is not None:
        links = merge_connectors(links, links_module.read_links(args.connectors))
    return links


def _warn_if_built_from_another_layer(
    network: RoutableNetwork, network_path: Path, links_path: Path
) -> None:
    """
    An explicit --network is used as given, but say so if --links has moved on.

    Results are drawn onto --links by link_id, so a graph built from an older
    version of the layer can put volumes on rows that have since changed.
    """
    stored = load_key(network.source_key)
    if stored is None or stored.get("links") == file_stamp(links_path):
        return
    logger.warning(
        "%s was not built from %s as it is now (edited since?). Results are "
        "joined back on link_id and may land on the wrong rows; rebuild with "
        "`valma build --links %s --mode %s`.",
        network_path,
        links_path,
        links_path,
        network.mode,
    )


def _network_and_links(
    args: argparse.Namespace, *, explicit_build: bool = False
) -> tuple[RoutableNetwork, Path | None]:
    """Load or build the network, and say which link layer it came from."""
    if args.network is not None:
        try:
            network = load_network(args.network)
        except NetworkFormatError as error:
            raise SystemExit(str(error)) from None
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
        if args.links is not None:
            _warn_if_built_from_another_layer(network, args.network, args.links)
        return network, args.links

    settings = _settings(args)

    if args.links is not None:
        if explicit_build:
            # `valma build`: a named, reusable deliverable, next to the results.
            path = _network_out(args, settings)
        else:
            # `matrix`/`assign` building it themselves, with no `valma build`
            # in between: a cache, not a deliverable, so callers need not know
            # it exists.
            path = settings.network_cache_path_for_links(
                args.mode, args.links, args.connectors
            )
        # Reused only while it was built from this very version of the layer;
        # an edit, `valma dem` or `valma traffic` since then rebuilds it.
        network = network_for_links(
            args.links,
            args.mode,
            path,
            force_reload=args.force_reload,
            compress=explicit_build,
            connectors_path=args.connectors,
        )
        return network, args.links

    if args.pbf is None:
        raise SystemExit(
            "Give one of --network (a built graph), --links (a link GeoPackage) "
            "or --pbf (to run both stages)."
        )
    if args.connectors is not None:
        raise SystemExit(
            "--connectors joins a link layer; run 'valma extract' and pass "
            "--links instead of --pbf."
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
    print("\nEdit it in QGIS if you like, then route against it directly:")
    print(
        f"  valma matrix --links {path} --mode {args.mode} "
        "--centroids points.csv --id-column id"
    )
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    links = links_module.read_links(args.links)
    connectors = island_connectors(
        links, min_nodes=args.min_nodes, speed_kmh=args.speed_kmh, gap_m=args.gap_m
    )
    out = args.out or args.links.with_name(
        f"{args.links.stem.removesuffix('_links')}_connectors.gpkg"
    )
    if connectors.empty:
        print("The network is already in one piece; no connectors needed.")
        return 0
    links_module.write_links(connectors, out)
    gaps = connectors[KIND_COLUMN] == GAP
    islands = connectors.loc[~gaps, "length_m"]
    print(f"{len(connectors):,} connectors -> {out}")
    print(
        f"  {int(gaps.sum()):,} gap(s) under {args.gap_m:g} m, at the mode's "
        "normal speed"
    )
    if not islands.empty:
        print(
            f"  {len(islands):,} island crossing(s), median {islands.median():,.0f} m, "
            f"longest {islands.max():,.0f} m, at {args.speed_kmh:g} km/h"
        )
    print("\nRoute with it by adding it to the link layer:")
    print(f"  valma matrix --links {args.links} --connectors {out} --mode <mode> ...")
    return 0


def cmd_overlay(args: argparse.Namespace) -> int:
    model_path = args.model or args.output_dir / f"travel_times_{args.mode}.npz"
    base_ids, base = read_omx_dense(
        args.baseline, args.baseline_matrix, args.baseline_mapping
    )
    speed_kmh = args.baseline_speed_kmh
    if speed_kmh is _MODE_BASE_SPEED:
        # Converted at the same pace the model walks or rides at, so a cell
        # does not change just for having come from the baseline.
        speed_kmh = profile_for(args.mode).base_kph
    if speed_kmh is not None:
        # A distance matrix, in km: turn it into minutes at a fixed speed.
        logger.info("Baseline taken as km, at %g km/h", speed_kmh)
        base *= np.float32(60.0 / speed_kmh)
    model_ids, model = read_model_minutes(model_path)

    result, replaced = overlay(
        base_ids, base, model_ids, model, max_minutes=args.max_minutes
    )
    name = args.matrix_name or args.mode
    write_omx_matrix(
        args.out,
        base_ids,
        result,
        matrix_name=name,
        mapping_name=args.baseline_mapping,
        units="minutes",
        append=True,
    )
    print(
        f"Wrote {name!r} to {args.out} ({result.shape[0]}x{result.shape[1]}, minutes)"
    )
    print(
        f"  {replaced:,} of {result.size:,} cells ({replaced / result.size:.1%}) "
        f"from {model_path}, the rest from {args.baseline} {args.baseline_matrix!r}"
    )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    network, links_path = _network_and_links(args, explicit_build=True)
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


def _dem_cache(args: argparse.Namespace, settings: Settings) -> DemCache:
    return DemCache(
        root=settings.dem_cache_dir,
        resolution_m=args.dem_resolution,
        tile_m=args.dem_tile,
        api_key=args.api_key,
        insecure=args.insecure or None,
    )


def _links_mode(args: argparse.Namespace) -> str:
    """
    The mode a link layer was extracted for, to refresh its derived columns.

    ``--mode`` if given, else the ``<mode>_links.gpkg`` name ``valma extract``
    writes by default, else bike. Only the styling columns depend on it; the
    graph recomputes speeds and times from the tags on every build anyway.
    """
    if args.mode is not None:
        return str(args.mode)
    prefix = Path(args.links).stem.split("_", 1)[0]
    return prefix if prefix in MODES else "bike"


def cmd_dem(args: argparse.Namespace) -> int:
    settings = _settings(args)
    cache = _dem_cache(args, settings)

    if args.links is not None:
        logger.info("Reading link layer from %s", args.links)
        links = links_module.read_links(args.links)
        links = add_elevation(
            links, cache, workers=args.dem_workers, remeasure=args.remeasure_all
        )
        links = links_module.normalise(links, _links_mode(args))
        out = args.out or args.links
        logger.info("Writing link layer to %s", out)
        links_module.write_links(links, out)
        print(f"Elevation written to {out} ({len(links):,} links)")
        rise, _ = links_module.link_climb(links)
        print(f"  total ascent: {np.nansum(rise):,.0f} m")
        print(f"  steepest link: {np.nanmax(rise):,.1f} m of climb")
    else:
        if args.bbox is None and args.area is None:
            raise SystemExit(
                "valma dem needs --links (to attach elevation to a link layer) "
                "or --bbox/--area (to pre-fill the tile cache)."
            )
        clip, _ = resolve_clip(settings, args.area, args.bbox)
        assert clip is not None
        bounds = gpd.GeoSeries([clip], crs=WGS84).to_crs(PROJECTED_CRS).total_bounds
        tiles = tiles_for_bounds(bounds, args.dem_resolution, args.dem_tile)
        cache.ensure(tiles, workers=args.dem_workers)
        print(f"{len(tiles):,} tile(s) cached in {cache.directory}")

    print("\nElevation data (c) National Land Survey of Finland, CC BY 4.0.")
    return 0


def cmd_traffic(args: argparse.Namespace) -> int:
    logger.info("Reading link layer from %s", args.links)
    links = links_module.read_links(args.links)
    if args.rebuild_volumes or not args.volumes.exists():
        _build_volumes(args)
        volumes = read_volumes(args.volumes)
    else:
        volumes = read_volumes(
            args.volumes, layer=args.volume_layer, column=args.volume_column
        )
    settings = MatchSettings(
        max_distance_m=args.max_distance,
        max_angle_deg=args.max_angle,
        min_coverage=args.min_coverage,
        spacing_m=args.sample_spacing,
        split_divided=not args.no_split_divided,
        adjacent_distance_m=args.adjacent_distance,
    )

    links, match = add_volumes(links, volumes, settings)

    out = args.out or args.links
    logger.info("Writing link layer to %s", out)
    links_module.write_links(links, out)
    print(f"Car volumes written to {out} ({len(links):,} links)")
    print(f"  {summarise_volumes(links, volumes, match)}")

    if args.unmatched_out is not None:
        missed = unmatched_sources(volumes, match)
        write_vector(missed, args.unmatched_out, driver="GPKG", layer="unmatched")
        print(
            f"  {len(missed):,} unmatched source link(s) -> {args.unmatched_out} "
            "(mostly roads the bike network drops, such as motorways)"
        )

    matched = links[VOLUME_COLUMN].notna()
    if matched.any():
        print(f"  busiest matched link: {links.loc[matched, VOLUME_COLUMN].max():,.0f}")
    return 0


def _build_volumes(args: argparse.Namespace) -> None:
    """Write --volumes from Valma's assignment results, both directions summed."""
    source = args.assignment_results
    if source is None:
        why = (
            "--rebuild-volumes was given"
            if args.volumes.exists()
            else f"{args.volumes} does not exist"
        )
        raise SystemExit(
            f"valma traffic: {why}, so the volumes have to be built, but no "
            "--assignment-results names Valma's assignment results to build them from."
        )
    if not source.exists():
        raise SystemExit(f"valma traffic: no assignment results at {source}.")
    built = volumes_from_assignment(source)
    args.volumes.parent.mkdir(parents=True, exist_ok=True)
    # Removed first: a second layer left in the file would make it ambiguous.
    args.volumes.unlink(missing_ok=True)
    write_vector(built, args.volumes, driver="GPKG", layer="link_volumes")
    print(f"Built {args.volumes} from {source} ({len(built):,} roads, both directions summed)")


def _analysis_points(
    args: argparse.Namespace, network: RoutableNetwork
) -> Centroids | Zones:
    """
    Whatever the run routes between: one point per zone, or several.

    ``--centroids`` is the original method and is untouched by any of this --
    one node per zone, no access time, intrazonal trips left at zero.
    ``--zones`` takes polygons and spreads each zone's trip ends over weighted
    access points inside it; see :mod:`valma_bike_and_walk.zones`.
    """
    if args.zones is not None:
        zones = load_zones(
            network,
            args.zones,
            id_column=args.id_column,
            crs=args.centroid_crs,
            weights_path=args.weights,
            weight_column=args.weight_column,
            weight_x_column=args.x_column,
            weight_y_column=args.y_column,
            weight_crs=args.centroid_crs,
            points_per_zone=args.points_per_zone,
            access_speed_kmh=args.access_speed_kmh,
            charge_access_time=not args.no_access_time,
            max_snap_distance=args.max_snap_distance,
        )
        if args.zone_points_gpkg is not None:
            args.zone_points_gpkg.parent.mkdir(parents=True, exist_ok=True)
            write_vector(
                zone_points_frame(zones), args.zone_points_gpkg, layer="zone_points", driver="GPKG"
            )
            print(
                f"Wrote {args.zone_points_gpkg} "
                f"({zones.n_points:,} access points across {zones.n_zones:,} zones)"
            )
        return zones

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
    points = _analysis_points(args, network)

    if len(points) == 0:
        logger.error("Nothing could be snapped to the network.")
        return 1

    max_seconds = args.max_minutes * 60 if args.max_minutes else None
    if isinstance(points, Zones):
        matrix = zone_travel_time_matrix(
            network,
            points,
            max_seconds=max_seconds,
            near_seconds=args.near_minutes * 60 if args.near_minutes else None,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
    else:
        matrix = travel_time_matrix(
            network,
            points.node_index,
            max_seconds=max_seconds,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"travel_times_{args.mode}.npz"
    np.savez_compressed(out, ids=points.ids, seconds=matrix)
    print(
        f"Wrote {out} ({matrix.shape[0]}x{matrix.shape[1]}, "
        f"{out.stat().st_size / 1e6:.1f} MB)"
    )

    omx_out = (
        args.output_dir / f"travel_times_{args.mode}.omx"
        if args.omx is None or args.omx is _OMX_DEFAULT_PATH
        else args.omx
    )
    omx_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Minutes, unlike the .npz: the OMX goes to other modelling tools,
        # and minutes is what they expect of a travel-time matrix.
        write_omx_matrix(
            omx_out,
            points.ids,
            matrix / 60.0,
            matrix_name=args.omx_matrix_name or args.mode,
            mapping_name=args.omx_mapping_name,
            units="minutes",
        )
    except ValueError as exc:
        # Written by default, so ids OMX cannot hold only fail a run that asked.
        if args.omx is not None:
            raise
        logger.info("Not writing %s: %s", omx_out, exc)
    else:
        print(
            f"Wrote {omx_out} ({matrix.shape[0]}x{matrix.shape[1]}, "
            f"{omx_out.stat().st_size / 1e6:.1f} MB)"
        )

    for key, value in summarise(matrix).items():
        print(f"  {key}: {value:,.3f}")
    return 0


def _assign_demand(args: argparse.Namespace, points: Centroids | Zones):
    """``--demand`` as a sparse matrix aligned with the analysis points."""
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
        demand = align_to_ids(ids, demand, points.ids)
    elif suffix == ".npz":
        ids, demand = read_demand_npz(args.demand)
        demand = align_to_ids(ids, demand, points.ids)
    else:
        long_demand = read_demand_long(
            args.demand,
            origin_column=args.origin_column,
            destination_column=args.destination_column,
            demand_column=args.demand_column,
        )
        demand = demand_matrix(long_demand, points.ids)
    return clip_minimum(demand, args.min_demand)


def cmd_assign(args: argparse.Namespace) -> int:
    network, links_path = _network_and_links(args)
    points = _analysis_points(args, network)

    if len(points) == 0:
        logger.error("Nothing could be snapped to the network.")
        return 1

    if args.gpkg and links_path is None:
        raise SystemExit(
            "--gpkg draws volumes on the link layer's geometry, so it needs "
            "--links pointing at the GeoPackage this network was built from."
        )

    demand = _assign_demand(args, points)
    max_seconds = args.max_minutes * 60 if args.max_minutes else None

    if args.select_link and isinstance(points, Zones):
        raise SystemExit(
            "--select-link works with --centroids for now; --zones spreads each "
            "zone over several access points and two search tiers."
        )

    select_demand = select_flow = None
    if args.select_link:
        link_volume, select_demand, select_flow = select_link_assignment(
            network,
            points.node_index,
            demand,
            args.select_link,
            max_seconds=max_seconds,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
    elif isinstance(points, Zones):
        link_volume = assign_zone_traffic(
            network,
            points,
            demand,
            max_seconds=max_seconds,
            near_seconds=args.near_minutes * 60 if args.near_minutes else None,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
    else:
        link_volume, trip_lengths = trip_length_assignment(
            network,
            points.node_index,
            demand,
            max_seconds=max_seconds,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_trip_lengths(args, points, trip_lengths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = link_volume_frame(network, link_volume)
    out = args.output_dir / f"link_volumes_{args.mode}.npz"
    np.savez_compressed(out, **frame)  # type: ignore[arg-type]
    print(f"Wrote {out} ({network.n_edges:,} links, {out.stat().st_size / 1e6:.1f} MB)")

    if args.gpkg:
        assert links_path is not None
        gpkg_path = args.output_dir / f"{args.mode}_volumes.gpkg"
        layer = links_module.read_links(args.links) if args.links else links_path
        edge_mask = None
        if args.connectors is not None:
            # Gap connectors stand in for real paths, so they are drawn, from
            # the connector layer's geometry. Island crossings carry no real
            # traffic -- nobody rows across -- so they stay in
            # link_volumes_<mode>.npz but are left off the map.
            layer = _link_layer(args)
            crossings = island_link_ids(links_module.read_links(args.connectors))
            edge_mask = ~np.isin(network.link_id, crossings)
        write_edges_gpkg(
            network,
            layer,
            gpkg_path,
            extra_columns={"volume": link_volume},
            edge_mask=edge_mask,
        )
        rows = network.n_edges if edge_mask is None else int(edge_mask.sum())
        print(f"Wrote {gpkg_path} ({rows:,} rows)")

    for key, value in summarise_assignment(link_volume).items():
        print(f"  {key}: {value:,.3f}")

    if select_demand is not None:
        _write_select_link(args, network, points, select_demand, select_flow)
    return 0


def _write_trip_lengths(args: argparse.Namespace, points: Centroids, trips) -> None:
    """
    ``trip_lengths_<mode>.npz``: every OD pair's demand and network length.

    Zone ids rather than positions, so ``valma view`` can group trips by where
    they start. Intrazonal pairs are kept, at length 0, for it to estimate.
    """
    ids = np.asarray(points.ids)
    out = args.output_dir / f"trip_lengths_{args.mode}.npz"
    np.savez_compressed(
        out,
        origin=ids[trips.origin],
        destination=ids[trips.destination],
        demand=trips.demand,
        length_m=trips.length_m.astype(np.float32),
    )
    reached = np.isfinite(trips.length_m)
    total = float(trips.demand.sum())
    mean = float((trips.demand * np.where(reached, trips.length_m, 0)).sum()) / max(
        float(trips.demand[reached].sum()), 1e-12
    )
    print(
        f"Wrote {out} ({trips.demand.size:,} OD pairs; mean network length "
        f"{mean / 1000:.2f} km over {float(trips.demand[reached].sum()):,.0f} of "
        f"{total:,.0f} trips)"
    )


def _write_select_link(
    args: argparse.Namespace,
    network: RoutableNetwork,
    points: Centroids,
    select_demand,
    flow: np.ndarray,
) -> None:
    """
    Write what a select-link run found: the OD pairs whose path uses the
    selected link(s), each zone's share as origin and as destination, and
    where those trips run on the rest of the network.
    """
    import pandas as pd

    stem = f"select_link_{args.mode}"
    coo = select_demand.tocoo()
    ids = np.asarray(points.ids)
    od = pd.DataFrame(
        {"origin": ids[coo.row], "destination": ids[coo.col], "demand": coo.data}
    ).sort_values("demand", ascending=False)
    od_path = args.output_dir / f"{stem}_od.csv"
    od.to_csv(od_path, index=False)

    as_origin = np.asarray(select_demand.sum(axis=1)).ravel()
    as_destination = np.asarray(select_demand.sum(axis=0)).ravel()
    zones = gpd.GeoDataFrame(
        {
            "zone_id": ids,
            "as_origin": as_origin,
            "as_destination": as_destination,
            "total": as_origin + as_destination,
        },
        geometry=gpd.points_from_xy(points.x, points.y),
        crs=network.crs,
    )
    zones = zones[zones["total"] > 0].sort_values("total", ascending=False)
    zones_path = args.output_dir / f"{stem}_zones.gpkg"
    write_vector(zones, zones_path, layer="zones", driver="GPKG")

    # Where the selected link's trips go: loaded in the same pass as the rest.
    flow_path = args.output_dir / f"{stem}.gpkg"
    write_edges_gpkg(
        network,
        _link_layer(args),
        flow_path,
        extra_columns={"volume": flow},
        edge_mask=flow > 0,
    )

    total = float(select_demand.sum())
    print(f"\nSelect link {args.select_link}: {total:,.1f} trips, {len(od):,} OD pair(s)")
    print(f"  {od_path}\n  {zones_path} ({len(zones):,} zones)\n  {flow_path}")
    if total > 0:
        print("  Largest origins / destinations (zone: trips, share):")
        for _, row in zones.head(10).iterrows():
            print(
                f"    {row['zone_id']}: {row['as_origin']:,.1f} from, "
                f"{row['as_destination']:,.1f} to ({row['total'] / total:.0%} of the trips)"
            )


def cmd_view(args: argparse.Namespace) -> int:
    """Serve a link layer and its runs to the browser (``docs/viewer-design.md``)."""
    try:
        from valma_bike_and_walk.viewer.data import (
            ViewerData,
            cache_key,
            discover_runs,
            load_counters,
            load_run,
            prepare_links,
        )
        from valma_bike_and_walk.viewer.server import serve
    except ImportError as error:
        raise SystemExit(
            "valma view needs the viewer extra: pip install "
            f"'valma_bike_and_walk[viewer]' (missing: {error.name})"
        ) from None

    mode = _links_mode(args)
    cache = args.cache_dir / "viewer"
    links = prepare_links(args.links, cache, args.connectors)

    runs = {}
    sources = args.run or [args.output_dir / f"link_volumes_{mode}.npz"]
    for path in discover_runs(sources, mode):
        run = load_run(path, links)
        runs[run.name] = run
        print(f"  run {run.name}: {float(run.total.sum()):,.0f} vehicle-links")

    data = ViewerData(links, mode, runs)
    if args.count_lines.exists() and args.observed.exists():
        data.counters = load_counters(data, args.count_lines, args.observed)
    if args.municipalities is not None and args.municipalities.exists():
        from valma_bike_and_walk.viewer.municipalities import load_municipalities

        data.municipalities = load_municipalities(
            links, args.municipalities, args.municipality_column, cache
        )
        print(f"  municipalities: {len(data.municipalities.names)}")
        share_demand = args.share_demand
        if share_demand is None and args.demand is not None and args.demand.suffix == ".omx":
            share_demand = args.demand
        if share_demand is not None:
            from valma_bike_and_walk.viewer.municipalities import load_trip_shares

            load_trip_shares(
                data.municipalities,
                share_demand,
                args.demand_matrix or mode,
                args.demand_mapping,
                args.municipalities,
                args.municipality_zone_column,
                args.municipality_column,
                cache,
            )
            share = data.municipalities.mode_trips.sum() / data.municipalities.all_trips.sum()
            print(f"  {mode} share of trips: {share:.1%} (from {share_demand.name})")
        from valma_bike_and_walk.viewer.trip_lengths import (
            TripLengthGroups,
            load_regions,
            load_zone_table,
        )

        regions = {}
        if args.regions is not None and args.regions.exists():
            regions = load_regions(args.regions)
            print(f"  regions: {len(set(regions.values()))} (from {args.regions.name})")
        data.trip_lengths = TripLengthGroups(
            load_zone_table(
                args.municipalities, args.municipality_zone_column, args.municipality_column
            ),
            data.municipalities.names,
            regions,
        )
    from valma_bike_and_walk.viewer.route import Router

    network = _view_network(args, mode)
    data.router = Router(network, links)
    if args.demand is not None:
        data.flows = _counter_flows(args, mode, links, cache, network)

    tile_key = cache_key(args.links)
    if args.connectors is not None:
        tile_key += "+" + cache_key(args.connectors)
    serve(
        data,
        cache / "tiles" / tile_key,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        observations_dir=args.output_dir,
    )
    return 0


def _view_assign_argv(args: argparse.Namespace, mode: str) -> list[str]:
    """``assign`` arguments describing the graph ``valma view`` was given."""
    argv = [
        "assign", "--links", str(args.links), "--mode", mode,
        "--cache-dir", str(args.cache_dir), "--workers", str(args.workers),
        "--centroids", str(args.centroids or "unused"),
        "--id-column", args.id_column,
        "--demand", str(args.demand or "unused"), "--demand-mapping", args.demand_mapping,
        "--min-demand", str(args.min_demand),
    ]
    if args.demand_matrix:
        argv += ["--demand-matrix", args.demand_matrix]
    if args.max_minutes:
        argv += ["--max-minutes", str(args.max_minutes)]
    if args.connectors is not None:
        argv += ["--connectors", str(args.connectors)]
    if args.chunk_size:
        argv += ["--chunk-size", str(args.chunk_size)]
    return argv


def _view_network(args: argparse.Namespace, mode: str) -> RoutableNetwork:
    """The routable graph, from the same cache ``assign`` builds and reuses."""
    network, _ = _network_and_links(build_parser().parse_args(_view_assign_argv(args, mode)))
    return network


def _counter_flows(
    args: argparse.Namespace, mode: str, links, cache: Path, network: RoutableNetwork
):
    """
    The select-link service behind counter clicks in ``valma view``.

    The graph, zones and demand are loaded through ``assign``'s own argument
    handling, so a click answers exactly what ``assign --select-link`` would.
    """
    import hashlib

    from valma_bike_and_walk.viewer.data import cache_key
    from valma_bike_and_walk.viewer.select import CounterFlows, SelectSetup

    if args.centroids is None:
        raise SystemExit("valma view --demand needs --centroids, as assign was run with.")
    argv = _view_assign_argv(args, mode)
    assign_args = build_parser().parse_args(argv)

    points = _analysis_points(assign_args, network)
    demand = _assign_demand(assign_args, points)
    description = " ".join(argv[1:]) + " | " + " ".join(
        cache_key(Path(p)) for p in (args.links, args.demand, args.centroids)
    ) + " | impedance " + hashlib.sha1(network.travel_time.tobytes()).hexdigest()
    setup = SelectSetup(
        network=network,
        zone_ids=np.asarray(points.ids),
        zone_nodes=np.asarray(points.node_index),
        demand=demand.tocsr(),
        max_seconds=args.max_minutes * 60 if args.max_minutes else None,
        workers=args.workers,
        chunk_size=args.chunk_size,
        description=description,
    )
    print(f"  counter flows: {demand.nnz:,} OD pairs over {len(points):,} zones")
    return CounterFlows(setup, links["link_id"].to_numpy(), cache / "counter_flows")


def _add_routing(parser: argparse.ArgumentParser) -> None:
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument(
        "--centroids",
        type=Path,
        help=(
            "CSV or vector file of points, one per zone. Every trip starts and "
            "ends at the nearest network node to the point."
        ),
    )
    where.add_argument(
        "--zones",
        type=Path,
        help=(
            "Vector file of zone polygons. Each zone's trip ends are spread "
            "over --points-per-zone weighted access points inside it, which "
            "keeps volumes off a single centroid node, makes short trips "
            "realistic and lets intrazonal demand load onto the network."
        ),
    )
    parser.add_argument("--id-column")
    parser.add_argument("--x-column", default="lon")
    parser.add_argument("--y-column", default="lat")
    parser.add_argument(
        "--centroid-crs",
        default="EPSG:4326",
        help=(
            "CRS to assume for a --centroids CSV, and for any --zones or "
            "--weights layer that does not carry one of its own."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help=(
            "With --zones: point or polygon layer saying where demand actually "
            "sits -- buildings, address points, a population grid as points. "
            "Without it, the network's own nodes are used, weighted by street "
            "length, which needs no extra data and is a fair proxy for where "
            "people are."
        ),
    )
    parser.add_argument(
        "--weight-column",
        help=(
            "Column in --weights giving each point's size (population, jobs, "
            "floor area). Without it every point counts equally, which is "
            "right for address points and wrong for a population grid."
        ),
    )
    parser.add_argument(
        "--points-per-zone",
        type=int,
        default=DEFAULT_POINTS_PER_ZONE,
        help=(
            "Access points per zone with --zones (default "
            f"{DEFAULT_POINTS_PER_ZONE}). This is the cost dial: it multiplies "
            "the number of shortest-path searches. 1 reproduces a single "
            "weighted centroid. Accuracy saturates quickly, so raising it far "
            "above the default buys little."
        ),
    )
    parser.add_argument(
        "--access-speed-kmh",
        type=float,
        help=(
            "Speed for the leg between an access point and the network node it "
            "snaps to (default: the mode's base speed). Snapping otherwise "
            "teleports a trip end up to --max-snap-distance for free."
        ),
    )
    parser.add_argument(
        "--no-access-time",
        action="store_true",
        help="Don't charge for the snap distance; snapping is free, as with --centroids.",
    )
    parser.add_argument(
        "--zone-points-gpkg",
        type=Path,
        help=(
            "Also write the generated access points here as a GeoPackage. Worth "
            "doing once for any new zone or weight layer -- bad placement shows "
            "up here and nowhere else."
        ),
    )
    parser.add_argument("--max-snap-distance", type=float, default=DEFAULT_MAX_SNAP_M)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Threads to route on (default: all cores but two).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Origins a worker takes at a time. Sized automatically; rarely worth setting.",
    )


def _add_dem(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dem-resolution",
        type=float,
        default=DEFAULT_RESOLUTION_M,
        choices=RESOLUTIONS_M,
        help=(
            "Grid spacing to fetch elevation at, in metres. The service scales "
            "its 2 m model to whatever is asked; these are the spacings that "
            "divide the default tile exactly. Coarser means much smaller tiles "
            "-- 10 m is a sixteenth of the bytes 2 m is."
        ),
    )
    parser.add_argument(
        "--dem-tile",
        type=int,
        default=DEFAULT_TILE_M,
        help=(
            "Edge of one cached tile in metres. The service refuses anything "
            "over 10000 m or 5000 px a side."
        ),
    )
    parser.add_argument("--dem-workers", type=int, default=4)
    parser.add_argument(
        "--api-key",
        help=(
            "National Land Survey API key. Defaults to the "
            f"{API_KEY_ENV} environment variable. Free, from "
            "https://www.maanmittauslaitos.fi/rajapinnat/api-avaimen-ohje"
        ),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Skip TLS certificate verification when downloading elevation "
            "tiles. For a network that inspects TLS and whose root cannot be "
            "reached otherwise; installing truststore is the better fix. The "
            "API key travels in the URL, so this exposes it. Also settable as "
            f"{INSECURE_ENV}=1."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="valma",
        description=__doc__,
        # The description is a worked example with its own layout; argparse
        # would otherwise reflow it into a wall of text.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    _add_config(parser)
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

    connect = sub.add_parser(
        "connect",
        help=(
            "Write connector links joining islands and other disconnected "
            "parts to the rest of the network, so the build keeps them."
        ),
    )
    connect.add_argument("--links", type=Path, required=True)
    connect.add_argument(
        "--out",
        type=Path,
        help="Where to write them (default: <links stem>_connectors.gpkg beside --links).",
    )
    connect.add_argument(
        "--min-nodes",
        type=int,
        default=DEFAULT_MIN_COMPONENT_NODES,
        help=(
            "Only connect parts with at least this many nodes (default: "
            "%(default)s). Smaller ones are mostly data gaps on the mainland, "
            "better dropped as before so zones snap to a real street."
        ),
    )
    connect.add_argument(
        "--speed-kmh",
        type=float,
        default=DEFAULT_CONNECTOR_SPEED_KMH,
        help=(
            "Speed on a connector (default: %(default)s km/h -- 100 m costs "
            "an hour). Deliberately prohibitive: the connectors keep islands "
            "routable within themselves, not reachable across the water."
        ),
    )
    connect.add_argument(
        "--gap-m",
        type=float,
        default=DEFAULT_GAP_M,
        help=(
            "Connectors shorter than this (default: %(default)s m) bridge a gap "
            "in the data, not water, and run at the mode's normal speed "
            "instead of --speed-kmh. 0 makes every connector an island crossing."
        ),
    )
    connect.set_defaults(func=cmd_connect)

    overlay_parser = sub.add_parser(
        "overlay",
        help=(
            "Write this model's travel times over a baseline OMX matrix, so the "
            "result has the baseline's full zone system."
        ),
    )
    overlay_parser.add_argument("--mode", choices=MODES, required=True)
    _add_output_dir(overlay_parser)
    overlay_parser.add_argument(
        "--model",
        type=Path,
        help=(
            "Routed matrix, as written by 'valma matrix' (default: "
            "<output-dir>/travel_times_<mode>.npz)."
        ),
    )
    overlay_parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="OMX file whose matrix sets the zones and fills every cell the model does not.",
    )
    overlay_parser.add_argument(
        "--baseline-matrix", required=True, help="Matrix inside --baseline, e.g. 'bike'."
    )
    overlay_parser.add_argument("--baseline-mapping", default="zone_number")
    overlay_parser.add_argument(
        "--baseline-speed-kmh",
        type=float,
        nargs="?",
        const=_MODE_BASE_SPEED,
        metavar="KMH",
        help=(
            "Treat --baseline as distances in km and turn them into minutes at "
            "this speed. Given bare, the mode's own base speed from its speed "
            "profile (4.8 km/h walking, 14 km/h cycling). Without it, "
            "--baseline is taken to be minutes already."
        ),
    )
    overlay_parser.add_argument(
        "--max-minutes",
        type=float,
        help=(
            "Keep the baseline where the model's time is longer than this -- "
            "e.g. trips that only route across a slow island connector."
        ),
    )
    overlay_parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "OMX file to write. Added to if it exists, so bike and walk runs "
            "can share one file; a matrix of the same name is replaced."
        ),
    )
    overlay_parser.add_argument(
        "--matrix-name", help="Name of the matrix to write (default: --mode)."
    )
    overlay_parser.set_defaults(func=cmd_overlay)

    build = sub.add_parser(
        "build",
        help=(
            "Turn a link GeoPackage into a routable graph, kept as a named "
            "file under --output-dir. Optional -- 'matrix'/'assign' do this "
            "themselves, as a cache, when given --links directly."
        ),
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

    dem = sub.add_parser(
        "dem",
        help="Fetch elevation tiles into the cache, and attach them to a link layer.",
    )
    # --mode is optional: the ground is the same height whichever way you travel
    # over it, and one cache serves both modes. It only picks the speeds that
    # refresh the layer's derived styling columns.
    dem.add_argument(
        "--mode",
        choices=MODES,
        help=(
            "Mode the --links layer was extracted for (default: read from a "
            "<mode>_links.gpkg file name, else bike)."
        ),
    )
    dem.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    dem.add_argument("--output-dir", type=Path, default=Path("output"))
    dem.add_argument("--force-reload", action="store_true")
    _add_extent(dem)
    _add_dem(dem)
    dem.add_argument(
        "--pbf",
        type=Path,
        help="Only needed when --area has to be looked up in the PBF.",
    )
    dem.add_argument(
        "--links",
        type=Path,
        help=(
            "Link GeoPackage to attach elevation to. Only the tiles these links "
            "actually touch are fetched. Without it, --bbox/--area just fills "
            "the tile cache."
        ),
    )
    dem.add_argument(
        "--out",
        type=Path,
        help="Where to write the elevated link layer (default: in place).",
    )
    dem.add_argument(
        "--remeasure-all",
        action="store_true",
        help=(
            "Read every link's height from the DEM again, not only those with "
            "ascent_m/descent_m empty -- after a link was moved, or the DEM "
            "settings changed. Links with both overrides set are still skipped."
        ),
    )
    dem.set_defaults(func=cmd_dem)

    traffic = sub.add_parser(
        "traffic",
        help=(
            "Attach car traffic volumes from another network's links to a link "
            "layer, matching the two geometries."
        ),
    )
    # Deliberately no --mode: the cars are on the road whichever way you are
    # travelling over it, and the same match serves both link layers.
    traffic.add_argument(
        "--links",
        type=Path,
        required=True,
        help="Link GeoPackage to attach the volumes to.",
    )
    traffic.add_argument(
        "--volumes",
        type=Path,
        required=True,
        help=(
            "Vector layer of the other network's links, each carrying a car "
            "volume. Any CRS; any format GDAL reads. If it does not exist, it "
            "is built from --assignment-results."
        ),
    )
    traffic.add_argument(
        "--assignment-results",
        type=Path,
        help=(
            "Valma's assignment results GeoPackage (its LINK layer). Used when "
            "--volumes does not exist, or with --rebuild-volumes: the daily "
            "bev, icev, phev, semi_trailer, trailer_truck, van and truck volumes "
            "are summed over both directions and written to --volumes."
        ),
    )
    traffic.add_argument(
        "--rebuild-volumes",
        action="store_true",
        help="Rebuild --volumes from --assignment-results even if it exists.",
    )
    traffic.add_argument(
        "--volume-layer", help="Layer inside --volumes (default: the only one)."
    )
    traffic.add_argument(
        "--volume-column",
        default=DEFAULT_VOLUME_COLUMN,
        help=f"Column holding the volume (default: {DEFAULT_VOLUME_COLUMN!r}).",
    )
    traffic.add_argument(
        "--out",
        type=Path,
        help="Where to write the link layer with volumes (default: in place).",
    )
    traffic.add_argument(
        "--unmatched-out",
        type=Path,
        help=(
            "Also write the source links whose volume went nowhere, with the "
            "share of each that found a road. Most of them are the roads the "
            "bike network drops; the rest are what to check."
        ),
    )
    traffic.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE_M,
        help=(
            "How far apart the two networks' lines may be and still be the same "
            "road, in metres (default: %(default)s). Raise it for a source "
            "network digitised more loosely, but not past the width of the gap "
            "to the next road along."
        ),
    )
    traffic.add_argument(
        "--max-angle",
        type=float,
        default=DEFAULT_MAX_ANGLE_DEG,
        help=(
            "How far apart in bearing they may be, in degrees "
            "(default: %(default)s). Keeps a crossing road from taking a "
            "volume just because it passes close."
        ),
    )
    traffic.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=(
            "Share of a source link that has to find a road before any of its "
            "volume is used (default: %(default)s). This is what keeps a "
            "motorway's volume off the service road beside it: the motorway is "
            "not in the bike network, so most of the link finds nothing, so "
            "none of it counts."
        ),
    )
    traffic.add_argument(
        "--sample-spacing",
        type=float,
        default=DEFAULT_SPACING_M,
        help="How often a source link is sampled, in metres (default: %(default)s).",
    )
    traffic.add_argument(
        "--no-split-divided",
        action="store_true",
        help=(
            "Give both carriageways of a divided road the source's full volume "
            "instead of half each. Right only if the source layer already holds "
            "one directed link per carriageway."
        ),
    )
    traffic.add_argument(
        "--adjacent-distance",
        type=float,
        default=DEFAULT_ADJACENT_DISTANCE_M,
        help=(
            "A path within this many metres of a road, and running alongside "
            "it, takes that road's volume as adjacent_car_volume and is charged "
            "for it like the road (default: %(default)s; 0 turns it off)."
        ),
    )
    traffic.set_defaults(func=cmd_traffic)

    matrix = sub.add_parser("matrix", help="Compute an OD travel-time matrix.")
    _add_common(matrix)
    _add_source(matrix)
    _add_routing(matrix)
    matrix.add_argument(
        "--max-minutes",
        type=float,
        help="Cut off the search here. The single biggest speed-up available.",
    )
    matrix.add_argument(
        "--near-minutes",
        type=float,
        help=(
            "With --zones, route all access points only within this cutoff and "
            "fall back to one representative point per zone beyond it. The "
            "single-point error falls as (R/d)^2 in zone radius over separation "
            "-- about 2 %% at five zone radii -- while a bounded search costs "
            "roughly the square of its cutoff, so this recovers nearly all of "
            "the accuracy for a fraction of the multi-point cost. Aim for "
            "five or so zone radii."
        ),
    )
    matrix.add_argument(
        "--omx",
        nargs="?",
        const=_OMX_DEFAULT_PATH,
        default=None,
        type=Path,
        metavar="PATH",
        help=(
            "Also write an OMX matrix of travel times in minutes, in the same "
            "matrix+lookup layout "
            "'valma assign' reads with --demand-matrix (needs integer "
            "centroid ids). Always written, to "
            "<output-dir>/travel_times_<mode>.omx unless a path is given here."
        ),
    )
    matrix.add_argument(
        "--omx-matrix-name",
        help="Name of the matrix inside the OMX file (default: --mode).",
    )
    matrix.add_argument(
        "--omx-mapping-name",
        default="zone_number",
        help=(
            "Name of the OMX lookup mapping row/column position to centroid "
            "id (default: 'zone_number')."
        ),
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
        "--select-link",
        type=int,
        nargs="+",
        metavar="LINK_ID",
        help=(
            "Select-link analysis of these link_id(s) from the link layer (with "
            "--centroids): writes select_link_<mode>_od.csv (the OD pairs whose "
            "path uses them), select_link_<mode>_zones.gpkg (each zone's trips "
            "as origin and destination) and select_link_<mode>.gpkg (where those "
            "trips run on the rest of the network). Found in the same pass as "
            "the ordinary assignment."
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
    assign.add_argument(
        "--near-minutes",
        type=float,
        help=(
            "With --zones, route all access points only within this cutoff and "
            "assign whatever they could not reach from one representative point "
            "per zone. Spreading trip ends over a zone only changes the route "
            "materially at short range, and a bounded search is far cheaper "
            "than an open one, so this brings a --points-per-zone run back "
            "towards the cost of a single-centroid one. Nothing is lost between "
            "the tiers: the second one is given exactly what the first "
            "reported it could not place."
        ),
    )
    assign.set_defaults(func=cmd_assign)

    view = sub.add_parser(
        "view",
        help="Open a link layer and its assignment results on a browser map.",
    )
    view.add_argument("--links", type=Path, required=True, help="Link GeoPackage.")
    view.add_argument(
        "--mode",
        choices=MODES,
        help="Mode of --links (default: from a <mode>_links.gpkg name, else bike).",
    )
    view.add_argument(
        "--run",
        type=Path,
        nargs="*",
        help=(
            "link_volumes_<mode>.npz files, or directories to search for them "
            "(default: <output-dir>/link_volumes_<mode>.npz). Several runs can "
            "be compared in the browser."
        ),
    )
    view.add_argument(
        "--connectors",
        type=Path,
        help="Connector layer the runs were assigned with, so gap connectors are drawn.",
    )
    view.add_argument(
        "--count-lines",
        type=Path,
        default=Path("data/count_lines.gpkg"),
        help="Counter lines to score against the runs (default: %(default)s, if present).",
    )
    view.add_argument(
        "--observed",
        type=Path,
        default=Path("output/bicycle_counter_annual_averages.csv"),
        help="Observed counter averages (default: %(default)s, if present).",
    )
    view.add_argument(
        "--municipalities",
        type=Path,
        default=Path("data/2023_zonedata--dev-template.gpkg"),
        help=(
            "Polygons naming their municipality, for the per-municipality speed "
            "table: the municipalities themselves or finer zones "
            "(default: %(default)s, if present)."
        ),
    )
    view.add_argument(
        "--municipality-column",
        default="municipality",
        help="Column of --municipalities holding the name (default: %(default)s).",
    )
    view.add_argument(
        "--share-demand",
        type=Path,
        help=(
            "OMX file with a demand matrix per mode, for each municipality's "
            "modal share and its fit against the cycling speed (default: "
            "--demand, if it is an OMX file). The mode's matrix is "
            "--demand-matrix, or the mode's name."
        ),
    )
    view.add_argument(
        "--regions",
        type=Path,
        default=Path("data/hlt2016_regions.csv"),
        help=(
            "CSV of municipality;region, for trip lengths by region "
            "(default: %(default)s, the HLT 2016 regional reports, if present)."
        ),
    )
    view.add_argument(
        "--municipality-zone-column",
        default="input_zone_id",
        help=(
            "Column of --municipalities holding the demand zone id, to put "
            "--share-demand's zones in municipalities (default: %(default)s)."
        ),
    )
    flows = view.add_argument_group(
        "counter flows",
        "Give the settings the run was assigned with, and clicking a counter "
        "maps the flows that pass it (a select-link analysis, cached per counter).",
    )
    flows.add_argument("--demand", type=Path, help="Demand the run was assigned with.")
    flows.add_argument("--demand-matrix", help="Matrix inside an OMX --demand file.")
    flows.add_argument("--demand-mapping", default="zone_number")
    flows.add_argument("--centroids", type=Path, help="Zone points the run used.")
    flows.add_argument("--id-column", default="id")
    flows.add_argument("--min-demand", type=float, default=0.0)
    flows.add_argument("--max-minutes", type=float)
    flows.add_argument("--workers", type=int, default=1)
    flows.add_argument("--chunk-size", type=int)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--no-browser", action="store_true", help="Do not open a browser.")
    view.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    _add_output_dir(view)
    view.set_defaults(func=cmd_view)

    pipeline = sub.add_parser(
        "pipeline",
        help=(
            "Run extract, dem, traffic, connect, matrix, overlay and assign from settings "
            "files, "
            "each only when its inputs or settings changed since it last ran."
        ),
        description=(
            "Each step runs exactly as its own command with the same -c files. "
            "What each step saw is kept in <output-dir>/pipeline_<mode>.json; a "
            "step is skipped when its settings, its input files and the pipeline "
            "files it reads are as they were. traffic is skipped when no volumes "
            "are configured, overlay when no baseline matrix is, assign when no "
            "demand is. A link layer edited by hand re-runs connect, matrix, "
            "overlay and assign, never extract, which would overwrite the edit."
        ),
    )
    pipeline.add_argument(
        "--dry-run", action="store_true", help="Say what would run, and why, without running it."
    )
    pipeline.add_argument("--force", action="store_true", help="Run every step regardless.")
    pipeline.add_argument(
        "--mark-done",
        action="store_true",
        help=(
            "Record the current outputs as up to date without running anything: "
            "for results made by running the steps by hand, so the first pipeline "
            "run does not redo them all."
        ),
    )
    pipeline.add_argument(
        "--skip",
        choices=("extract", "dem", "traffic", "connect", "matrix", "overlay", "assign"),
        action="append",
        help="Leave a step out (repeatable), e.g. --skip dem when no DEM is wanted.",
    )
    pipeline.set_defaults(func=cmd_pipeline)

    for command in sub.choices.values():
        _add_config(command)
    return parser


def _take_config_options(argv: list[str]) -> tuple[list[Path], bool, list[str]]:
    """
    ``-c`` files and ``--show-config``, and the command line without them.

    Read before the real parse, because the files decide what it requires.
    Abbreviations are not recognised here, so that ``--con`` cannot be taken
    for ``--config`` when it was meant as ``--connectors``.
    """
    early = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    early.add_argument("-c", "--config", action="append", type=Path, default=[])
    early.add_argument("--show-config", action="store_true")
    known, rest = early.parse_known_args(argv)
    return known.config, known.show_config, rest


def _command_in(argv: list[str], parser: argparse.ArgumentParser) -> str | None:
    """The command named on the command line: its first positional argument."""
    for token in argv:
        if not token.startswith("-"):
            return token if token in config_file.subparsers(parser) else None
    return None


def _record(args: argparse.Namespace, parser, command: str, files: list[Path]) -> Path:
    """Leave the settings a run used next to its results, for ``-c`` to repeat."""
    mode = getattr(args, "mode", None)
    path = args.output_dir / f"valma_{command}{f'_{mode}' if mode else ''}.toml"
    header = [
        f"Settings `valma {command}` ran with, {datetime.now():%Y-%m-%d %H:%M}.",
        f"Run it again with: valma {command} -c {path.as_posix()}",
    ]
    if files:
        header.append("Layered from: " + ", ".join(p.as_posix() for p in files))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        config_file.dump(args, parser, command, header=header), encoding="utf-8"
    )
    return path


def cmd_pipeline(args: argparse.Namespace) -> int:
    """extract, dem, traffic, connect, matrix, overlay and assign, each only when its inputs changed."""
    from valma_bike_and_walk import __version__
    from valma_bike_and_walk.pipeline import build_plan, run_pipeline

    files: list[Path] = list(getattr(args, "config_files", []))
    if not files:
        raise SystemExit("valma pipeline needs settings files: valma pipeline -c scenarios/<name>.toml")
    flags = [token for path in files for token in ("-c", str(path))]

    def resolve(argv: list[str]) -> argparse.Namespace | None:
        errors = io.StringIO()
        try:
            with contextlib.redirect_stderr(errors):
                return _parse(argv, files).args
        except SystemExit:
            if argv[0] == "traffic":
                return None  # no volumes configured: nothing to match
            if argv[0] == "overlay":
                return None  # no baseline matrix configured: nothing to lay over
            if argv[0] == "assign":
                return None  # no demand configured: nothing to assign
            raise SystemExit(
                f"valma pipeline: the {argv[0]} step cannot run with these settings:\n"
                + errors.getvalue().strip()
            ) from None

    try:
        plan = build_plan(resolve, skip=set(args.skip or []))
    except ValueError as error:
        raise SystemExit(f"valma pipeline: {error}") from None
    print(f"Pipeline over {plan.files['links']} (state in {plan.state_path.name}):")
    return run_pipeline(
        plan,
        lambda argv: main([*argv, *flags]),
        __version__,
        force=args.force,
        dry_run=args.dry_run,
        mark_done=args.mark_done,
    )


@dataclass
class _Parsed:
    args: argparse.Namespace
    parser: argparse.ArgumentParser
    command: str
    files: list[Path]
    config: config_file.Config | None
    applied: dict[str, config_file.Setting]
    given: set[str]


def _parse(argv: list[str], files: list[Path], show_config: bool = False) -> _Parsed:
    """A command's arguments as the settings ``files`` and ``argv`` make them."""
    parser = build_parser()
    command = _command_in(argv, parser)

    applied: dict[str, config_file.Setting] = {}
    config: config_file.Config | None = None
    if files and command is not None:
        try:
            config = config_file.load(files)
            applied = config_file.apply(config, parser, command)
        except config_file.ConfigError as error:
            parser.exit(2, f"valma: error: {error}\n")

    args = parser.parse_args(argv)
    command = args.command

    given: set[str] = set()
    if applied or show_config:
        given = config_file.given_on_command_line(build_parser, argv, command)
    if applied:
        try:
            config_file.settle(args, parser, command, applied, given, config)
        except config_file.ConfigError as error:
            parser.exit(2, f"valma: error: {error}\n")
    if getattr(args, "scenario", None):
        args.output_dir = args.output_dir / args.scenario
    return _Parsed(args, parser, command, files, config, applied, given)


def main(argv: list[str] | None = None) -> int:
    files, show_config, argv = _take_config_options(
        list(sys.argv[1:] if argv is None else argv)
    )
    parsed = _parse(argv, files, show_config)
    args, parser, command = parsed.args, parsed.parser, parsed.command
    config, applied, given = parsed.config, parsed.applied, parsed.given
    args.config_files = files
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if show_config:
        sources = config_file.sources_of(args, applied, given)
        if getattr(args, "scenario", None):
            sources["output_dir"] = (
                f"{sources.get('output_dir', 'default')}, "
                f"+ scenario from {sources.get('scenario', 'default')}"
            )
        header = []
        if config is not None:
            header.append("Layered from: " + ", ".join(p.as_posix() for p in config.files))
        print(config_file.dump(args, parser, command, sources=sources, header=header), end="")
        return 0

    code = int(args.func(args))
    if code == 0 and command in _RECORDED:
        files_used = config.files if config is not None else []
        logger.info("Settings recorded in %s", _record(args, parser, command, files_used))
    return code


if __name__ == "__main__":
    sys.exit(main())
