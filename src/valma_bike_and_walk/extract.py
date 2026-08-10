"""Everything that talks to pyrosm and the .osm.pbf file.

Keeping the pyrosm calls in one module means the rest of the package deals only
with plain DataFrames and numpy arrays, and a pyrosm upgrade can only break
here.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Sequence

import geopandas as gpd
import pandas as pd
from pyrosm import OSM
from pyrosm.graph_simplify import simplify_graph
from pyrosm.graphs import get_directed_edges
from shapely.geometry import base as shapely_base
from shapely.geometry import box

from valma_bike_and_walk.config import PYROSM_NETWORK_TYPES, Settings, validate_mode

logger = logging.getLogger(__name__)


def resolve_engine(engine: str) -> str:
    """Turn the "auto" sentinel into a concrete pyrosm engine name."""
    return "out_of_core" if engine == "auto" else engine


def _bbox_geometry(bbox: Sequence[float]) -> shapely_base.BaseGeometry:
    if len(bbox) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat].")
    return box(bbox[0], bbox[1], bbox[2], bbox[3])


def find_boundary(
    settings: Settings,
    area: str,
    search_bbox: Sequence[float] | None = None,
    force_reload: bool = False,
) -> gpd.GeoDataFrame:
    """
    Look up an administrative boundary inside the PBF, with no network access.

    pyrosm has to walk every relation in the extract to do this, which on a
    whole-country file costs on the order of 15 minutes and >10 GB of memory, so
    the answer is cached as GeoJSON and paid only once per area. ``search_bbox``
    restricts the scan and cuts that down sharply, but it must fully contain the
    area or the boundary comes back truncated -- note that Finnish
    municipalities include their sea territory and reach much further south than
    their built-up area does.
    """
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    name = area.split(",")[0].strip()
    cache_file = settings.boundary_cache_path(area)

    if cache_file.exists() and not force_reload:
        logger.info("Loading boundary from cache: %s", cache_file)
        return gpd.read_file(cache_file)

    logger.info("Scanning %s for the boundary of %r (slow)", settings.pbf_path, name)
    osm = OSM(
        str(settings.pbf_path),
        bounding_box=list(search_bbox) if search_bbox is not None else None,
        engine=settings.engine,
        workers=settings.osm_workers,
    )
    boundaries = osm.get_boundaries(boundary_type="administrative", name=name)

    if boundaries is None or len(boundaries) == 0:
        raise ValueError(
            f"No administrative boundary named {name!r} in {settings.pbf_path}."
        )

    # pyrosm matches on substring, so "Espoo" also returns "Pohjois-Espoo" and
    # "Espoonlahti". Prefer an exact name, then the largest such area.
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


def read_network_tables(
    settings: Settings,
    mode: str,
    clip: shapely_base.BaseGeometry | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the PBF and return (nodes, edges) for one travel mode.

    ``clip`` is applied while parsing, so a small extent is much cheaper in
    memory than reading the whole country -- though pyrosm still has to scan the
    entire file either way.
    """
    validate_mode(mode)
    network_type = PYROSM_NETWORK_TYPES[mode]
    settings = replace(settings, engine=resolve_engine(settings.engine))

    logger.info(
        "Reading %s network from %s (engine=%s, workers=%s)",
        network_type,
        settings.pbf_path,
        settings.engine,
        settings.osm_workers,
    )

    def read(engine: str, workers: int | str | None):
        osm = OSM(
            str(settings.pbf_path),
            bounding_box=clip,
            engine=engine,
            workers=workers,
        )
        # No extra_attributes: pyrosm's cycling filter already returns
        # "oneway:bicycle" wherever ways carry it (verified against central
        # Helsinki), and asking for it again only costs another pass over the
        # tags.
        return osm.get_network(network_type=network_type, nodes=True)

    try:
        nodes, edges = read(settings.engine, settings.osm_workers)
    except RuntimeError as exc:
        # The out_of_core engine parses in a process pool. On Windows (and
        # anywhere else that spawns rather than forks) the children re-import
        # the main module, so a caller whose script runs build_network() at
        # import time makes every child re-run the whole script. Python catches
        # that and raises here. Falling back keeps such a script working, at
        # single-core speed, instead of drowning it in tracebacks.
        if "bootstrapping phase" not in str(exc) and "freeze_support" not in str(exc):
            raise
        logger.warning(
            "Multi-core parsing needs the calling script to guard its entry point with "
            '`if __name__ == "__main__":`. Falling back to the slower single-core '
            "reader. Add the guard, or pass engine='in_memory' to silence this."
        )
        nodes, edges = read("in_memory", None)

    if nodes is None or edges is None or len(edges) == 0:
        raise ValueError(
            f"No {network_type} network found in {settings.pbf_path} for this extent."
        )

    logger.info("Parsed %d nodes and %d ways", len(nodes), len(edges))
    return nodes, edges


def build_directed_edges(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    mode: str,
    travel_time_col: str = "travel_time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expand ways into directed edges and topologically simplify them.

    The network type is passed explicitly rather than left to pyrosm's sniffing
    of GeoDataFrame ``_metadata``, which any reshaping of the frame can drop. It
    decides the direction rules, and getting them wrong is silent: walking is
    bidirectional (people ignore one-way signs), while cycling honours ``oneway``
    together with ``oneway:bicycle`` so contraflow cycle lanes are modelled.

    ``travel_time`` must already be on ``edges``; it is summed along each
    collapsed chain exactly like ``length`` is. Computing it before
    simplification is what keeps us clear of pyrosm's list-valued merged tags.
    """
    validate_mode(mode)
    network_type = PYROSM_NETWORK_TYPES[mode]

    nodes, directed = get_directed_edges(
        nodes,
        edges,
        network_type=network_type,
    )

    nodes, directed = simplify_graph(
        nodes,
        directed,
        length_cols=("length", travel_time_col),
    )
    return nodes, directed


def resolve_clip(
    settings: Settings,
    area: str | None,
    bbox: Sequence[float] | None,
    search_bbox: Sequence[float] | None = None,
    force_reload: bool = False,
) -> tuple[shapely_base.BaseGeometry | None, str]:
    """Work out the clipping geometry and a stable cache key for it."""
    if bbox is not None:
        return _bbox_geometry(bbox), "bbox_" + "_".join(f"{c:.4f}" for c in bbox)
    if area is not None:
        boundary = find_boundary(settings, area, search_bbox, force_reload=force_reload)
        key = area.split(",")[0].strip().replace(" ", "_")
        return boundary.geometry.iloc[0], key
    return None, "full"
