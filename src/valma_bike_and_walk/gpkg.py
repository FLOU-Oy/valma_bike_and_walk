"""Export a routable network's links to a GeoPackage for visualization.

Only meaningful for networks built with ``keep_geometry=True`` (``valma build
--gpkg``): the compact .npz cache normally keeps nothing but node coordinates
and a CSR adjacency, which is enough to route but not to draw a road on a map.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

from valma_bike_and_walk.config import WGS84
from valma_bike_and_walk.network import RoutableNetwork

logger = logging.getLogger(__name__)


def edges_to_geodataframe(
    network: RoutableNetwork, extra_columns: dict[str, np.ndarray] | None = None
) -> gpd.GeoDataFrame:
    """
    One row per directed edge: geometry, OSM endpoint ids, length, time, speed.

    Geometry is the merged road shape captured before simplification discarded
    it, in WGS84 -- the CRS it was read from the PBF in, unrelated to the
    projected metres the network routes in.

    ``extra_columns``, if given, is merged in as-is -- e.g. an assignment's
    per-edge volume array from
    :func:`valma_bike_and_walk.assignment.link_volume_frame`, which is aligned
    with the network's edges the same way this function's own columns are.
    """
    if network.geometry is None:
        raise ValueError(
            "This network was built without geometry. Rebuild with "
            "`valma build --gpkg` (add --force-reload if a plain network is "
            "already cached for this extent) to export its links."
        )

    rows = np.repeat(np.arange(network.n_nodes), np.diff(network.indptr))
    cols = network.indices

    length_m = network.length.astype(np.float64)
    travel_time_s = network.travel_time
    with np.errstate(invalid="ignore", divide="ignore"):
        speed_kmh = np.where(travel_time_s > 0, length_m / travel_time_s * 3.6, np.nan)

    columns = {
        "u": network.node_ids[rows],
        "v": network.node_ids[cols],
        "length_m": length_m,
        "travel_time_s": travel_time_s,
        "speed_kmh": speed_kmh,
    }
    if extra_columns:
        columns.update(extra_columns)

    return gpd.GeoDataFrame(
        columns,
        geometry=shapely.from_wkb(network.geometry),
        crs=WGS84,
    )


def write_edges_gpkg(
    network: RoutableNetwork,
    path: Path,
    layer: str = "links",
    extra_columns: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write the network's links (geometry + speed/cost columns) to a GeoPackage."""
    gdf = edges_to_geodataframe(network, extra_columns=extra_columns)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG", layer=layer)

    logger.info(
        "Saved %d links to %s (%.1f MB)", len(gdf), path, path.stat().st_size / 1e6
    )
    return path
