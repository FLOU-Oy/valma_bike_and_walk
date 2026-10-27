"""Draw a per-edge result back onto the link layer, for GIS.

The routable network holds no geometry -- the link GeoPackage does, and every
directed edge remembers the ``link_id`` it came from. So an export is a join:
take an array with one value per directed edge, look up each edge's link, and
write the link's shape out with the value attached.

That indirection is what makes the workflow work. The GeoPackage you edited in
QGIS is the same one the results come back on, row for row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from valma_bike_and_walk.links import LINKS_LAYER, traffic_light_waits
from valma_bike_and_walk.links import read_links as read_links_gpkg
from valma_bike_and_walk.network import RoutableNetwork
from valma_bike_and_walk.vector_io import write_vector

logger = logging.getLogger(__name__)


def edges_to_geodataframe(
    network: RoutableNetwork,
    links: gpd.GeoDataFrame,
    extra_columns: dict[str, np.ndarray] | None = None,
    edge_mask: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """
    One row per directed edge: the link's geometry, its endpoints and its costs.

    ``links`` is the layer the network was built from. A link that carries
    traffic both ways produces two rows here, distinguished by ``direction``
    (+1 along the link as digitised, -1 against it) and sharing one geometry.
    Links that the build dropped -- outside the largest connected component, or
    a slower parallel duplicate -- have no row.

    ``extra_columns``, if given, is merged in as-is: an assignment's per-edge
    volume array from :func:`valma_bike_and_walk.assignment.link_volume_frame`
    is aligned with the network's edges exactly as this function's own columns
    are.

    ``edge_mask``, if given, is a boolean per edge: only the edges it marks get
    a row, and ``links`` need only hold theirs. This is how island connectors
    are left off a map without being left out of the graph.
    """
    rows = np.repeat(np.arange(network.n_nodes), np.diff(network.indptr))
    cols = network.indices
    keep = (
        np.ones(network.n_edges, dtype=bool)
        if edge_mask is None
        else np.asarray(edge_mask, dtype=bool)
    )
    rows, cols, link_id = rows[keep], cols[keep], network.link_id[keep]

    geometry_by_link = links.set_index("link_id").geometry
    missing = ~np.isin(link_id, geometry_by_link.index.to_numpy())
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} of {link_id.size} edges reference a link_id that "
            "is not in this link layer. It is not the layer this network was built "
            "from, or it has been edited since."
        )

    length_m = network.length[keep].astype(np.float64)
    travel_time_s = network.travel_time[keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        speed_kmh = np.where(travel_time_s > 0, length_m / travel_time_s * 3.6, np.nan)

    columns = {
        "link_id": link_id,
        "direction": network.direction[keep],
        "u": network.node_ids[rows],
        "v": network.node_ids[cols],
        "length_m": length_m,
        "travel_time_s": travel_time_s,
        "speed_kmh": speed_kmh,
        "signal_wait_s": _signal_wait_per_edge(
            links, network.mode, link_id, network.direction[keep]
        ),
    }
    if extra_columns:
        columns.update({k: np.asarray(v)[keep] for k, v in extra_columns.items()})

    return gpd.GeoDataFrame(
        columns,
        geometry=geometry_by_link.loc[link_id].to_numpy(),
        crs=links.crs,
    )


def _signal_wait_per_edge(
    links: gpd.GeoDataFrame, mode: str, link_id: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Seconds of each edge's ``travel_time_s`` spent waiting at traffic lights."""
    forward, reverse = traffic_light_waits(links, mode)
    row = pd.Index(links["link_id"]).get_indexer(link_id)
    return np.where(direction >= 0, forward[row], reverse[row])


def write_edges_gpkg(
    network: RoutableNetwork,
    links: gpd.GeoDataFrame | Path | str,
    path: Path,
    layer: str = LINKS_LAYER,
    extra_columns: dict[str, np.ndarray] | None = None,
    edge_mask: np.ndarray | None = None,
) -> Path:
    """
    Write the network's edges, with their link geometry, to a GeoPackage.

    ``links`` is either the layer itself or a path to the GeoPackage
    ``valma extract`` wrote.
    """
    if isinstance(links, (str, Path)):
        links = read_links_gpkg(Path(links))
    gdf = edges_to_geodataframe(
        network, links, extra_columns=extra_columns, edge_mask=edge_mask
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_vector(gdf, path, driver="GPKG", layer=layer)

    logger.info(
        "Saved %d edges to %s (%.1f MB)", len(gdf), path, path.stat().st_size / 1e6
    )
    return path
