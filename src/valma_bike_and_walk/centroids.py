"""Loading analysis points and attaching them to the network."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from valma_bike_and_walk.config import PROJECTED_CRS, WGS84
from valma_bike_and_walk.network import RoutableNetwork

logger = logging.getLogger(__name__)

#: Beyond this, a point is not really "on" the network any more.
DEFAULT_MAX_SNAP_M = 1000.0


@dataclass
class Centroids:
    """Analysis points, projected, plus where they attach to the network."""

    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    node_index: np.ndarray  # -1 where snapping failed
    snap_distance: np.ndarray

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    @property
    def snapped(self) -> np.ndarray:
        """Boolean mask of points that found a node within the limit."""
        return self.node_index >= 0

    def drop_unsnapped(self) -> "Centroids":
        keep = self.snapped
        if keep.all():
            return self
        logger.warning(
            "Dropping %d centroid(s) with no node nearby", int((~keep).sum())
        )
        return Centroids(
            ids=self.ids[keep],
            x=self.x[keep],
            y=self.y[keep],
            node_index=self.node_index[keep],
            snap_distance=self.snap_distance[keep],
        )


def normalise_ids(ids: np.ndarray) -> np.ndarray:
    """
    Turn object-dtype ids into fixed-width strings.

    A text column out of pandas or a GeoPackage arrives as ``dtype=object``.
    ``np.savez`` stores that as a pickle, and ``np.load`` then refuses to read
    it back without ``allow_pickle`` -- so a matrix keyed on string zone ids
    would write successfully and be unreadable. Nothing downstream needs object
    ids, so they are converted once, here at the edge.
    """
    ids = np.asarray(ids)
    return ids.astype(str) if ids.dtype == object else ids


def read_points(
    path: Path,
    id_column: str | None = None,
    x_column: str = "lon",
    y_column: str = "lat",
    crs: str = WGS84,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read centroids from CSV or any vector format GeoPandas understands.

    Returns (ids, x, y) with coordinates in :data:`PROJECTED_CRS` metres.
    """
    path = Path(path)

    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        frame = pd.read_csv(path, sep=None, engine="python")
        missing = [c for c in (x_column, y_column) if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path} has no column(s) {missing}; available: {list(frame.columns)}"
            )
        points = gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame[x_column], frame[y_column]),
            crs=crs,
        )
    else:
        points = gpd.read_file(path)
        if points.crs is None:
            points = points.set_crs(crs)

    points = points.to_crs(PROJECTED_CRS)

    if id_column is not None:
        if id_column not in points.columns:
            raise ValueError(f"{path} has no id column {id_column!r}.")
        ids = normalise_ids(points[id_column].to_numpy())
    else:
        ids = np.arange(len(points))

    geometry = points.geometry
    if not (geometry.geom_type == "Point").all():
        geometry = geometry.representative_point()

    return ids, geometry.x.to_numpy(dtype=float), geometry.y.to_numpy(dtype=float)


def snap_to_network(
    network: RoutableNetwork,
    ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_snap_distance: float = DEFAULT_MAX_SNAP_M,
) -> Centroids:
    """Attach points to their nearest network node."""
    node_index, distance = network.nearest_nodes(x, y, max_distance=max_snap_distance)

    unsnapped = int((node_index < 0).sum())
    if unsnapped:
        logger.warning(
            "%d of %d centroids are further than %.0f m from any %s node",
            unsnapped,
            len(ids),
            max_snap_distance,
            network.mode,
        )
    snapped_distance = distance[node_index >= 0]
    if snapped_distance.size:
        logger.info(
            "Snap distance: median %.0f m, max %.0f m",
            float(np.median(snapped_distance)),
            float(snapped_distance.max()),
        )

    return Centroids(
        ids=np.asarray(ids),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        node_index=node_index,
        snap_distance=distance,
    )


def load_centroids(
    network: RoutableNetwork,
    path: Path,
    id_column: str | None = None,
    x_column: str = "lon",
    y_column: str = "lat",
    crs: str = WGS84,
    max_snap_distance: float = DEFAULT_MAX_SNAP_M,
) -> Centroids:
    ids, x, y = read_points(path, id_column, x_column, y_column, crs)
    logger.info("Read %d centroids from %s", len(ids), path)
    return snap_to_network(network, ids, x, y, max_snap_distance)
