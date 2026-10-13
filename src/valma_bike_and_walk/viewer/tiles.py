"""Mapbox vector tiles of the link layer: geometry and a link index only.

A tile carries nothing but each link's row index (``i``). Everything the map
colours by is fetched separately as an array indexed the same way, which is
what lets one set of cached tiles serve every run and every measure.
"""

from __future__ import annotations

import math
from pathlib import Path

import mapbox_vector_tile
import numpy as np
import shapely

#: Half the Web Mercator world's width, metres.
ORIGIN = 20037508.342789244
EXTENT = 4096
#: Tile-unit margin, so lines crossing a tile edge join up without a seam.
BUFFER = 64


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Web Mercator bounds of an XYZ tile (y counted from the top)."""
    size = 2 * ORIGIN / (2**z)
    minx = -ORIGIN + x * size
    maxy = ORIGIN - y * size
    return minx, maxy - size, minx + size, maxy


def render_tile(
    tree: shapely.STRtree,
    geometry: np.ndarray,
    min_zoom: np.ndarray,
    z: int,
    x: int,
    y: int,
) -> bytes:
    """Encode the links visible at zoom ``z`` in tile ``(x, y)``."""
    minx, miny, maxx, maxy = bounds = tile_bounds(z, x, y)
    unit = (maxx - minx) / EXTENT
    pad = BUFFER * unit
    window = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    index = tree.query(shapely.box(*window))
    index = index[min_zoom[index] <= z]
    if index.size == 0:
        return b""
    clipped = shapely.clip_by_rect(geometry[index], *window)
    # About half a tile unit: invisible at this zoom, and it keeps tiles small.
    clipped = shapely.simplify(clipped, unit * 0.5, preserve_topology=False)
    keep = ~shapely.is_empty(clipped)
    features = [
        {"geometry": geom, "properties": {"i": int(i)}}
        for geom, i in zip(clipped[keep], index[keep])
    ]
    if not features:
        return b""
    return mapbox_vector_tile.encode(
        [{"name": "links", "features": features}],
        default_options={"quantize_bounds": bounds, "extents": EXTENT},
    )


class TileCache:
    """Tiles on disk, keyed on the prepared data they were cut from."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path(self, z: int, x: int, y: int) -> Path:
        return self.directory / str(z) / str(x) / f"{y}.mvt"

    def get(self, z: int, x: int, y: int) -> bytes | None:
        path = self.path(z, x, y)
        return path.read_bytes() if path.exists() else None

    def put(self, z: int, x: int, y: int, tile: bytes) -> None:
        path = self.path(z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(tile)
        tmp.replace(path)


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    """The XYZ tile containing a point, for tests and warm-up."""
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y
