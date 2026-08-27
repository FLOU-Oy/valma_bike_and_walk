"""Elevation: fetched from the National Land Survey, tiled and cached on disk.

MML publishes no whole-country GeoTIFF to download. What it publishes is a WCS
2.0.1 query service that cuts an arbitrary rectangle out of the 2 m laser-scanned
elevation model, and a request is capped at 10 km x 10 km and 5000 pixels a side.
So a "download the DEM" step is really a *tiling* step: cut the area we need into
tiles the service will serve, fetch the ones we do not already have, and keep
them.

The cache is the whole point. Tiles are cut on a fixed grid aligned to the
EPSG:3067 origin, so the same patch of Finland is always the same tile with the
same filename no matter which extent asked for it. Two runs over overlapping
areas share every tile they have in common, and a re-run downloads nothing.

Why a tile grid rather than one request per extent
--------------------------------------------------
An extent-shaped request is unshareable: shift the bbox by a metre and none of it
can be reused. Fixed tiles make the cache additive -- each run can only ever add
to what the next one finds already there.

Which tiles get fetched is decided from the *links*, not from their bounding box.
A rural network is mostly empty space, and asking per link means we pay for the
tiles the roads actually cross instead of the rectangle that contains them.

Coordinates
-----------
The service's native CRS is EPSG:3067, which is already
:data:`~valma_bike_and_walk.config.PROJECTED_CRS`. Nothing here reprojects a
raster; tile bounds, sample points and link geometry are all in the same metres.

Access
------
The service is free and unmetered but identified by a per-user API key, from
https://www.maanmittauslaitos.fi/rajapinnat/api-avaimen-ohje . Pass it as
``--api-key`` or set ``MML_API_KEY``. The data is CC BY 4.0: attribute the
National Land Survey of Finland in anything you publish from it.

TLS
---
On a corporate network the connection is often intercepted, and the certificate
that arrives is the inspector's rather than MML's. Python does not read the
Windows certificate store the way a browser does, so a root that Windows trusts
can still fail here. :func:`tls_context` prefers ``truststore``, which hands
verification to the operating system and so trusts whatever the machine already
trusts; verification can be turned off outright as a last resort.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from valma_bike_and_walk.config import PROJECTED_CRS
from valma_bike_and_walk.progress import Progress

logger = logging.getLogger(__name__)

#: Environment variable read when no key is passed explicitly.
API_KEY_ENV = "MML_API_KEY"

API_KEY_URL = "https://www.maanmittauslaitos.fi/rajapinnat/api-avaimen-ohje"

#: Environment variable that turns certificate verification off, for a network
#: whose TLS-inspecting root cannot be reached any other way. See
#: :func:`tls_context`.
INSECURE_ENV = "VALMA_INSECURE_TLS"


@dataclass(frozen=True)
class DemService:
    """A WCS coverage we can cut elevation tiles out of."""

    endpoint: str
    coverage: str
    #: Grid spacing of the coverage as published, in metres.
    native_resolution_m: float
    #: Largest rectangle the service will cut, per axis, in metres.
    max_extent_m: float
    #: Largest raster it will return, per axis, in pixels.
    max_pixels: int


#: The National Land Survey's 2 m elevation model, and the only elevation
#: coverage this service publishes -- the 10 m national model (KM10) is a
#: separate product of the file download service, not a coverage here. Coarser
#: grids come off this one coverage via WCS scaling.
MML_DEM = DemService(
    endpoint="https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2",
    coverage="korkeusmalli_2m",
    native_resolution_m=2.0,
    max_extent_m=10_000.0,
    max_pixels=5_000,
)

#: Grid spacings offered on the command line. The service will scale by any
#: factor, not just powers of two, so this list is a convenience rather than the
#: limit -- what actually has to hold is that the spacing divides the tile
#: exactly, which :func:`check_tile_size` is what enforces. Every entry here
#: divides :data:`DEFAULT_TILE_M`.
RESOLUTIONS_M: tuple[float, ...] = (
    2.0,
    4.0,
    5.0,
    8.0,
    10.0,
    16.0,
    20.0,
    25.0,
    32.0,
    50.0,
    100.0,
)

#: Default grid spacing to fetch at. 2 m is the native model and the only one
#: that resolves a street's own gradient rather than the hillside it sits on.
DEFAULT_RESOLUTION_M = 2.0

#: Tile edge in metres. Comfortably inside both service caps at every resolution
#: in :data:`RESOLUTIONS_M` (4000 px a side at 2 m), and a whole multiple of each
#: of them, so a tile never lands on a fractional pixel.
DEFAULT_TILE_M = 8_000

#: Tiles are fetched a few pixels larger than their grid cell.
#:
#: Not to close a seam -- there is no seam to close, because a point is looked up
#: in ``floor(coord / tile_m)`` and tiles are fetched over that same floor of
#: each link's bounds, so the tile a sample lands in is always one that was
#: fetched. The margin is for the *raster* edge: the service snaps a coverage to
#: its own pixel grid, and a request for exactly the cell can come back a pixel
#: short of it. Two pixels of slack means a sample right against the cell edge
#: still has data under it.
TILE_OVERLAP_PX = 2

#: Sample spacing along a link when reading its height profile.
PROFILE_SPACING_M = 25.0

#: Height changes below this between two consecutive samples are read as DEM
#: noise and dropped. Without it, sampling a flat road picks up the ditch and the
#: kerb either side of it and reports a climb that is not there -- the same
#: phantom-ascent problem a GPS track has.
DEAD_BAND_M = 0.5

# A terrain-model spike at a bridge or tunnel can spill into its neighbouring
# link. Only steep changes are treated as that artefact; gentler grades remain.
STRUCTURE_ADJACENT_GRADE = 0.15

#: Value MML writes where it has no measurement.
NODATA = -9999.0

_HTTP_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = 2.0
_TIMEOUT_SECONDS = 120.0

#: Big-endian and little-endian TIFF magic. A WCS error comes back as an XML
#: document, often under a 200, so the body has to be checked rather than trusted.
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")


# --------------------------------------------------------------------------
# The tile grid
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """One cell of the fixed EPSG:3067 grid, at one resolution."""

    col: int
    row: int
    tile_m: int
    resolution_m: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_e, min_n, max_e, max_n) of the grid cell itself."""
        return (
            float(self.col * self.tile_m),
            float(self.row * self.tile_m),
            float((self.col + 1) * self.tile_m),
            float((self.row + 1) * self.tile_m),
        )

    @property
    def request_bounds(self) -> tuple[float, float, float, float]:
        """The cell grown by :data:`TILE_OVERLAP_PX`, which is what we ask for."""
        pad = TILE_OVERLAP_PX * self.resolution_m
        min_e, min_n, max_e, max_n = self.bounds
        return (min_e - pad, min_n - pad, max_e + pad, max_n + pad)

    @property
    def filename(self) -> str:
        min_e, min_n, _, _ = self.bounds
        return f"E{int(min_e)}_N{int(min_n)}.tif"


def validate_resolution(resolution_m: float, service: DemService = MML_DEM) -> float:
    """
    A grid spacing the service can produce.

    Only the floor is fixed: the coverage is 2 m, so anything finer would be the
    service inventing detail. Above that it will scale by whatever factor is
    asked, which is why the pairing with a tile size -- not membership of
    :data:`RESOLUTIONS_M` -- is the check that matters.
    """
    resolution = float(resolution_m)
    if resolution < service.native_resolution_m:
        raise ValueError(
            f"Resolution {resolution:g} m is finer than the {service.coverage} "
            f"coverage's own {service.native_resolution_m:g} m grid."
        )
    return resolution


def check_tile_size(tile_m: int, resolution_m: float, service: DemService) -> int:
    """Fail early on a tile the service would refuse, or would round."""
    pixels = tile_m / resolution_m + 2 * TILE_OVERLAP_PX
    if tile_m > service.max_extent_m:
        raise ValueError(
            f"A {tile_m} m tile exceeds the service's {service.max_extent_m:.0f} m "
            "limit per axis."
        )
    if pixels > service.max_pixels:
        raise ValueError(
            f"A {tile_m} m tile at {resolution_m} m is {pixels:.0f} px, over the "
            f"service's {service.max_pixels} px limit. Use a coarser resolution "
            "or a smaller tile."
        )
    # The service fits whole pixels to the rectangle it is given. If the spacing
    # does not divide the tile, it does not refuse -- it silently returns a grid
    # of a slightly different spacing (3 m over an 8 km tile comes back at
    # 3.0007 m), and every sample after that is read off a raster whose geometry
    # is not the one the cache is named for.
    steps = tile_m / resolution_m
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(
            f"A {resolution_m:g} m grid does not divide a {tile_m} m tile "
            f"exactly ({steps:.4f} px), so the service would return a slightly "
            "different spacing than the one asked for. Pick a resolution that "
            "divides the tile, or a tile that is a whole multiple of it."
        )
    return int(tile_m)


def tiles_for_bounds(
    bounds: Sequence[float],
    resolution_m: float = DEFAULT_RESOLUTION_M,
    tile_m: int = DEFAULT_TILE_M,
) -> list[Tile]:
    """Every tile a projected (min_e, min_n, max_e, max_n) box touches."""
    min_e, min_n, max_e, max_n = (float(v) for v in bounds)
    cols = range(int(np.floor(min_e / tile_m)), int(np.floor(max_e / tile_m)) + 1)
    rows = range(int(np.floor(min_n / tile_m)), int(np.floor(max_n / tile_m)) + 1)
    return [Tile(c, r, tile_m, resolution_m) for c in cols for r in rows]


def tiles_for_links(
    links: gpd.GeoDataFrame,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    tile_m: int = DEFAULT_TILE_M,
) -> list[Tile]:
    """
    Every tile the links themselves touch -- not their bounding box.

    A link's own bounds are used rather than its exact geometry: cheap,
    vectorised, and at worst it fetches a tile a diagonal link only clips the
    corner of. The saving over the whole extent's bounding box is the point, and
    on a sparse network it is most of the download.
    """
    geometry = _projected(links).geometry
    min_e, min_n, max_e, max_n = (a / tile_m for a in geometry.bounds.to_numpy().T)

    cells: set[tuple[int, int]] = set()
    for c0, r0, c1, r1 in zip(
        np.floor(min_e).astype(int),
        np.floor(min_n).astype(int),
        np.floor(max_e).astype(int),
        np.floor(max_n).astype(int),
    ):
        for col in range(c0, c1 + 1):
            for row in range(r0, r1 + 1):
                cells.add((col, row))

    tiles = [Tile(c, r, tile_m, resolution_m) for c, r in sorted(cells)]
    logger.info("%d link(s) touch %d DEM tile(s)", len(links), len(tiles))
    return tiles


def _projected(links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if links.crs is None:
        raise ValueError("The link layer has no CRS, so it cannot be located.")
    if str(links.crs).upper() == PROJECTED_CRS:
        return links
    return links.to_crs(PROJECTED_CRS)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def coverage_url(tile: Tile, service: DemService, api_key: str) -> str:
    """The WCS GetCoverage request for one tile."""
    min_e, min_n, max_e, max_n = tile.request_bounds
    params = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("request", "GetCoverage"),
        ("CoverageID", service.coverage),
        ("SUBSET", f"E({min_e:.0f},{max_e:.0f})"),
        ("SUBSET", f"N({min_n:.0f},{max_n:.0f})"),
        ("format", "image/tiff"),
        ("geotiff:compression", "LZW"),
    ]
    # SCALEFACTOR 1 is the native grid; the service is happier without the
    # parameter at all than with a no-op one.
    scale = service.native_resolution_m / tile.resolution_m
    if scale != 1.0:
        params.append(("SCALEFACTOR", f"{scale:g}"))
    params.append(("api-key", api_key))

    query = urllib.parse.urlencode(params, safe="(),/:")
    return f"{service.endpoint}?{query}"


def resolve_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(
            "The elevation service needs an API key. Pass --api-key or set "
            f"{API_KEY_ENV}. Keys are free, from {API_KEY_URL}."
        )
    return key


def resolve_insecure(insecure: bool | None = None) -> bool:
    """Explicit flag if given, otherwise whatever :data:`INSECURE_ENV` says."""
    if insecure is not None:
        return bool(insecure)
    return os.environ.get(INSECURE_ENV, "").strip().lower() in {"1", "true", "yes"}


def _strip_key(url: str) -> str:
    """The request URL with the key taken out, so it is safe to log or raise."""
    split = urllib.parse.urlsplit(url)
    kept = [
        (k, v) for k, v in urllib.parse.parse_qsl(split.query) if k.lower() != "api-key"
    ]
    return urllib.parse.urlunsplit(
        split._replace(query=urllib.parse.urlencode(kept, safe="(),/:"))
    )


@lru_cache(maxsize=None)
def tls_context(insecure: bool = False) -> ssl.SSLContext:
    """
    How certificates are verified, built once and shared by every request.

    The default asks ``truststore`` to verify against the operating system's own
    store, which is what makes this work behind TLS inspection: the appliance's
    root is already trusted by the machine (that is how a browser gets through),
    but Python's bundled roots have never heard of it. Without truststore
    installed this falls back to those bundled roots, and ``SSL_CERT_FILE``
    pointed at a CA bundle is the manual equivalent.

    ``insecure`` skips verification entirely. It is a genuine escape hatch --
    the payload is public open data -- but the API key travels in the query
    string, so whoever is in the middle gets to keep it.
    """
    if insecure:
        logger.warning(
            "Certificate verification is OFF for elevation requests. The "
            "response is unauthenticated and the API key is exposed to anyone "
            "in the path."
        )
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    try:
        import truststore
    except ImportError:
        return ssl.create_default_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def _certificate_error(error: Exception) -> ssl.SSLCertVerificationError | None:
    """The verification failure inside a URLError, if that is what this is."""
    for candidate in (error, getattr(error, "reason", None)):
        if isinstance(candidate, ssl.SSLCertVerificationError):
            return candidate
    return None


def _fetch(
    url: str,
    timeout: float = _TIMEOUT_SECONDS,
    context: ssl.SSLContext | None = None,
) -> bytes:
    """GET the URL, retrying the failures that are worth retrying."""
    request = urllib.request.Request(url, headers={"User-Agent": "valma_bike_and_walk"})
    context = context if context is not None else tls_context()
    last: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=context
            ) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            # A 4xx that is not rate limiting will not become a 200 by asking
            # again -- a bad key or a bad request needs the caller to change.
            if error.code not in _HTTP_RETRY_STATUS:
                raise RuntimeError(
                    f"{error.code} {error.reason} from the elevation service for "
                    f"{_strip_key(url)}"
                ) from error
            last = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # A rejected certificate is not going to be accepted on the fourth
            # go, and retrying only buries the one line that says what to do.
            certificate = _certificate_error(error)
            if certificate is not None:
                raise RuntimeError(
                    f"The elevation service's certificate could not be verified "
                    f"({getattr(certificate, 'verify_message', None) or certificate}"
                    "). On a "
                    "network that inspects TLS, the certificate that arrives is "
                    "the inspector's, and Python does not trust it just because "
                    "the machine does. Fixes, best first:\n"
                    "  * pip install truststore -- verify against the operating "
                    "system's own store, which already trusts it\n"
                    "  * set SSL_CERT_FILE to a CA bundle that includes your "
                    "organisation's root\n"
                    f"  * --insecure, or {INSECURE_ENV}=1, to skip verification "
                    "altogether -- this hands the API key to whoever is in the "
                    "middle"
                ) from error
            last = error

        if attempt < _MAX_ATTEMPTS:
            delay = _BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.warning(
                "Elevation request failed (%s), retrying in %.0fs [%d/%d]",
                last,
                delay,
                attempt,
                _MAX_ATTEMPTS,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Elevation service unreachable after {_MAX_ATTEMPTS} attempts: {last}"
    )


def _check_is_tiff(payload: bytes, tile: Tile) -> None:
    if payload[:4] in _TIFF_MAGIC:
        return
    # WCS reports its errors as an XML document, sometimes under a 200 status.
    excerpt = payload[:400].decode("utf-8", errors="replace").strip()
    raise RuntimeError(
        f"The elevation service returned no GeoTIFF for tile {tile.filename}. "
        f"It said:\n{excerpt}"
    )


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


class DemCache:
    """A directory of elevation tiles, filled on demand and reused thereafter."""

    def __init__(
        self,
        root: Path,
        resolution_m: float = DEFAULT_RESOLUTION_M,
        tile_m: int = DEFAULT_TILE_M,
        service: DemService = MML_DEM,
        api_key: str | None = None,
        insecure: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.resolution_m = validate_resolution(resolution_m, service)
        self.tile_m = check_tile_size(tile_m, self.resolution_m, service)
        self.service = service
        self._api_key = api_key
        self.insecure = resolve_insecure(insecure)

    @property
    def directory(self) -> Path:
        """Tiles of one coverage at one resolution live together."""
        return self.root / f"{self.service.coverage}_{self.resolution_m:g}m"

    def path(self, tile: Tile) -> Path:
        return self.directory / tile.filename

    def tile_at(self, east: float, north: float) -> Tile:
        return Tile(
            int(np.floor(east / self.tile_m)),
            int(np.floor(north / self.tile_m)),
            self.tile_m,
            self.resolution_m,
        )

    def missing(self, tiles: Iterable[Tile]) -> list[Tile]:
        return [tile for tile in tiles if not self.path(tile).exists()]

    # -- fetching ---------------------------------------------------------

    def fetch(self, tile: Tile) -> Path:
        """
        Download one tile, unless it is already here.

        The write is atomic: the body lands in a ``.part`` beside the target and
        is renamed only once it is complete and has been confirmed to be a
        GeoTIFF. An interrupted run therefore leaves no half-file that a later
        run would mistake for a cache hit.
        """
        destination = self.path(tile)
        if destination.exists():
            return destination

        url = coverage_url(tile, self.service, resolve_api_key(self._api_key))
        payload = _fetch(url, context=tls_context(self.insecure))
        _check_is_tiff(payload, tile)

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".tif.part")
        partial.write_bytes(payload)
        os.replace(partial, destination)

        logger.debug("Fetched %s (%.1f MB)", destination.name, len(payload) / 1e6)
        return destination

    def ensure(self, tiles: Sequence[Tile], workers: int = 4) -> list[Path]:
        """
        Have every one of these tiles on disk, downloading only what is absent.

        Downloads run on a small thread pool because they are entirely I/O bound;
        the pool is deliberately modest, since this is a free public service and
        the point is to be a good citizen of it rather than to saturate it.
        """
        wanted = list(tiles)
        outstanding = self.missing(wanted)
        if not outstanding:
            logger.info(
                "All %d DEM tile(s) already cached in %s", len(wanted), self.directory
            )
            return [self.path(tile) for tile in wanted]

        # Fail on a missing key now rather than once per tile inside the pool.
        resolve_api_key(self._api_key)

        area_km2 = len(outstanding) * (self.tile_m / 1000.0) ** 2
        logger.info(
            "Fetching %d of %d DEM tile(s) at %g m (%.0f km2) into %s",
            len(outstanding),
            len(wanted),
            self.resolution_m,
            area_km2,
            self.directory,
        )
        if len(outstanding) > 500:
            logger.warning(
                "That is %d requests. A coarser --dem-resolution covers the same "
                "ground in fewer, smaller tiles.",
                len(outstanding),
            )

        with Progress(len(outstanding), "  downloading tiles") as progress:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for _ in pool.map(self.fetch, outstanding):
                    progress.advance()

        return [self.path(tile) for tile in wanted]

    # -- reading ----------------------------------------------------------

    def sample(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        """
        Height at each projected point, NaN where the DEM does not cover it.

        Points are grouped by tile so each raster is opened once, however the
        caller happened to order them.
        """
        rasterio = _require_rasterio()

        east = np.asarray(east, dtype=float)
        north = np.asarray(north, dtype=float)
        heights = np.full(east.shape, np.nan, dtype=float)
        if east.size == 0:
            return heights

        cols = np.floor(east / self.tile_m).astype(np.int64)
        rows = np.floor(north / self.tile_m).astype(np.int64)
        keys = np.column_stack([cols, rows])
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)

        logger.info("Reading %d height(s) from %d DEM tile(s)", east.size, len(unique))
        with Progress(len(unique), "  reading tiles") as progress:
            for i, (col, row) in enumerate(unique):
                tile = Tile(int(col), int(row), self.tile_m, self.resolution_m)
                path = self.path(tile)
                if not path.exists():
                    logger.warning(
                        "No cached DEM tile for %s; leaving those NaN", tile.filename
                    )
                    progress.advance()
                    continue

                selected = np.flatnonzero(inverse == i)
                with rasterio.open(path) as dataset:
                    values = np.fromiter(
                        (
                            v[0]
                            for v in dataset.sample(
                                zip(east[selected], north[selected])
                            )
                        ),
                        dtype=float,
                        count=selected.size,
                    )
                    nodata = dataset.nodata
                values[values == NODATA] = np.nan
                if nodata is not None:
                    values[values == nodata] = np.nan
                heights[selected] = values
                progress.advance()

        return heights


def _require_rasterio():
    try:
        import rasterio
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading elevation tiles needs rasterio, which is an optional "
            "dependency: pip install 'valma_bike_and_walk[dem]'. Downloading "
            "them does not."
        ) from error
    _prepare_gdal(rasterio)
    return rasterio


# --------------------------------------------------------------------------
# Making GDAL usable on a machine that has other GIS software on it
# --------------------------------------------------------------------------

#: Where GDAL looks for its PROJ database, newest name first.
PROJ_DATA_ENV: tuple[str, ...] = ("PROJ_DATA", "PROJ_LIB")

_gdal_prepared = False


def _prepare_gdal(rasterio) -> None:
    """Repair the PROJ path and stop GDAL repeating itself. Runs once."""
    global _gdal_prepared
    if _gdal_prepared:
        return
    _gdal_prepared = True
    _repair_proj_data(rasterio)
    _quieten_repeated_gdal_messages()


def _proj_layout_version(directory: Path) -> tuple[int, int] | None:
    """
    Layout version of the ``proj.db`` in this directory, None if there is none.

    PROJ refuses a database whose layout is older than the library reading it,
    and records that layout in the database itself, so this is the same number
    the complaint in the GDAL warning quotes.
    """
    database = directory / "proj.db"
    if not database.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            rows = dict(
                connection.execute(
                    "SELECT key, value FROM metadata "
                    "WHERE key LIKE 'DATABASE.LAYOUT.VERSION.%'"
                )
            )
        finally:
            connection.close()
        return (
            int(rows["DATABASE.LAYOUT.VERSION.MAJOR"]),
            int(rows["DATABASE.LAYOUT.VERSION.MINOR"]),
        )
    except (sqlite3.Error, KeyError, ValueError, OSError):
        return None


def _repair_proj_data(rasterio) -> None:
    """
    Point GDAL at a PROJ database it can actually read.

    ``PROJ_DATA`` (and the older ``PROJ_LIB``) is a machine-wide setting, and
    on Windows more or less anything that ships PROJ sets it -- PostgreSQL with
    PostGIS notably does. GDAL then reads *that* proj.db rather than the one
    rasterio brings with it, and if it is the older of the two, every single
    raster we open complains twice: once that the database layout is too old,
    and once that it therefore could not match the file's CRS against the EPSG
    registry. Six thousand tiles, twelve thousand warnings, and the CRS comes
    back as an unnamed local system instead of EPSG:3067.

    So when the database the environment points at is older than the one beside
    rasterio, we prefer rasterio's. GDAL reads the variable the first time it
    needs PROJ, which has not happened yet at import time, so setting it here is
    still early enough. A newer database is left alone -- someone who installed
    a current PROJ, with grid shifts we do not ship, should keep it. Nothing is
    set that was not already set: an environment that says nothing about PROJ
    already gets rasterio's own copy.
    """
    bundled = Path(rasterio.__file__).parent / "proj_data"
    bundled_version = _proj_layout_version(bundled)
    if bundled_version is None:  # pragma: no cover - a non-wheel rasterio
        return

    for variable in PROJ_DATA_ENV:
        current = os.environ.get(variable)
        if not current:
            continue
        # The variable is a search path, so one usable entry is enough.
        versions = [
            _proj_layout_version(Path(entry))
            for entry in current.split(os.pathsep)
            if entry
        ]
        if any(
            version is not None and version >= bundled_version for version in versions
        ):
            continue
        logger.info(
            "%s=%s has no PROJ database GDAL can read; using rasterio's at %s",
            variable,
            current,
            bundled,
        )
        os.environ[variable] = str(bundled)


class _SayItOnce(logging.Filter):
    """Pass each distinct message through the first time and no more."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if message in self._seen:
            return False
        self._seen.add(message)
        return True


def _quieten_repeated_gdal_messages() -> None:
    """
    Let GDAL say each thing once instead of once per tile.

    GDAL's complaints are about the environment, not about the file in hand, so
    reading a few thousand tiles reproduces the same two or three lines a few
    thousand times and buries everything else. Deduplicating is not the same as
    silencing: whatever GDAL has to say is still said, in full, the first time
    it says it -- and ``--verbose`` is unaffected, since the filter drops
    repeats rather than lowering a level.
    """
    logging.getLogger("rasterio._env").addFilter(_SayItOnce())


# --------------------------------------------------------------------------
# Height profiles along links
# --------------------------------------------------------------------------

#: Sampler signature: projected (east, north) in, heights out.
Sampler = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _sample_positions(
    links: gpd.GeoDataFrame, spacing_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Points at a fixed spacing along every link.

    Returns (link_index, east, north, samples_per_link). Sampling at a fixed
    spacing rather than at the geometry's own vertices matters both ways round:
    a long straight link has only two vertices and would otherwise hide a hill
    between them, and a densely drawn curve would otherwise be sampled far more
    heavily than it needs.
    """
    geometry = _projected(links).geometry.to_numpy()
    length = shapely.length(geometry)

    counts = np.maximum(2, np.ceil(length / spacing_m).astype(np.int64) + 1)
    total = int(counts.sum())

    offsets = np.zeros(counts.shape[0], dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])

    link_index = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    within = np.arange(total, dtype=np.int64) - np.repeat(offsets, counts)
    # Fractions 0..1 along each link, so the first and last sample land exactly
    # on its endpoints and z_u/z_v come out of the same pass.
    fraction = within / (np.repeat(counts, counts) - 1)

    points = shapely.line_interpolate_point(
        np.repeat(geometry, counts), fraction * np.repeat(length, counts)
    )
    return link_index, shapely.get_x(points), shapely.get_y(points), counts


def link_profiles(
    links: gpd.GeoDataFrame,
    sampler: Sampler,
    spacing_m: float = PROFILE_SPACING_M,
    dead_band_m: float = DEAD_BAND_M,
) -> pd.DataFrame:
    """
    Read each link's height profile and reduce it to four columns.

    ``z_u`` / ``z_v``       height at the link's first and last vertex, metres.
    ``ascent_m``            metres climbed following the digitised direction.
    ``descent_m``           metres dropped following it.

    Ascent and descent are kept apart rather than netted off, because the cost of
    a link that goes up ten metres and back down again is nothing like the cost
    of a flat one, and because the two swap places when the link is traversed
    backwards -- which is exactly the asymmetry the routing needs.

    ``sampler`` is injected rather than fixed so the reduction can be tested
    against a known surface, and so a caller with elevation from somewhere other
    than :class:`DemCache` can use it.

    Links tagged ``bridge`` or ``tunnel`` are forced flat. MML's model is a
    *terrain* model: it reports the valley floor under a bridge and the hilltop
    over a tunnel, so sampling one produces a violent climb that is not on the
    route. Flat is wrong for a bridge that genuinely rises, but it is wrong by
    far less, and ``grade_override`` is there for the ones that matter.
    """
    logger.info(
        "Laying out sample points every %g m along %d link(s)", spacing_m, len(links)
    )
    link_index, east, north, counts = _sample_positions(links, spacing_m)
    z = np.asarray(sampler(east, north), dtype=float)
    if z.shape != east.shape:
        raise ValueError(
            f"The sampler returned {z.shape} heights for {east.shape} points."
        )

    n = len(links)
    first = np.zeros(counts.shape[0], dtype=np.int64)
    np.cumsum(counts[:-1], out=first[1:])
    last = first + counts - 1

    step = z[1:] - z[:-1]
    same_link = link_index[1:] == link_index[:-1]
    step = step[same_link]
    owner = link_index[1:][same_link]

    unknown = ~np.isfinite(step)
    if unknown.any():
        logger.warning(
            "%d of %d profile steps had no DEM value and were read as flat",
            int(unknown.sum()),
            step.shape[0],
        )
        step = np.where(unknown, 0.0, step)

    # The dead band goes on each step, not on the total: it is there to stop
    # noise accumulating, and a genuine slope survives it intact.
    step = np.where(np.abs(step) < dead_band_m, 0.0, step)

    profile = pd.DataFrame(
        {
            "z_u": z[first],
            "z_v": z[last],
            "ascent_m": np.bincount(owner, weights=np.maximum(step, 0.0), minlength=n),
            "descent_m": np.bincount(
                owner, weights=np.maximum(-step, 0.0), minlength=n
            ),
        },
        index=links.index,
    )

    flat = _structure_mask(links)
    if flat.any():
        logger.info("Forcing %d bridge/tunnel link(s) flat", int(flat.sum()))
        profile.loc[flat, ["ascent_m", "descent_m"]] = 0.0
        profile.loc[flat, "z_v"] = profile.loc[flat, "z_u"]

    _remove_structure_adjacent_spikes(links, profile, flat)

    return profile


def _structure_mask(links: gpd.GeoDataFrame) -> np.ndarray:
    """Links carried on a bridge or through a tunnel, where the DEM lies."""
    mask = np.zeros(len(links), dtype=bool)
    for column in ("bridge", "tunnel"):
        if column in links.columns:
            values = links[column].astype("object")
            mask |= values.notna().to_numpy() & (values != "no").to_numpy()
    return mask


def _remove_structure_adjacent_spikes(
    links: gpd.GeoDataFrame, profile: pd.DataFrame, structure: np.ndarray
) -> None:
    """Discard steep DEM changes on links touching a bridge or tunnel."""
    if not structure.any() or not {"u", "v"}.issubset(links.columns):
        return

    endpoints = links[["u", "v"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    structure_nodes = np.unique(endpoints[structure][np.isfinite(endpoints[structure])])
    adjacent = np.isin(endpoints, structure_nodes).any(axis=1) & ~structure
    if not adjacent.any():
        return

    length = shapely.length(_projected(links).geometry.to_numpy())
    ascent = profile["ascent_m"].to_numpy(dtype=float)
    descent = profile["descent_m"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ascent_grade = np.divide(
            ascent, length, out=np.zeros_like(ascent), where=length > 0
        )
        descent_grade = np.divide(
            descent, length, out=np.zeros_like(descent), where=length > 0
        )

    remove_ascent = adjacent & (ascent_grade > STRUCTURE_ADJACENT_GRADE)
    remove_descent = adjacent & (descent_grade > STRUCTURE_ADJACENT_GRADE)
    profile.loc[remove_ascent, "ascent_m"] = 0.0
    profile.loc[remove_descent, "descent_m"] = 0.0
    removed = int(remove_ascent.sum() + remove_descent.sum())
    if removed:
        logger.info(
            "Removed %d steep ascent/descent artefact(s) next to bridges/tunnels",
            removed,
        )


def add_elevation(
    links: gpd.GeoDataFrame,
    cache: DemCache,
    spacing_m: float = PROFILE_SPACING_M,
    workers: int = 4,
) -> gpd.GeoDataFrame:
    """
    Attach the elevation columns to a link layer, downloading what it needs.

    This is the one call the pipeline makes: it works out which tiles the links
    touch, fills any gaps in the cache, reads the profiles and returns the layer
    with ``z_u``, ``z_v``, ``ascent_m``, ``descent_m`` and an empty
    ``grade_override`` added -- the same "an empty column is an invitation"
    convention ``speed_override_kmh`` follows.
    """
    cache.ensure(tiles_for_links(links, cache.resolution_m, cache.tile_m), workers)

    links = links.copy()
    profile = link_profiles(links, cache.sample, spacing_m=spacing_m)
    for column in ("z_u", "z_v", "ascent_m", "descent_m"):
        links[column] = profile[column].to_numpy()
    if "grade_override" not in links.columns:
        links["grade_override"] = np.nan

    covered = np.isfinite(profile["z_u"].to_numpy())
    logger.info(
        "Elevation attached to %d of %d link(s); mean ascent %.2f m/km",
        int(covered.sum()),
        len(links),
        _ascent_per_km(links),
    )
    return links


def _ascent_per_km(links: gpd.GeoDataFrame) -> float:
    total_m = float(np.nansum(links["ascent_m"].to_numpy()))
    total_km = float(np.nansum(links["length_m"].to_numpy())) / 1000.0
    return total_m / total_km if total_km > 0 else 0.0


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_RESOLUTION_M",
    "DEFAULT_TILE_M",
    "INSECURE_ENV",
    "RESOLUTIONS_M",
    "DemCache",
    "DemService",
    "MML_DEM",
    "Tile",
    "add_elevation",
    "coverage_url",
    "link_profiles",
    "resolve_api_key",
    "resolve_insecure",
    "tiles_for_bounds",
    "tiles_for_links",
    "tls_context",
]
