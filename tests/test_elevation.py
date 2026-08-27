"""Tiling, caching and profile reduction.

Nothing here touches the network. The download is exercised against a stub
``urlopen``, which is what lets the tests assert the things that actually matter
about a cache -- that a hit costs no request, that an error response never lands
on disk as if it were data -- without a key or a connection.
"""

from __future__ import annotations

import ssl
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from valma_bike_and_walk import elevation
from valma_bike_and_walk.config import PROJECTED_CRS
from valma_bike_and_walk.elevation import DemCache, Tile, coverage_url, tiles_for_links

from .conftest import links_frame

TIFF_HEADER = b"II*\x00"


def fake_tiff(payload: bytes = b"pixels") -> bytes:
    return TIFF_HEADER + payload


class StubResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class StubServer:
    """Records every request and answers with whatever it was told to."""

    def __init__(self, body: bytes | Exception = fake_tiff()) -> None:
        self.body = body
        self.urls: list[str] = []
        self.contexts: list[object] = []

    def __call__(self, request, timeout=None, context=None):
        self.urls.append(request.full_url)
        self.contexts.append(context)
        if isinstance(self.body, Exception):
            raise self.body
        return StubResponse(self.body)


@pytest.fixture
def server(monkeypatch):
    stub = StubServer()
    monkeypatch.setattr(elevation.urllib.request, "urlopen", stub)
    monkeypatch.setenv(elevation.API_KEY_ENV, "test-key")
    # Retries would otherwise make a failure test sleep for its whole backoff.
    monkeypatch.setattr(elevation, "_BACKOFF_SECONDS", 0.0)
    return stub


def cache_at(tmp_path: Path, **kwargs) -> DemCache:
    return DemCache(root=tmp_path / "dem", **kwargs)


# --------------------------------------------------------------------------
# The tile grid
# --------------------------------------------------------------------------


def test_tiles_snap_to_a_fixed_national_grid():
    """The same ground is the same tile however the query was framed."""
    wide = elevation.tiles_for_bounds((380_100.0, 6_670_100.0, 380_200.0, 6_670_200.0))
    narrow = elevation.tiles_for_bounds(
        (380_150.0, 6_670_150.0, 380_160.0, 6_670_160.0)
    )
    assert wide == narrow
    assert len(wide) == 1
    assert wide[0].filename == "E376000_N6664000.tif"


def test_tile_bounds_are_the_grid_cell_and_the_request_is_padded():
    tile = Tile(col=47, row=833, tile_m=8000, resolution_m=2.0)
    assert tile.bounds == (376_000.0, 6_664_000.0, 384_000.0, 6_672_000.0)

    # Padded by TILE_OVERLAP_PX pixels, so a sample hard against the cell edge
    # still has a pixel under it if the service snaps the coverage to its own
    # grid and comes back a pixel short.
    min_e, min_n, max_e, max_n = tile.request_bounds
    assert min_e == 376_000.0 - 4.0
    assert max_n == 6_672_000.0 + 4.0


def test_a_box_spanning_the_grid_gets_every_tile_it_touches():
    tiles = elevation.tiles_for_bounds(
        (7_900.0, 7_900.0, 8_100.0, 8_100.0), tile_m=8000
    )
    assert {(t.col, t.row) for t in tiles} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_tiles_follow_the_links_not_their_bounding_box():
    """Two far-apart links must not drag in the empty tiles between them."""
    links = links_frame(
        [
            {"coords": [(1_000.0, 1_000.0), (2_000.0, 1_000.0)]},
            {"coords": [(41_000.0, 1_000.0), (42_000.0, 1_000.0)]},
        ],
        crs=PROJECTED_CRS,
    )
    tiles = tiles_for_links(links, tile_m=8000)
    assert {(t.col, t.row) for t in tiles} == {(0, 0), (5, 0)}

    # The bounding box of the same two links spans the lot.
    assert len(elevation.tiles_for_bounds(links.total_bounds, tile_m=8000)) == 6


def test_a_tile_the_service_would_refuse_is_caught_before_any_request():
    with pytest.raises(ValueError, match="10000 m limit"):
        elevation.check_tile_size(20_000, 2.0, elevation.MML_DEM)
    with pytest.raises(ValueError, match="px limit"):
        elevation.check_tile_size(10_000, 2.0, elevation.MML_DEM)


def test_a_resolution_finer_than_the_coverage_is_refused():
    assert elevation.validate_resolution(8) == 8.0
    # Not a power of two, but the service scales by any factor and 10 m divides
    # the default tile, so it is a real option.
    assert elevation.validate_resolution(10) == 10.0
    with pytest.raises(ValueError, match="finer than"):
        elevation.validate_resolution(1)


def test_a_resolution_that_does_not_divide_the_tile_is_refused():
    """The service would round it rather than refuse, which is worse."""
    assert elevation.check_tile_size(8000, 10.0, elevation.MML_DEM) == 8000
    with pytest.raises(ValueError, match="does not divide"):
        elevation.check_tile_size(8000, 3.0, elevation.MML_DEM)


# --------------------------------------------------------------------------
# Request building
# --------------------------------------------------------------------------


def test_coverage_url_carries_the_subsets_and_the_key():
    url = coverage_url(Tile(0, 0, 8000, 2.0), elevation.MML_DEM, "secret")
    assert "CoverageID=korkeusmalli_2m" in url
    assert "SUBSET=E(-4,8004)" in url
    assert "SUBSET=N(-4,8004)" in url
    assert "format=image/tiff" in url
    assert "api-key=secret" in url
    # Native resolution: no scaling parameter at all.
    assert "SCALEFACTOR" not in url


def test_a_coarser_resolution_asks_the_service_to_scale():
    url = coverage_url(Tile(0, 0, 8000, 8.0), elevation.MML_DEM, "secret")
    assert "SCALEFACTOR=0.25" in url


def test_the_key_is_stripped_from_anything_we_report():
    url = coverage_url(Tile(0, 0, 8000, 2.0), elevation.MML_DEM, "secret")
    assert "secret" not in elevation._strip_key(url)


def test_a_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(elevation.API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match="needs an API key"):
        elevation.resolve_api_key(None)


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


def test_a_tile_is_downloaded_once_and_then_reused(tmp_path, server):
    cache = cache_at(tmp_path)
    tiles = [Tile(0, 0, 8000, 2.0)]

    cache.ensure(tiles)
    assert len(server.urls) == 1
    assert cache.path(tiles[0]).read_bytes() == fake_tiff()

    # Second run, same tile: the cache answers and nothing is requested.
    cache.ensure(tiles)
    assert len(server.urls) == 1


def test_only_the_missing_tiles_of_a_set_are_fetched(tmp_path, server):
    cache = cache_at(tmp_path)
    cache.ensure([Tile(0, 0, 8000, 2.0)])
    server.urls.clear()

    cache.ensure([Tile(0, 0, 8000, 2.0), Tile(1, 0, 8000, 2.0)])
    assert len(server.urls) == 1
    assert "SUBSET=E(7996,16004)" in server.urls[0]


def test_resolutions_are_cached_side_by_side(tmp_path, server):
    """Refetching at 8 m must not be mistaken for the 2 m tile already held."""
    coarse = cache_at(tmp_path, resolution_m=8.0)
    fine = cache_at(tmp_path, resolution_m=2.0)
    assert coarse.directory != fine.directory

    coarse.ensure([Tile(0, 0, 8000, 8.0)])
    fine.ensure([Tile(0, 0, 8000, 2.0)])
    assert len(server.urls) == 2


def test_an_xml_error_never_lands_in_the_cache(tmp_path, server):
    """WCS reports failures as XML, sometimes under a 200. Trusting the status
    code would poison the cache with a file that looks like a hit forever."""
    server.body = (
        b'<?xml version="1.0"?><ExceptionReport>Invalid api-key</ExceptionReport>'
    )
    cache = cache_at(tmp_path)
    tile = Tile(0, 0, 8000, 2.0)

    with pytest.raises(RuntimeError, match="Invalid api-key"):
        cache.ensure([tile])

    assert not cache.path(tile).exists()
    assert list(cache.root.rglob("*.part")) == []


def test_a_failed_download_leaves_no_partial_file(tmp_path, server):
    server.body = urllib.error.URLError("connection reset")
    cache = cache_at(tmp_path)
    tile = Tile(0, 0, 8000, 2.0)

    with pytest.raises(RuntimeError, match="unreachable"):
        cache.ensure([tile])
    assert not cache.path(tile).exists()


def test_a_rejected_certificate_is_not_retried(tmp_path, server):
    """It will not be accepted on the fourth attempt, and retrying buries the
    one message that says what to do about it."""
    verification = ssl.SSLCertVerificationError(
        "certificate verify failed: self-signed certificate in certificate chain"
    )
    server.body = urllib.error.URLError(verification)

    with pytest.raises(RuntimeError, match="certificate could not be verified"):
        cache_at(tmp_path).ensure([Tile(0, 0, 8000, 2.0)])
    assert len(server.urls) == 1


def test_the_insecure_context_is_the_one_that_reaches_the_request(tmp_path, server):
    cache_at(tmp_path, insecure=True).ensure([Tile(0, 0, 8000, 2.0)])
    assert server.contexts[0].verify_mode == ssl.CERT_NONE

    server.urls.clear()
    server.contexts.clear()
    cache_at(tmp_path / "verified", insecure=False).ensure([Tile(0, 0, 8000, 2.0)])
    assert server.contexts[0].verify_mode == ssl.CERT_REQUIRED


def test_the_environment_can_turn_verification_off(monkeypatch):
    monkeypatch.delenv(elevation.INSECURE_ENV, raising=False)
    assert elevation.resolve_insecure() is False
    monkeypatch.setenv(elevation.INSECURE_ENV, "1")
    assert elevation.resolve_insecure() is True
    # An explicit flag still wins over the environment.
    assert elevation.resolve_insecure(False) is False


def test_a_bad_key_is_not_retried(tmp_path, server):
    """403 will not become 200 by asking again; only the caller can fix it."""
    server.body = urllib.error.HTTPError(
        url="http://x", code=403, msg="Forbidden", hdrs=None, fp=None
    )
    with pytest.raises(RuntimeError, match="403"):
        cache_at(tmp_path).ensure([Tile(0, 0, 8000, 2.0)])
    assert len(server.urls) == 1


def test_a_rate_limit_is_retried(tmp_path, server):
    server.body = urllib.error.HTTPError(
        url="http://x", code=429, msg="Too Many Requests", hdrs=None, fp=None
    )
    with pytest.raises(RuntimeError, match="unreachable"):
        cache_at(tmp_path).ensure([Tile(0, 0, 8000, 2.0)])
    assert len(server.urls) == elevation._MAX_ATTEMPTS


def test_a_missing_key_fails_before_any_tile_is_attempted(
    tmp_path, server, monkeypatch
):
    monkeypatch.delenv(elevation.API_KEY_ENV, raising=False)
    with pytest.raises(ValueError, match="needs an API key"):
        cache_at(tmp_path).ensure([Tile(0, 0, 8000, 2.0), Tile(1, 0, 8000, 2.0)])
    assert server.urls == []


# --------------------------------------------------------------------------
# Height profiles
# --------------------------------------------------------------------------


def slope_sampler(gradient: float, base: float = 100.0):
    """A plane rising ``gradient`` metres per metre of easting."""

    def sample(east: np.ndarray, north: np.ndarray) -> np.ndarray:
        return base + gradient * np.asarray(east, dtype=float)

    return sample


def test_a_constant_climb_becomes_ascent_and_nothing_else():
    links = links_frame([{"coords": [(0.0, 0.0), (1_000.0, 0.0)]}], crs=PROJECTED_CRS)
    profile = elevation.link_profiles(links, slope_sampler(0.05))

    assert profile["z_u"].iloc[0] == pytest.approx(100.0)
    assert profile["z_v"].iloc[0] == pytest.approx(150.0)
    assert profile["ascent_m"].iloc[0] == pytest.approx(50.0)
    assert profile["descent_m"].iloc[0] == pytest.approx(0.0)


def test_ascent_and_descent_are_kept_apart_over_a_hill():
    """A link up and back down nets to zero rise but is not a flat link."""

    def over_a_hill(east, north):
        return 100.0 + 20.0 * np.sin(np.pi * np.asarray(east, dtype=float) / 1_000.0)

    links = links_frame([{"coords": [(0.0, 0.0), (1_000.0, 0.0)]}], crs=PROJECTED_CRS)
    profile = elevation.link_profiles(links, over_a_hill)

    assert profile["z_v"].iloc[0] == pytest.approx(profile["z_u"].iloc[0], abs=1e-6)
    # A little under the true 20 m: the steps either side of the crest are
    # shallower than the dead band, so they are read as flat. That is the dead
    # band working, not an error -- it costs a metre on a hill and saves several
    # on every flat road in the country.
    assert profile["ascent_m"].iloc[0] == pytest.approx(19.0, abs=0.5)
    assert profile["descent_m"].iloc[0] == pytest.approx(19.0, abs=0.5)


def test_the_dead_band_absorbs_dem_noise():
    """Sampling a flat road picks up the kerb and the ditch. It is not a climb."""
    rng = np.random.default_rng(0)

    def noisy_flat(east, north):
        return 100.0 + rng.normal(0.0, 0.15, size=np.asarray(east).shape)

    links = links_frame([{"coords": [(0.0, 0.0), (2_000.0, 0.0)]}], crs=PROJECTED_CRS)
    profile = elevation.link_profiles(links, noisy_flat)
    assert profile["ascent_m"].iloc[0] == 0.0
    assert profile["descent_m"].iloc[0] == 0.0

    # The same noise with no dead band invents metres of climbing out of nothing.
    inflated = elevation.link_profiles(links, noisy_flat, dead_band_m=0.0)
    assert inflated["ascent_m"].iloc[0] > 1.0


def test_a_real_slope_survives_the_dead_band():
    links = links_frame([{"coords": [(0.0, 0.0), (1_000.0, 0.0)]}], crs=PROJECTED_CRS)
    # 25 m spacing at a 2% grade is 0.5 m a step -- right on the dead band, which
    # is the case worth pinning down.
    profile = elevation.link_profiles(links, slope_sampler(0.02), spacing_m=25.0)
    assert profile["ascent_m"].iloc[0] == pytest.approx(20.0)


def test_long_links_are_sampled_along_their_length_not_just_at_the_ends():
    """A two-vertex link across a valley is not the flat link its ends suggest."""

    def valley(east, north):
        return 100.0 + np.abs(np.asarray(east, dtype=float) - 500.0) / 10.0

    links = links_frame([{"coords": [(0.0, 0.0), (1_000.0, 0.0)]}], crs=PROJECTED_CRS)
    profile = elevation.link_profiles(links, valley)

    assert profile["z_u"].iloc[0] == pytest.approx(profile["z_v"].iloc[0])
    assert profile["descent_m"].iloc[0] == pytest.approx(50.0, abs=1.0)
    assert profile["ascent_m"].iloc[0] == pytest.approx(50.0, abs=1.0)


def test_bridges_and_tunnels_are_forced_flat():
    """The DEM is a terrain model: under the bridge, over the tunnel."""
    links = links_frame(
        [
            {"coords": [(0.0, 0.0), (500.0, 0.0)]},
            {"coords": [(500.0, 0.0), (1_000.0, 0.0)]},
            {"coords": [(1_000.0, 0.0), (1_500.0, 0.0)]},
        ],
        crs=PROJECTED_CRS,
    )
    links["bridge"] = [None, "yes", "no"]
    profile = elevation.link_profiles(links, slope_sampler(0.05))

    assert profile["ascent_m"].iloc[0] == pytest.approx(25.0)
    assert profile["ascent_m"].iloc[1] == 0.0
    assert profile["z_v"].iloc[1] == profile["z_u"].iloc[1]
    # bridge="no" is not a bridge.
    assert profile["ascent_m"].iloc[2] == pytest.approx(25.0)


def test_steep_profile_changes_next_to_structures_are_removed():
    links = links_frame(
        [
            {"u": 1, "v": 2, "coords": [(0.0, 0.0), (500.0, 0.0)]},
            {
                "u": 2,
                "v": 3,
                "coords": [(500.0, 0.0), (1_000.0, 0.0)],
                "bridge": "yes",
            },
            {"u": 3, "v": 4, "coords": [(1_000.0, 0.0), (1_500.0, 0.0)]},
        ],
        crs=PROJECTED_CRS,
    )
    links["bridge"] = [None, "yes", None]
    profile = elevation.link_profiles(links, slope_sampler(0.20))

    assert profile["ascent_m"].to_numpy() == pytest.approx([0.0, 0.0, 0.0])
    assert profile["descent_m"].to_numpy() == pytest.approx([0.0, 0.0, 0.0])


def test_moderate_profile_changes_next_to_structures_are_kept():
    links = links_frame(
        [
            {"u": 1, "v": 2, "coords": [(0.0, 0.0), (500.0, 0.0)]},
            {
                "u": 2,
                "v": 3,
                "coords": [(500.0, 0.0), (1_000.0, 0.0)],
                "bridge": "yes",
            },
        ],
        crs=PROJECTED_CRS,
    )
    links["bridge"] = [None, "yes"]
    profile = elevation.link_profiles(links, slope_sampler(0.10))

    assert profile["ascent_m"].iloc[0] == pytest.approx(50.0)


def test_a_gap_in_the_dem_is_read_as_flat_rather_than_poisoning_the_total():
    def holey(east, north):
        z = 100.0 + 0.05 * np.asarray(east, dtype=float)
        z[np.asarray(east) > 500.0] = np.nan
        return z

    links = links_frame([{"coords": [(0.0, 0.0), (1_000.0, 0.0)]}], crs=PROJECTED_CRS)
    profile = elevation.link_profiles(links, holey)
    assert np.isfinite(profile["ascent_m"].iloc[0])
    assert profile["ascent_m"].iloc[0] == pytest.approx(25.0, abs=1.0)


def test_profiles_are_read_in_one_pass_over_every_link():
    """The reduction has to stay aligned when links have different lengths."""
    links = links_frame(
        [
            {"coords": [(0.0, 0.0), (100.0, 0.0)]},
            {"coords": [(0.0, 0.0), (1_000.0, 0.0)]},
            {"coords": [(0.0, 0.0), (10.0, 0.0)]},
        ],
        crs=PROJECTED_CRS,
    )
    profile = elevation.link_profiles(links, slope_sampler(0.05))
    assert profile["ascent_m"].to_numpy() == pytest.approx([5.0, 50.0, 0.5])


def test_a_sampler_returning_the_wrong_shape_is_rejected():
    links = links_frame([{"coords": [(0.0, 0.0), (100.0, 0.0)]}], crs=PROJECTED_CRS)
    with pytest.raises(ValueError, match="heights for"):
        elevation.link_profiles(links, lambda e, n: np.zeros(3))


# --------------------------------------------------------------------------
# End to end, against the stub server
# --------------------------------------------------------------------------


def test_add_elevation_fetches_only_what_the_links_touch(tmp_path, server, monkeypatch):
    links = links_frame(
        [{"coords": [(1_000.0, 1_000.0), (2_000.0, 1_000.0)]}], crs=PROJECTED_CRS
    )
    links["length_m"] = 1_000.0

    cache = cache_at(tmp_path)
    monkeypatch.setattr(
        DemCache, "sample", lambda self, e, n: slope_sampler(0.05)(e, n)
    )
    elevated = elevation.add_elevation(links, cache)

    assert len(server.urls) == 1
    for column in ("z_u", "z_v", "ascent_m", "descent_m", "grade_override"):
        assert column in elevated.columns
    assert elevated["ascent_m"].iloc[0] == pytest.approx(50.0)
    assert elevated["grade_override"].isna().all()


def write_tile(cache: DemCache, tile: Tile, surface, nodata_at=None) -> None:
    """Put a real GeoTIFF in the cache, so the rasterio path can be exercised.

    Written with no CRS on purpose: sampling reads the affine transform and
    nothing else, and leaving the CRS off keeps the test clear of whichever PROJ
    database happens to be first on the machine running it.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    min_e, min_n, max_e, max_n = tile.request_bounds
    res = tile.resolution_m
    width = int(round((max_e - min_e) / res))
    height = int(round((max_n - min_n) / res))

    east = min_e + (np.arange(width) + 0.5) * res
    north = max_n - (np.arange(height) + 0.5) * res
    grid = surface(*np.meshgrid(east, north)).astype("float32")
    if nodata_at is not None:
        e, n = nodata_at
        grid[int((max_n - n) / res), int((e - min_e) / res)] = elevation.NODATA

    path = cache.path(tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        nodata=elevation.NODATA,
        transform=from_origin(min_e, max_n, res, res),
    ) as dataset:
        dataset.write(grid, 1)


def test_sampling_reads_a_real_geotiff(tmp_path):
    cache = cache_at(tmp_path, resolution_m=8.0)
    tile = cache.tile_at(380_000.0, 6_668_000.0)
    write_tile(cache, tile, lambda e, n: e * 0.05, nodata_at=(382_000.0, 6_666_000.0))

    heights = cache.sample(
        np.array([380_000.0, 383_999.0, 382_000.0, 999_000.0]),
        np.full(4, 6_668_000.0),
    )
    assert heights[0] == pytest.approx(19_000.0, abs=0.5)
    # Hard against the cell edge: this is what the overlap margin is for.
    assert np.isfinite(heights[1])
    # An uncached tile reads NaN rather than silently borrowing a neighbour's.
    assert np.isnan(heights[3])

    nodata = cache.sample(np.array([382_000.0]), np.array([6_666_000.0]))
    assert np.isnan(nodata[0])
    # Its neighbour still reads, so that NaN was the nodata and not a cache miss.
    assert np.isfinite(cache.sample(np.array([382_050.0]), np.array([6_666_000.0]))[0])


def test_add_elevation_over_a_real_tile(tmp_path, server):
    """The whole path end to end: fetch decision, raster read, reduction."""
    cache = cache_at(tmp_path, resolution_m=8.0)
    write_tile(cache, cache.tile_at(378_500.0, 6_668_000.0), lambda e, n: e * 0.05)

    links = links_frame(
        [{"coords": [(378_000.0, 6_668_000.0), (379_000.0, 6_668_000.0)]}],
        crs=PROJECTED_CRS,
    )
    links["length_m"] = 1_000.0
    elevated = elevation.add_elevation(links, cache)

    assert server.urls == []  # already cached, so nothing was requested
    assert elevated["z_v"].iloc[0] - elevated["z_u"].iloc[0] == pytest.approx(
        50.0, abs=1
    )
    assert elevated["ascent_m"].iloc[0] == pytest.approx(50.0, abs=1.0)
    assert elevated["descent_m"].iloc[0] == 0.0


def test_geometry_is_reprojected_before_it_is_located(tmp_path, server):
    """A WGS84 link layer must still land on the right EPSG:3067 tile."""
    wgs84 = links_frame([{"coords": [(24.94, 60.17), (24.95, 60.17)]}])
    projected = wgs84.to_crs(PROJECTED_CRS)

    assert tiles_for_links(wgs84) == tiles_for_links(projected)
