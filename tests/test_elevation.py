"""Tiling, caching and profile reduction.

Nothing here touches the network. The download is exercised against a stub
``urlopen``, which is what lets the tests assert the things that actually matter
about a cache -- that a hit costs no request, that an error response never lands
on disk as if it were data -- without a key or a connection.
"""

from __future__ import annotations

import logging
import os
import sqlite3
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
    monkeypatch.setattr(elevation, "_saved_user_variable", lambda name: None)
    with pytest.raises(ValueError, match="needs an API key"):
        elevation.resolve_api_key(None)


def test_a_key_saved_after_the_terminal_opened_is_still_found(monkeypatch):
    monkeypatch.delenv(elevation.API_KEY_ENV, raising=False)
    monkeypatch.setattr(elevation, "_saved_user_variable", lambda name: "saved")
    assert elevation.resolve_api_key(None) == "saved"
    monkeypatch.setenv(elevation.API_KEY_ENV, "live")
    assert elevation.resolve_api_key(None) == "live"


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
    monkeypatch.setattr(elevation, "_saved_user_variable", lambda name: None)
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


def test_a_lone_sample_on_a_cliff_is_interpolated_out():
    """One sample landing on the rock cut beside the road is not a hill."""

    def cliff_beside_the_road(east, north):
        z = 100.0 + 0.05 * np.asarray(east, dtype=float)
        z[np.isclose(east, 500.0)] += 15.0
        return z

    links = links_frame([{"coords": [(0.0, 0.0), (1_000.0, 0.0)]}], crs=PROJECTED_CRS)
    profile = elevation.link_profiles(links, cliff_beside_the_road)

    assert profile["ascent_m"].iloc[0] == pytest.approx(50.0)
    assert profile["descent_m"].iloc[0] == pytest.approx(0.0)


def test_a_one_sided_jump_is_clipped_to_the_cap_not_dropped():
    """A step down off a pixel edge keeps its direction but not its height."""

    def step_down(east, north):
        east = np.asarray(east, dtype=float)
        return np.where(east < 512.0, 120.0, 100.0)

    links = links_frame([{"coords": [(0.0, 0.0), (1_000.0, 0.0)]}], crs=PROJECTED_CRS)
    profile = elevation.link_profiles(links, step_down, spacing_m=25.0)

    assert profile["ascent_m"].iloc[0] == pytest.approx(0.0)
    assert profile["descent_m"].iloc[0] == pytest.approx(
        elevation.MAX_PROFILE_GRADE * 25.0
    )


def test_the_cap_follows_the_actual_step_on_a_short_link():
    """A 2 m jump over a 5 m link is 40 %; the same jump over 25 m is not."""

    def jump(east, north):
        return np.where(np.asarray(east, dtype=float) > 1.0, 102.0, 100.0)

    links = links_frame(
        [
            {"coords": [(0.0, 0.0), (5.0, 0.0)]},
            {"coords": [(0.0, 0.0), (25.0, 0.0)]},
        ],
        crs=PROJECTED_CRS,
    )
    profile = elevation.link_profiles(links, jump)

    assert profile["ascent_m"].to_numpy() == pytest.approx(
        [elevation.MAX_PROFILE_GRADE * 5.0, 2.0]
    )


def test_a_steep_but_believable_hill_is_untouched():
    links = links_frame([{"coords": [(0.0, 0.0), (200.0, 0.0)]}], crs=PROJECTED_CRS)
    profile = elevation.link_profiles(links, slope_sampler(0.25))
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


def tile_payload(tile: Tile, surface) -> bytes:
    """The GeoTIFF the service would send for ``tile``, of a known surface."""
    pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    min_e, min_n, max_e, max_n = tile.request_bounds
    res = tile.resolution_m
    width = int(round((max_e - min_e) / res))
    height = int(round((max_n - min_n) / res))
    east = min_e + (np.arange(width) + 0.5) * res
    north = max_n - (np.arange(height) + 0.5) * res
    grid = surface(*np.meshgrid(east, north)).astype("float32")
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff", height=height, width=width, count=1, dtype="float32",
            nodata=elevation.NODATA, transform=from_origin(min_e, max_n, res, res),
        ) as dataset:
            dataset.write(grid, 1)
        return memory.read()


def ramp(e, n):
    return 100.0 + 0.05 * (e - 380_000.0)


#: Two links in one 8 km tile, 3 km apart.
NEAR = [(378_000.0, 6_668_000.0), (379_000.0, 6_668_000.0)]
FAR = [(378_000.0, 6_671_000.0), (379_000.0, 6_671_000.0)]


def projected_links(*lines):
    links = links_frame([{"coords": c} for c in lines], crs=PROJECTED_CRS)
    links["length_m"] = 1_000.0
    return links


def test_add_elevation_fetches_only_what_the_links_touch(tmp_path, server):
    cache = cache_at(tmp_path, resolution_m=8.0)
    tile = cache.tile_at(*NEAR[0])
    server.body = tile_payload(tile, ramp)
    elevated = elevation.add_elevation(projected_links(NEAR), cache)

    assert len(server.urls) == 1
    for column in ("z_u", "z_v", "ascent_m", "descent_m", "ascent_override_m", "descent_override_m"):
        assert column in elevated.columns
    assert "grade_override" not in elevated.columns
    assert elevated["ascent_m"].iloc[0] == pytest.approx(50.0, abs=0.5)
    assert elevated["ascent_override_m"].isna().all()
    assert elevated["descent_override_m"].isna().all()


def test_add_elevation_leaves_links_with_both_overrides_alone(tmp_path, server):
    """No tile is fetched for them, and their DEM columns are kept as they were."""
    cache = cache_at(tmp_path, resolution_m=8.0)
    links = projected_links(NEAR, FAR)
    links["ascent_m"] = [np.nan, 7.0]
    links["descent_m"] = [np.nan, 3.0]
    links["ascent_override_m"] = [np.nan, 12.0]
    links["descent_override_m"] = [np.nan, 0.0]
    links["grade_override"] = np.nan
    server.body = tile_payload(cache.tile_at(*NEAR[0]), ramp)

    elevated = elevation.add_elevation(links, cache)

    # Both links sit in one tile, so only the strip along NEAR was fetched.
    assert len(server.urls) == 1
    far = elevation.corridors_for_links(projected_links(FAR), 8.0)[cache.tile_at(*FAR[0])]
    assert not (cache.stored_corridor(cache.tile_at(*FAR[0])) & far).any()
    assert elevated["ascent_m"].iloc[0] == pytest.approx(50.0, abs=0.5)
    assert elevated["ascent_m"].iloc[1] == 7.0
    assert elevated["descent_m"].iloc[1] == 3.0
    assert elevated["ascent_override_m"].iloc[1] == 12.0
    assert "grade_override" not in elevated.columns


def test_add_elevation_keeps_links_already_measured(tmp_path, server):
    """A layer shared without its cache: only the link with no climb is fetched."""
    cache = cache_at(tmp_path, resolution_m=8.0)
    links = projected_links(NEAR, FAR)
    links["ascent_m"] = [np.nan, 7.0]
    links["descent_m"] = [np.nan, 3.0]
    server.body = tile_payload(cache.tile_at(*NEAR[0]), ramp)

    elevated = elevation.add_elevation(links, cache)

    assert len(server.urls) == 1
    far = elevation.corridors_for_links(projected_links(FAR), 8.0)[cache.tile_at(*FAR[0])]
    assert not (cache.stored_corridor(cache.tile_at(*FAR[0])) & far).any()
    assert elevated["ascent_m"].iloc[0] == pytest.approx(50.0, abs=0.5)
    assert elevated["ascent_m"].iloc[1] == 7.0
    assert elevated["descent_m"].iloc[1] == 3.0


def test_add_elevation_fetches_nothing_when_every_link_is_measured(tmp_path, server):
    cache = cache_at(tmp_path, resolution_m=8.0)
    links = projected_links(NEAR)
    links["ascent_m"] = [7.0]
    links["descent_m"] = [3.0]

    elevated = elevation.add_elevation(links, cache)

    assert server.urls == []
    assert elevated["ascent_m"].iloc[0] == 7.0


def test_remeasure_reads_every_link_again(tmp_path, server):
    cache = cache_at(tmp_path, resolution_m=8.0)
    links = projected_links(NEAR)
    links["ascent_m"] = [7.0]
    links["descent_m"] = [3.0]
    server.body = tile_payload(cache.tile_at(*NEAR[0]), ramp)

    elevated = elevation.add_elevation(links, cache, remeasure=True)

    assert len(server.urls) == 1
    assert elevated["ascent_m"].iloc[0] == pytest.approx(50.0, abs=0.5)


def test_add_elevation_still_measures_a_link_with_one_override(tmp_path, server):
    cache = cache_at(tmp_path, resolution_m=8.0)
    links = projected_links(NEAR)
    links["ascent_override_m"] = [5.0]
    server.body = tile_payload(cache.tile_at(*NEAR[0]), ramp)

    elevated = elevation.add_elevation(links, cache)

    assert len(server.urls) == 1
    assert elevated["ascent_m"].iloc[0] == pytest.approx(50.0, abs=0.5)
    assert elevated["descent_override_m"].isna().all()


def test_a_tile_is_stored_as_the_strip_along_the_links(tmp_path, server):
    """Compact: heights in decimetres, and only near the links."""
    rasterio = pytest.importorskip("rasterio")
    cache = cache_at(tmp_path, resolution_m=8.0)
    tile = cache.tile_at(*NEAR[0])
    server.body = tile_payload(tile, ramp)
    cache.ensure_corridor(projected_links(NEAR))

    with rasterio.open(cache.path(tile)) as dataset:
        assert dataset.dtypes[0] == "int16"
        assert elevation.CORRIDOR_TAG in dataset.tags()
    on_link = cache.sample(np.array([378_500.0]), np.array([6_668_000.0]))
    assert on_link[0] == pytest.approx(ramp(378_500.0, 0.0), abs=0.06)
    # Far from any link: not kept.
    assert np.isnan(cache.sample(np.array([378_500.0]), np.array([6_671_000.0]))[0])
    assert cache.stored_corridor(tile).mean() < 0.05


def test_the_same_links_again_fetch_nothing(tmp_path, server):
    cache = cache_at(tmp_path, resolution_m=8.0)
    server.body = tile_payload(cache.tile_at(*NEAR[0]), ramp)
    cache.ensure_corridor(projected_links(NEAR))
    # Nudged a few metres, as an edit in QGIS would: still inside the strip.
    nudged = [(e + 3.0, n + 4.0) for e, n in NEAR]
    assert cache.ensure_corridor(projected_links(NEAR, nudged)) == 0
    assert len(server.urls) == 1


def test_a_link_added_outside_the_strip_fetches_the_tile_again(tmp_path, server, caplog):
    cache = cache_at(tmp_path, resolution_m=8.0)
    tile = cache.tile_at(*NEAR[0])
    server.body = tile_payload(tile, ramp)
    cache.ensure_corridor(projected_links(NEAR))

    # Before `valma dem` runs again, the new link reads nothing, and says why.
    with caplog.at_level(logging.WARNING):
        assert np.isnan(cache.sample(np.array([378_500.0]), np.array([6_671_000.0]))[0])
    assert "outside the strip" in caplog.text

    elevated = elevation.add_elevation(projected_links(NEAR, FAR), cache)
    assert len(server.urls) == 2
    # Both strips held now: the new link and the old one.
    assert elevated["ascent_m"].tolist() == pytest.approx([50.0, 50.0], abs=0.5)
    assert cache.ensure_corridor(projected_links(NEAR, FAR)) == 0


def test_a_tile_cached_whole_is_never_fetched_again(tmp_path, server):
    cache = cache_at(tmp_path, resolution_m=8.0)
    write_tile(cache, cache.tile_at(*NEAR[0]), ramp)
    assert cache.ensure_corridor(projected_links(NEAR, FAR)) == 0
    assert server.urls == []


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


def test_sampling_picks_the_pixel_rasterio_would(tmp_path):
    """Whole-tile indexing gives what dataset.sample gives, point for point."""
    rasterio = pytest.importorskip("rasterio")
    cache = cache_at(tmp_path, resolution_m=8.0)
    tiles = [cache.tile_at(380_000.0 + dx, 6_668_000.0 + dy)
             for dx in (0, 8_000) for dy in (0, 8_000)]
    for tile in tiles:
        write_tile(cache, tile, lambda e, n: np.sin(e / 97.0) * 40 + n * 1e-3)

    rng = np.random.default_rng(1)
    east = rng.uniform(376_000.0, 391_990.0, 4_000)
    north = rng.uniform(6_664_000.0, 6_679_990.0, 4_000)
    # Exactly on pixel edges too, where floor vs round would differ.
    east[:500] = np.round(east[:500] / 8.0) * 8.0
    north[500:1000] = np.round(north[500:1000] / 8.0) * 8.0

    expected = np.full(east.size, np.nan)
    for i, (e, n) in enumerate(zip(east, north)):
        tile = cache.tile_at(e, n)
        with rasterio.open(cache.path(tile)) as dataset:
            expected[i] = next(dataset.sample([(e, n)]))[0]
    for workers in (1, 3):
        np.testing.assert_array_equal(cache.sample(east, north, workers=workers), expected)


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


# --------------------------------------------------------------------------
# Making GDAL usable on a machine that has other GIS software on it
# --------------------------------------------------------------------------


def proj_data_dir(directory: Path, version: tuple[int, int] | None) -> Path:
    """A directory that looks like a PROJ data dir of the given layout."""
    directory.mkdir(parents=True, exist_ok=True)
    if version is None:
        return directory
    connection = sqlite3.connect(directory / "proj.db")
    with connection:
        connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("DATABASE.LAYOUT.VERSION.MAJOR", str(version[0])),
                ("DATABASE.LAYOUT.VERSION.MINOR", str(version[1])),
            ],
        )
    connection.close()
    return directory


class FakeRasterio:
    """Just enough rasterio for the repair: somewhere to find proj_data."""

    def __init__(self, package: Path) -> None:
        self.__file__ = str(package / "__init__.py")


@pytest.fixture
def rasterio_at(tmp_path):
    def build(bundled_version: tuple[int, int] | None) -> FakeRasterio:
        package = tmp_path / "rasterio"
        proj_data_dir(package / "proj_data", bundled_version)
        return FakeRasterio(package)

    return build


def test_an_older_proj_database_is_replaced_by_rasterio_s(
    tmp_path, rasterio_at, monkeypatch
):
    """The PostGIS-on-Windows case: the machine-wide PROJ is too old for GDAL."""
    stale = proj_data_dir(tmp_path / "postgis", (1, 2))
    monkeypatch.setenv("PROJ_LIB", str(stale))
    monkeypatch.delenv("PROJ_DATA", raising=False)

    rasterio = rasterio_at((1, 6))
    elevation._repair_proj_data(rasterio)

    assert os.environ["PROJ_LIB"] == str(Path(rasterio.__file__).parent / "proj_data")
    assert "PROJ_DATA" not in os.environ  # nothing said, nothing invented


@pytest.mark.parametrize("version", [(1, 6), (2, 0)])
def test_a_current_proj_database_is_left_alone(
    tmp_path, rasterio_at, monkeypatch, version
):
    """Someone else's PROJ may carry grid shifts ours does not; keep it."""
    theirs = proj_data_dir(tmp_path / "theirs", version)
    monkeypatch.setenv("PROJ_DATA", str(theirs))

    elevation._repair_proj_data(rasterio_at((1, 6)))

    assert os.environ["PROJ_DATA"] == str(theirs)


def test_one_usable_entry_in_the_search_path_is_enough(
    tmp_path, rasterio_at, monkeypatch
):
    stale = proj_data_dir(tmp_path / "postgis", (1, 2))
    good = proj_data_dir(tmp_path / "good", (1, 6))
    path = os.pathsep.join([str(stale), str(good)])
    monkeypatch.setenv("PROJ_DATA", path)

    elevation._repair_proj_data(rasterio_at((1, 6)))

    assert os.environ["PROJ_DATA"] == path


def test_a_directory_without_a_database_is_replaced(tmp_path, rasterio_at, monkeypatch):
    empty = proj_data_dir(tmp_path / "empty", None)
    monkeypatch.setenv("PROJ_DATA", str(empty))

    rasterio = rasterio_at((1, 6))
    elevation._repair_proj_data(rasterio)

    assert os.environ["PROJ_DATA"] == str(Path(rasterio.__file__).parent / "proj_data")


def test_nothing_is_set_when_the_environment_says_nothing(rasterio_at, monkeypatch):
    """No variable means GDAL already finds rasterio's own copy."""
    for variable in elevation.PROJ_DATA_ENV:
        monkeypatch.delenv(variable, raising=False)

    elevation._repair_proj_data(rasterio_at((1, 6)))

    assert not any(variable in os.environ for variable in elevation.PROJ_DATA_ENV)


def test_gdal_says_each_thing_once_rather_than_once_per_tile(caplog):
    """Six thousand tiles must not mean six thousand copies of one warning."""
    logger = logging.getLogger("rasterio._env")
    filters = list(logger.filters)
    try:
        elevation._quieten_repeated_gdal_messages()
        with caplog.at_level(logging.WARNING, logger="rasterio._env"):
            for _ in range(5):
                logger.warning("CPLE_AppDefined in %s", "the CRS is not the EPSG one")
            logger.warning("CPLE_AppDefined in something else entirely")
    finally:
        logger.filters = filters

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "CPLE_AppDefined in the CRS is not the EPSG one",
        "CPLE_AppDefined in something else entirely",
    ]
