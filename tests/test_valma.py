"""The in-process interface VALMA calls: skims over its zones, then assignment."""

import geopandas as gpd
import numpy as np
import pytest

from valma_bike_and_walk import valma
from valma_bike_and_walk.demand import write_omx_matrix
from valma_bike_and_walk.links import LINKS_LAYER

from .test_pipeline import chain_pbf, extract_chain, grid_zone_layer  # noqa: F401

# VALMA's zones: the two the grid rolls up into, and an external one we never
# route. EMME's baseline is distinctive everywhere and 0 on the diagonal, as
# EMME leaves it.
ZONES = np.array([10, 20, 50107])


def baseline(value):
    matrix = np.full((3, 3), value, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    return matrix


@pytest.fixture
def setup(chain_pbf, tmp_path):  # noqa: F811
    _, cache, links = extract_chain(chain_pbf, tmp_path)
    config_dir = tmp_path / "bike_and_walk"
    config_dir.mkdir()
    # Settings files for a real run name their own zones; VALMA's must win.
    lines = [
        f'links = "{links.as_posix()}"',
        f'cache_dir = "{cache.as_posix()}"',
        "points_per_zone = 4",
        'zones = "elsewhere.gpkg"',
        'id_column = "cell_id"',
    ]
    (config_dir / "walk.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return valma.Setup(
        config_dir=config_dir,
        zone_data_file=grid_zone_layer(tmp_path / "zonedata.gpkg"),
        submodel="submodel",
        output_dir=tmp_path / "results" / "bike_and_walk",
        workers=1,
    )


def test_skims_lay_routed_cells_over_valmas_matrices(setup):
    time, dist = baseline(999.0), baseline(888.0)
    result = valma.skims(setup, ZONES, {"time": {"walk": time}, "dist": {"walk": dist}})

    for kind, base in (("time", time), ("dist", dist)):
        matrix = result[kind]["walk"]
        assert matrix.shape == (3, 3)
        # The routed zones are ours, intrazonal trips included: positive where
        # EMME left 0, which is what VALMA keeps over its zone-data fallback.
        assert (matrix[:2, :2] > 0).all()
        assert (matrix[:2, :2] < 100).all()
        # The external zone keeps EMME's values, its diagonal included.
        np.testing.assert_array_equal(matrix[2], base[2])
        np.testing.assert_array_equal(matrix[:, 2], base[:, 2])

    # Units: a walk of a few hundred metres is minutes, not seconds, and km.
    assert result["dist"]["walk"][0, 1] < 1.0
    assert result["time"]["walk"][0, 1] > result["dist"]["walk"][0, 1]
    # The baseline is VALMA's; it is not written into.
    assert (time[:2, :2][~np.eye(2, dtype=bool)] == 999.0).all()


def test_skims_use_the_overlay_cutoffs_from_the_settings(setup):
    with open(setup.config_dir / "walk.toml", "a", encoding="utf-8") as f:
        f.write("\n[overlay]\nmax_km = 0.000001\n")

    result = valma.skims(
        setup,
        ZONES,
        {"time": {"walk": baseline(999.0)}, "dist": {"walk": baseline(888.0)}},
    )

    np.testing.assert_array_equal(result["dist"]["walk"], baseline(888.0))
    assert (result["time"]["walk"][:2, :2] < 100).all()


def test_skims_only_route_the_modes_valma_asks_for(setup):
    # No bike.toml: asking only for walk must not need one.
    assert set(valma.skims(setup, ZONES, {"dist": {"walk": baseline(1.0)}})) == {"dist"}


def test_a_missing_settings_file_is_named(setup):
    with pytest.raises(FileNotFoundError, match="bike.toml"):
        valma.skims(setup, ZONES, {"dist": {"bike": baseline(1.0)}})


def test_skims_refuse_a_baseline_of_the_wrong_size(setup):
    with pytest.raises(ValueError, match="not square"):
        valma.skims(setup, ZONES, {"dist": {"walk": np.zeros((2, 2))}})


def write_demand(path, trips):
    write_omx_matrix(
        path,
        ZONES,
        trips,
        matrix_name="walk",
        mapping_name="zone_number",
        units="trips",
    )
    return path


def test_assign_writes_one_layer_per_mode_with_a_column_per_period(setup, tmp_path):
    trips = np.array([[4.0, 6.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    aht = write_demand(tmp_path / "demand_aht.omx", trips)
    vrk = write_demand(tmp_path / "demand_vrk.omx", 2.0 * trips)

    # Both modes asked for, as VALMA does; the files hold walk only, so bike is
    # skipped -- before it could ask for the bike.toml this setup lacks.
    result = valma.assign(setup, {"aht": aht, "vrk": vrk})

    assert set(result) == {"walk"}
    walk = result["walk"]
    assert walk.gpkg == setup.output_dir / "walk_volumes.gpkg"
    volumes = gpd.read_file(walk.gpkg, layer=LINKS_LAYER)
    # Intrazonal trips load too, so the network carries more than the 6
    # trips between zones would on their own.
    assert volumes["volume_aht"].max() >= 6.0
    assert volumes["volume_aht"].sum() > 6.0
    np.testing.assert_allclose(volumes["volume_vrk"], 2.0 * volumes["volume_aht"])

    assert walk.trips == {"aht": 10.0, "vrk": 20.0}
    assert walk.kilometres["aht"] > 0
    assert walk.kilometres["vrk"] == pytest.approx(2.0 * walk.kilometres["aht"])


def test_assign_writes_only_loaded_links_unless_asked(setup, tmp_path):
    # Demand inside zone 10 only: the links over in zone 20 carry nothing.
    trips = np.zeros((3, 3))
    trips[0, 0] = 5.0
    demand = write_demand(tmp_path / "demand_pt.omx", trips)

    loaded = valma.assign(setup, {"pt": demand})["walk"]
    rows = gpd.read_file(loaded.gpkg, layer=LINKS_LAYER)
    assert (rows["volume_pt"] > 0).all()

    with np.load(loaded.npz) as everything:
        assert everything["volume_pt"].size > len(rows)
        assert (everything["volume_pt"] == 0).any()

    every = valma.assign(setup, {"pt": demand}, all_links=True)["walk"]
    assert len(gpd.read_file(every.gpkg, layer=LINKS_LAYER)) > len(rows)


def test_skims_route_once_however_many_periods_ask(setup, monkeypatch):
    runs = []
    real_run = valma._run
    monkeypatch.setattr(valma, "_run", lambda *a: runs.append(a[0]) or real_run(*a))

    pt = valma.skims(setup, ZONES, {"dist": {"walk": baseline(888.0)}})
    aht = valma.skims(setup, ZONES, {"dist": {"walk": baseline(777.0)}})

    assert runs == ["matrix"]
    # Same routing, each laid over its own period's baseline.
    np.testing.assert_array_equal(
        pt["dist"]["walk"][:2, :2], aht["dist"]["walk"][:2, :2]
    )
    assert pt["dist"]["walk"][2, 0] == 888.0 and aht["dist"]["walk"][2, 0] == 777.0
