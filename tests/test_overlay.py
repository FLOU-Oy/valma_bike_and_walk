"""Model travel times written over a baseline matrix's full zone system."""

import numpy as np
import pytest

from valma_bike_and_walk.cli import main
from valma_bike_and_walk.demand import write_omx_matrix
from valma_bike_and_walk.overlay import overlay, read_omx_dense

omx = pytest.importorskip("openmatrix")

BASE_IDS = np.array([100, 101, 102, 103])


def test_model_times_replace_the_baseline_where_they_exist():
    base = np.full((4, 4), 99.0, dtype=np.float32)
    # Zones 101 and 103 were routed; 500 is not in the baseline at all.
    model_ids = np.array([103, 101, 500])
    model = np.array(
        [[0.0, 7.0, 1.0], [8.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32
    )
    model[0, 1] = np.inf  # 103 -> 101 beyond the cutoff

    result, replaced = overlay(BASE_IDS, base, model_ids, model)

    assert result.shape == (4, 4)
    assert result[3, 3] == 0.0 and result[1, 1] == 0.0
    assert result[1, 3] == 8.0  # 101 -> 103
    assert result[3, 1] == 99.0  # unreachable: baseline kept
    assert result[0, 2] == 99.0  # zones the model never routed
    assert replaced == 3
    assert (base == 99.0).all()


def test_times_over_the_limit_keep_the_baseline():
    base = np.full((4, 4), 50.0, dtype=np.float32)
    model = np.array([[0.0, 30.0], [400.0, 0.0]], dtype=np.float32)

    result, _ = overlay(BASE_IDS, base, np.array([100, 101]), model, max_minutes=120)

    assert result[0, 1] == 30.0
    assert result[1, 0] == 50.0


def test_cli_writes_bike_and_walk_into_one_file(tmp_path):
    times = tmp_path / "time_aht.omx"
    dists = tmp_path / "dist_aht.omx"
    write_omx_matrix(times, BASE_IDS, np.full((4, 4), 40.0), matrix_name="bike")
    write_omx_matrix(dists, BASE_IDS, np.full((4, 4), 2.0), matrix_name="bike")

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    seconds = np.array([[0.0, 600.0], [660.0, 0.0]])
    for mode in ("bike", "walk"):
        np.savez(
            out_dir / f"travel_times_{mode}.npz", ids=np.array([101, 102]), seconds=seconds
        )

    out = out_dir / "kapy_time_vrk.omx"
    common = ["--output-dir", str(out_dir), "--out", str(out)]
    assert main(["overlay", "--mode", "bike", "--baseline", str(times),
                 "--baseline-matrix", "bike", *common]) == 0  # fmt: skip
    assert main(["overlay", "--mode", "walk", "--baseline", str(dists),
                 "--baseline-matrix", "bike", "--baseline-speed-kmh", "5",
                 *common]) == 0  # fmt: skip

    ids, bike = read_omx_dense(out, "bike")
    _, walk = read_omx_dense(out, "walk")
    np.testing.assert_array_equal(ids, BASE_IDS)
    assert bike[1, 2] == pytest.approx(10.0)  # routed: 600 s
    assert bike[0, 3] == pytest.approx(40.0)  # baseline minutes
    assert walk[2, 1] == pytest.approx(11.0)  # routed: 660 s
    assert walk[0, 3] == pytest.approx(24.0)  # 2 km at 5 km/h
    with omx.open_file(str(out)) as f:
        assert sorted(f.list_matrices()) == ["bike", "walk"]
        assert f["walk"].attrs["units"] == "minutes"

    # Running a mode again replaces its matrix rather than failing.
    assert main(["overlay", "--mode", "bike", "--baseline", str(times),
                 "--baseline-matrix", "bike", *common]) == 0  # fmt: skip
