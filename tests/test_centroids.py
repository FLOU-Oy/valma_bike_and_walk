import numpy as np
import pytest

from valma_bike_and_walk.centroids import read_points, snap_to_network
from tests.test_routing import make_network


@pytest.fixture
def net():
    # Three nodes at x = 0, 100, 200 metres, y = 0.
    return make_network([(0, 1, 10.0), (1, 2, 10.0)], 3)


def test_snapping_picks_the_closest_node(net):
    cent = snap_to_network(
        net, np.array(["a", "b"]), np.array([5.0, 190.0]), np.array([0.0, 0.0])
    )
    np.testing.assert_array_equal(cent.node_index, [0, 2])
    np.testing.assert_allclose(cent.snap_distance, [5.0, 10.0])


def test_far_points_are_marked_unsnapped_not_silently_moved(net):
    cent = snap_to_network(
        net,
        np.array(["far"]),
        np.array([50_000.0]),
        np.array([0.0]),
        max_snap_distance=1000.0,
    )
    assert cent.node_index[0] == -1
    assert not cent.snapped.any()

    assert len(cent.drop_unsnapped()) == 0


def test_drop_unsnapped_keeps_alignment(net):
    cent = snap_to_network(
        net,
        np.array(["near", "far", "near2"]),
        np.array([0.0, 90_000.0, 200.0]),
        np.array([0.0, 0.0, 0.0]),
        max_snap_distance=500.0,
    )
    kept = cent.drop_unsnapped()
    assert len(kept) == 2
    np.testing.assert_array_equal(kept.ids, ["near", "near2"])
    np.testing.assert_array_equal(kept.node_index, [0, 2])


def test_read_points_from_csv_reprojects_to_metres(tmp_path):
    csv = tmp_path / "points.csv"
    csv.write_text("id,lon,lat\nA,24.75,60.22\nB,24.76,60.23\n", encoding="utf-8")

    ids, x, y = read_points(csv, id_column="id")

    np.testing.assert_array_equal(ids, ["A", "B"])
    # EPSG:3067 easting is ~370 km, northing ~6.68 Mm around Helsinki.
    assert 300_000 < x[0] < 450_000
    assert 6_600_000 < y[0] < 6_750_000


def test_read_points_reports_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("id,easting,northing\nA,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no column"):
        read_points(csv)
