"""Placing weighted access points inside zone polygons."""

import geopandas as gpd
import numpy as np
import pytest
import shapely

from valma_bike_and_walk.config import PROJECTED_CRS
from valma_bike_and_walk.zones import (
    _reduce_points,
    load_zones,
    network_node_weights,
    zone_points_frame,
)

from .conftest import make_grid_network

# The grid network spans (0, 0) to (400, 400) at 100 m spacing, so these two
# boxes split it down the middle: nodes with x <= 200 on the left, x >= 300 on
# the right. Nothing lands on a boundary.
LEFT = shapely.box(-50, -50, 250, 450)
RIGHT = shapely.box(250, -50, 450, 450)


@pytest.fixture
def net():
    return make_grid_network(side=5, spacing=100.0, seconds=10.0)


def write_zones(path, geometries, ids=None):
    frame = gpd.GeoDataFrame(
        {"zone_id": ids if ids is not None else list(range(len(geometries)))},
        geometry=list(geometries),
        crs=PROJECTED_CRS,
    )
    frame.to_file(path, layer="zones", driver="GPKG")
    return path


def test_access_points_stay_inside_their_own_zone(net, tmp_path):
    path = write_zones(tmp_path / "zones.gpkg", [LEFT, RIGHT], ids=["L", "R"])
    zones = load_zones(net, path, id_column="zone_id", points_per_zone=4)

    assert zones.n_zones == 2
    np.testing.assert_array_equal(zones.ids, ["L", "R"])

    left = zones.node_index[zones.indptr[0] : zones.indptr[1]]
    right = zones.node_index[zones.indptr[1] : zones.indptr[2]]
    assert net.x[left].max() <= 200
    assert net.x[right].min() >= 300


def test_weights_are_normalised_within_each_zone(net, tmp_path):
    path = write_zones(tmp_path / "zones.gpkg", [LEFT, RIGHT])
    zones = load_zones(net, path, id_column="zone_id", points_per_zone=4)

    totals = np.add.reduceat(zones.weight, zones.indptr[:-1])
    np.testing.assert_allclose(totals, 1.0)
    assert (zones.points_per_zone <= 4).all()
    assert (zones.points_per_zone >= 1).all()


def test_one_point_per_zone_lands_on_the_weighted_centroid(net, tmp_path):
    path = write_zones(tmp_path / "zones.gpkg", [LEFT, RIGHT])
    zones = load_zones(net, path, id_column="zone_id", points_per_zone=1)

    assert zones.n_points == 2

    # The one point has to be the street-length-weighted mean of every network
    # node in the polygon -- not the polygon's own centre, and not the mean of
    # the nodes counted equally.
    nx, ny, nw = network_node_weights(net)
    inside = nx <= 250
    np.testing.assert_allclose(zones.x[0], (nw[inside] * nx[inside]).sum() / nw[inside].sum())
    np.testing.assert_allclose(zones.y[0], (nw[inside] * ny[inside]).sum() / nw[inside].sum())
    np.testing.assert_allclose(zones.representative.x[0], zones.x[0])


def test_more_points_per_zone_spreads_them_out(net, tmp_path):
    path = write_zones(tmp_path / "zones.gpkg", [LEFT, RIGHT])
    one = load_zones(net, path, id_column="zone_id", points_per_zone=1)
    many = load_zones(net, path, id_column="zone_id", points_per_zone=6)

    assert many.n_points > one.n_points
    # The representative point stays put to within the access-point spacing: it
    # is the point nearest the weighted mean, and the mean itself does not move
    # with K because the grid reduction preserves it.
    assert np.abs(many.representative.x - one.representative.x).max() <= 100.0
    assert np.abs(many.representative.y - one.representative.y).max() <= 100.0


def test_the_representative_is_one_of_the_access_points(net, tmp_path):
    """
    A zone in two separated pieces must not be represented by the gap between
    them. The mean of the two clusters lands mid-network with nothing to route
    from; the representative has to be somewhere a trip could start.
    """
    split = shapely.MultiPolygon([shapely.box(-50, -50, 50, 450), shapely.box(350, -50, 450, 450)])
    path = write_zones(tmp_path / "zones.gpkg", [split], ids=["SPLIT"])
    zones = load_zones(net, path, id_column="zone_id", points_per_zone=6)

    points = np.column_stack([zones.x, zones.y])
    rep = np.array([zones.representative.x[0], zones.representative.y[0]])
    assert (np.abs(points - rep).sum(axis=1) == 0).any()

    # The access points sit at x <= 50 and x >= 350, so their mean is near 200 --
    # the middle of the network, in neither piece of the zone.
    assert 150 <= float((zones.weight * zones.x).sum()) <= 250
    assert rep[0] <= 50 or rep[0] >= 350
    # And it snapped to the node it actually sits on.
    assert net.x[zones.representative.node_index[0]] == pytest.approx(rep[0], abs=1e-6)


def test_reduce_points_reaches_k_when_candidates_are_clustered(net, tmp_path):
    """
    Two tight clusters far apart must still yield K points, not one per cluster.

    Sizing the grid from the bounding box alone made the very first grid too
    coarse to separate anything inside a cluster, and coarsening could only make
    it worse -- so a zone like this collapsed to two points however large K was.
    """
    x = np.concatenate([np.linspace(0, 40, 20), np.linspace(9960, 10000, 20)])
    y = np.zeros(40)
    w = np.ones(40)

    for k in (2, 4, 8, 16):
        rx, _, _ = _reduce_points(x, y, w, k)
        assert rx.shape[0] == k, f"k={k} gave {rx.shape[0]} point(s)"
        # Both clusters keep representation rather than one swallowing the grid.
        assert (rx < 1000).any() and (rx > 9000).any()


def test_external_weight_points_beat_network_density(net, tmp_path):
    """A population layer piled into one corner should drag the points there."""
    path = write_zones(tmp_path / "zones.gpkg", [LEFT])
    weights = tmp_path / "pop.csv"
    weights.write_text(
        "x,y,pop\n0,0,1000\n100,0,5\n200,0,5\n0,400,5\n",
        encoding="utf-8",
    )

    zones = load_zones(
        net,
        path,
        id_column="zone_id",
        weights_path=weights,
        weight_column="pop",
        weight_x_column="x",
        weight_y_column="y",
        weight_crs=PROJECTED_CRS,
        points_per_zone=1,
    )
    np.testing.assert_allclose(zones.x[0], 0.0, atol=30.0)
    np.testing.assert_allclose(zones.y[0], 0.0, atol=30.0)


def test_a_zone_with_no_candidate_inside_uses_its_own_centre(net, tmp_path):
    """An empty polygon still gets a zone, as long as it can reach the network."""
    gap = shapely.box(430, 430, 470, 470)  # past the last node at (400, 400)
    path = write_zones(tmp_path / "zones.gpkg", [LEFT, gap], ids=["L", "gap"])

    zones = load_zones(net, path, id_column="zone_id", points_per_zone=4)
    assert set(zones.ids) == {"L", "gap"}
    assert zones.points_per_zone[list(zones.ids).index("gap")] == 1


def test_a_zone_far_from_the_network_is_dropped_not_moved(net, tmp_path):
    far = shapely.box(90_000, 90_000, 91_000, 91_000)
    path = write_zones(tmp_path / "zones.gpkg", [LEFT, far], ids=["L", "far"])

    zones = load_zones(net, path, id_column="zone_id", points_per_zone=4, max_snap_distance=500.0)
    np.testing.assert_array_equal(zones.ids, ["L"])


def test_access_time_charges_for_the_snap_distance(net, tmp_path):
    """A zone centred between nodes pays for getting to one."""
    offset = shapely.box(30, 30, 70, 70)  # 40 m box, nearest node is (0, 0)
    path = write_zones(tmp_path / "zones.gpkg", [offset])

    charged = load_zones(net, path, id_column="zone_id", points_per_zone=1, access_speed_kmh=3.6)
    free = load_zones(
        net,
        path,
        id_column="zone_id",
        points_per_zone=1,
        access_speed_kmh=3.6,
        charge_access_time=False,
    )

    # 3.6 km/h is 1 m/s, so access seconds equal the snap distance in metres.
    assert charged.access_seconds[0] == pytest.approx(float(np.hypot(50.0, 50.0)), rel=1e-6)
    assert free.access_seconds[0] == 0.0


def test_node_weights_follow_street_length_not_node_count(net):
    _, _, weight = network_node_weights(net)
    # Corner nodes have two edges, edge nodes three, interior four.
    assert weight[0] < weight[2]  # (0, 0) against (200, 0)
    assert weight[2] < weight[12]  # (200, 0) against the middle of the grid


def test_reduce_points_preserves_the_weighted_centroid():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1000, 500)
    y = rng.uniform(0, 1000, 500)
    w = rng.uniform(0.1, 10, 500)

    for k in (1, 4, 16):
        rx, ry, rw = _reduce_points(x, y, w, k)
        assert rx.shape[0] <= k
        assert rw.sum() == pytest.approx(w.sum())
        assert (rw * rx).sum() == pytest.approx((w * x).sum())
        assert (rw * ry).sum() == pytest.approx((w * y).sum())


def test_zone_points_frame_round_trips_to_a_geopackage(net, tmp_path):
    path = write_zones(tmp_path / "zones.gpkg", [LEFT, RIGHT], ids=["L", "R"])
    zones = load_zones(net, path, id_column="zone_id", points_per_zone=3)

    frame = zone_points_frame(zones)
    assert len(frame) == zones.n_points
    assert set(frame["zone_id"]) == {"L", "R"}

    out = tmp_path / "points.gpkg"
    frame.to_file(out, layer="zone_points", driver="GPKG")
    assert len(gpd.read_file(out)) == zones.n_points
