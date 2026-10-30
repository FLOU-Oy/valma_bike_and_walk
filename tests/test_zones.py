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
    snap = float(np.hypot(50.0, 50.0))
    assert charged.access_seconds[0] == pytest.approx(snap, rel=1e-6)
    assert free.access_seconds[0] == 0.0

    # The same leg in metres, for a distance matrix: the snap distance itself,
    # and zero exactly where the time is, so the two charge for access alike.
    assert charged.access_metres[0] == pytest.approx(snap, rel=1e-6)
    assert charged.representative_access_metres[0] == pytest.approx(snap, rel=1e-6)
    assert free.access_metres[0] == 0.0
    assert free.representative_access_metres[0] == 0.0


def test_access_metres_survive_merging_points_on_one_node(net, tmp_path):
    """Two candidates on the same node merge; both legs average together."""
    # A box covering a single network node, asked for more points than the
    # network can give it distinct nodes for.
    path = write_zones(tmp_path / "zones.gpkg", [shapely.box(-40, -40, 40, 40)])

    zones = load_zones(
        net, path, id_column="zone_id", points_per_zone=4, access_speed_kmh=3.6
    )

    # 1 m/s again, so metres and seconds have to come out of the merge equal.
    np.testing.assert_allclose(
        zones.access_metres, zones.access_seconds, rtol=1e-6, atol=1e-9
    )


def street_and_park(tmp_path):
    """A street 0-300 m along y = 0, and a park east of it full of footways."""
    from valma_bike_and_walk.network import network_from_links

    from .conftest import links_frame

    rows = [
        {"u": 1, "v": 2, "coords": [(0, 0), (100, 0)]},
        {"u": 2, "v": 3, "coords": [(100, 0), (200, 0)]},
        {"u": 3, "v": 4, "coords": [(200, 0), (300, 0)], "highway": "service"},
    ]
    # Footways: a ladder from x = 300 to 600, two rails and a rung every 100 m.
    park = [
        (4, (300, 0)),
        (5, (400, 0)),
        (6, (500, 0)),
        (7, (600, 0)),
        (8, (300, 100)),
        (9, (400, 100)),
        (10, (500, 100)),
        (11, (600, 100)),
    ]
    xy = dict(park)
    for a, b in [
        (4, 5),
        (5, 6),
        (6, 7),
        (8, 9),
        (9, 10),
        (10, 11),
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
    ]:
        rows.append({"u": a, "v": b, "coords": [xy[a], xy[b]], "highway": "footway"})
    for i, row in enumerate(rows):
        row["link_id"] = i
    return network_from_links(links_frame(rows, crs=PROJECTED_CRS), "walk")


def test_paths_do_not_draw_access_points(tmp_path):
    network = street_and_park(tmp_path)
    everything = shapely.box(-50, -50, 650, 150)
    path = write_zones(tmp_path / "zones.gpkg", [everything])
    zones = load_zones(network, path, id_column="zone_id", points_per_zone=1)
    # Streets and the service road only: their length-weighted mean is on the street.
    assert zones.x[0] <= 300
    assert zones.y[0] == pytest.approx(0.0, abs=1e-6)


def test_a_zone_with_no_street_takes_its_points_from_the_paths(tmp_path):
    network = street_and_park(tmp_path)
    park_only = shapely.box(350, -50, 650, 150)
    path = write_zones(tmp_path / "zones.gpkg", [park_only])
    zones = load_zones(network, path, id_column="zone_id", points_per_zone=4)
    assert zones.n_points > 1
    assert (zones.x >= 350).all()


def test_the_demand_flag_survives_a_save(tmp_path):
    from valma_bike_and_walk.network import RoutableNetwork

    network = street_and_park(tmp_path)
    assert network.demand_edge is not None and not network.demand_edge.all()
    loaded = RoutableNetwork.load(network.save(tmp_path / "net.npz"))
    np.testing.assert_array_equal(loaded.demand_edge, network.demand_edge)


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


# --------------------------------------------------------------------------
# Rows sharing an id are one zone
# --------------------------------------------------------------------------


def write_grid_zones(path, column="taz", mapping=None):
    """
    The 5x5 grid network cut into 4 cells of 2x2 nodes, plus a mapping column.

    Cell k covers nodes in a 200 m box; ``mapping[k]`` is the analysis zone it
    rolls up into, so the default maps the four cells onto two zones.
    """
    cells = [
        shapely.box(-50, -50, 150, 150),  # nodes with x,y <= 100
        shapely.box(150, -50, 450, 150),
        shapely.box(-50, 150, 150, 450),
        shapely.box(150, 150, 450, 450),
    ]
    frame = gpd.GeoDataFrame(
        {
            "cell_id": [0, 1, 2, 3],
            column: mapping if mapping is not None else ["A", "A", "B", "B"],
        },
        geometry=cells,
        crs=PROJECTED_CRS,
    )
    frame.to_file(path, layer="zones", driver="GPKG")
    return path


def test_rows_sharing_an_id_become_one_zone(net, tmp_path):
    path = write_grid_zones(tmp_path / "zones.gpkg")

    cells = load_zones(net, path, id_column="cell_id", points_per_zone=1)
    grouped = load_zones(net, path, id_column="taz", points_per_zone=1)

    assert cells.n_zones == 4
    assert grouped.n_zones == 2
    np.testing.assert_array_equal(grouped.ids, ["A", "B"])


def test_a_grouped_zone_covers_every_node_of_its_parts(net, tmp_path):
    """Access points are placed over the whole zone, not one part of it."""
    path = write_grid_zones(tmp_path / "zones.gpkg")
    grouped = load_zones(net, path, id_column="taz", points_per_zone=8)

    # Zone A is the bottom half of the grid: y <= 100, x anywhere.
    a = slice(grouped.indptr[0], grouped.indptr[1])
    assert grouped.y[a].max() <= 100
    assert grouped.x[a].max() >= 300  # reaches into the second cell
    # And its weights still normalise within the zone, not within a part.
    np.testing.assert_allclose(grouped.weight[a].sum(), 1.0)


def test_a_grouped_zone_takes_the_area_of_all_its_parts(net, tmp_path):
    path = write_grid_zones(tmp_path / "zones.gpkg")
    cells = load_zones(net, path, id_column="cell_id", points_per_zone=1)
    grouped = load_zones(net, path, id_column="taz", points_per_zone=1)

    # Cells 0 and 1 make zone A; the equal-area-circle intrazonal estimate
    # has to see the whole zone, or it would understate every grouped zone.
    np.testing.assert_allclose(grouped.area_m2[0], cells.area_m2[0] + cells.area_m2[1])


def test_grouping_keeps_the_order_the_ids_first_appear_in(net, tmp_path):
    path = write_grid_zones(
        tmp_path / "zones.gpkg", mapping=["late", "early", "early", "late"]
    )
    grouped = load_zones(net, path, id_column="taz", points_per_zone=1)

    np.testing.assert_array_equal(grouped.ids, ["late", "early"])


def test_unique_ids_are_left_exactly_as_they_were(net, tmp_path):
    """The ungrouped path must not change: no dissolve, no reordering."""
    path = write_grid_zones(tmp_path / "zones.gpkg", mapping=["d", "c", "b", "a"])
    zones = load_zones(net, path, id_column="taz", points_per_zone=4)

    np.testing.assert_array_equal(zones.ids, ["d", "c", "b", "a"])


def test_rows_with_no_id_are_dropped_rather_than_made_into_a_zone(net, tmp_path):
    path = write_grid_zones(tmp_path / "zones.gpkg", mapping=["A", None, "A", "B"])
    zones = load_zones(net, path, id_column="taz", points_per_zone=4)

    assert zones.n_zones == 2
    np.testing.assert_array_equal(sorted(zones.ids), ["A", "B"])
    # The dropped cell is the one covering x >= 150, y <= 100; zone A keeps
    # only its other part.
    a = slice(zones.indptr[0], zones.indptr[1])
    assert zones.x[a].max() <= 100


def test_a_grouped_zone_in_separated_pieces_still_routes(net, tmp_path):
    """Two disjoint cells under one id: a MultiPolygon, not an error."""
    path = write_grid_zones(tmp_path / "zones.gpkg", mapping=["X", "Y", "Y", "X"])
    zones = load_zones(net, path, id_column="taz", points_per_zone=4)

    assert zones.n_zones == 2
    # The representative point has to sit on an access point, not in the gap
    # between the two diagonal pieces (see the module docstring).
    for z in range(2):
        block = slice(zones.indptr[z], zones.indptr[z + 1])
        points = np.column_stack([zones.x[block], zones.y[block]])
        rep = np.array([zones.representative.x[z], zones.representative.y[z]])
        assert (np.abs(points - rep).sum(axis=1) < 1e-6).any()
