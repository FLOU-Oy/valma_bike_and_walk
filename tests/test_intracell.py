"""Trips between two people in the same access point's cell: measured, priced, not loaded."""

import numpy as np
import pytest

from valma_bike_and_walk.assignment import expand_demand
from valma_bike_and_walk.matrix import zone_travel_time_matrix
from valma_bike_and_walk.speeds import profile_for
from valma_bike_and_walk.zones import (
    DISC_MEAN_DISTANCE,
    INTRACELL_DETOUR,
    _merge_shared_nodes,
    _reduce_points,
)

from .conftest import make_network, make_zones

#: Five nodes in a row, 10 s and 100 m between neighbours both ways.
LINE = [(0, 1, 10.0), (1, 0, 10.0), (1, 2, 10.0), (2, 1, 10.0),
        (2, 3, 10.0), (3, 2, 10.0), (3, 4, 10.0), (4, 3, 10.0)]  # fmt: skip


@pytest.fixture
def net():
    return make_network(LINE, 5, lengths=[100.0] * len(LINE))


def intracell(radius):
    return INTRACELL_DETOUR * DISC_MEAN_DISTANCE * radius


def base_speed_ms(net):
    return profile_for(net.mode).base_kph / 3.6


def disc(rng, centre_x, radius, n):
    r = radius * np.sqrt(rng.uniform(size=n))
    theta = rng.uniform(0, 2 * np.pi, size=n)
    return centre_x + r * np.cos(theta), r * np.sin(theta)


def test_reduced_points_remember_how_spread_their_cell_was():
    # Two far-apart uniform discs of radius 100 m, reduced to two points: each
    # point's cell is one disc, and its recorded spread gives that radius back.
    rng = np.random.default_rng(1)
    ax, ay = disc(rng, 0.0, 100.0, 4000)
    bx, by = disc(rng, 50_000.0, 100.0, 4000)
    x, y = np.concatenate([ax, bx]), np.concatenate([ay, by])

    rx, _, _, moment = _reduce_points(x, y, np.ones_like(x), 2)

    assert rx.shape == (2,)
    np.testing.assert_allclose(np.sqrt(2.0 * moment), 100.0, rtol=0.03)


def test_a_point_that_is_one_candidate_has_no_spread():
    x, y = np.array([0.0, 500.0]), np.array([0.0, 0.0])
    *_, moment = _reduce_points(x, y, np.ones(2), 8)
    np.testing.assert_array_equal(moment, 0.0)


def test_merging_points_on_one_node_keeps_the_spread_between_them():
    # Two single-candidate points 10 m apart, snapped to the same node: the
    # merged point stands for both, 5 m either side of its centroid.
    zone = np.array([0, 0])
    node = np.array([7, 7])
    out = _merge_shared_nodes(
        zone, node, np.array([0.5, 0.5]), np.zeros(2), np.zeros(2),
        np.array([0.0, 10.0]), np.zeros(2), np.zeros(2),
    )  # fmt: skip
    merged_x, merged_moment = out[5], out[7]
    np.testing.assert_allclose(merged_x, [5.0])
    np.testing.assert_allclose(merged_moment, [25.0])


def test_a_cell_with_spread_prices_its_self_pair(net):
    # Zone A: nodes 0 and 1, each standing for a 200 m cell. Half its point
    # pairs are distinct (0<->1, 100 m, 10 s); half are within a cell.
    zones = make_zones(
        [[0, 1], [3, 4]], network=net, cell_radius_per_zone=[[200.0, 200.0], [0.0, 0.0]]
    )
    seconds, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=False, return_distance=True
    )

    within = intracell(200.0)
    assert metres[0, 0] == pytest.approx(0.5 * 100.0 + 0.5 * within)
    assert seconds[0, 0] == pytest.approx(
        0.5 * 10.0 + 0.5 * within / base_speed_ms(net)
    )
    # Zone B's points are single candidates: its self pairs are dropped, as before.
    assert metres[1, 1] == pytest.approx(100.0)
    assert seconds[1, 1] == pytest.approx(10.0)
    # Nothing off the diagonal moves.
    assert metres[0, 1] == pytest.approx(300.0)


def test_access_legs_still_come_on_top_of_a_within_cell_trip(net):
    zones = make_zones(
        [[0], [4]],
        network=net,
        access_per_zone=[[3.0], [0.0]],
        access_metres_per_zone=[[12.0], [0.0]],
        cell_radius_per_zone=[[200.0], [0.0]],
    )
    seconds, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=False, return_distance=True
    )
    assert metres[0, 0] == pytest.approx(2 * 12.0 + intracell(200.0))
    assert seconds[0, 0] == pytest.approx(
        2 * 3.0 + intracell(200.0) / base_speed_ms(net)
    )


def test_one_point_with_a_cell_is_measured_not_guessed_from_area(net):
    # Both zones have one access point and the same area; only the first has
    # a cell spread to go on. The second still falls back to its equal-area
    # circle, the first does not.
    area = np.pi * 500.0**2
    zones = make_zones(
        [[0], [4]],
        network=net,
        area_m2=[area, area],
        cell_radius_per_zone=[[150.0], [0.0]],
    )
    _, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=True, return_distance=True
    )
    assert metres[0, 0] == pytest.approx(intracell(150.0))
    assert metres[1, 1] == pytest.approx(DISC_MEAN_DISTANCE * 500.0, rel=1e-5)


def test_within_cell_demand_is_not_moved_onto_longer_paths(net):
    # 8 trips inside each zone, two equal points apiece. In zone A the points
    # have cells: a quarter of the trips go each way between them and half stay
    # inside a cell, unloaded. Zone B's points do not: its self pairs are
    # dropped and all 8 trips go between the two points, as before.
    zones = make_zones(
        [[0, 1], [3, 4]], network=net, cell_radius_per_zone=[[50.0, 50.0], [0.0, 0.0]]
    )
    expanded = expand_demand(np.diag([8.0, 8.0]), zones).toarray()

    assert expanded[0, 1] == pytest.approx(2.0) and expanded[1, 0] == pytest.approx(2.0)
    assert expanded[2, 3] == pytest.approx(4.0) and expanded[3, 2] == pytest.approx(4.0)
    np.testing.assert_array_equal(np.diag(expanded), 0.0)
