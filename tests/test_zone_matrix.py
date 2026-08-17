"""Aggregating point-to-point times into a zone-level matrix."""

import numpy as np
import pytest

from valma_bike_and_walk.matrix import (
    multipoint_travel_time_matrix,
    travel_time_matrix,
    zone_travel_time_matrix,
)

from .conftest import make_network, make_zones

#: Five nodes in a row, 100 m apart, 10 s between neighbours both ways.
LINE = [
    (0, 1, 10.0),
    (1, 0, 10.0),
    (1, 2, 10.0),
    (2, 1, 10.0),
    (2, 3, 10.0),
    (3, 2, 10.0),
    (3, 4, 10.0),
    (4, 3, 10.0),
]


@pytest.fixture
def net():
    return make_network(LINE, 5)


def test_interzonal_time_is_the_weighted_mean_over_point_pairs(net):
    # Zone A holds nodes 0 and 1, zone B holds nodes 3 and 4. The four crossing
    # times are 30, 40, 20 and 30 seconds, so an even average is 30.
    zones = make_zones([[0, 1], [3, 4]], network=net)
    matrix = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)

    assert matrix[0, 1] == pytest.approx(30.0)
    assert matrix[1, 0] == pytest.approx(30.0)


def test_point_weights_tilt_the_average(net):
    zones = make_zones([[0, 1], [3, 4]], [[0.25, 0.75], [0.5, 0.5]], network=net)
    matrix = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)

    # 0.25*0.5*30 + 0.25*0.5*40 + 0.75*0.5*20 + 0.75*0.5*30
    assert matrix[0, 1] == pytest.approx(27.5)


def test_intrazonal_time_is_not_zero(net):
    """The whole point: a zone's own diagonal comes from real paths inside it."""
    zones = make_zones([[0, 1], [3, 4]], network=net)
    matrix = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)

    # The only distinct pairs inside zone A are 0->1 and 1->0, both 10 s.
    assert matrix[0, 0] == pytest.approx(10.0)
    assert matrix[1, 1] == pytest.approx(10.0)


def test_keeping_self_pairs_halves_the_intrazonal_time(net):
    """Averaging in the zero-length 'trip' from a point to itself is the bug."""
    zones = make_zones([[0, 1], [3, 4]], network=net)
    kept = zone_travel_time_matrix(
        net, zones, exclude_self_pairs=False, intrazonal_fallback=False
    )
    assert kept[0, 0] == pytest.approx(5.0)


def test_access_time_is_charged_at_both_ends(net):
    zones = make_zones(
        [[0, 1], [3, 4]], access_per_zone=[[1.0, 2.0], [3.0, 4.0]], network=net
    )
    matrix = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)

    # 30 s between the zones, plus the mean access at each end.
    assert matrix[0, 1] == pytest.approx(30.0 + 1.5 + 3.5)


def test_one_point_per_zone_reproduces_the_plain_matrix(net):
    """--zones with K=1 is the single-centroid method, not a different one."""
    zones = make_zones([[0], [2], [4]], network=net)

    zoned = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)
    plain = travel_time_matrix(net, [0, 2, 4])

    off_diagonal = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(zoned[off_diagonal], plain[off_diagonal])


def test_a_single_point_zone_has_no_intrazonal_path(net):
    """One point cannot say how far apart two people in a zone are."""
    zones = make_zones([[0], [4]], network=net)
    matrix = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)

    assert not np.isfinite(matrix[0, 0])


def test_the_equal_area_circle_fills_a_missing_diagonal(net):
    # A 1 km^2 zone: R = 564 m, mean separation 0.9054 R = 511 m, and the bike
    # profile's 14 km/h base makes that ~131 s.
    zones = make_zones([[0], [4]], area_m2=[1e6, 1e6], network=net)
    matrix = zone_travel_time_matrix(net, zones, intrazonal_fallback=True)

    expected = (128.0 / (45.0 * np.pi)) * np.sqrt(1e6 / np.pi) / (14.0 / 3.6)
    assert matrix[0, 0] == pytest.approx(expected, rel=1e-4)


def test_the_fallback_does_not_overrule_a_routed_diagonal():
    """A zone whose demand sits in one corner really does have shorter trips."""
    net = make_network(LINE, 5)
    # Two access points 10 s apart, in a zone the size of a 1 km^2 disc whose
    # crow-flies estimate would be ~131 s. Routing measured it; trust that.
    zones = make_zones([[0, 1], [3, 4]], area_m2=[1e6, 1e6], network=net)
    matrix = zone_travel_time_matrix(net, zones, intrazonal_fallback=True)

    assert matrix[0, 0] == pytest.approx(10.0)


def test_a_two_tier_run_does_not_report_a_near_zero_diagonal():
    """An unresolved diagonal must not silently take the far tier's zero."""
    # One long zone: its two access points are 60 s apart, well past a 25 s
    # near cutoff, so the multi-point pass cannot resolve the diagonal. The far
    # tier would route the representative point to itself in no time at all.
    net = make_network(LINE, 5)
    zones = make_zones([[0, 4], [2]], area_m2=[1e6, 1e6], network=net)
    matrix = zone_travel_time_matrix(net, zones, near_seconds=25.0)

    expected = (128.0 / (45.0 * np.pi)) * np.sqrt(1e6 / np.pi) / (14.0 / 3.6)
    assert matrix[0, 0] == pytest.approx(expected, rel=1e-4)


def test_a_cutoff_averages_over_what_it_reached(net):
    zones = make_zones([[0, 1], [3, 4]], network=net)

    # Only 1->3 (20 s) is inside a 25 s cutoff; 0->3, 0->4 and 1->4 are not.
    reached = multipoint_travel_time_matrix(
        net, zones, max_seconds=25.0, min_reachable_fraction=0.0
    )
    assert reached[0, 1] == pytest.approx(20.0)

    # Demanding every pair instead leaves the cell unreachable rather than
    # reporting an average biased towards the closest points.
    strict = multipoint_travel_time_matrix(
        net, zones, max_seconds=25.0, min_reachable_fraction=1.0
    )
    assert not np.isfinite(strict[0, 1])


def test_two_tiers_stitch_near_and_far(net):
    """Short pairs come from every point, long ones from the representative."""
    zones = make_zones([[0, 1], [3, 4]], network=net)
    matrix = zone_travel_time_matrix(
        net, zones, near_seconds=25.0, intrazonal_fallback=False
    )

    # The diagonal is fully inside the near cutoff, so it keeps the multi-point
    # answer that a single centroid could not have produced at all.
    assert matrix[0, 0] == pytest.approx(10.0)

    # The zone pair is not, so it falls back to representative point to
    # representative point: node 0 to node 3, 30 s.
    assert matrix[0, 1] == pytest.approx(30.0)


def test_two_tiers_match_one_tier_when_the_near_cutoff_covers_everything(net):
    zones = make_zones([[0, 1], [3, 4]], network=net)

    one = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)
    two = zone_travel_time_matrix(
        net, zones, near_seconds=10_000.0, intrazonal_fallback=False
    )
    np.testing.assert_allclose(one, two)


def test_logsum_aggregation_sits_below_the_arithmetic_mean(net):
    """Jensen's inequality, and the reason a gravity model wants the logsum."""
    zones = make_zones([[0, 1], [3, 4]], network=net)

    mean = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)
    logsum = zone_travel_time_matrix(
        net, zones, decay_mu=0.05, intrazonal_fallback=False
    )
    assert logsum[0, 1] < mean[0, 1]


def test_multiple_workers_match_single_worker():
    rng = np.random.default_rng(3)
    n = 30
    edges = [(int(i), int(i + 1), float(rng.uniform(1, 10))) for i in range(n - 1)]
    edges += [(int(i + 1), int(i), float(rng.uniform(1, 10))) for i in range(n - 1)]
    net = make_network(edges, n)

    zones = make_zones([[i, i + 1, i + 2] for i in range(0, n - 2, 6)], network=net)
    one = zone_travel_time_matrix(net, zones, workers=1, intrazonal_fallback=False)
    two = zone_travel_time_matrix(
        net, zones, workers=2, chunk_size=3, intrazonal_fallback=False
    )

    np.testing.assert_allclose(one, two, rtol=1e-6)


def test_chunking_on_zone_boundaries_does_not_change_the_result(net):
    zones = make_zones([[0, 1], [2], [3, 4]], network=net)

    whole = zone_travel_time_matrix(
        net, zones, chunk_size=100, intrazonal_fallback=False
    )
    split = zone_travel_time_matrix(net, zones, chunk_size=1, intrazonal_fallback=False)
    np.testing.assert_allclose(whole, split)


def test_unreachable_zones_stay_unreachable():
    net = make_network([(0, 1, 10.0), (1, 0, 10.0), (2, 3, 10.0), (3, 2, 10.0)], 4)
    zones = make_zones([[0, 1], [2, 3]], network=net)

    matrix = zone_travel_time_matrix(net, zones, intrazonal_fallback=False)
    assert not np.isfinite(matrix[0, 1])
    assert matrix[0, 0] == pytest.approx(10.0)
