import numpy as np
import pandas as pd
import pytest

from valma_bike_and_walk.speeds import (
    BIKE_PROFILE,
    CYCLE_LANE_FACTOR,
    MIXED_TRAFFIC_FACTOR,
    WALK_PROFILE,
    carriageway_divisor,
    profile_for,
)


def test_unknown_tags_fall_back_to_base_speed():
    highway = pd.Series(["not_a_real_highway_value", None])
    speeds = WALK_PROFILE.speeds_kph(highway)
    assert np.allclose(speeds, WALK_PROFILE.base_kph)


def test_surface_slows_cycling_more_than_walking():
    highway = pd.Series(["path"])
    surface = pd.Series(["mud"])

    walk_penalty = (
        WALK_PROFILE.speeds_kph(highway, surface)[0]
        / WALK_PROFILE.speeds_kph(highway)[0]
    )
    bike_penalty = (
        BIKE_PROFILE.speeds_kph(highway, surface)[0]
        / BIKE_PROFILE.speeds_kph(highway)[0]
    )

    assert bike_penalty < walk_penalty


def test_cycleway_is_preferred_to_primary_and_secondary_for_cycling():
    highway = pd.Series(["cycleway", "primary", "secondary"])
    speeds = BIKE_PROFILE.speeds_kph(highway, pd.Series(["asphalt"] * 3))
    assert speeds[0] > speeds[1]
    assert speeds[0] > speeds[2]


def test_primary_road_with_cycleway_lane_uses_lane_multiplier():
    highway = pd.Series(["primary", "primary"])
    surface = pd.Series(["asphalt", "asphalt"])
    lane_speed = BIKE_PROFILE.speeds_kph(highway, surface, cycleway=pd.Series(["lane", None]))
    no_lane_speed = BIKE_PROFILE.speeds_kph(highway, surface, cycleway=pd.Series([None, None]))
    assert lane_speed[0] > no_lane_speed[0]
    assert lane_speed[0] > lane_speed[1]


def test_segregated_cycleway_is_preferred_to_shared_cycleway():
    highway = pd.Series(["cycleway", "cycleway"])
    segregated = pd.Series(["yes", "no"])
    speeds = BIKE_PROFILE.speeds_kph(highway, pd.Series(["asphalt"] * 2), segregated)
    assert speeds[0] > speeds[1]


def test_car_volume_uses_one_speed_multiplier_for_all_volumes():
    highway = pd.Series(["cycleway"] * 4)
    surface = pd.Series(["asphalt"] * 4)
    volume = pd.Series([0.0, 100.0, 1000.0, 1001.0])
    base_speed = BIKE_PROFILE.speeds_kph(highway, surface)

    speeds = BIKE_PROFILE.speeds_kph(
        highway, surface, car_volume=volume
    )
    # The per-route volume term, spread over the average 5.6 km route.
    expected = base_speed * np.array(
        [
            7.5 / (7.5 + 0.008 * 0.0 / 5.6),
            7.5 / (7.5 + 0.008 * 100.0 / 5.6),
            7.5 / (7.5 + 0.008 * 1000.0 / 5.6),
            7.5 / (7.5 + 0.008 * 1001.0 / 5.6),
        ]
    )

    assert speeds == pytest.approx(expected)


def test_missing_car_volume_does_not_penalize_bike_speed():
    highway = pd.Series(["primary", "cycleway"])
    surface = pd.Series(["asphalt", "asphalt"])

    speeds = BIKE_PROFILE.speeds_kph(
        highway, surface, car_volume=pd.Series([None, np.nan])
    )

    assert speeds == pytest.approx(BIKE_PROFILE.speeds_kph(highway, surface))


def test_speeds_are_clamped_to_profile_range():
    highway = pd.Series(["steps", "cycleway"])
    speeds = BIKE_PROFILE.speeds_kph(highway)
    assert speeds.min() >= BIKE_PROFILE.min_kph
    assert speeds.max() <= BIKE_PROFILE.max_kph


def test_walking_never_reaches_car_speeds():
    """The bug this profile exists to fix: motor speed limits leaking in."""
    highway = pd.Series(["primary", "trunk", "secondary"])
    assert WALK_PROFILE.speeds_kph(highway).max() <= 6.0


def test_steps_are_slow_for_both_modes():
    highway = pd.Series(["steps"])
    assert WALK_PROFILE.speeds_kph(highway)[0] < 2.0
    assert BIKE_PROFILE.speeds_kph(highway)[0] < 2.0


def test_travel_time_matches_hand_calculation():
    # 1000 m of cycleway at 18 km/h = 200 s
    seconds = BIKE_PROFILE.travel_times_seconds(
        np.array([1000.0]), pd.Series(["cycleway"]), pd.Series(["asphalt"])
    )
    assert seconds[0] == pytest.approx(1000 / (18 * 1000 / 3600))


def test_profile_lookup_rejects_unknown_mode():
    with pytest.raises(ValueError, match="No speed profile"):
        profile_for("hovercraft")


def test_cycle_lane_replaces_the_mixed_traffic_factor():
    highway = pd.Series(["secondary", "secondary"])
    speeds = BIKE_PROFILE.speeds_kph(highway, cycleway=pd.Series(["lane", None]))
    assert speeds[0] == pytest.approx(18.0 / carriageway_divisor("secondary", CYCLE_LANE_FACTOR))
    assert speeds[1] == pytest.approx(18.0 / carriageway_divisor("secondary", MIXED_TRAFFIC_FACTOR))


def test_cycle_lane_does_not_speed_up_roads_without_a_carriageway_factor():
    highway = pd.Series(["service", "unclassified", "trunk"])
    lane = BIKE_PROFILE.speeds_kph(highway, cycleway=pd.Series(["lane"] * 3))
    none = BIKE_PROFILE.speeds_kph(highway, cycleway=pd.Series([None] * 3))
    assert lane == pytest.approx(none)


def test_cycle_track_counts_as_a_separate_cycleway():
    highway = pd.Series(["secondary", "secondary", "secondary"])
    speeds = BIKE_PROFILE.speeds_kph(
        highway, cycleway=pd.Series(["track", "lane", None])
    )
    assert speeds[0] == pytest.approx(18.0)
    assert speeds[0] > speeds[1] > speeds[2]


def test_road_class_and_facility_costs_add_up():
    """Tarkkala's per-km terms are summed, not multiplied."""
    highway = pd.Series(["secondary", "residential", "secondary"])
    speeds = BIKE_PROFILE.speeds_kph(highway, cycleway=pd.Series([None, None, "track"]))
    # Mixed traffic on a main street: (4.14 + 5.03) / (4.14 + 2.61).
    assert speeds[0] == pytest.approx(18.0 * 6.75 / (4.14 + 2.61 * MIXED_TRAFFIC_FACTOR))
    # On a local street the two costs still add rather than multiply.
    assert speeds[1] == pytest.approx(18.0 * 6.75 / (4.96 + 2.61 * MIXED_TRAFFIC_FACTOR))
    # A track along a main street is the reference itself.
    assert speeds[2] == pytest.approx(18.0)


def test_a_segregated_path_is_quicker_and_untagged_counts_as_shared():
    from valma_bike_and_walk.speeds import SEGREGATED_PATH_FACTOR

    highway = pd.Series(["cycleway", "cycleway", "cycleway", "path", "secondary"])
    segregated = pd.Series(["yes", "no", None, "yes", "yes"])
    speeds = BIKE_PROFILE.speeds_kph(highway, segregated=segregated)
    assert speeds[0] == pytest.approx(18.0 * SEGREGATED_PATH_FACTOR)
    # Shared, whether tagged so or untagged: the cycleway's own speed.
    assert speeds[1] == pytest.approx(18.0)
    assert speeds[2] == pytest.approx(18.0)
    assert speeds[3] == pytest.approx(11.0 * SEGREGATED_PATH_FACTOR)
    # A road is not a foot-and-cycle path; the tag changes nothing there.
    road = BIKE_PROFILE.speeds_kph(pd.Series(["secondary"]))
    assert speeds[4] == pytest.approx(road[0])
