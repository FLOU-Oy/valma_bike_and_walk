import numpy as np
import pandas as pd
import pytest

from valma_bike_and_walk.speeds import BIKE_PROFILE, WALK_PROFILE, profile_for


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
