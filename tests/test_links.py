"""The editable link layer: round-tripping, repair after a QGIS edit, direction."""

import geopandas as gpd
import numpy as np
import pytest

from valma_bike_and_walk import links as links_module
from valma_bike_and_walk.config import PROJECTED_CRS
from valma_bike_and_walk.links import (
    LINKS_LAYER,
    directed_edges,
    normalise,
    read_links,
    resolve_endpoints,
    write_links,
)

from .conftest import links_frame

# Two links in a row: 1 -> 2 -> 3, each roughly 111 m long.
CHAIN = [
    {"link_id": 0, "u": 1, "v": 2, "coords": [(24.000, 60.0), (24.002, 60.0)]},
    {"link_id": 1, "u": 2, "v": 3, "coords": [(24.002, 60.0), (24.004, 60.0)]},
]


# --------------------------------------------------------------------------
# GeoPackage round trip
# --------------------------------------------------------------------------


def test_write_and_read_round_trips_through_a_real_file(tmp_path):
    links = links_module.annotate(_with_lengths(links_frame(CHAIN)), "walk")
    path = write_links(links, tmp_path / "links.gpkg")

    back = read_links(path)
    assert len(back) == 2
    assert set(back.columns) >= {
        "link_id",
        "u",
        "v",
        "highway",
        "length_m",
        "speed_kmh",
        "travel_time_s",
        "speed_override_kmh",
        "geometry",
    }
    assert back.crs.to_epsg() == 4326


def test_reading_a_layer_without_the_required_columns_says_so(tmp_path):
    frame = links_frame(CHAIN).drop(columns=["u"])
    path = tmp_path / "bad.gpkg"
    frame.to_file(path, driver="GPKG", layer=LINKS_LAYER)

    with pytest.raises(ValueError, match="missing required column"):
        read_links(path)


def _with_lengths(links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    links = links.copy()
    links["length_m"] = links_module.geometry_lengths(links)
    return links


# --------------------------------------------------------------------------
# Normalising an edited layer
# --------------------------------------------------------------------------


def test_length_is_recomputed_from_the_geometry_not_trusted():
    """Move a vertex in QGIS and the cost must follow, untouched column or not."""
    edited = links_frame(
        [
            {
                "link_id": 0,
                "u": 1,
                "v": 2,
                "coords": [(24.0, 60.0), (24.004, 60.0)],
                "length_m": 1.0,
            }
        ]
    )
    normalised = normalise(edited, "walk")
    assert normalised["length_m"].iloc[0] == pytest.approx(222.0, rel=0.05)


def test_editing_the_highway_tag_changes_the_speed():
    slow = normalise(links_frame([{**CHAIN[0], "highway": "steps"}]), "walk")
    fast = normalise(links_frame([{**CHAIN[0], "highway": "footway"}]), "walk")
    assert slow["speed_kmh"].iloc[0] < fast["speed_kmh"].iloc[0]
    assert slow["travel_time_s"].iloc[0] > fast["travel_time_s"].iloc[0]


SIGNALLED_CYCLEWAY = {"highway": "cycleway", "crossing": "traffic_signals"}


def test_traffic_light_penalty_applies_in_both_bike_directions():
    """A red light is paid whichever way you ride through it."""
    signalled = normalise(links_frame([{**CHAIN[0], **SIGNALLED_CYCLEWAY}]), "bike")
    plain = normalise(links_frame([{**CHAIN[0], "highway": "cycleway"}]), "bike")
    expected = (
        plain["travel_time_s"].iloc[0] + links_module.TRAFFIC_LIGHT_PENALTY_SECONDS
    )

    assert signalled["travel_time_s"].iloc[0] == pytest.approx(expected)
    assert signalled["travel_time_reverse_s"].iloc[0] == pytest.approx(expected)

    edges = directed_edges(signalled, "bike")
    assert edges.loc[edges["direction"] == 1, "travel_time_s"].iloc[0] == pytest.approx(
        expected
    )
    assert edges.loc[edges["direction"] == -1, "travel_time_s"].iloc[
        0
    ] == pytest.approx(expected)


def test_walking_pays_no_traffic_light_penalty():
    """Pedestrian crossings are already in the walk speeds; charging again double-counts."""
    signalled = normalise(links_frame([{**CHAIN[0], **SIGNALLED_CYCLEWAY}]), "walk")
    plain = normalise(links_frame([{**CHAIN[0], "highway": "cycleway"}]), "walk")

    assert signalled["travel_time_s"].iloc[0] == pytest.approx(
        plain["travel_time_s"].iloc[0]
    )
    edges = directed_edges(signalled, "walk")
    assert edges["travel_time_s"].tolist() == pytest.approx(
        [plain["travel_time_s"].iloc[0]] * 2
    )


def test_non_signal_crossing_has_no_bike_penalty():
    signalled = normalise(
        links_frame([{**CHAIN[0], "highway": "cycleway", "crossing": "uncontrolled"}]),
        "bike",
    )
    plain = normalise(links_frame([{**CHAIN[0], "highway": "cycleway"}]), "bike")
    assert signalled["travel_time_s"].iloc[0] == pytest.approx(
        plain["travel_time_s"].iloc[0]
    )


def test_adjacent_signal_gets_penalty_without_crossing_tag():
    links = normalise(
        links_frame(
            [
                {
                    **CHAIN[0],
                    "traffic_signal_at_end": True,
                }
            ]
        ),
        "bike",
    )
    plain = normalise(links_frame([CHAIN[0]]), "bike")
    assert links["travel_time_s"].iloc[0] == pytest.approx(
        plain["travel_time_s"].iloc[0] + links_module.TRAFFIC_LIGHT_PENALTY_SECONDS
    )


def test_car_lane_gets_adjacent_signal_penalty():
    signal = normalise(
        links_frame(
            [
                {
                    **CHAIN[0],
                    "highway": "primary",
                    "crossing": "traffic_signals",
                    "traffic_signal_at_start": True,
                    "traffic_signal_at_end": True,
                }
            ]
        ),
        "bike",
    )
    plain = normalise(
        links_frame([{**CHAIN[0], "highway": "primary"}]),
        "bike",
    )
    assert signal["travel_time_s"].iloc[0] == pytest.approx(
        plain["travel_time_s"].iloc[0] + links_module.TRAFFIC_LIGHT_PENALTY_SECONDS
    )


def test_car_lane_crossing_tag_without_adjacent_signal_has_no_penalty():
    signal_tag_only = normalise(
        links_frame(
            [{**CHAIN[0], "highway": "primary", "crossing": "traffic_signals"}]
        ),
        "bike",
    )
    plain = normalise(
        links_frame([{**CHAIN[0], "highway": "primary"}]),
        "bike",
    )
    assert signal_tag_only["travel_time_s"].iloc[0] == pytest.approx(
        plain["travel_time_s"].iloc[0]
    )


def _graded_links(profiles) -> gpd.GeoDataFrame:
    """One roughly kilometre-long link per ``(up, down)`` pair of gradients.

    ``normalise`` recomputes ``length_m`` from the geometry rather than trusting
    the column, so a test that wants a known gradient has to derive the ascent
    from the real length instead of writing both down.
    """
    frame = links_frame(
        [
            {
                "link_id": i,
                "u": 2 * i + 1,
                "v": 2 * i + 2,
                "coords": [(24.000, 60.0 + 0.01 * i), (24.020, 60.0 + 0.01 * i)],
            }
            for i in range(len(profiles))
        ]
    )
    length = links_module.geometry_lengths(frame)
    frame["ascent_m"] = length * np.array([up for up, _ in profiles])
    frame["descent_m"] = length * np.array([down for _, down in profiles])
    return frame


def _flat_seconds(links: gpd.GeoDataFrame) -> np.ndarray:
    """What each link would cost with no penalty at all."""
    return links["length_m"].to_numpy() / (links["speed_kmh"].to_numpy() / 3.6)


def test_elevation_penalty_uses_ascent_thresholds_and_not_descent():
    """Below 2.5 % costs nothing; past 5 % the steeper factor takes over."""
    normalised = normalise(
        _graded_links([(0.02, 0.02), (0.03, 0.03), (0.06, 0.06)]), "bike"
    )
    length = normalised["length_m"].to_numpy()
    expected = (
        _flat_seconds(normalised) + np.array([0.0, 2.32 * 0.03, 2.50 * 0.06]) * length
    )

    assert normalised["travel_time_s"].to_numpy() == pytest.approx(expected)
    assert normalised["travel_time_reverse_s"].to_numpy() == pytest.approx(expected)


def test_elevation_penalty_is_directional_for_an_asymmetric_profile():
    normalised = normalise(_graded_links([(0.03, 0.01)]), "bike")
    flat = _flat_seconds(normalised)[0]
    length = normalised["length_m"].iloc[0]

    assert normalised["travel_time_s"].iloc[0] == pytest.approx(
        flat + 2.32 * 0.03 * length
    )
    # 1 % downhill is below the threshold, so the way back is just the flat time.
    assert normalised["travel_time_reverse_s"].iloc[0] == pytest.approx(flat)


def test_unrealistic_twenty_percent_grade_is_ignored():
    """A grade that steep is a bad elevation sample, not a hill worth pricing."""
    too_steep = links_module.GRADE_MAX_VALID + 0.02
    steep = links_module.GRADE_MAX_VALID - 0.01
    normalised = normalise(_graded_links([(too_steep, steep)]), "bike")
    flat = _flat_seconds(normalised)[0]
    length = normalised["length_m"].iloc[0]

    assert normalised["travel_time_s"].iloc[0] == pytest.approx(flat)
    assert normalised["travel_time_reverse_s"].iloc[0] == pytest.approx(
        flat + 2.50 * steep * length
    )


def test_segregated_cycleway_changes_the_bike_link_speed():
    shared = normalise(
        links_frame([{**CHAIN[0], "highway": "cycleway", "segregated": "no"}]),
        "bike",
    )
    separated = normalise(
        links_frame([{**CHAIN[0], "highway": "cycleway", "segregated": "yes"}]),
        "bike",
    )
    assert separated["speed_kmh"].iloc[0] > shared["speed_kmh"].iloc[0]


def test_a_speed_override_wins_over_the_tags():
    forced = normalise(
        links_frame([{**CHAIN[0], "highway": "steps", "speed_override_kmh": 20.0}]),
        "walk",
    )
    assert forced["speed_kmh"].iloc[0] == pytest.approx(20.0)
    # 111 m at 20 km/h = 20 s.
    assert forced["travel_time_s"].iloc[0] == pytest.approx(
        111.0 / (20 / 3.6), rel=0.05
    )


def test_an_empty_override_leaves_the_profile_in_charge():
    rows = [{**CHAIN[0], "highway": "steps", "speed_override_kmh": np.nan}]
    normalised = normalise(links_frame(rows), "walk")
    assert normalised["speed_kmh"].iloc[0] == pytest.approx(1.2)


def test_links_drawn_without_ids_snap_onto_the_existing_network():
    """A new link drawn in QGIS from an existing junction must join there."""
    rows = CHAIN + [
        # Starts exactly on node 3 and heads north; no u/v filled in.
        {"link_id": 2, "coords": [(24.004, 60.0), (24.004, 60.002)]}
    ]
    u, v = resolve_endpoints(links_frame(rows))

    assert u[2] == 3, "the drawn link should adopt node 3, not invent one"
    assert v[2] < 0, "its far end is new, so it gets a fresh negative id"


def test_two_hand_drawn_links_meeting_end_to_end_connect():
    rows = [
        {"link_id": 0, "coords": [(24.000, 60.0), (24.002, 60.0)]},
        {"link_id": 1, "coords": [(24.002, 60.0), (24.004, 60.0)]},
    ]
    u, v = resolve_endpoints(links_frame(rows))
    assert v[0] == u[1] != 0


def test_a_far_away_drawn_link_does_not_get_welded_to_anything():
    rows = CHAIN + [{"link_id": 2, "coords": [(25.0, 61.0), (25.001, 61.0)]}]
    u, v = resolve_endpoints(links_frame(rows))
    assert u[2] < 0 and v[2] < 0
    assert u[2] != v[2]


def test_new_rows_without_a_link_id_are_given_one():
    rows = CHAIN + [{"link_id": np.nan, "coords": [(24.004, 60.0), (24.006, 60.0)]}]
    normalised = normalise(links_frame(rows), "walk")
    ids = normalised["link_id"].to_numpy()
    assert len(set(ids)) == 3
    assert ids[2] == 2


def test_duplicated_link_ids_are_made_unique():
    rows = [
        {"link_id": 7, "u": 1, "v": 2, "coords": [(24.000, 60.0), (24.002, 60.0)]},
        {"link_id": 7, "u": 2, "v": 3, "coords": [(24.002, 60.0), (24.004, 60.0)]},
    ]
    normalised = normalise(links_frame(rows), "walk")
    assert len(set(normalised["link_id"])) == 2


def test_a_deleted_row_is_simply_gone():
    normalised = normalise(links_frame(CHAIN[:1]), "walk")
    assert len(normalised) == 1
    assert set(normalised["v"]) == {2}


def test_self_looping_links_are_dropped():
    rows = CHAIN + [
        {
            "link_id": 2,
            "u": 3,
            "v": 3,
            "coords": [(24.004, 60.0), (24.005, 60.001), (24.004, 60.0)],
        }
    ]
    normalised = normalise(links_frame(rows), "walk")
    assert len(normalised) == 2


def test_a_projected_layer_is_measured_in_its_own_metres():
    """QGIS users in Finland often reproject; lengths must not change."""
    wgs84 = links_frame(CHAIN)
    projected = wgs84.to_crs(PROJECTED_CRS)
    np.testing.assert_allclose(
        normalise(wgs84, "walk")["length_m"].to_numpy(),
        normalise(projected, "walk")["length_m"].to_numpy(),
        rtol=1e-6,
    )


# --------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------


def test_walking_ignores_oneway():
    links = normalise(links_frame([{**CHAIN[0], "oneway": "yes"}]), "walk")
    edges = directed_edges(links, "walk")
    assert sorted(zip(edges["u"], edges["v"])) == [(1, 2), (2, 1)]


def test_cycling_honours_oneway():
    links = normalise(links_frame([{**CHAIN[0], "oneway": "yes"}]), "bike")
    edges = directed_edges(links, "bike")
    assert list(zip(edges["u"], edges["v"])) == [(1, 2)]


def test_oneway_minus_one_runs_against_the_digitised_direction():
    links = normalise(links_frame([{**CHAIN[0], "oneway": "-1"}]), "bike")
    edges = directed_edges(links, "bike")
    assert list(zip(edges["u"], edges["v"])) == [(2, 1)]
    assert list(edges["direction"]) == [-1]


def test_a_contraflow_cycle_lane_is_two_way_for_bikes():
    """oneway=yes + oneway:bicycle=no is the whole reason this tag is read."""
    row = {**CHAIN[0], "oneway": "yes", "oneway_bicycle": "no"}
    edges = directed_edges(normalise(links_frame([row]), "bike"), "bike")
    assert sorted(zip(edges["u"], edges["v"])) == [(1, 2), (2, 1)]


def test_a_roundabout_is_oneway_without_being_tagged_so():
    row = {**CHAIN[0], "junction": "roundabout"}
    edges = directed_edges(normalise(links_frame([row]), "bike"), "bike")
    assert list(zip(edges["u"], edges["v"])) == [(1, 2)]


def test_every_edge_remembers_the_link_it_came_from():
    links = normalise(links_frame(CHAIN), "walk")
    edges = directed_edges(links, "walk")
    assert len(edges) == 4
    assert sorted(edges["link_id"]) == [0, 0, 1, 1]
    assert set(edges["direction"]) == {1, -1}
