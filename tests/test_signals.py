"""Signalled junctions: one wait per junction on the road, by how much green."""

import numpy as np
import pytest

from valma_bike_and_walk import links as links_module
from valma_bike_and_walk.links import normalise, signal_waits
from valma_bike_and_walk.signals import EQUAL, MAIN, MIDBLOCK, MINOR, NONE, signal_codes

from .conftest import links_frame, write_pbf

# A signalled crossroads, in metres. The main road (secondary) runs west-east
# through the centre 3: a stop-line light at 2, a signalled crossing on the far
# arm at 4. The side road (residential) comes from the north: a signalled
# crossing at 11, a stop-line light at 7. Path 9-4-10 crosses the main road;
# path 12-11-13 crosses the side road. Far away, road 20-21-22 has a mid-block
# pedestrian light at 21, crossed by path 23-21-24.
NODES = {
    1: (-200, 0), 2: (-15, 0), 3: (0, 0), 4: (20, 0), 5: (200, 0),
    6: (0, 200), 11: (0, 25), 7: (0, 12),
    9: (20, -20), 10: (20, 20), 12: (-20, 25), 13: (20, 25),
    20: (0, 1000), 21: (100, 1000), 22: (200, 1000), 23: (100, 980), 24: (100, 1020),
}
LINKS = [  # u, v, highway
    (1, 2, "secondary"), (2, 3, "secondary"), (3, 4, "secondary"), (4, 5, "secondary"),
    (6, 11, "residential"), (11, 7, "residential"), (7, 3, "residential"),
    (9, 4, "cycleway"), (4, 10, "cycleway"),
    (12, 11, "footway"), (11, 13, "footway"),
    (20, 21, "residential"), (21, 22, "residential"),
    (23, 21, "cycleway"), (21, 24, "cycleway"),
]
SIGNALS = [2, 4, 7, 11, 21]
LIGHTS = [2, 7]


def codes():
    u = np.array([a for a, _, _ in LINKS])
    v = np.array([b for _, b, _ in LINKS])
    highway = np.array([h for _, _, h in LINKS], dtype=object)
    ids = np.array(sorted(NODES))
    return {
        (a, b): row
        for (a, b, _), row in zip(
            LINKS,
            zip(*signal_codes(
                u, v, highway, np.isin(highway, list(links_module.PATH_HIGHWAYS)),
                ids, np.array([NODES[i][0] for i in ids], float),
                np.array([NODES[i][1] for i in ids], float),
                np.array(SIGNALS), np.array(LIGHTS),
            ).values()),
        )
    }  # (u, v) -> (entry, entry_reverse, cross_start, cross_end)


def test_driving_through_a_junction_enters_it_once():
    c = codes()
    # West to east: the stop-line link enters; centre, far crossing and exit do not.
    assert [c[(1, 2)][0], c[(2, 3)][0], c[(3, 4)][0], c[(4, 5)][0]] == [MAIN, NONE, NONE, NONE]
    # East to west enters on the far arm instead.
    assert [c[(4, 5)][1], c[(3, 4)][1], c[(2, 3)][1], c[(1, 2)][1]] == [MAIN, NONE, NONE, NONE]


def test_a_side_road_approach_gets_the_minor_share():
    c = codes()
    assert c[(6, 11)][0] == MINOR
    assert c[(11, 7)][0] == NONE and c[(7, 3)][0] == NONE


def test_paths_wait_by_the_road_they_cross():
    c = codes()
    # Crossing the main road waits like a side road does.
    assert c[(9, 4)][3] == MINOR and c[(4, 10)][2] == MINOR
    # Crossing the side road goes with the main road's green.
    assert c[(12, 11)][3] == MAIN and c[(11, 13)][2] == MAIN
    # Roads carry no crossing codes; paths no entry codes.
    assert c[(3, 4)][2:] == (NONE, NONE) and c[(9, 4)][:2] == (NONE, NONE)


def test_a_mid_block_pedestrian_light():
    c = codes()
    assert c[(20, 21)][0] == MIDBLOCK and c[(21, 22)][1] == MIDBLOCK
    assert c[(23, 21)][3] == MIDBLOCK and c[(21, 24)][2] == MIDBLOCK


def _codes_for(nodes, links, signals, lights):
    u = np.array([a for a, _, _ in links])
    v = np.array([b for _, b, _ in links])
    highway = np.array([h for _, _, h in links], dtype=object)
    ids = np.array(sorted(nodes))
    out = signal_codes(
        u, v, highway, np.isin(highway, list(links_module.PATH_HIGHWAYS)), ids,
        np.array([nodes[i][0] for i in ids], float), np.array([nodes[i][1] for i in ids], float),
        np.array(signals), np.array(lights),
    )
    return {(a, b): (out["signal_entry"][k], out["signal_cross_end"][k])
            for k, (a, b, _) in enumerate(links)}


def test_a_crossroads_of_equals_shares_the_green():
    # Two secondaries cross at 0; lights on all four approaches; a path
    # crosses the east arm at signalled crossing 5.
    nodes = {0: (0, 0), 1: (-200, 0), 2: (-12, 0), 3: (200, 0), 4: (12, 0),
             6: (0, -200), 7: (0, -12), 8: (0, 200), 9: (0, 12),
             5: (20, 0), 10: (20, -20)}
    links = [(1, 2, "secondary"), (2, 0, "secondary"), (3, 5, "secondary"),
             (5, 4, "secondary"), (4, 0, "secondary"), (6, 7, "secondary"),
             (7, 0, "secondary"), (8, 9, "secondary"), (9, 0, "secondary"),
             (10, 5, "cycleway")]
    c = _codes_for(nodes, links, [2, 4, 7, 9, 5], [2, 4, 7, 9])
    assert c[(1, 2)][0] == EQUAL and c[(6, 7)][0] == EQUAL and c[(3, 5)][0] == EQUAL
    assert c[(10, 5)][1] == EQUAL


def test_a_divided_main_road_is_one_arm_each_way():
    """Two one-way carriageways 8 m apart are one direction, not two."""
    nodes = {1: (-200, 4), 2: (-12, 4), 0: (0, 4), 3: (200, -4), 4: (12, -4), 5: (0, -4),
             6: (0, 200), 7: (0, 12)}
    links = [(1, 2, "primary"), (2, 0, "primary"),  # westbound carriageway, eastward drawn
             (3, 4, "primary"), (4, 5, "primary"),  # the other carriageway
             (0, 5, "primary_link"), (6, 7, "residential"), (7, 0, "residential")]
    c = _codes_for(nodes, links, [2, 4, 7], [2, 4, 7])
    assert c[(1, 2)][0] == MAIN and c[(3, 4)][0] == MAIN
    assert c[(6, 7)][0] == MINOR


def test_waits_follow_the_green_share():
    road, cross = signal_waits()
    cycle = links_module.SIGNAL_CYCLE_S
    red = (1 - links_module.SIGNAL_GREEN_SHARE_MAIN) * cycle
    assert road[MAIN] == pytest.approx(red**2 / (2 * cycle))
    assert road[NONE] == 0 and road[MAIN] < road[MINOR]
    assert road[MIDBLOCK] < cross[MIDBLOCK]


def test_codes_become_seconds_in_the_direction_that_enters(monkeypatch):
    row = {"link_id": 0, "u": 1, "v": 2, "coords": [(24.0, 60.0), (24.002, 60.0)],
           "highway": "secondary"}
    timed = normalise(links_frame([{**row, "signal_entry": MAIN, "signal_entry_reverse": NONE,
                                    "signal_cross_start": 0, "signal_cross_end": 0}]), "bike")
    plain = normalise(links_frame([{**row, "signal_entry": 0, "signal_entry_reverse": 0,
                                    "signal_cross_start": 0, "signal_cross_end": 0}]), "bike")
    road, _ = signal_waits()
    assert timed["travel_time_s"].iloc[0] - plain["travel_time_s"].iloc[0] == pytest.approx(road[MAIN])
    assert timed["travel_time_reverse_s"].iloc[0] == pytest.approx(plain["travel_time_reverse_s"].iloc[0])

    # The constants are read when times are computed, so calibration can vary them.
    monkeypatch.setattr(links_module, "SIGNAL_GREEN_SHARE_MAIN", 1.0)
    timed = normalise(links_frame([{**row, "signal_entry": MAIN, "signal_entry_reverse": NONE,
                                    "signal_cross_start": 0, "signal_cross_end": 0}]), "bike")
    assert timed["travel_time_s"].iloc[0] == pytest.approx(plain["travel_time_s"].iloc[0])


def test_extract_writes_the_codes(tmp_path):
    """Through a real PBF: the stop line enters, the far arm does not."""
    from valma_bike_and_walk.config import Settings
    from valma_bike_and_walk.osm import read_links

    def lonlat(xy):  # metres to degrees near 60 N, close enough for a test
        return (24.0 + xy[0] / 55_800.0, 60.0 + xy[1] / 111_300.0)

    nodes = {i: lonlat(NODES[i]) for i in (1, 2, 3, 4, 5, 6, 7, 9, 10)}
    ways = [
        (1, [1, 2, 3, 4, 5], {"highway": "secondary"}),
        (2, [6, 7, 3], {"highway": "residential"}),
        (3, [9, 4, 10], {"highway": "cycleway"}),
    ]
    tags = {2: {"highway": "traffic_signals"}, 7: {"highway": "traffic_signals"},
            4: {"highway": "crossing", "crossing": "traffic_signals"}}
    pbf = write_pbf(tmp_path / "junction.osm.pbf", nodes, ways, node_tags=tags)
    settings = Settings(pbf_path=pbf, cache_dir=tmp_path / "cache", output_dir=tmp_path)
    links = read_links(settings, "bike").set_index(["u", "v"])
    assert links.loc[(1, 2), "signal_entry"] == MAIN
    assert links.loc[(3, 4), "signal_entry"] == NONE
    assert links.loc[(6, 7), "signal_entry"] == MINOR
    assert links.loc[(9, 4), "signal_cross_end"] == MINOR
