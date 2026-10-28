"""Signalled junctions: roads pay entering, paths pay crossing, once each."""

import numpy as np
import pytest

from valma_bike_and_walk import links as links_module
from valma_bike_and_walk.links import normalise, signal_waits
from valma_bike_and_walk.signals import JUNCTION, MIDBLOCK, NONE, signal_codes

from .conftest import links_frame, write_pbf


def codes_for(nodes, links, signals, lights, crossings=()):
    """(u, v) -> {"entry", "entry_reverse", "cross_start", "cross_end"}."""
    u = np.array([a for a, _, _ in links])
    v = np.array([b for _, b, _ in links])
    highway = np.array([h for _, _, h in links], dtype=object)
    ids = np.array(sorted(nodes))
    out = signal_codes(
        u, v, highway, np.isin(highway, list(links_module.PATH_HIGHWAYS)), ids,
        np.array([nodes[i][0] for i in ids], float), np.array([nodes[i][1] for i in ids], float),
        np.array(signals), np.array(lights), np.array(crossings, dtype=np.int64),
    )
    return {
        (a, b): {key.removeprefix("signal_"): out[key][k] for key in out}
        for k, (a, b, _) in enumerate(links)
    }


def route(c, *nodes):
    """The codes a route pays, in order: one per charge, travelling node to node."""
    paid = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        if (a, b) in c:
            row = c[(a, b)]
            paid += [row["entry"], row["cross_end"]]
        else:
            row = c[(b, a)]
            paid += [row["entry_reverse"], row["cross_start"]]
    return [p for p in paid if p != NONE]


# A signalled crossroads, in metres. The main road (secondary) runs west-east
# through the centre 3: a stop-line light at 2, a signalled crossing on the far
# arm at 4. The side road (residential) comes from the north: a signalled
# crossing at 11, a stop-line light at 7. Path 9-4-10 crosses the main road;
# path 12-11-13 crosses the side road. A sidewalk 50-51-52 goes round the
# south-west corner without crossing anything.
NODES = {
    1: (-200, 0), 2: (-15, 0), 3: (0, 0), 4: (20, 0), 5: (200, 0),
    6: (0, 200), 11: (0, 25), 7: (0, 12),
    9: (20, -60), 10: (20, 60), 12: (-60, 25), 13: (60, 25),
    50: (-60, -12), 51: (-12, -12), 52: (-12, -60),
}
LINKS = [  # u, v, highway
    (1, 2, "secondary"), (2, 3, "secondary"), (3, 4, "secondary"), (4, 5, "secondary"),
    (6, 11, "residential"), (11, 7, "residential"), (7, 3, "residential"),
    (9, 4, "cycleway"), (4, 10, "cycleway"),
    (12, 11, "footway"), (11, 13, "footway"),
    (50, 51, "footway"), (51, 52, "footway"),
]


def crossroads():
    return codes_for(NODES, LINKS, [2, 4, 7, 11], [2, 7])


def test_driving_through_a_junction_pays_once():
    c = crossroads()
    assert route(c, 1, 2, 3, 4, 5) == [JUNCTION]  # stop line, centre, far crossing
    assert route(c, 5, 4, 3, 2, 1) == [JUNCTION]
    assert route(c, 6, 11, 7, 3, 4, 5) == [JUNCTION]  # in from the side road
    assert c[(1, 2)]["entry"] == JUNCTION and c[(4, 5)]["entry"] == NONE


def test_crossing_on_a_path_pays_once_either_way():
    c = crossroads()
    assert route(c, 9, 4, 10) == [JUNCTION] and route(c, 10, 4, 9) == [JUNCTION]
    assert route(c, 12, 11, 13) == [JUNCTION] and route(c, 13, 11, 12) == [JUNCTION]
    # Paid on the way onto the crossing, not off it.
    assert c[(9, 4)]["cross_end"] == JUNCTION and c[(4, 10)]["cross_end"] == NONE


def test_a_right_turn_on_the_sidewalk_is_free():
    """Inside the junction, but crossing nothing and on no road."""
    c = crossroads()
    assert route(c, 50, 51, 52) == [] and route(c, 52, 51, 50) == []


# A divided main road: carriageways at y = +6 and y = -6, each with a
# stop-line light; a side road from the north joins at 30. A path crosses both
# carriageways east of the junction, with a signal node on each (31 and 32).
# A cycleway runs along the north side and reaches the side road at 35, inside
# the junction; a service road ends on the southern carriageway at 41.
DIVIDED = {
    1: (-200, 6), 2: (-10, 6), 30: (0, 6), 31: (20, 6), 3: (200, 6),
    4: (200, -6), 41: (10, -6), 5: (15, -6), 32: (20, -6), 6: (-200, -6),
    7: (0, 200), 35: (0, 20),
    33: (20, 40), 34: (20, -40),
    36: (150, 20), 37: (60, 20),
    42: (10, -200), 43: (10, -25),
}
DIVIDED_LINKS = [
    (3, 31, "primary"), (31, 30, "primary"), (30, 2, "primary"), (2, 1, "primary"),
    (6, 5, "primary"), (5, 41, "primary"), (41, 32, "primary"), (32, 4, "primary"),
    (7, 35, "residential"), (35, 30, "residential"),
    (33, 31, "footway"), (31, 32, "footway"), (32, 34, "footway"),
    (36, 37, "cycleway"), (37, 35, "cycleway"),
    (42, 43, "service"), (43, 41, "service"),
]


def divided(crossings=()):
    return codes_for(DIVIDED, DIVIDED_LINKS, [2, 5, 31, 32], [2, 5], crossings)


def test_a_crossing_over_a_divided_road_is_one_wait():
    c = divided()
    assert route(c, 33, 31, 32, 34) == [JUNCTION]
    assert route(c, 34, 32, 31, 33) == [JUNCTION]


def test_joining_the_road_inside_the_junction_pays_as_the_road_does():
    c = divided()
    # Cycleway in, then the side road through: the same one wait as the side road.
    assert route(c, 36, 37, 35, 30, 2, 1) == [JUNCTION]
    assert route(c, 7, 35, 30, 2, 1) == [JUNCTION]
    # Onto the crossing, then off it onto the main road: one wait too.
    assert route(c, 33, 31, 30, 2, 1) == [JUNCTION]
    # A service road reaching the main road inside the junction.
    assert route(c, 42, 43, 41, 32, 4) == [JUNCTION]


def test_leaving_the_road_inside_the_junction_is_free():
    c = divided()
    assert route(c, 3, 31, 30, 35, 37, 36) == [JUNCTION]  # paid entering on the road


def test_a_path_reaching_the_road_at_an_unsignalled_crossing_crosses_it():
    c = divided(crossings=[35])
    assert c[(37, 35)]["entry"] == NONE


# A divided road with a stop-line light on each carriageway, 2 and 5. A
# cycleway 24-21-22-25 crosses both carriageways at untagged nodes, inside the
# junction. Another, 27-26-28-29-30, has a signal node of its own at 26 (the
# crossing of something outside the network) before it reaches the road.
ACROSS = {
    1: (-200, 6), 2: (-10, 6), 21: (5, 6), 28: (25, 6), 3: (200, 6),
    4: (200, -6), 29: (25, -6), 5: (10, -6), 22: (5, -6), 6: (-200, -6),
    24: (5, 60), 25: (5, -60), 27: (25, 60), 26: (25, 15), 30: (25, -60),
}
ACROSS_LINKS = [
    (1, 2, "secondary"), (2, 21, "secondary"), (21, 28, "secondary"), (28, 3, "secondary"),
    (4, 29, "secondary"), (29, 5, "secondary"), (5, 22, "secondary"), (22, 6, "secondary"),
    (24, 21, "cycleway"), (21, 22, "cycleway"), (22, 25, "cycleway"),
    (27, 26, "cycleway"), (26, 28, "cycleway"), (28, 29, "cycleway"), (29, 30, "cycleway"),
]


def across(crossings=()):
    return codes_for(ACROSS, ACROSS_LINKS, [2, 5, 26], [2, 5], crossings)


def test_a_path_over_untagged_carriageway_nodes_pays_once():
    c = across()
    assert route(c, 24, 21, 22, 25) == [JUNCTION]
    assert route(c, 25, 22, 21, 24) == [JUNCTION]


def test_a_path_off_a_signalled_crossing_onto_the_road_pays_once():
    c = across()
    assert route(c, 27, 26, 28, 29, 30) == [JUNCTION]
    assert route(c, 30, 29, 28, 26, 27) == [JUNCTION]


def test_a_zebra_is_no_free_way_onto_the_junction_roads():
    c = across(crossings=[21])
    assert route(c, 24, 21, 22, 25) == [JUNCTION]


# Mid-block pedestrian lights, far from any junction. Road 20-21-22 with a
# light at 21, crossed by path 23-21-24; a sidewalk 25-26-27 runs past. A
# divided road, carriageways at y = 506 and 494, has a crossing with a signal
# node on each, 61 and 62, crossed by path 63-61-62-64.
MIDBLOCK_NODES = {
    20: (0, 1000), 21: (100, 1000), 22: (200, 1000), 23: (100, 980), 24: (100, 1040),
    25: (0, 1020), 26: (100, 1020), 27: (200, 1020),
    65: (-200, 506), 61: (0, 506), 66: (200, 506),
    67: (200, 494), 62: (0, 494), 68: (-200, 494),
    63: (0, 540), 64: (0, 460),
}
MIDBLOCK_LINKS = [
    (20, 21, "residential"), (21, 22, "residential"),
    (23, 21, "cycleway"), (21, 24, "cycleway"),
    (25, 26, "footway"), (26, 27, "footway"),
    (65, 61, "secondary"), (61, 66, "secondary"),
    (67, 62, "secondary"), (62, 68, "secondary"),
    (63, 61, "footway"), (61, 62, "footway"), (62, 64, "footway"),
]


def midblock():
    return codes_for(MIDBLOCK_NODES, MIDBLOCK_LINKS, [21, 61, 62], [])


def test_a_mid_block_light_stops_road_and_crossing_once_each():
    c = midblock()
    assert route(c, 20, 21, 22) == [MIDBLOCK] and route(c, 22, 21, 20) == [MIDBLOCK]
    assert route(c, 23, 21, 24) == [MIDBLOCK] and route(c, 24, 21, 23) == [MIDBLOCK]


def test_a_mid_block_crossing_over_a_divided_road_is_one_wait():
    c = midblock()
    assert route(c, 65, 61, 66) == [MIDBLOCK] and route(c, 67, 62, 68) == [MIDBLOCK]
    assert route(c, 63, 61, 62, 64) == [MIDBLOCK] and route(c, 64, 62, 61, 63) == [MIDBLOCK]


def test_walking_past_a_mid_block_light_is_free():
    """Its inside is the signal node itself: the sidewalk 20 m away is outside."""
    c = midblock()
    assert route(c, 25, 26, 27) == [] and route(c, 27, 26, 25) == []


def test_waits_by_code():
    waits = signal_waits()
    assert waits[NONE] == 0
    assert waits[JUNCTION] == links_module.SIGNAL_WAIT_S
    assert waits[MIDBLOCK] == links_module.MIDBLOCK_SIGNAL_WAIT_S


def test_codes_become_seconds_in_the_direction_that_pays(monkeypatch):
    row = {"link_id": 0, "u": 1, "v": 2, "coords": [(24.0, 60.0), (24.002, 60.0)],
           "highway": "cycleway", "signal_entry": NONE, "signal_entry_reverse": NONE,
           "signal_cross_start": NONE, "signal_cross_end": NONE}
    plain = normalise(links_frame([row]), "bike")
    stepping_on = normalise(links_frame([{**row, "signal_cross_end": JUNCTION}]), "bike")
    wait = links_module.SIGNAL_WAIT_S
    assert stepping_on["travel_time_s"].iloc[0] - plain["travel_time_s"].iloc[0] == pytest.approx(wait)
    assert stepping_on["signal_wait_s"].iloc[0] == pytest.approx(wait)
    assert stepping_on["travel_time_reverse_s"].iloc[0] == pytest.approx(plain["travel_time_reverse_s"].iloc[0])

    joining = normalise(links_frame([{**row, "signal_entry_reverse": MIDBLOCK}]), "bike")
    assert joining["signal_wait_reverse_s"].iloc[0] == pytest.approx(links_module.MIDBLOCK_SIGNAL_WAIT_S)
    assert joining["signal_wait_s"].iloc[0] == 0.0

    # The constants are read when times are computed, so calibration can vary them.
    monkeypatch.setattr(links_module, "SIGNAL_WAIT_S", 0.0)
    stepping_on = normalise(links_frame([{**row, "signal_cross_end": JUNCTION}]), "bike")
    assert stepping_on["travel_time_s"].iloc[0] == pytest.approx(plain["travel_time_s"].iloc[0])


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
    assert links.loc[(1, 2), "signal_entry"] == JUNCTION
    assert links.loc[(3, 4), "signal_entry"] == NONE
    assert links.loc[(6, 7), "signal_entry"] == JUNCTION
    assert links.loc[(9, 4), "signal_cross_end"] == JUNCTION
    assert links.loc[(9, 4), "signal_entry"] == NONE
    assert links.loc[(4, 10), "signal_cross_end"] == NONE


# A side street crossed by a cycleway at an unsignalled crossing inside a
# signalled junction -- Sallinkatu/Mannerheimintie. Main road 1-2-3-4-5 with
# stop-line lights at 2 and 4; side road 6-7-3 with the crossing node 7 at
# (0, 20), 25 m from the light at 2; cycleway 8-7-9 crosses the side road at 7.
SIDE = {
    1: (-200, 0), 2: (-15, 0), 3: (0, 0), 4: (15, 0), 5: (200, 0),
    6: (0, 200), 7: (0, 20), 8: (-60, 20), 9: (60, 20),
}
SIDE_LINKS = [
    (1, 2, "secondary"), (2, 3, "secondary"), (3, 4, "secondary"), (4, 5, "secondary"),
    (6, 7, "residential"), (7, 3, "residential"),
    (8, 7, "cycleway"), (7, 9, "cycleway"),
]


def split_side():
    """The SIDE junction after the split, as codes, plus the path-side twin of 7."""
    from valma_bike_and_walk.signals import CROSSING_ENTRY_HIGHWAY, split_crossing_entries

    u = np.array([a for a, _, _ in SIDE_LINKS])
    v = np.array([b for _, b, _ in SIDE_LINKS])
    highway = np.array([h for _, _, h in SIDE_LINKS], dtype=object)
    ids = np.array(sorted(SIDE))
    split = split_crossing_entries(
        u, v, highway, np.isin(highway, list(links_module.PATH_HIGHWAYS)), ids,
        np.array([SIDE[i][0] for i in ids], float), np.array([SIDE[i][1] for i in ids], float),
        np.array([2, 4]), np.array([2, 4]), np.array([7]),
    )
    assert split.count == 1 and split.connector_to.tolist() == [7]
    twin = int(split.connector_from[0])
    nodes = {**SIDE, twin: SIDE[7]}
    links = [(int(a), int(b), h) for a, b, h in zip(split.u, split.v, highway)]
    links.append((twin, 7, CROSSING_ENTRY_HIGHWAY))
    return codes_for(nodes, links, [2, 4], [2, 4], [7]), twin


def test_the_paths_at_an_unsignalled_crossing_in_a_junction_get_their_own_node():
    _, twin = split_side()
    assert twin < 0


def test_crossing_the_side_street_on_the_path_stays_free():
    c, twin = split_side()
    assert route(c, 8, twin, 9) == [] and route(c, 9, twin, 8) == []


def test_stepping_off_the_path_onto_the_junction_roads_pays_once():
    c, twin = split_side()
    assert route(c, 8, twin, 7, 3, 4, 5) == [JUNCTION]
    assert route(c, 9, twin, 7, 3, 2, 1) == [JUNCTION]


def test_leaving_the_road_onto_the_path_pays_only_the_road_entry():
    c, twin = split_side()
    assert route(c, 6, 7, twin, 9) == [JUNCTION]  # the side road entering
    assert route(c, 1, 2, 3, 7, twin, 8) == [JUNCTION]


def test_a_crossing_outside_any_junction_is_not_split():
    from valma_bike_and_walk.signals import split_crossing_entries

    u, v = np.array([1, 2, 10]), np.array([2, 3, 2])
    highway = np.array(["residential", "residential", "cycleway"], dtype=object)
    split = split_crossing_entries(
        u, v, highway, np.array([False, False, True]), np.array([1, 2, 3, 10]),
        np.array([0.0, 100.0, 200.0, 100.0]), np.array([0.0, 0.0, 0.0, 50.0]),
        np.array([3]), np.array([3]), np.array([2]),
    )
    assert split.count == 0 and split.u.tolist() == [1, 2, 10]


def test_extract_splits_the_crossing_and_writes_the_connector(tmp_path):
    from valma_bike_and_walk.config import Settings
    from valma_bike_and_walk.osm import read_links
    from valma_bike_and_walk.signals import CROSSING_ENTRY_HIGHWAY

    def lonlat(xy):
        return (24.0 + xy[0] / 55_800.0, 60.0 + xy[1] / 111_300.0)

    nodes = {i: lonlat(SIDE[i]) for i in SIDE}
    ways = [
        (1, [1, 2, 3, 4, 5], {"highway": "secondary"}),
        (2, [6, 7, 3], {"highway": "residential"}),
        (3, [8, 7, 9], {"highway": "cycleway"}),
    ]
    tags = {2: {"highway": "traffic_signals"}, 4: {"highway": "traffic_signals"},
            7: {"highway": "crossing", "crossing": "unmarked"}}
    pbf = write_pbf(tmp_path / "side.osm.pbf", nodes, ways, node_tags=tags)
    settings = Settings(pbf_path=pbf, cache_dir=tmp_path / "cache", output_dir=tmp_path)
    links = read_links(settings, "bike")

    connector = links[links["highway"] == CROSSING_ENTRY_HIGHWAY]
    assert len(connector) == 1
    twin = int(connector["u"].iloc[0])
    assert twin < 0 and int(connector["v"].iloc[0]) == 7
    assert connector["signal_entry"].iloc[0] == JUNCTION       # path -> road pays
    assert connector["signal_entry_reverse"].iloc[0] == NONE   # road -> path is free
    cycle = links[links["highway"] == "cycleway"]
    assert set(cycle["u"]).union(cycle["v"]) == {8, 9, twin}
    assert links["link_id"].is_unique
    # The whole layer still normalises and routes.
    normalised = normalise(links, "bike")
    assert np.isfinite(normalised["travel_time_s"]).all()
