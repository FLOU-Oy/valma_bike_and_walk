"""Distance skims: how far the route the travel time chose actually is.

The one thing worth stating twice, because every test here turns on it: the
distance is measured *along the time-optimal path*, not along the shortest path
by length. For a bike network those are routinely different routes, and a model
given a time from one and a distance from the other would infer a speed nobody
rides at.
"""

import dataclasses

import numpy as np
import pytest

from valma_bike_and_walk.cli import main
from valma_bike_and_walk.matrix import (
    multipoint_travel_time_matrix,
    summarise_metres,
    travel_time_matrix,
    zone_travel_time_matrix,
)

from .conftest import make_network, make_zones, write_pbf

#: Five nodes in a row, 10 s and 100 m between neighbours both ways. Unlike the
#: default fixture, length and time are deliberately different numbers here, so
#: a test cannot pass by reporting one where it means the other.
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
LINE_METRES = [100.0] * len(LINE)


@pytest.fixture
def net():
    return make_network(LINE, 5, lengths=LINE_METRES)


# --------------------------------------------------------------------------
# Point to point
# --------------------------------------------------------------------------


def detour_network():
    """
    Two ways from 0 to 2: slow and short, or quick and long.

    0 -1-> 1 -1-> 2  costs 20 s over 100 m -- the shortest route by length.
    0 -3-> 3 -3-> 2  costs  4 s over 1000 m -- the quickest, and the one a
    travel-time search takes. A distance skim has to report 1000 m.
    """
    return make_network(
        [(0, 1, 10.0), (1, 2, 10.0), (0, 3, 2.0), (3, 2, 2.0)],
        4,
        lengths=[50.0, 50.0, 500.0, 500.0],
    )


def test_distance_follows_the_quickest_route_not_the_shortest_one():
    net = detour_network()
    seconds, metres = travel_time_matrix(net, [0], [2], return_distance=True)

    assert seconds[0, 0] == pytest.approx(4.0)
    assert metres[0, 0] == pytest.approx(1000.0)


def test_asking_for_distance_does_not_change_the_times():
    net = detour_network()
    plain = travel_time_matrix(net, [0, 1, 2, 3])
    seconds, _ = travel_time_matrix(net, [0, 1, 2, 3], return_distance=True)

    np.testing.assert_array_equal(plain, seconds)


def test_distance_sums_the_edges_of_a_multi_hop_path(net):
    seconds, metres = travel_time_matrix(net, [0], [3], return_distance=True)

    assert seconds[0, 0] == pytest.approx(30.0)
    assert metres[0, 0] == pytest.approx(300.0)


def test_unreachable_pairs_are_infinite_in_both(net):
    broken = make_network([(0, 1, 10.0)], 3, lengths=[70.0])
    seconds, metres = travel_time_matrix(broken, [0], [1, 2], return_distance=True)

    assert seconds[0, 0] == pytest.approx(10.0) and metres[0, 0] == pytest.approx(70.0)
    assert np.isinf(seconds[0, 1]) and np.isinf(metres[0, 1])


def test_a_pair_cut_off_by_max_seconds_has_no_distance_either():
    net = make_network([(0, 1, 10.0), (1, 2, 10.0)], 3, lengths=[100.0, 100.0])

    near, near_m = travel_time_matrix(
        net, [0], [2], max_seconds=100, return_distance=True
    )
    far, far_m = travel_time_matrix(net, [0], [2], max_seconds=15, return_distance=True)

    assert near[0, 0] == pytest.approx(20.0) and near_m[0, 0] == pytest.approx(200.0)
    assert np.isinf(far[0, 0]) and np.isinf(far_m[0, 0])


def test_a_zero_length_trip_to_itself_is_zero(net):
    _, metres = travel_time_matrix(net, [2], [2], return_distance=True)
    assert metres[0, 0] == pytest.approx(0.0)


def test_chunking_and_workers_do_not_change_the_distances():
    rng = np.random.default_rng(0)
    n = 60
    edges = [(int(i), int(i + 1), float(rng.uniform(1, 10))) for i in range(n - 1)]
    edges += [(int(i + 1), int(i), float(rng.uniform(1, 10))) for i in range(n - 1)]
    lengths = rng.uniform(5, 500, size=len(edges))
    net = make_network(edges, n, lengths=lengths)
    nodes = np.arange(0, n, 7, dtype=np.int32)

    one = travel_time_matrix(net, nodes, chunk_size=len(nodes), return_distance=True)
    many = travel_time_matrix(net, nodes, chunk_size=2, workers=3, return_distance=True)

    np.testing.assert_allclose(one[0], many[0])
    np.testing.assert_allclose(one[1], many[1])


def test_distances_match_a_scipy_predecessor_walk():
    """An independent oracle: SciPy's tree, walked by hand, summing lengths."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    from scipy.sparse.csgraph import dijkstra

    rng = np.random.default_rng(7)
    n = 40
    edges: list[tuple[int, int, float]] = []
    for i in range(n - 1):
        edges.append((i, i + 1, float(rng.uniform(1, 10))))
        edges.append((i + 1, i, float(rng.uniform(1, 10))))
    # A handful of long-range shortcuts, so the quickest route is not simply
    # the one with fewest hops and the two measures genuinely come apart.
    for _ in range(25):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            edges.append((int(u), int(v), float(rng.uniform(1, 30))))
    lengths = rng.uniform(1, 1000, size=len(edges))
    net = make_network(edges, n, lengths=lengths)

    sources = np.arange(0, n, 5, dtype=np.int64)
    seconds, metres = travel_time_matrix(
        net, sources, np.arange(n), return_distance=True, dtype=np.float64
    )

    graph = scipy_sparse.csr_matrix(
        (net.travel_time, net.indices, net.indptr), shape=(n, n)
    )
    edge_length = {}
    for tail in range(n):
        for k in range(net.indptr[tail], net.indptr[tail + 1]):
            edge_length[(tail, int(net.indices[k]))] = float(net.length[k])

    for row, source in enumerate(sources):
        dist, pred = dijkstra(graph, indices=int(source), return_predecessors=True)
        for target in range(n):
            if not np.isfinite(dist[target]):
                assert np.isinf(metres[row, target])
                continue
            total, node = 0.0, target
            while node != source:
                parent = int(pred[node])
                total += edge_length[(parent, node)]
                node = parent
            assert seconds[row, target] == pytest.approx(dist[target])
            assert metres[row, target] == pytest.approx(total)


# --------------------------------------------------------------------------
# Zone to zone
# --------------------------------------------------------------------------


def test_zone_distance_is_the_weighted_mean_over_the_same_pairs(net):
    # Zone A holds nodes 0 and 1, zone B holds nodes 3 and 4: the four crossing
    # distances are 300, 400, 200 and 300 m, so an even average is 300.
    zones = make_zones([[0, 1], [3, 4]], network=net)
    seconds, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=False, return_distance=True
    )

    assert seconds[0, 1] == pytest.approx(30.0)
    assert metres[0, 1] == pytest.approx(300.0)


def test_point_weights_tilt_the_distance_average_too(net):
    zones = make_zones([[0, 1], [3, 4]], [[0.25, 0.75], [0.5, 0.5]], network=net)
    _, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=False, return_distance=True
    )

    # 0.25*0.5*300 + 0.25*0.5*400 + 0.75*0.5*200 + 0.75*0.5*300
    assert metres[0, 1] == pytest.approx(275.0)


def test_access_legs_are_charged_in_metres(net):
    """The snap gap is part of the trip, and a distance matrix pays for it."""
    zones = make_zones(
        [[0], [3]],
        access_per_zone=[[5.0], [1.0]],
        access_metres_per_zone=[[70.0], [14.0]],
        network=net,
    )
    seconds, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=False, return_distance=True
    )

    assert seconds[0, 1] == pytest.approx(30.0 + 5.0 + 1.0)
    assert metres[0, 1] == pytest.approx(300.0 + 70.0 + 14.0)


def test_the_intrazonal_diagonal_drops_self_pairs_in_metres_as_well(net):
    zones = make_zones([[0, 1], [3, 4]], network=net)
    _, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=False, return_distance=True
    )

    # The only distinct pairs inside zone A are 0->1 and 1->0, both 100 m. With
    # the a==b "trips" left in, the mean would be halved to 50.
    assert metres[0, 0] == pytest.approx(100.0)
    assert metres[1, 1] == pytest.approx(100.0)


def test_keeping_self_pairs_halves_the_intrazonal_distance(net):
    zones = make_zones([[0, 1], [3, 4]], network=net)
    _, metres = multipoint_travel_time_matrix(
        net, zones, exclude_self_pairs=False, return_distance=True
    )

    assert metres[0, 0] == pytest.approx(50.0)


def test_an_unreachable_cell_is_infinite_in_both_matrices():
    net = make_network(
        [(0, 1, 10.0), (1, 0, 10.0), (2, 3, 10.0), (3, 2, 10.0)],
        4,
        lengths=[100.0] * 4,
    )
    zones = make_zones([[0, 1], [2, 3]], network=net)
    seconds, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=False, return_distance=True
    )

    assert np.isinf(seconds[0, 1]) and np.isinf(metres[0, 1])
    assert np.isfinite(seconds[0, 0]) and np.isfinite(metres[0, 0])


def test_the_two_tiers_agree_on_which_tier_a_cell_came_from(net):
    """
    Whichever tier resolved the time resolved the distance: one journey, two
    measures. Zone A and zone C are 40 s apart, past a 25 s near cutoff, so
    that pair falls through to the single representative point in both.
    """
    zones = make_zones([[0, 1], [3, 4]], network=net)
    seconds, metres = zone_travel_time_matrix(
        net, zones, near_seconds=25.0, return_distance=True
    )

    # Far tier: representative points are nodes 0 and 3, 30 s and 300 m apart.
    assert seconds[0, 1] == pytest.approx(30.0)
    assert metres[0, 1] == pytest.approx(300.0)
    # Near tier still resolves the diagonal, from inside each zone.
    assert metres[0, 0] == pytest.approx(100.0)


def test_the_far_tier_charges_the_representative_access_leg_in_metres(net):
    zones = dataclasses.replace(
        make_zones([[0, 1], [3, 4]], network=net),
        representative_access_seconds=np.array([2.0, 3.0]),
        representative_access_metres=np.array([28.0, 42.0]),
    )
    seconds, metres = zone_travel_time_matrix(
        net, zones, near_seconds=25.0, return_distance=True
    )

    assert seconds[0, 1] == pytest.approx(30.0 + 2.0 + 3.0)
    assert metres[0, 1] == pytest.approx(300.0 + 28.0 + 42.0)


def test_an_unroutable_diagonal_falls_back_to_the_equal_area_circle(net):
    """One access point has no distinct pair, so the disc estimate fills in."""
    area = np.pi * 500.0**2  # radius 500 m
    zones = make_zones([[0], [3]], area_m2=[area, area], network=net)
    seconds, metres = zone_travel_time_matrix(
        net, zones, intrazonal_fallback=True, return_distance=True
    )

    expected_m = 128.0 / (45.0 * np.pi) * 500.0
    assert metres[0, 0] == pytest.approx(expected_m, rel=1e-5)
    # And the time fallback is that same distance at the mode's base speed,
    # so the pair still divides into a speed the profile knows about.
    assert expected_m / seconds[0, 0] == pytest.approx(14.0 * 1000.0 / 3600.0, rel=1e-5)


def test_no_zones_returns_two_empty_matrices(net):
    seconds, metres = zone_travel_time_matrix(net, make_zones([]), return_distance=True)
    assert seconds.shape == (0, 0) and metres.shape == (0, 0)


def test_summarise_metres_reports_kilometres():
    matrix = np.array([[0.0, 1000.0], [3000.0, np.inf]])
    stats = summarise_metres(matrix)

    assert stats["reachable_fraction"] == pytest.approx(0.75)
    assert stats["min_km"] == pytest.approx(1.0)
    assert stats["max_km"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Through the CLI, and on to a model
# --------------------------------------------------------------------------


NODES = {
    1: (24.000, 60.000),
    2: (24.002, 60.000),
    3: (24.004, 60.000),
    4: (24.006, 60.000),
}
FOOTWAY = {"highway": "footway", "surface": "asphalt"}
WAYS = [(100, [1, 2], FOOTWAY), (101, [2, 3], FOOTWAY), (102, [3, 4], FOOTWAY)]


@pytest.fixture
def chain_pbf(tmp_path):
    return write_pbf(tmp_path / "chain.osm.pbf", NODES, WAYS)


def run_matrix(chain_pbf, tmp_path, *extra):
    out_dir, cache = tmp_path / "out", tmp_path / "cache"
    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\n101,24.000,60.000\n102,24.006,60.000\n")
    code = main(
        [
            "matrix",
            "--pbf", str(chain_pbf),
            "--mode", "walk",
            "--centroids", str(centroids),
            "--id-column", "id",
            "--output-dir", str(out_dir),
            "--cache-dir", str(cache),
            "--workers", "1",
            *extra,
        ]
    )  # fmt: skip
    assert code == 0
    return out_dir


def test_cli_without_distance_writes_times_only(chain_pbf, tmp_path):
    out_dir = run_matrix(chain_pbf, tmp_path)

    with np.load(out_dir / "travel_times_walk.npz") as data:
        assert "metres" not in data
    assert not (out_dir / "travel_distances_walk.omx").exists()


def test_cli_distance_writes_metres_beside_the_seconds(chain_pbf, tmp_path):
    out_dir = run_matrix(chain_pbf, tmp_path, "--distance")

    with np.load(out_dir / "travel_times_walk.npz") as data:
        seconds, metres = data["seconds"], data["metres"]

    # 0.006 degrees of longitude at 60 N is some 335 m of chain. What matters
    # here is that the two agree on one journey, at the walk profile's own pace.
    assert metres[0, 1] == pytest.approx(335.0, rel=0.02)
    assert metres[0, 1] / seconds[0, 1] == pytest.approx(
        4.8 * 1000.0 / 3600.0, rel=0.05
    )
    assert metres[0, 0] == pytest.approx(0.0)


def test_cli_distance_omx_is_in_kilometres(chain_pbf, tmp_path):
    omx = pytest.importorskip("openmatrix")
    out_dir = run_matrix(chain_pbf, tmp_path, "--distance")

    npz = np.load(out_dir / "travel_times_walk.npz")
    with omx.open_file(str(out_dir / "travel_distances_walk.omx")) as f:
        assert f["walk"].attrs["units"] == "kilometers"
        np.testing.assert_allclose(
            np.array(f["walk"]), npz["metres"] / 1000.0, rtol=1e-5
        )
    with omx.open_file(str(out_dir / "travel_times_walk.omx")) as f:
        assert f["walk"].attrs["units"] == "minutes"


def test_overlay_lays_distances_over_a_distance_baseline(chain_pbf, tmp_path):
    pytest.importorskip("openmatrix")
    from valma_bike_and_walk.demand import write_omx_matrix
    from valma_bike_and_walk.overlay import read_omx_dense

    out_dir = run_matrix(chain_pbf, tmp_path, "--distance")
    base_ids = np.array([101, 102, 103])
    baseline = tmp_path / "dist_vrk.omx"
    write_omx_matrix(baseline, base_ids, np.full((3, 3), 9.0), matrix_name="walk")

    out = out_dir / "skims_dist.omx"
    assert (
        main(
            [
                "overlay",
                "--mode", "walk",
                "--quantity", "distance",
                "--output-dir", str(out_dir),
                "--baseline", str(baseline),
                "--baseline-matrix", "walk",
                "--out", str(out),
            ]
        )  # fmt: skip
        == 0
    )

    ids, result = read_omx_dense(out, "walk")
    np.testing.assert_array_equal(ids, base_ids)
    routed = np.load(out_dir / "travel_times_walk.npz")["metres"]
    # Routed zones take the model's km; zone 103 was never routed, so it keeps
    # the baseline -- which is the whole point of overlaying rather than
    # replacing.
    assert result[0, 1] == pytest.approx(routed[0, 1] / 1000.0, rel=1e-5)
    assert result[0, 2] == pytest.approx(9.0)


def test_overlay_says_so_when_the_matrix_run_measured_no_distances(chain_pbf, tmp_path):
    pytest.importorskip("openmatrix")
    from valma_bike_and_walk.demand import write_omx_matrix

    out_dir = run_matrix(chain_pbf, tmp_path)  # no --distance
    baseline = tmp_path / "dist_vrk.omx"
    write_omx_matrix(
        baseline, np.array([101, 102]), np.full((2, 2), 9.0), matrix_name="walk"
    )

    with pytest.raises(ValueError, match="--distance"):
        main(
            [
                "overlay",
                "--mode", "walk",
                "--quantity", "distance",
                "--output-dir", str(out_dir),
                "--baseline", str(baseline),
                "--baseline-matrix", "walk",
                "--out", str(out_dir / "skims.omx"),
            ]
        )  # fmt: skip


def test_overlay_rejects_a_speed_conversion_on_a_distance_overlay(chain_pbf, tmp_path):
    pytest.importorskip("openmatrix")
    from valma_bike_and_walk.demand import write_omx_matrix

    out_dir = run_matrix(chain_pbf, tmp_path, "--distance")
    baseline = tmp_path / "dist_vrk.omx"
    write_omx_matrix(
        baseline, np.array([101, 102]), np.full((2, 2), 9.0), matrix_name="walk"
    )

    with pytest.raises(SystemExit, match="already in km"):
        main(
            [
                "overlay",
                "--mode", "walk",
                "--quantity", "distance",
                "--baseline-speed-kmh",
                "--output-dir", str(out_dir),
                "--baseline", str(baseline),
                "--baseline-matrix", "walk",
                "--out", str(out_dir / "skims.omx"),
            ]
        )  # fmt: skip


def test_cli_distance_through_zone_polygons(chain_pbf, tmp_path):
    """The --zones path, where access legs come from load_zones rather than a fixture."""
    import geopandas as gpd
    import shapely

    out_dir, cache = tmp_path / "out", tmp_path / "cache"
    zones = tmp_path / "zones.gpkg"
    gpd.GeoDataFrame(
        {"zone_id": [1, 2]},
        geometry=[
            shapely.box(23.9990, 59.9990, 24.0030, 60.0010),
            shapely.box(24.0030, 59.9990, 24.0070, 60.0010),
        ],
        crs="EPSG:4326",
    ).to_file(zones, layer="zones", driver="GPKG")

    assert (
        main(
            [
                "matrix",
                "--pbf", str(chain_pbf),
                "--mode", "walk",
                "--zones", str(zones),
                "--id-column", "zone_id",
                "--points-per-zone", "2",
                "--distance", "--workers", "1",
                "--output-dir", str(out_dir),
                "--cache-dir", str(cache),
            ]
        )  # fmt: skip
        == 0
    )

    with np.load(out_dir / "travel_times_walk.npz") as data:
        seconds, metres = data["seconds"], data["metres"]

    # Spread over the zone, so the diagonal is a real trip in both measures --
    # the headline reason for routing zones rather than centroids.
    assert (metres.diagonal() > 0).all()
    assert (metres > 0).all()
    # One journey: every cell divides into the walk profile's own pace.
    np.testing.assert_allclose(metres / seconds, 4.8 * 1000.0 / 3600.0, rtol=0.05)
