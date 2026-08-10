"""Routing tests, including a check against NetworkX as an independent oracle."""

import numpy as np
import pytest

from valma_bike_and_walk.matrix import choose_chunk_size, summarise, travel_time_matrix
from valma_bike_and_walk.network import RoutableNetwork, _build_csr


def make_network(edges, n_nodes, mode="bike"):
    """edges: list of (u, v, seconds)."""
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    t = np.array([e[2] for e in edges], dtype=float)
    length = t.astype(np.float32)

    indptr, indices, travel_time, length, _geometry = _build_csr(
        u, v, t, length, n_nodes
    )
    return RoutableNetwork(
        mode=mode,
        node_ids=np.arange(n_nodes, dtype=np.int64),
        x=np.arange(n_nodes, dtype=float) * 100.0,
        y=np.zeros(n_nodes, dtype=float),
        indptr=indptr,
        indices=indices,
        travel_time=travel_time,
        length=length,
    )


def test_path_travel_time_is_sum_of_edges():
    net = make_network([(0, 1, 10.0), (1, 2, 20.0), (0, 2, 100.0)], 3)
    matrix = travel_time_matrix(net, [0], [2])
    assert matrix[0, 0] == pytest.approx(30.0)


def test_parallel_edges_keep_the_fastest():
    net = make_network([(0, 1, 50.0), (0, 1, 5.0)], 2)
    assert net.n_edges == 1
    assert travel_time_matrix(net, [0], [1])[0, 0] == pytest.approx(5.0)


def test_unreachable_pairs_are_infinite():
    net = make_network([(0, 1, 10.0), (2, 3, 10.0)], 4)
    assert np.isinf(travel_time_matrix(net, [0], [3])[0, 0])


def test_direction_is_respected():
    net = make_network([(0, 1, 10.0)], 2)
    assert travel_time_matrix(net, [0], [1])[0, 0] == pytest.approx(10.0)
    assert np.isinf(travel_time_matrix(net, [1], [0])[0, 0])


def test_max_seconds_cuts_off_distant_pairs():
    net = make_network([(0, 1, 10.0), (1, 2, 10.0)], 3)
    assert travel_time_matrix(net, [0], [2], max_seconds=100)[0, 0] == pytest.approx(
        20.0
    )
    assert np.isinf(travel_time_matrix(net, [0], [2], max_seconds=15)[0, 0])


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(0)
    n = 60
    edges = [(int(i), int(i + 1), float(rng.uniform(1, 10))) for i in range(n - 1)]
    edges += [(int(i + 1), int(i), float(rng.uniform(1, 10))) for i in range(n - 1)]
    net = make_network(edges, n)

    nodes = np.arange(0, n, 7, dtype=np.int32)
    one_chunk = travel_time_matrix(net, nodes, chunk_size=len(nodes))
    many_chunks = travel_time_matrix(net, nodes, chunk_size=2)
    np.testing.assert_allclose(one_chunk, many_chunks)


def test_nearest_nodes_respects_max_distance():
    net = make_network([(0, 1, 1.0)], 3)
    idx, dist = net.nearest_nodes(np.array([0.0]), np.array([0.0]))
    assert idx[0] == 0 and dist[0] == pytest.approx(0.0)

    idx, _ = net.nearest_nodes(np.array([10_000.0]), np.array([0.0]), max_distance=50.0)
    assert idx[0] == -1


def test_save_and_load_round_trip(tmp_path):
    net = make_network([(0, 1, 10.0), (1, 2, 20.0)], 3)
    path = net.save(tmp_path / "net.npz")
    loaded = RoutableNetwork.load(path)

    assert loaded.mode == net.mode
    assert loaded.n_nodes == net.n_nodes
    np.testing.assert_array_equal(loaded.indptr, net.indptr)
    np.testing.assert_allclose(
        travel_time_matrix(loaded, [0], [2]), travel_time_matrix(net, [0], [2])
    )


def test_rejects_unsnapped_indices():
    net = make_network([(0, 1, 10.0)], 2)
    with pytest.raises(ValueError, match="non-negative"):
        travel_time_matrix(net, np.array([-1]), np.array([1]))


def test_choose_chunk_size_respects_budget():
    # 1e6 nodes at float64 = 8 MB per source, so a 64 MB budget allows 8.
    assert choose_chunk_size(1_000_000, budget_bytes=64 * 1024**2) == 8
    assert choose_chunk_size(10**9, budget_bytes=1024) >= 1


def test_summarise_ignores_unreachable():
    matrix = np.array([[0.0, 60.0], [np.inf, 0.0]], dtype=np.float32)
    stats = summarise(matrix)
    assert stats["reachable_fraction"] == pytest.approx(0.75)
    assert stats["median_minutes"] == pytest.approx(1.0)


def test_matches_networkx_on_a_random_graph():
    """SciPy CSR routing must agree with NetworkX, our reference implementation."""
    nx = pytest.importorskip("networkx")

    rng = np.random.default_rng(42)
    n = 120
    edges = set()
    while len(edges) < 400:
        u, v = rng.integers(0, n, size=2)
        if u != v:
            edges.add((int(u), int(v)))
    weighted = [(u, v, float(rng.uniform(1, 100))) for u, v in sorted(edges)]

    net = make_network(weighted, n)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v, t in weighted:
        if G.has_edge(u, v):
            G[u][v]["travel_time"] = min(G[u][v]["travel_time"], t)
        else:
            G.add_edge(u, v, travel_time=t)

    sources = np.arange(0, n, 11, dtype=np.int32)
    ours = travel_time_matrix(
        net, sources, np.arange(n, dtype=np.int32), dtype=np.float64
    )

    for row, source in enumerate(sources):
        reference = nx.single_source_dijkstra_path_length(
            G, int(source), weight="travel_time"
        )
        for target in range(n):
            expected = reference.get(target, np.inf)
            assert ours[row, target] == pytest.approx(expected), f"{source}->{target}"
