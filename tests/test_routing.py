"""Routing tests, including a check against NetworkX as an independent oracle."""

import numpy as np
import pytest

from valma_bike_and_walk.matrix import summarise, travel_time_matrix
from valma_bike_and_walk.network import RoutableNetwork

from .conftest import make_network


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


def test_summarise_ignores_unreachable():
    matrix = np.array([[0.0, 60.0], [np.inf, 0.0]], dtype=np.float32)
    stats = summarise(matrix)
    assert stats["reachable_fraction"] == pytest.approx(0.75)
    assert stats["median_minutes"] == pytest.approx(1.0)


def _random_network(seed: int, n: int = 120, m: int = 400):
    rng = np.random.default_rng(seed)
    edges = set()
    while len(edges) < m:
        u, v = rng.integers(0, n, size=2)
        if u != v:
            edges.add((int(u), int(v)))
    return make_network([(u, v, float(rng.uniform(1, 100))) for u, v in sorted(edges)], n)


def _scipy_graph(net):
    from scipy.sparse import csr_matrix

    return csr_matrix((net.travel_time, net.indices, net.indptr), shape=(net.n_nodes,) * 2)


def test_searcher_paths_are_shortest_and_connected():
    from scipy.sparse.csgraph import dijkstra

    from valma_bike_and_walk.routing import Searcher

    net = _random_network(1)
    searcher = Searcher(net.routing_graph())
    graph = searcher.graph
    reference = dijkstra(_scipy_graph(net), directed=True, indices=np.arange(0, 120, 13))
    for row, source in enumerate(range(0, 120, 13)):
        for target in range(0, 120, 7):
            edges, cost = searcher.path(source, target)
            assert cost == pytest.approx(reference[row, target])
            if not np.isfinite(cost) or source == target:
                assert edges.size == 0
                continue
            # A chain from source to target whose edges add up to the cost.
            assert graph.edge_tail[edges[0]] == source and graph.indices[edges[-1]] == target
            np.testing.assert_array_equal(graph.indices[edges[:-1]], graph.edge_tail[edges[1:]])
            assert net.travel_time[edges].sum() == pytest.approx(cost)


def test_searcher_costs_match_scipy_min_only_both_ways():
    from scipy.sparse.csgraph import dijkstra

    from valma_bike_and_walk.routing import Searcher

    net = _random_network(2)
    sources = np.array([3, 40, 77, 40])
    targets = np.arange(0, 120, 3)
    forward = Searcher(net.routing_graph())
    backward = Searcher(net.routing_graph().reversed())
    scipy_graph = _scipy_graph(net)
    for limit in (None, 60.0):
        bound = np.inf if limit is None else limit
        ahead = dijkstra(scipy_graph, indices=sources, min_only=True, limit=bound)
        behind = dijkstra(scipy_graph.T.tocsr(), indices=sources, min_only=True, limit=bound)
        np.testing.assert_allclose(forward.costs(sources, targets, limit), ahead[targets])
        np.testing.assert_allclose(backward.costs(sources, targets, limit), behind[targets])


def test_searcher_rejects_nodes_outside_the_graph():
    from valma_bike_and_walk.routing import Searcher

    searcher = Searcher(make_network([(0, 1, 1.0)], 2).routing_graph())
    with pytest.raises(ValueError, match="node indices"):
        searcher.path(0, 2)
    with pytest.raises(ValueError, match="node indices"):
        searcher.costs(np.array([-1]), np.array([0]))


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
