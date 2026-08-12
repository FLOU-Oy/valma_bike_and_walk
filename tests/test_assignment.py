"""Traffic assignment tests."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from valma_bike_and_walk.assignment import (
    _as_demand_csr,
    _edge_keys,
    _solve_chunk,
    assign_traffic,
    link_volume_frame,
    summarise,
)
from valma_bike_and_walk.network import RoutableNetwork, _build_csr


def make_network(edges, n_nodes, mode="bike"):
    """edges: list of (u, v, seconds). Node ids are just 0..n_nodes-1."""
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


def volume_of(network, link_volume, u, v):
    """Look up the assigned volume on edge u->v, or 0.0 if it isn't an edge."""
    frame = link_volume_frame(network, link_volume)
    match = (frame["u"] == u) & (frame["v"] == v)
    if not match.any():
        return 0.0
    return float(frame["volume"][match][0])


def test_demand_loads_onto_every_link_of_the_shortest_path():
    # 0->1->2->3 is the fast route; 0->2 direct is slow and should carry nothing.
    net = make_network([(0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0), (0, 2, 100.0)], 4)
    demand = np.zeros((1, 4))
    demand[0, 3] = 5.0

    volume = assign_traffic(net, [0], demand, targets=[0, 1, 2, 3])

    assert volume_of(net, volume, 0, 1) == pytest.approx(5.0)
    assert volume_of(net, volume, 1, 2) == pytest.approx(5.0)
    assert volume_of(net, volume, 2, 3) == pytest.approx(5.0)
    assert volume_of(net, volume, 0, 2) == pytest.approx(0.0)


def test_demand_from_shared_tail_sums_on_the_shared_link():
    # 0->1->3 and 2->1->3: both origins converge on node 1 then share the rest.
    net = make_network([(0, 1, 5.0), (2, 1, 5.0), (1, 3, 5.0)], 4)
    demand = np.zeros((2, 1))
    demand[0, 0] = 4.0  # 0 -> 3
    demand[1, 0] = 6.0  # 2 -> 3

    volume = assign_traffic(net, [0, 2], demand, targets=[3])

    assert volume_of(net, volume, 0, 1) == pytest.approx(4.0)
    assert volume_of(net, volume, 2, 1) == pytest.approx(6.0)
    assert volume_of(net, volume, 1, 3) == pytest.approx(10.0)


def test_diagonal_demand_is_ignored():
    net = make_network([(0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0)], 4)
    demand = np.zeros((4, 4))
    np.fill_diagonal(demand, 5.0)

    volume = assign_traffic(net, [0, 1, 2, 3], demand)

    assert volume.sum() == pytest.approx(0.0)


def test_unreachable_pairs_are_dropped_not_raised(caplog):
    net = make_network([(0, 1, 10.0), (2, 3, 10.0)], 4)
    demand = np.zeros((4, 4))
    demand[0, 3] = 5.0  # 0 and 3 are in different components

    with caplog.at_level("WARNING"):
        volume = assign_traffic(net, [0, 1, 2, 3], demand)

    assert volume.sum() == pytest.approx(0.0)
    assert any("dropped" in message for message in caplog.messages)


def test_negative_and_zero_demand_are_ignored():
    net = make_network([(0, 1, 10.0), (1, 2, 10.0)], 3)
    demand = np.array([[0.0, 0.0, -3.0]])

    volume = assign_traffic(net, [0], demand, targets=[0, 1, 2])

    assert volume.sum() == pytest.approx(0.0)


def test_sparse_and_dense_demand_agree():
    net = make_network([(0, 1, 10.0), (1, 2, 10.0), (0, 2, 100.0)], 3)
    dense = np.array([[0.0, 0.0, 7.0]])
    sparse = csr_matrix(dense)

    from_dense = assign_traffic(net, [0], dense, targets=[0, 1, 2])
    from_sparse = assign_traffic(net, [0], sparse, targets=[0, 1, 2])

    np.testing.assert_allclose(from_dense, from_sparse)


def test_multiple_workers_match_single_worker():
    rng = np.random.default_rng(0)
    n = 30
    edges = [(int(i), int(i + 1), float(rng.uniform(1, 10))) for i in range(n - 1)]
    edges += [(int(i + 1), int(i), float(rng.uniform(1, 10))) for i in range(n - 1)]
    net = make_network(edges, n)

    nodes = np.arange(0, n, 5, dtype=np.int64)
    demand = rng.uniform(0, 10, size=(nodes.size, nodes.size))
    np.fill_diagonal(demand, 0.0)

    one_worker = assign_traffic(net, nodes, demand, workers=1)
    two_workers = assign_traffic(net, nodes, demand, workers=2, chunk_size=2)

    np.testing.assert_allclose(one_worker, two_workers)


def test_many_small_chunks_match_one_big_chunk():
    # Regression test: chunk_size=1 forces one Dijkstra call (and, with
    # workers=2, one process-pool task) per source -- the shape of thing that
    # used to leak a full-network-sized array per chunk. Correctness across
    # many chunks is the observable half of that fix (the memory bound isn't
    # something a unit test can watch directly).
    rng = np.random.default_rng(1)
    n = 25
    edges = [(int(i), int(i + 1), float(rng.uniform(1, 10))) for i in range(n - 1)]
    edges += [(int(i + 1), int(i), float(rng.uniform(1, 10))) for i in range(n - 1)]
    net = make_network(edges, n)

    nodes = np.arange(0, n, 4, dtype=np.int64)
    demand = rng.uniform(0, 10, size=(nodes.size, nodes.size))
    np.fill_diagonal(demand, 0.0)

    one_chunk = assign_traffic(net, nodes, demand, workers=1, chunk_size=nodes.size)
    many_chunks_serial = assign_traffic(net, nodes, demand, workers=1, chunk_size=1)
    many_chunks_parallel = assign_traffic(net, nodes, demand, workers=2, chunk_size=1)

    np.testing.assert_allclose(one_chunk, many_chunks_serial)
    np.testing.assert_allclose(one_chunk, many_chunks_parallel)


def test_solve_chunk_returns_only_touched_edges():
    net = make_network([(0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0), (0, 2, 100.0)], 4)
    demand = _as_demand_csr(np.array([[0.0, 0.0, 0.0, 5.0]]), 1, 4)
    edge_keys = _edge_keys(net)

    idx, vals, dropped = _solve_chunk(
        net.csr(),
        np.array([0], dtype=np.int64),
        np.arange(4, dtype=np.int64),
        demand,
        0,
        edge_keys,
        net.n_nodes,
        None,
        net.n_edges,
    )

    # Only the 3 edges on the shortest path 0->1->2->3 were touched, not all
    # 4 edges in the network (the untouched 0->2 shortcut is excluded).
    assert idx.size == 3
    assert vals.size == 3
    np.testing.assert_allclose(vals, 5.0)
    assert dropped == 0


def test_max_seconds_drops_far_pairs():
    net = make_network([(0, 1, 10.0), (1, 2, 10.0)], 3)
    demand = np.array([[0.0, 0.0, 5.0]])

    reached = assign_traffic(net, [0], demand, targets=[0, 1, 2], max_seconds=100)
    assert reached.sum() == pytest.approx(10.0)  # 5 on each of the two links

    cutoff = assign_traffic(net, [0], demand, targets=[0, 1, 2], max_seconds=15)
    assert cutoff.sum() == pytest.approx(0.0)


def test_link_volume_frame_is_edge_aligned():
    net = make_network([(0, 1, 10.0), (1, 2, 10.0)], 3)
    volume = np.zeros(net.n_edges)
    frame = link_volume_frame(net, volume)

    assert set(frame) == {"u", "v", "length_m", "travel_time_s", "volume"}
    for array in frame.values():
        assert array.shape == (net.n_edges,)


def test_summarise_reports_loaded_links():
    volume = np.array([0.0, 5.0, 0.0, 15.0])
    stats = summarise(volume)
    assert stats["loaded_edges"] == pytest.approx(2.0)
    assert stats["total_volume"] == pytest.approx(20.0)
    assert stats["max_volume"] == pytest.approx(15.0)


def test_empty_sources_or_targets_returns_zeros():
    net = make_network([(0, 1, 10.0)], 2)
    volume = assign_traffic(net, np.array([], dtype=np.int64), np.zeros((0, 2)))
    assert volume.shape == (net.n_edges,)
    assert volume.sum() == pytest.approx(0.0)


def test_rejects_wrong_shaped_demand():
    net = make_network([(0, 1, 10.0)], 2)
    with pytest.raises(ValueError, match="expected"):
        assign_traffic(net, [0], np.zeros((1, 5)), targets=[0, 1])


def test_rejects_unsnapped_indices():
    net = make_network([(0, 1, 10.0)], 2)
    with pytest.raises(ValueError, match="non-negative"):
        assign_traffic(net, np.array([-1]), np.zeros((1, 2)), targets=[0, 1])
