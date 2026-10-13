"""The Numba routing against the frozen SciPy implementation it replaced.

Edge weights are drawn from a continuous distribution, so shortest paths are
unique and the two implementations have no ties to break differently: every
matrix cell and every link volume has exactly one right answer.
"""

import numpy as np
import pytest
from scipy.sparse import random as sparse_random

from valma_bike_and_walk.assignment import assign_traffic, assign_zone_traffic
from valma_bike_and_walk.matrix import travel_time_matrix, zone_travel_time_matrix
from valma_bike_and_walk.network import (
    FORMAT_VERSION,
    NetworkFormatError,
    RoutableNetwork,
    hilbert_order,
)

from .conftest import make_network, make_zones
from .reference import scipy_assignment, scipy_matrix


def random_network(n: int, seed: int) -> RoutableNetwork:
    """A sparse random digraph over a ring, so most of it is reachable."""
    rng = np.random.default_rng(seed)
    ring = [(i, (i + 1) % n) for i in range(n)] + [((i + 1) % n, i) for i in range(n)]
    extra = sparse_random(n, n, density=3.0 / n, random_state=seed, format="coo")
    pairs = ring + [(int(u), int(v)) for u, v in zip(extra.row, extra.col) if u != v]
    edges = [(u, v, float(rng.uniform(1.0, 60.0))) for u, v in pairs]
    return make_network(edges, n)


def random_zones(network: RoutableNetwork, n_zones: int, seed: int):
    rng = np.random.default_rng(seed)
    nodes = rng.permutation(network.n_nodes)
    counts = rng.integers(1, 6, size=n_zones)
    start = 0
    nodes_per_zone, weights_per_zone, access_per_zone = [], [], []
    for count in counts:
        zone_nodes = nodes[start : start + count].tolist()
        start += count
        w = rng.uniform(0.1, 1.0, size=count)
        nodes_per_zone.append(zone_nodes)
        weights_per_zone.append((w / w.sum()).tolist())
        access_per_zone.append(rng.uniform(0, 30, size=count).tolist())
    return make_zones(
        nodes_per_zone,
        weights_per_zone,
        access_per_zone,
        area_m2=rng.uniform(1e4, 1e6, size=n_zones).tolist(),
        network=network,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("max_seconds", [None, 120.0])
def test_point_matrix_matches_reference(seed, max_seconds):
    net = random_network(400, seed)
    rng = np.random.default_rng(seed)
    sources = rng.choice(net.n_nodes, 60, replace=False)
    targets = rng.choice(net.n_nodes, 80, replace=False)

    ours = travel_time_matrix(
        net, sources, targets, max_seconds=max_seconds, workers=3, chunk_size=7
    )
    theirs = scipy_matrix.travel_time_matrix(
        net, sources, targets, max_seconds=max_seconds
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"max_seconds": 600.0, "near_seconds": 150.0},
        {"decay_mu": 0.01},
    ],
)
def test_zone_matrix_matches_reference(seed, kwargs):
    net = random_network(300, seed)
    zones = random_zones(net, 40, seed)

    ours = zone_travel_time_matrix(net, zones, workers=3, chunk_size=5, **kwargs)
    theirs = scipy_matrix.zone_travel_time_matrix(net, zones, **kwargs)
    np.testing.assert_allclose(ours, theirs, rtol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("max_seconds", [None, 150.0])
def test_assignment_matches_reference(seed, max_seconds):
    net = random_network(400, seed)
    rng = np.random.default_rng(seed)
    nodes = rng.choice(net.n_nodes, 50, replace=False)
    demand = rng.uniform(0, 10, size=(50, 50)) * (rng.uniform(size=(50, 50)) < 0.3)

    ours = assign_traffic(
        net, nodes, demand, max_seconds=max_seconds, workers=3, chunk_size=4
    )
    theirs = scipy_assignment.assign_traffic(
        net, nodes, demand, max_seconds=max_seconds
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("near_seconds", [None, 150.0])
def test_zone_assignment_matches_reference(seed, near_seconds):
    net = random_network(300, seed)
    zones = random_zones(net, 30, seed)
    rng = np.random.default_rng(seed)
    demand = rng.uniform(0, 10, size=(30, 30)) * (rng.uniform(size=(30, 30)) < 0.4)

    ours = assign_zone_traffic(
        net, zones, demand, near_seconds=near_seconds, workers=3, chunk_size=6
    )
    theirs = scipy_assignment.assign_zone_traffic(
        net, zones, demand, near_seconds=near_seconds
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def test_loaded_cost_equals_demand_times_od_cost():
    """Sum of volume x edge time is the demand-weighted sum of OD times."""
    net = random_network(300, 5)
    rng = np.random.default_rng(5)
    nodes = rng.choice(net.n_nodes, 40, replace=False)
    demand = rng.uniform(0, 5, size=(40, 40))
    np.fill_diagonal(demand, 0.0)

    volume = assign_traffic(net, nodes, demand, workers=2)
    times = travel_time_matrix(net, nodes, dtype=np.float64)
    reachable = np.isfinite(times)
    assert float(volume @ net.travel_time) == pytest.approx(
        float((demand * np.where(reachable, times, 0.0)).sum()), rel=1e-9
    )


def test_hilbert_order_is_a_permutation_that_keeps_neighbours_close():
    side = 64
    rows, cols = np.divmod(np.arange(side * side), side)
    order = hilbert_order(cols.astype(float), rows.astype(float))
    assert np.array_equal(np.sort(order), np.arange(side * side))
    steps = np.abs(np.diff(cols[order])) + np.abs(np.diff(rows[order]))
    assert steps.max() == 1


def test_graph_from_another_format_is_refused(tmp_path):
    net = make_network([(0, 1, 10.0)], 2)
    path = net.save(tmp_path / "net.npz")

    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if k != "format_version"}
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **arrays)

    assert RoutableNetwork.load(path).n_nodes == 2
    with pytest.raises(NetworkFormatError, match=f"needs format {FORMAT_VERSION}"):
        RoutableNetwork.load(legacy)
