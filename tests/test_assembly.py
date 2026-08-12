"""Assembling edge arrays into a network, and merging separately read extents.

pyosmium reads any extent in one streaming pass, so nothing here is tiled any
more -- but ``_assemble`` still has to stitch on OSM node id, because that is
how two regional extracts (or one link layer concatenated with another) join up.
"""

import numpy as np
import pytest

from valma_bike_and_walk.matrix import travel_time_matrix
from valma_bike_and_walk.network import _assemble, _EdgeArrays


def part(node_ids, lons, edges, first_link_id=0):
    """edges: list of (u_osmid, v_osmid, seconds)."""
    return _EdgeArrays(
        node_ids=np.array(node_ids, dtype=np.int64),
        lon=np.array(lons, dtype=float),
        lat=np.full(len(node_ids), 60.0),
        u=np.array([e[0] for e in edges], dtype=np.int64),
        v=np.array([e[1] for e in edges], dtype=np.int64),
        travel_time=np.array([e[2] for e in edges], dtype=float),
        length=np.array([e[2] for e in edges], dtype=np.float32),
        link_id=np.arange(first_link_id, first_link_id + len(edges), dtype=np.int64),
        direction=np.ones(len(edges), dtype=np.int8),
    )


def index_of(net, osm_id):
    return int(np.searchsorted(net.node_ids, osm_id))


def test_extents_join_on_the_shared_boundary_node():
    """A route crossing the seam must be traversable in the merged network."""
    west = part(
        [1, 2, 3],
        [24.0, 24.1, 24.2],
        [(1, 2, 10.0), (2, 3, 10.0), (3, 2, 10.0), (2, 1, 10.0)],
    )
    # Node 3 is the shared seam node, present in both extents.
    east = part(
        [3, 4, 5],
        [24.2, 24.3, 24.4],
        [(3, 4, 10.0), (4, 5, 10.0), (5, 4, 10.0), (4, 3, 10.0)],
        first_link_id=100,
    )

    net = _assemble("walk", [west, east])

    assert net.n_nodes == 5
    seconds = travel_time_matrix(net, [index_of(net, 1)], [index_of(net, 5)])
    assert seconds[0, 0] == pytest.approx(40.0)


def test_duplicate_edges_across_an_overlap_collapse():
    a = part([1, 2], [24.0, 24.1], [(1, 2, 10.0), (2, 1, 10.0)])
    b = part([1, 2], [24.0, 24.1], [(1, 2, 10.0), (2, 1, 10.0)], first_link_id=100)

    net = _assemble("walk", [a, b])
    assert net.n_nodes == 2
    assert net.n_edges == 2  # not 4


def test_the_fastest_of_several_parallel_links_is_the_one_kept():
    fast = part([1, 2], [24.0, 24.1], [(1, 2, 5.0), (2, 1, 5.0)])
    slow = part([1, 2], [24.0, 24.1], [(1, 2, 50.0), (2, 1, 50.0)], first_link_id=100)

    net = _assemble("walk", [slow, fast])
    assert net.n_edges == 2
    np.testing.assert_allclose(net.travel_time, 5.0)
    # And the surviving edge names the link it actually came from.
    assert set(net.link_id) == {0, 1}


def test_a_coarser_extent_does_not_change_travel_time():
    """One side may hold u->v directly where the other holds u->x->v."""
    fine = part(
        [1, 2, 3],
        [24.0, 24.1, 24.2],
        [(1, 2, 30.0), (2, 3, 70.0), (3, 2, 70.0), (2, 1, 30.0)],
    )
    coarse = part(
        [1, 3], [24.0, 24.2], [(1, 3, 100.0), (3, 1, 100.0)], first_link_id=100
    )

    net = _assemble("walk", [fine, coarse])
    seconds = travel_time_matrix(net, [index_of(net, 1)], [index_of(net, 3)])
    assert seconds[0, 0] == pytest.approx(100.0)


def test_edges_pointing_outside_the_kept_nodes_are_dropped():
    """A link clipped at the extent edge can reference a node we did not keep."""
    dangling = part([1, 2], [24.0, 24.1], [(1, 2, 10.0), (2, 1, 10.0), (2, 999, 10.0)])
    net = _assemble("walk", [dangling])
    assert net.n_nodes == 2
    assert net.n_edges == 2


def test_only_the_largest_component_survives():
    """An island unreachable from the main network would only break the matrix."""
    main = part(
        [1, 2, 3],
        [24.0, 24.1, 24.2],
        [(1, 2, 10.0), (2, 1, 10.0), (2, 3, 10.0), (3, 2, 10.0)],
    )
    island = part([7, 8], [25.0, 25.1], [(7, 8, 10.0), (8, 7, 10.0)], first_link_id=100)

    net = _assemble("walk", [main, island])
    assert net.n_nodes == 3
    assert set(net.node_ids) == {1, 2, 3}


def test_per_edge_arrays_stay_aligned_after_a_subset():
    """The invariant every downstream consumer relies on."""
    main = part(
        [1, 2, 3],
        [24.0, 24.1, 24.2],
        [(1, 2, 10.0), (2, 1, 10.0), (2, 3, 20.0), (3, 2, 20.0)],
    )
    island = part([7, 8], [25.0, 25.1], [(7, 8, 99.0), (8, 7, 99.0)], first_link_id=100)

    net = _assemble("walk", [main, island])
    for array in (net.travel_time, net.length, net.link_id, net.direction):
        assert array.shape[0] == net.n_edges
    # The dropped island's link ids went with it.
    assert not set(net.link_id) & {100, 101}
