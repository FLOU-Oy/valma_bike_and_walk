"""Tiled building: tiles must stitch back into one network across their seams."""

import numpy as np
import pytest

from valma_bike_and_walk.matrix import travel_time_matrix
from valma_bike_and_walk.network import _assemble, _EdgeArrays, iter_tiles


def part(node_ids, lons, edges):
    """edges: list of (u_osmid, v_osmid, seconds)."""
    return _EdgeArrays(
        node_ids=np.array(node_ids, dtype=np.int64),
        lon=np.array(lons, dtype=float),
        lat=np.full(len(node_ids), 60.0),
        u=np.array([e[0] for e in edges], dtype=np.int64),
        v=np.array([e[1] for e in edges], dtype=np.int64),
        travel_time=np.array([e[2] for e in edges], dtype=float),
        length=np.array([e[2] for e in edges], dtype=np.float32),
    )


def index_of(net, osm_id):
    return int(np.searchsorted(net.node_ids, osm_id))


def test_tiles_join_on_the_shared_boundary_node():
    """A route crossing a tile seam must be traversable in the merged network."""
    west = part(
        [1, 2, 3],
        [24.0, 24.1, 24.2],
        [(1, 2, 10.0), (2, 3, 10.0), (3, 2, 10.0), (2, 1, 10.0)],
    )
    # Node 3 is the shared seam node, present in both tiles.
    east = part(
        [3, 4, 5],
        [24.2, 24.3, 24.4],
        [(3, 4, 10.0), (4, 5, 10.0), (5, 4, 10.0), (4, 3, 10.0)],
    )

    net = _assemble("walk", [west, east])

    assert net.n_nodes == 5
    seconds = travel_time_matrix(net, [index_of(net, 1)], [index_of(net, 5)])
    assert seconds[0, 0] == pytest.approx(40.0)


def test_duplicate_edges_across_overlapping_tiles_collapse():
    a = part([1, 2], [24.0, 24.1], [(1, 2, 10.0), (2, 1, 10.0)])
    b = part([1, 2], [24.0, 24.1], [(1, 2, 10.0), (2, 1, 10.0)])

    net = _assemble("walk", [a, b])
    assert net.n_nodes == 2
    assert net.n_edges == 2  # not 4


def test_a_tile_simplifying_further_does_not_change_travel_time():
    """One tile may collapse u->x->v into u->v; the merged cost must still match."""
    fine = part(
        [1, 2, 3],
        [24.0, 24.1, 24.2],
        [(1, 2, 30.0), (2, 3, 70.0), (3, 2, 70.0), (2, 1, 30.0)],
    )
    coarse = part([1, 3], [24.0, 24.2], [(1, 3, 100.0), (3, 1, 100.0)])

    net = _assemble("walk", [fine, coarse])
    seconds = travel_time_matrix(net, [index_of(net, 1)], [index_of(net, 3)])
    assert seconds[0, 0] == pytest.approx(100.0)


def test_edges_pointing_outside_the_kept_nodes_are_dropped():
    """Ways clipped at a tile edge can reference a node the tile did not keep."""
    dangling = part([1, 2], [24.0, 24.1], [(1, 2, 10.0), (2, 1, 10.0), (2, 999, 10.0)])
    net = _assemble("walk", [dangling])
    assert net.n_nodes == 2
    assert net.n_edges == 2


def test_iter_tiles_covers_the_whole_box_with_overlap():
    tiles = iter_tiles([24.0, 60.0, 26.0, 61.0], tile_degrees=1.0, overlap_degrees=0.1)
    assert len(tiles) == 2 * 1

    # Every tile stays inside the requested bounds.
    for min_lon, min_lat, max_lon, max_lat in tiles:
        assert 24.0 <= min_lon < max_lon <= 26.0
        assert 60.0 <= min_lat < max_lat <= 61.0

    # Neighbouring tiles genuinely overlap rather than merely touching.
    left, right = sorted(tiles, key=lambda t: t[0])[:2]
    assert right[0] < left[2]


def test_iter_tiles_handles_a_box_smaller_than_one_tile():
    tiles = iter_tiles([24.0, 60.0, 24.5, 60.5], tile_degrees=2.0)
    assert len(tiles) == 1
