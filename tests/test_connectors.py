"""Island connectors: disconnected parts joined, not dropped."""

import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra

from valma_bike_and_walk.cli import main
from valma_bike_and_walk.connectors import (
    CONNECTOR_HIGHWAY,
    island_connectors,
    merge_connectors,
)
from valma_bike_and_walk.links import read_links, write_links
from valma_bike_and_walk.network import network_from_links

from .conftest import links_frame

# At 60 N, 0.001 degrees of longitude is about 56 m.
LAT = 60.0


def chain(first_node, lon0, n_nodes=4, first_link_id=0):
    """A straight east-west chain of ``n_nodes`` nodes, 56 m apart."""
    return [
        {
            "link_id": first_link_id + k,
            "u": first_node + k,
            "v": first_node + k + 1,
            "coords": [(lon0 + 0.001 * k, LAT), (lon0 + 0.001 * (k + 1), LAT)],
        }
        for k in range(n_nodes - 1)
    ]


@pytest.fixture
def archipelago():
    """
    Mainland (nodes 1-4), island A about 1 km east of it (11-14), island B about
    1 km east of A (21-24), and a two-node fragment just west of the mainland.
    """
    return links_frame(
        chain(1, 24.000, first_link_id=0)
        + chain(11, 24.020, first_link_id=10)
        + chain(21, 24.040, first_link_id=20)
        + chain(31, 23.990, n_nodes=2, first_link_id=30)
    )


def pairs(connectors):
    return {frozenset((int(u), int(v))) for u, v in zip(connectors.u, connectors.v)}


def test_each_island_connects_to_its_nearest_neighbour(archipelago):
    connectors = island_connectors(archipelago, min_nodes=3)

    # B reaches for A, the next island along -- not across A to the mainland.
    assert pairs(connectors) == {frozenset((4, 11)), frozenset((14, 21))}
    assert (connectors.link_id < 0).all()
    assert (connectors.highway == CONNECTOR_HIGHWAY).all()
    assert connectors.crs == archipelago.crs


def test_parts_below_the_minimum_are_left_out(archipelago):
    connectors = island_connectors(archipelago, min_nodes=3)
    assert not {31, 32} & (set(connectors.u) | set(connectors.v))

    everything = island_connectors(archipelago, min_nodes=1)
    assert frozenset((32, 1)) in pairs(everything)


def test_islands_survive_the_build_behind_a_prohibitive_crossing(archipelago):
    speed_kmh = 0.1
    connectors = island_connectors(archipelago, min_nodes=3, speed_kmh=speed_kmh)

    alone = network_from_links(archipelago, "walk")
    joined = network_from_links(merge_connectors(archipelago, connectors), "walk")
    assert alone.n_nodes == 4
    assert joined.n_nodes == 12
    assert 31 not in set(joined.node_ids)

    seconds = dijkstra(joined.csr(), indices=[0])[0]
    island_b = int(np.flatnonzero(joined.node_ids == 21)[0])
    crossing_s = connectors.length_m.sum() / (speed_kmh / 3.6)
    assert seconds[island_b] == pytest.approx(crossing_s, rel=0.01, abs=120)
    # Hours, not minutes: nobody is meant to take this route.
    assert seconds[island_b] > 10 * 3600


def test_a_network_in_one_piece_needs_no_connectors():
    assert island_connectors(links_frame(chain(1, 24.0))).empty


def test_connector_ids_must_not_clash_with_the_link_layer(archipelago):
    connectors = island_connectors(archipelago, min_nodes=3)
    connectors["link_id"] = [0, 1]
    with pytest.raises(ValueError, match="link_id"):
        merge_connectors(archipelago, connectors)


def test_connect_then_route_from_the_command_line(archipelago, tmp_path):
    links_path = write_links(archipelago, tmp_path / "walk_links.gpkg")
    assert main(["connect", "--links", str(links_path), "--min-nodes", "3"]) == 0

    connectors_path = tmp_path / "walk_connectors.gpkg"
    assert len(read_links(connectors_path)) == 2

    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\n1,24.0005,60.0\n2,24.0405,60.0\n")
    common = [
        "--mode", "walk",
        "--links", str(links_path),
        "--centroids", str(centroids),
        "--id-column", "id",
        "--max-snap-distance", "100",
        "--cache-dir", str(tmp_path / "cache"),
        "--output-dir", str(tmp_path / "out"),
        "--workers", "1",
    ]  # fmt: skip

    assert main(["matrix", *common]) == 0
    without = np.load(tmp_path / "out" / "travel_times_walk.npz")
    assert list(without["ids"]) == [1]

    assert main(["matrix", *common, "--connectors", str(connectors_path)]) == 0
    with_islands = np.load(tmp_path / "out" / "travel_times_walk.npz")
    assert list(with_islands["ids"]) == [1, 2]
    # Kept apart in its own cache file, not mistaken for the plain graph.
    assert (tmp_path / "cache" / "walk+walk_connectors.npz").exists()
