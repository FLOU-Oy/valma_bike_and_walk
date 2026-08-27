"""Synthetic fixtures: hand-built .osm.pbf files and link layers.

pyosmium can write a PBF as well as read one, so the extract stage is testable
without downloading a country. Everything here is a few nodes across, so the
whole suite still runs in seconds.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import osmium
import osmium.osm.mutable as mutable
import pytest
import shapely

from valma_bike_and_walk.centroids import Centroids
from valma_bike_and_walk.config import WGS84
from valma_bike_and_walk.network import RoutableNetwork, _build_csr
from valma_bike_and_walk.zones import Zones


def make_network(edges, n_nodes, mode="bike") -> RoutableNetwork:
    """
    A network straight from an edge list, skipping both pipeline stages.

    ``edges`` is a list of ``(u, v, seconds)`` over node ids 0..n_nodes-1, laid
    out along the x axis 100 m apart. For tests about routing rather than about
    where the graph came from.
    """
    u = np.array([e[0] for e in edges], dtype=np.int64)
    v = np.array([e[1] for e in edges], dtype=np.int64)
    seconds = np.array([e[2] for e in edges], dtype=float)

    indptr, indices, travel_time, length, link_id, direction = _build_csr(
        u,
        v,
        seconds,
        seconds.astype(np.float32),
        np.arange(len(edges), dtype=np.int64),
        np.ones(len(edges), dtype=np.int8),
        n_nodes,
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
        link_id=link_id,
        direction=direction,
    )


def make_grid_network(
    side: int = 5,
    spacing: float = 100.0,
    seconds: float = 10.0,
    mode: str = "bike",
) -> RoutableNetwork:
    """
    A ``side`` x ``side`` grid of nodes, 4-connected, every edge the same cost.

    Node ``row * side + col`` sits at ``(col * spacing, row * spacing)`` in
    PROJECTED_CRS metres. Zone tests need a network with two dimensions to it --
    a polygon over a network laid out along a line cannot say much.
    """
    u: list[int] = []
    v: list[int] = []
    for row in range(side):
        for col in range(side):
            node = row * side + col
            if col + 1 < side:
                u += [node, node + 1]
                v += [node + 1, node]
            if row + 1 < side:
                u += [node, node + side]
                v += [node + side, node]

    n_nodes = side * side
    count = len(u)
    indptr, indices, travel_time, length, link_id, direction = _build_csr(
        np.array(u, dtype=np.int64),
        np.array(v, dtype=np.int64),
        np.full(count, seconds, dtype=float),
        np.full(count, spacing, dtype=np.float32),
        np.arange(count, dtype=np.int64),
        np.ones(count, dtype=np.int8),
        n_nodes,
    )
    rows, cols = np.divmod(np.arange(n_nodes), side)
    return RoutableNetwork(
        mode=mode,
        node_ids=np.arange(n_nodes, dtype=np.int64),
        x=cols.astype(float) * spacing,
        y=rows.astype(float) * spacing,
        indptr=indptr,
        indices=indices,
        travel_time=travel_time,
        length=length,
        link_id=link_id,
        direction=direction,
    )


def make_zones(
    nodes_per_zone: list[list[int]],
    weights_per_zone: list[list[float]] | None = None,
    access_per_zone: list[list[float]] | None = None,
    ids: list | None = None,
    area_m2: list[float] | None = None,
    network: RoutableNetwork | None = None,
) -> Zones:
    """
    Build a :class:`Zones` straight from node lists, skipping :func:`load_zones`.

    For tests about the aggregation arithmetic rather than about where the
    access points came from. Weights default to uniform within each zone and
    access time to zero; the representative point of a zone is its first node.
    """
    n_zones = len(nodes_per_zone)
    if weights_per_zone is None:
        weights_per_zone = [[1.0 / len(n)] * len(n) for n in nodes_per_zone]
    if access_per_zone is None:
        access_per_zone = [[0.0] * len(n) for n in nodes_per_zone]

    counts = np.array([len(n) for n in nodes_per_zone], dtype=np.int64)
    indptr = np.zeros(n_zones + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    node_index = np.array([n for zone in nodes_per_zone for n in zone], dtype=np.int64)
    weight = np.array([w for zone in weights_per_zone for w in zone], dtype=float)
    access = np.array([a for zone in access_per_zone for a in zone], dtype=float)

    if network is not None:
        x, y = network.x[node_index], network.y[node_index]
    else:
        x = y = np.zeros(node_index.shape[0])

    zone_ids = np.array(ids if ids is not None else range(n_zones))
    first = np.array([zone[0] for zone in nodes_per_zone], dtype=np.int64)
    return Zones(
        ids=zone_ids,
        indptr=indptr,
        node_index=node_index,
        weight=weight,
        access_seconds=access,
        x=x,
        y=y,
        area_m2=np.array(area_m2 if area_m2 is not None else [0.0] * n_zones),
        representative=Centroids(
            ids=zone_ids,
            x=x[indptr[:-1]],
            y=y[indptr[:-1]],
            node_index=first,
            snap_distance=np.zeros(n_zones),
        ),
        representative_access_seconds=np.zeros(n_zones),
    )


def write_pbf(
    path, nodes: dict[int, tuple[float, float]], ways, relations=(), node_tags=None
) -> str:
    """
    Write a tiny .osm.pbf.

    ``ways`` is a list of ``(way_id, [node_id, ...], tags)``; ``relations`` a
    list of ``(relation_id, [(type, ref, role), ...], tags)``.
    """
    writer = osmium.SimpleWriter(str(path))
    try:
        for node_id, (lon, lat) in nodes.items():
            writer.add_node(
                mutable.Node(
                    id=node_id,
                    location=(lon, lat),
                    tags=(node_tags or {}).get(node_id, {}),
                )
            )
        for way_id, refs, tags in ways:
            writer.add_way(mutable.Way(id=way_id, nodes=list(refs), tags=dict(tags)))
        for relation_id, members, tags in relations:
            writer.add_relation(
                mutable.Relation(id=relation_id, members=list(members), tags=dict(tags))
            )
    finally:
        writer.close()
    return str(path)


#: A small grid, roughly 500 m of streets around (24.00, 60.00).
#:
#:     1 --- 2 --- 3        way 100, residential
#:           |
#:           4              way 101, path, oneway=yes
#:           |
#:           5              way 102, steps  (walkable and cyclable)
#:
#: Node 2 is the only junction, so way 100 splits into two links there.
GRID_NODES = {
    1: (24.000, 60.000),
    2: (24.002, 60.000),
    3: (24.004, 60.000),
    4: (24.002, 60.001),
    5: (24.002, 60.002),
}

GRID_WAYS = [
    (100, [1, 2, 3], {"highway": "residential", "surface": "asphalt"}),
    (101, [2, 4], {"highway": "path", "oneway": "yes"}),
    (102, [4, 5], {"highway": "steps"}),
]


@pytest.fixture
def grid_pbf(tmp_path):
    """A .osm.pbf holding :data:`GRID_NODES` / :data:`GRID_WAYS`."""
    return write_pbf(tmp_path / "grid.osm.pbf", GRID_NODES, GRID_WAYS)


def links_frame(rows, crs: str = WGS84) -> gpd.GeoDataFrame:
    """
    Build a link layer by hand.

    ``rows`` is a list of dicts with at least ``coords``; ``link_id``, ``u``,
    ``v`` and ``highway`` default to something sensible so a test only has to
    say what it is actually about. Any other key -- ``crossing``, ``ascent_m``,
    a traffic-signal flag -- is carried onto the frame as its own column.
    """
    records = []
    geometries = []
    for i, row in enumerate(rows):
        coords = row["coords"]
        record = {
            "link_id": row.get("link_id", i),
            "u": row.get("u", np.nan),
            "v": row.get("v", np.nan),
            "highway": row.get("highway", "residential"),
            "surface": row.get("surface", None),
            "segregated": row.get("segregated", None),
            "oneway": row.get("oneway", None),
            "oneway_bicycle": row.get("oneway_bicycle", None),
            "junction": row.get("junction", None),
            "length_m": row.get("length_m", np.nan),
        }
        record.update(
            {k: v for k, v in row.items() if k not in record and k != "coords"}
        )
        records.append(record)
        geometries.append(shapely.LineString(coords))
    return gpd.GeoDataFrame(records, geometry=geometries, crs=crs)
