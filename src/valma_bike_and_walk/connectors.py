"""Island connectors: keep disconnected parts of the network in the graph.

The graph build keeps only the largest connected component
(:func:`valma_bike_and_walk.network._largest_component`), which is right for
stray fragments but wrong for islands: an archipelago's roads are a network of
their own, and dropping them leaves every zone on them either snapped across the
water to the mainland or dropped from the matrix altogether.

This module writes a second link layer holding one straight connector per
island, from its nearest node to the nearest node of another part of the
network. Merged into the graph (``--connectors``), the islands become part of
the largest component and survive. The connectors are given a deliberately
absurd speed, so a route across one costs hours: trips *within* an island route
normally, and trips across the water are effectively impossible -- beyond any
``--max-minutes`` cutoff, and never a shortcut for anyone else.

The layer is an ordinary link layer (same columns, same ``links`` layer name),
so it opens in QGIS next to the one it was built from, and any connector can be
deleted or redrawn by hand. Connectors have ``highway = "connector"``, negative
``link_id``s so they cannot collide with the extracted links, and their speed in
``speed_override_kmh``.

Parts are joined the way Boruvka's algorithm builds a spanning tree: every part
reaches for its nearest neighbouring part, those that touch merge, and the
merged groups reach again until one group is left. An island in a chain of
islands therefore connects to the next island along, not straight across the
whole archipelago to the mainland.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from valma_bike_and_walk.config import PROJECTED_CRS
from valma_bike_and_walk.links import OVERRIDE_COLUMN, resolve_endpoints

logger = logging.getLogger(__name__)

CONNECTOR_HIGHWAY = "connector"

#: Slow enough that crossing even a narrow strait costs longer than any sensible
#: trip: 100 m of connector is an hour. Trips that would need one fall outside
#: the search cutoff instead of loading onto a route that does not exist.
DEFAULT_CONNECTOR_SPEED_KMH = 0.1

#: Parts smaller than this are left out, and dropped by the build as before.
#: The fragments below it are almost all data gaps on the mainland -- a
#: driveway whose junction the extract lost -- and connecting them would pull a
#: zone that now snaps to a real street onto a fragment behind a connector.
DEFAULT_MIN_COMPONENT_NODES = 20

#: Parts at most this big find their nearest neighbour through one shared tree
#: (asking for one more neighbour than they have nodes guarantees an outside
#: one); bigger parts get a tree built without them.
_SMALL_QUERY_LIMIT = 64


class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return int(root)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def component_labels(
    links: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (node_ids, x, y, label): which connected part each node belongs to.

    Connectivity ignores direction. An island is a matter of geography, not of
    one-way streets, and a connector is two-way anyway.
    """
    u, v = resolve_endpoints(links)
    geometry = links.geometry.to_crs(PROJECTED_CRS).to_numpy()
    head = shapely.get_point(geometry, 0)
    tail = shapely.get_point(geometry, -1)

    ids = np.concatenate([u, v])
    x = np.concatenate([shapely.get_x(head), shapely.get_x(tail)])
    y = np.concatenate([shapely.get_y(head), shapely.get_y(tail)])
    ids, first = np.unique(ids, return_index=True)
    x, y = x[first], y[first]

    u_idx = np.searchsorted(ids, u)
    v_idx = np.searchsorted(ids, v)
    n = ids.shape[0]
    graph = coo_matrix(
        (np.ones(u_idx.shape[0], dtype=np.int8), (u_idx, v_idx)), shape=(n, n)
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    return ids, x, y, labels


def _nearest_outside(
    points: np.ndarray,
    main_tree: cKDTree,
    small_tree: cKDTree,
    small_points: np.ndarray,
    small_group: np.ndarray,
    group: int,
) -> tuple[float, int, int, bool]:
    """
    Nearest (distance, point row, target row, target_is_main) from ``points``
    to any node outside ``group``.

    The main part is never in ``group`` -- it is the one group that does not go
    looking -- so its tree can be asked directly. Other parts share
    ``small_tree``, where the group's own nodes have to be skipped.
    """
    main_d, main_i = main_tree.query(points, k=1)
    best_row = int(np.argmin(main_d))
    best = (float(main_d[best_row]), best_row, int(main_i[best_row]), True)

    outside = small_group != group
    if not outside.any():
        return best

    size = points.shape[0]
    if size < _SMALL_QUERY_LIMIT:
        k = min(size + 1, small_points.shape[0])
        d, i = small_tree.query(points, k=k)
        d = d.reshape(size, k)
        i = i.reshape(size, k)
        valid = i < small_points.shape[0]
        foreign = np.zeros_like(valid)
        foreign[valid] = small_group[i[valid]] != group
        d = np.where(foreign, d, np.inf)
    else:
        rows = np.flatnonzero(outside)
        tree = cKDTree(small_points[rows])
        d, i = tree.query(points, k=1)
        d, i = d[:, None], rows[i][:, None]

    flat = int(np.argmin(d))
    row, col = np.unravel_index(flat, d.shape)
    if np.isfinite(d[row, col]) and d[row, col] < best[0]:
        best = (float(d[row, col]), int(row), int(i[row, col]), False)
    return best


def island_connectors(
    links: gpd.GeoDataFrame,
    min_nodes: int = DEFAULT_MIN_COMPONENT_NODES,
    speed_kmh: float = DEFAULT_CONNECTOR_SPEED_KMH,
) -> gpd.GeoDataFrame:
    """
    Connector links joining every part of at least ``min_nodes`` to the rest.

    Returns a link layer in ``links``' CRS, empty if the network is already in
    one piece.
    """
    if speed_kmh <= 0:
        raise ValueError("Connector speed must be positive.")

    ids, x, y, labels = component_labels(links)
    counts = np.bincount(labels)
    main = int(counts.argmax())
    kept = np.flatnonzero(counts >= max(min_nodes, 1))
    kept = kept[kept != main]
    logger.info(
        "%d connected part(s); connecting %d of at least %d node(s) "
        "(%d nodes) to the main part (%d nodes)",
        counts.shape[0],
        kept.shape[0],
        min_nodes,
        int(counts[kept].sum()),
        int(counts[main]),
    )

    xy = np.column_stack([x, y])
    main_nodes = np.flatnonzero(labels == main)
    main_tree = cKDTree(xy[main_nodes])

    small_nodes = np.flatnonzero(np.isin(labels, kept))
    small_points = xy[small_nodes]
    small_tree = cKDTree(small_points) if small_nodes.size else None

    # Groups are indexed by component label; the main part is its own group.
    groups = _DisjointSet(counts.shape[0])
    edges: list[tuple[int, int, float]] = []

    while small_nodes.size:
        roots = np.array([groups.find(int(c)) for c in labels[small_nodes]])
        main_root = groups.find(main)
        searching = np.unique(roots[roots != main_root])
        if searching.size == 0:
            break

        order = np.argsort(roots, kind="stable")
        bounds = np.searchsorted(roots[order], searching)
        ends = np.searchsorted(roots[order], searching, side="right")

        candidates: list[tuple[float, int, int, int]] = []
        for group, start, end in zip(searching, bounds, ends):
            rows = order[start:end]
            distance, row, target, target_is_main = _nearest_outside(
                small_points[rows], main_tree, small_tree, small_points, roots, group
            )
            source = int(small_nodes[rows[row]])
            target = int(main_nodes[target] if target_is_main else small_nodes[target])
            candidates.append((distance, int(group), source, target))

        # Shortest first, so of two groups reaching for each other only one
        # connector is laid.
        for distance, group, source, target in sorted(candidates):
            if groups.union(int(labels[target]), group):
                edges.append((source, target, distance))
        logger.info(
            "  %d group(s) searched, %d connector(s) so far", searching.size, len(edges)
        )

    return _connector_frame(ids, xy, edges, speed_kmh, links.crs, labels, counts)


def _connector_frame(
    ids: np.ndarray,
    xy: np.ndarray,
    edges: list[tuple[int, int, float]],
    speed_kmh: float,
    crs,
    labels: np.ndarray,
    counts: np.ndarray,
) -> gpd.GeoDataFrame:
    source = np.array([e[0] for e in edges], dtype=np.int64)
    target = np.array([e[1] for e in edges], dtype=np.int64)
    geometry = gpd.GeoSeries(
        shapely.linestrings(
            np.stack([xy[source], xy[target]], axis=1).reshape(-1, 2),
            indices=np.repeat(np.arange(source.shape[0]), 2),
        )
        if source.size
        else [],
        crs=PROJECTED_CRS,
    )
    return gpd.GeoDataFrame(
        {
            "link_id": -np.arange(1, source.shape[0] + 1, dtype=np.int64),
            "u": ids[source],
            "v": ids[target],
            "highway": pd.Series([CONNECTOR_HIGHWAY] * source.shape[0], dtype=object),
            OVERRIDE_COLUMN: np.full(source.shape[0], float(speed_kmh)),
            "length_m": np.array([e[2] for e in edges], dtype=float),
            # How big the part on each end is -- the thing to sort by in QGIS
            # when checking whether a connector should be there at all.
            "u_part_nodes": counts[labels[source]].astype(np.int64),
            "v_part_nodes": counts[labels[target]].astype(np.int64),
        },
        geometry=geometry.to_crs(crs).to_numpy(),
        crs=crs,
    )


def merge_connectors(
    links: gpd.GeoDataFrame, connectors: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """The link layer with connectors appended, ready for the graph build."""
    if connectors.crs != links.crs:
        connectors = connectors.to_crs(links.crs)
    clash = np.intersect1d(
        pd.to_numeric(links["link_id"], errors="coerce").dropna().to_numpy(),
        pd.to_numeric(connectors["link_id"], errors="coerce").dropna().to_numpy(),
    )
    if clash.size:
        raise ValueError(
            f"{clash.size} connector link_id(s) are also in the link layer "
            f"(e.g. {int(clash[0])}). Connectors should have negative ids."
        )
    logger.info("Adding %d connector link(s)", len(connectors))
    return gpd.GeoDataFrame(
        pd.concat([links, connectors], ignore_index=True), crs=links.crs
    )
