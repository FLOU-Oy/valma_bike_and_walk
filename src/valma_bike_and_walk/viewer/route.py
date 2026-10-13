"""Point-to-point routing for the viewer: the path ``assign`` would load.

The route is the shortest path by the graph's own travel times -- the same
perceived costs (road class and facility, surface, traffic, penalties) that
``assign`` and ``matrix`` route on -- so a route drawn here is exactly what an
OD pair between those two points is assigned to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from valma_bike_and_walk.network import RoutableNetwork

#: Furthest a click may be from the network before the route is refused.
MAX_SNAP_M = 500.0
_NO_PREDECESSOR = -9999


class Router:
    """Shortest paths on a routable network, drawn on the viewer's links."""

    def __init__(self, network: RoutableNetwork, links) -> None:
        self.network = network
        self.graph = csr_matrix(
            (network.travel_time, network.indices, network.indptr),
            shape=(network.n_nodes, network.n_nodes),
        )
        n = network.n_nodes
        tails = np.repeat(np.arange(n, dtype=np.int64), np.diff(network.indptr))
        self.edge_keys = tails * n + network.indices.astype(np.int64)
        self.edge_rows = pd.Index(links["link_id"].to_numpy()).get_indexer(network.link_id)
        self.links = links
        self.geometry = links.geometry.to_numpy()  # Web Mercator
        self.from_wgs84 = Transformer.from_crs("EPSG:4326", network.crs, always_xy=True)
        self.net_to_wgs84 = Transformer.from_crs(network.crs, "EPSG:4326", always_xy=True)
        self.merc_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def snap(self, lon: float, lat: float) -> tuple[int, float]:
        x, y = self.from_wgs84.transform(lon, lat)
        node, distance = self.network.nearest_nodes(np.array([x]), np.array([y]))
        return int(node[0]), float(distance[0])

    def route(self, origin: tuple[float, float], destination: tuple[float, float]) -> dict:
        return self.route_via([origin, destination])

    def route_via(self, points: list[tuple[float, float]]) -> dict:
        """The shortest path through ``points`` in order, leg by leg."""
        edges, u, v, snaps = self.path_edges(points)
        return self._describe(edges, u, v, snaps[0], snaps[-1])

    def path_edges(
        self, points: list[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Edge indices (and their tail/head nodes) of the path through ``points``."""
        if len(points) < 2:
            raise ValueError("A route needs an origin and a destination.")
        snapped = [self.snap(*p) for p in points]
        worst = max(d for _, d in snapped)
        if worst > MAX_SNAP_M:
            raise ValueError(
                f"A point is {worst:,.0f} m from the network (limit {MAX_SNAP_M:,.0f} m)."
            )
        nodes_all: list[int] = [snapped[0][0]]
        for (a, _), (b, _) in zip(snapped[:-1], snapped[1:]):
            if a == b:
                continue
            nodes_all.extend(self._leg(a, b)[1:])
        if len(nodes_all) < 2:
            raise ValueError("Origin and destination snap to the same node.")
        u = np.asarray(nodes_all[:-1], dtype=np.int64)
        v = np.asarray(nodes_all[1:], dtype=np.int64)
        edges = np.searchsorted(self.edge_keys, u * self.network.n_nodes + v)
        return edges, u, v, [d for _, d in snapped]

    def _leg(self, a: int, b: int) -> list[int]:
        dist, pred = dijkstra(self.graph, directed=True, indices=a, return_predecessors=True)
        if not np.isfinite(dist[b]):
            raise ValueError("No route: a point cannot be reached from the one before it.")
        nodes = [b]
        while nodes[-1] != a:
            p = int(pred[nodes[-1]])
            if p == _NO_PREDECESSOR:
                raise ValueError("Broken shortest-path tree.")
            nodes.append(p)
        nodes.reverse()
        return nodes

    def _describe(self, edges, u, v, snap_a: float, snap_b: float) -> dict:
        net = self.network
        rows = self.edge_rows[edges]
        times = net.travel_time[edges].astype(float)
        lengths = net.length[edges].astype(float)

        coords: list[list[float]] = []
        for edge, row, tail, head in zip(edges, rows, u, v):
            if row >= 0:
                line = np.asarray(shapely.get_coordinates(self.geometry[row]))
                if net.direction[edge] < 0:
                    line = line[::-1]
                lon, lat = self.merc_to_wgs84.transform(line[:, 0], line[:, 1])
            else:  # a link the viewer does not draw (an island crossing)
                lon, lat = self.net_to_wgs84.transform(
                    net.x[[tail, head]], net.y[[tail, head]]
                )
            part = np.column_stack([lon, lat]).tolist()
            coords.extend(part[1:] if coords else part)

        frame = self.links
        def tag(column: str, row: int):
            if row < 0 or column not in frame.columns:
                return None
            value = frame[column].iloc[row]
            return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)

        steps: list[dict] = []
        for edge, row, t, length in zip(edges, rows, times, lengths):
            name = tag("name", row) or ""
            highway = tag("highway", row) or ("connector" if row < 0 else "")
            if steps and steps[-1]["name"] == name and steps[-1]["highway"] == highway:
                steps[-1]["length_m"] += length
                steps[-1]["time_s"] += t
                steps[-1]["rows"].append(int(row))
            else:
                steps.append({"name": name, "highway": highway, "length_m": length,
                              "time_s": t, "rows": [int(row)]})
        for step in steps:
            step["speed_kmh"] = step["length_m"] / step["time_s"] * 3.6 if step["time_s"] > 0 else None

        by_type = (
            pd.DataFrame({"highway": [s["highway"] for s in steps],
                          "length_m": [s["length_m"] for s in steps]})
            .groupby("highway")["length_m"].sum().sort_values(ascending=False)
        )
        total_t, total_l = float(times.sum()), float(lengths.sum())
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "length_m": total_l,
                "time_s": total_t,
                "speed_kmh": total_l / total_t * 3.6 if total_t > 0 else None,
                "links": int(len(edges)),
                "snap_m": [snap_a, snap_b],
                "by_highway": {k: float(val) for k, val in by_type.items()},
                "steps": steps,
            },
        }
