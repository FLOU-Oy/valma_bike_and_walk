"""Point-to-point routing for the viewer: the path ``assign`` would load.

The route is the shortest path by the graph's own travel times -- the same
perceived costs (road class and facility, surface, traffic, penalties) that
``assign`` and ``matrix`` route on -- so a route drawn here is exactly what an
OD pair between those two points is assigned to. It is searched by the same
compiled kernels too (:class:`~valma_bike_and_walk.routing.Searcher`), each leg
stopping as soon as it reaches its end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from valma_bike_and_walk import routing
from valma_bike_and_walk.network import RoutableNetwork

#: Furthest a click may be from the network before the route is refused.
MAX_SNAP_M = 500.0


def _column(frame, name: str) -> np.ndarray | None:
    return frame[name].to_numpy(dtype=object) if name in frame.columns else None


def _tag(column: np.ndarray | None, row: int) -> str | None:
    if column is None or row < 0:
        return None
    value = column[row]
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


class Router:
    """Shortest paths on a routable network, drawn on the viewer's links."""

    def __init__(self, network: RoutableNetwork, links) -> None:
        self.network = network
        self.searcher = routing.Searcher(network.routing_graph())
        self.edge_rows = _edge_rows(network, links)
        self.links = links
        self.geometry = links.geometry.to_numpy()  # Web Mercator
        # Plain arrays: a route reads them once per edge.
        self._names = _column(links, "name")
        self._highways = _column(links, "highway")
        self.from_wgs84 = Transformer.from_crs("EPSG:4326", network.crs, always_xy=True)
        self.net_to_wgs84 = Transformer.from_crs(network.crs, "EPSG:4326", always_xy=True)
        self.merc_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def snap(self, points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Nearest node and its distance in metres, for every ``(lon, lat)``."""
        lon, lat = np.asarray(points, dtype=float).reshape(-1, 2).T
        x, y = self.from_wgs84.transform(lon, lat)
        return self.network.nearest_nodes(x, y)

    def route(self, origin: tuple[float, float], destination: tuple[float, float]) -> dict:
        return self.route_via([origin, destination])

    def route_via(self, points: list[tuple[float, float]]) -> dict:
        """The shortest path through ``points`` in order, leg by leg."""
        return self.describe(*self.path_edges(points))

    def path_edges(
        self, points: list[tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Edge indices (and their tail/head nodes) of the path through ``points``."""
        if len(points) < 2:
            raise ValueError("A route needs an origin and a destination.")
        nodes, distances = self.snap(points)
        worst = float(distances.max())
        if worst > MAX_SNAP_M:
            raise ValueError(
                f"A point is {worst:,.0f} m from the network (limit {MAX_SNAP_M:,.0f} m)."
            )
        legs: list[np.ndarray] = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            if a == b:
                continue
            edges, cost = self.searcher.path(int(a), int(b))
            if not np.isfinite(cost):
                raise ValueError("No route: a point cannot be reached from the one before it.")
            legs.append(edges)
        if not legs:
            raise ValueError("Origin and destination snap to the same node.")
        edges = np.concatenate(legs)
        graph = self.searcher.graph
        u = graph.edge_tail[edges].astype(np.int64)
        v = graph.indices[edges].astype(np.int64)
        return edges, u, v, distances.tolist()

    def _coordinates(self, edges, rows, u, v) -> list[list[float]]:
        """The path's line in WGS84: each edge's points in travel order, joined end to end."""
        net = self.network
        parts: list[np.ndarray] = [np.empty((0, 2))] * len(edges)

        drawn = np.flatnonzero(rows >= 0)
        if drawn.size:
            xy, owner = shapely.get_coordinates(self.geometry[rows[drawn]], return_index=True)
            lon, lat = self.merc_to_wgs84.transform(xy[:, 0], xy[:, 1])
            lonlat = np.column_stack([lon, lat])
            bounds = np.cumsum(np.bincount(owner, minlength=drawn.size))[:-1]
            against = net.direction[edges[drawn]] < 0
            for k, part, back in zip(drawn, np.split(lonlat, bounds), against):
                parts[k] = part[::-1] if back else part

        hidden = np.flatnonzero(rows < 0)  # links the viewer does not draw (island crossings)
        if hidden.size:
            lon, lat = self.net_to_wgs84.transform(
                np.concatenate([net.x[u[hidden]], net.x[v[hidden]]]),
                np.concatenate([net.y[u[hidden]], net.y[v[hidden]]]),
            )
            n = hidden.size
            for i, k in enumerate(hidden):
                parts[k] = np.array([[lon[i], lat[i]], [lon[n + i], lat[n + i]]])

        # Consecutive edges share an end point; keep it once.
        return np.concatenate([parts[0]] + [p[1:] for p in parts[1:]]).tolist()

    def describe(self, edges, u, v, snaps: list[float]) -> dict:
        """The path as a GeoJSON feature, with its steps and totals."""
        net = self.network
        rows = self.edge_rows[edges]
        times = net.travel_time[edges].astype(float)
        lengths = net.length[edges].astype(float)
        coords = self._coordinates(edges, rows, u, v)

        steps: list[dict] = []
        for row, t, length in zip(rows.tolist(), times.tolist(), lengths.tolist()):
            name = _tag(self._names, row) or ""
            highway = _tag(self._highways, row) or ("connector" if row < 0 else "")
            if steps and steps[-1]["name"] == name and steps[-1]["highway"] == highway:
                steps[-1]["length_m"] += length
                steps[-1]["time_s"] += t
                steps[-1]["rows"].append(row)
            else:
                steps.append({"name": name, "highway": highway, "length_m": length,
                              "time_s": t, "rows": [row]})
        by_highway: dict[str, float] = {}
        for step in steps:
            step["speed_kmh"] = step["length_m"] / step["time_s"] * 3.6 if step["time_s"] > 0 else None
            by_highway[step["highway"]] = by_highway.get(step["highway"], 0.0) + step["length_m"]

        total_t, total_l = float(times.sum()), float(lengths.sum())
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "length_m": total_l,
                "time_s": total_t,
                "speed_kmh": total_l / total_t * 3.6 if total_t > 0 else None,
                "links": int(len(edges)),
                "snap_m": [snaps[0], snaps[-1]],
                "by_highway": dict(sorted(by_highway.items(), key=lambda kv: -kv[1])),
                "steps": steps,
            },
        }


def _edge_rows(network: RoutableNetwork, links) -> np.ndarray:
    """Link-layer row of every directed edge, -1 where the viewer lacks it."""
    return pd.Index(links["link_id"].to_numpy()).get_indexer(network.link_id)
