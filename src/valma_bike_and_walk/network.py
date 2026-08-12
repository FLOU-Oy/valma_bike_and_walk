"""A compact, routable representation of a travel network.

A NetworkX ``MultiDiGraph`` stores a Python dict per node and per edge. That is
convenient at city scale and hopeless at country scale: Finland's walking
network runs to millions of nodes, which costs tens of gigabytes as dicts and
makes every shortest-path call slow.

This module keeps the same information in a handful of numpy arrays plus a SciPy
CSR matrix -- roughly two orders of magnitude smaller, and routable with SciPy's
compiled Dijkstra instead of NetworkX's pure-Python one.

This is stage two of the pipeline. Its input is the link layer
(:mod:`valma_bike_and_walk.links`), not the PBF: geometry, tags and every
editable decision stay in the GeoPackage, and what lands here is the minimum
needed to route. Each directed edge keeps its ``link_id``, so a result computed
here maps straight back onto the rows you edited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from valma_bike_and_walk import links as links_module
from valma_bike_and_walk.config import (
    PROJECTED_CRS,
    WGS84,
    Settings,
    validate_mode,
)
from valma_bike_and_walk.osm import read_links, resolve_clip

logger = logging.getLogger(__name__)


@dataclass
class RoutableNetwork:
    """Nodes, coordinates and a CSR adjacency weighted by travel time."""

    mode: str
    node_ids: np.ndarray  # int64, original OSM node ids
    x: np.ndarray  # float64, PROJECTED_CRS metres
    y: np.ndarray
    indptr: np.ndarray  # int32/int64 CSR row pointers
    indices: np.ndarray  # int32 CSR column indices
    travel_time: np.ndarray  # float64 seconds, one per directed edge
    length: np.ndarray  # float32 metres, one per directed edge
    #: link_id of the link layer row this edge came from, one per directed edge.
    #: This is how an assignment's volumes are drawn back onto the GeoPackage.
    link_id: np.ndarray  # int64
    #: +1 if the edge runs along the link's digitised direction, -1 against it.
    direction: np.ndarray  # int8
    crs: str = PROJECTED_CRS

    def __post_init__(self) -> None:
        self._tree: cKDTree | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.indices.shape[0])

    def csr(self) -> csr_matrix:
        """Adjacency weighted by travel time in seconds."""
        return csr_matrix(
            (self.travel_time, self.indices, self.indptr),
            shape=(self.n_nodes, self.n_nodes),
        )

    def distance_csr(self) -> csr_matrix:
        """Adjacency weighted by length in metres."""
        return csr_matrix(
            (self.length.astype(np.float64), self.indices, self.indptr),
            shape=(self.n_nodes, self.n_nodes),
        )

    @property
    def tree(self) -> cKDTree:
        """KD-tree over node coordinates, built once and reused."""
        if self._tree is None:
            self._tree = cKDTree(np.column_stack([self.x, self.y]))
        return self._tree

    def nearest_nodes(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_distance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Snap projected points to the closest network node.

        Returns (node_index, distance_m). Points further than ``max_distance``
        get index -1 so the caller can decide what to do rather than silently
        routing from somewhere kilometres away.
        """
        points = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
        distances, idx = self.tree.query(points, k=1)
        idx = np.asarray(idx, dtype=np.int64)
        distances = np.asarray(distances, dtype=float)

        if max_distance is not None:
            idx = np.where(distances <= max_distance, idx, -1)
        return idx, distances

    def save(self, path: Path) -> Path:
        """Persist as a compressed .npz -- far smaller and faster than GraphML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mode=np.array(self.mode),
            crs=np.array(self.crs),
            node_ids=self.node_ids,
            x=self.x,
            y=self.y,
            indptr=self.indptr,
            indices=self.indices,
            travel_time=self.travel_time,
            length=self.length,
            link_id=self.link_id,
            direction=self.direction,
        )
        logger.info("Saved network to %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @classmethod
    def load(cls, path: Path) -> "RoutableNetwork":
        path = Path(path)
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                mode=str(data["mode"]),
                crs=str(data["crs"]),
                node_ids=data["node_ids"],
                x=data["x"],
                y=data["y"],
                indptr=data["indptr"],
                indices=data["indices"],
                travel_time=data["travel_time"],
                length=data["length"],
                link_id=data["link_id"],
                direction=data["direction"],
            )


def _largest_component(
    indptr: np.ndarray,
    indices: np.ndarray,
    travel_time: np.ndarray,
    n_nodes: int,
    directed: bool,
) -> np.ndarray:
    """
    Boolean mask of the biggest component.

    For a directed network we need *strong* connectivity: a weakly connected
    node can be reachable without being able to get back, which would leave
    one-way holes in the OD matrix.
    """
    graph = csr_matrix((travel_time, indices, indptr), shape=(n_nodes, n_nodes))
    n_components, labels = connected_components(
        graph, directed=directed, connection="strong" if directed else "weak"
    )
    if n_components == 1:
        return np.ones(n_nodes, dtype=bool)

    counts = np.bincount(labels)
    biggest = int(counts.argmax())
    keep = labels == biggest
    logger.info(
        "Keeping largest component: %d of %d nodes (%d components total)",
        int(keep.sum()),
        n_nodes,
        n_components,
    )
    return keep


def _build_csr(
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    travel_time: np.ndarray,
    length: np.ndarray,
    link_id: np.ndarray,
    direction: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, ...]:
    """
    Build CSR arrays, keeping only the fastest edge for any repeated node pair.

    Parallel links between the same two nodes are common in OSM, and a shortest
    path can only ever use the quickest one. ``link_id`` and ``direction`` ride
    through the same sort and dedup, so every per-edge array stays aligned with
    the CSR and the surviving edge names the link it actually came from.
    """
    if u_idx.shape[0] == 0:
        return (
            np.zeros(n_nodes + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int8),
        )

    order = np.lexsort((travel_time, v_idx, u_idx))
    u_idx, v_idx = u_idx[order], v_idx[order]
    travel_time, length = travel_time[order], length[order]
    link_id, direction = link_id[order], direction[order]

    first = np.empty(u_idx.shape[0], dtype=bool)
    first[0] = True
    first[1:] = (u_idx[1:] != u_idx[:-1]) | (v_idx[1:] != v_idx[:-1])

    u_idx, v_idx = u_idx[first], v_idx[first]
    travel_time, length = travel_time[first], length[first]
    link_id, direction = link_id[first], direction[first]

    # int32 throughout: SciPy wants indptr and indices to share a dtype, and a
    # mismatch makes it silently copy both arrays -- which on a country-sized
    # network is hundreds of megabytes we would rather not spend.
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(u_idx, minlength=n_nodes), out=indptr[1:])
    index_dtype = np.int32 if indptr[-1] <= np.iinfo(np.int32).max else np.int64

    return (
        indptr.astype(index_dtype),
        v_idx.astype(index_dtype),
        travel_time.astype(np.float64),
        length.astype(np.float32),
        link_id.astype(np.int64),
        direction.astype(np.int8),
    )


@dataclass
class _EdgeArrays:
    """One extent's worth of network, still keyed by original OSM node ids."""

    node_ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    u: np.ndarray
    v: np.ndarray
    travel_time: np.ndarray
    length: np.ndarray
    link_id: np.ndarray
    direction: np.ndarray


def edge_arrays(links: gpd.GeoDataFrame, mode: str) -> _EdgeArrays:
    """
    Reduce a normalised link layer to the plain arrays the graph is built from.

    The link layer's geometry is dropped here and never enters the network: the
    GeoPackage keeps it, and ``link_id`` is enough to find it again.
    """
    directed = links_module.directed_edges(links, mode)

    # Node coordinates come from the link ends. Both ends of every link are in
    # the directed frame (each link contributes at least one row, and u/v are
    # swapped on the reverse row), so taking first and last vertex of each link
    # covers every node the edges can reference.
    geometry = links.geometry
    if links.crs is not None and str(links.crs).upper() != WGS84:
        geometry = geometry.to_crs(WGS84)
    coords = geometry.to_numpy()

    head = shapely.get_point(coords, 0)
    tail = shapely.get_point(coords, -1)

    node_ids = np.concatenate(
        [links["u"].to_numpy(dtype=np.int64), links["v"].to_numpy(dtype=np.int64)]
    )
    lon = np.concatenate([shapely.get_x(head), shapely.get_x(tail)])
    lat = np.concatenate([shapely.get_y(head), shapely.get_y(tail)])
    node_ids, first = np.unique(node_ids, return_index=True)

    return _EdgeArrays(
        node_ids=node_ids,
        lon=lon[first],
        lat=lat[first],
        u=directed["u"].to_numpy(dtype=np.int64),
        v=directed["v"].to_numpy(dtype=np.int64),
        travel_time=directed["travel_time_s"].to_numpy(dtype=float),
        length=directed["length_m"].to_numpy(dtype=np.float32),
        link_id=directed["link_id"].to_numpy(dtype=np.int64),
        direction=directed["direction"].to_numpy(dtype=np.int8),
    )


def _assemble(mode: str, parts: list[_EdgeArrays]) -> RoutableNetwork:
    """
    Turn one or more extents into a single routable network.

    Extents are stitched purely on OSM node id: a node on the seam carries the
    same id in both, so concatenating and de-duplicating reconnects them. Links
    duplicated across an overlap collapse in :func:`_build_csr`, which keeps
    only the fastest edge per node pair. Callers merging several extents must
    make ``link_id`` unique across them first, or results will join back onto
    the wrong rows.
    """
    node_ids = np.concatenate([p.node_ids for p in parts])
    lon = np.concatenate([p.lon for p in parts])
    lat = np.concatenate([p.lat for p in parts])

    node_ids, first = np.unique(node_ids, return_index=True)
    lon, lat = lon[first], lat[first]

    u_osm = np.concatenate([p.u for p in parts])
    v_osm = np.concatenate([p.v for p in parts])
    travel_time = np.concatenate([p.travel_time for p in parts])
    length = np.concatenate([p.length for p in parts])
    link_id = np.concatenate([p.link_id for p in parts])
    direction = np.concatenate([p.direction for p in parts])

    # An edge can only survive if both of its endpoints did.
    u_idx = np.searchsorted(node_ids, u_osm)
    v_idx = np.searchsorted(node_ids, v_osm)
    valid = (
        (u_idx < node_ids.size)
        & (v_idx < node_ids.size)
        & (node_ids[np.clip(u_idx, 0, node_ids.size - 1)] == u_osm)
        & (node_ids[np.clip(v_idx, 0, node_ids.size - 1)] == v_osm)
    )
    u_idx, v_idx = u_idx[valid], v_idx[valid]
    travel_time, length = travel_time[valid], length[valid]
    link_id, direction = link_id[valid], direction[valid]

    transformer = Transformer.from_crs(WGS84, PROJECTED_CRS, always_xy=True)
    x, y = transformer.transform(lon, lat)

    indptr, indices, travel_time, length, link_id, direction = _build_csr(
        u_idx, v_idx, travel_time, length, link_id, direction, len(node_ids)
    )

    # Walking is modelled bidirectionally, so weak and strong components agree;
    # cycling is genuinely directed.
    keep = _largest_component(
        indptr, indices, travel_time, len(node_ids), directed=(mode != "walk")
    )
    return _subset(
        RoutableNetwork(
            mode=mode,
            node_ids=node_ids,
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            indptr=indptr,
            indices=indices,
            travel_time=travel_time,
            length=length,
            link_id=link_id,
            direction=direction,
        ),
        keep,
    )


def network_from_links(links: gpd.GeoDataFrame, mode: str) -> RoutableNetwork:
    """
    Build a routable network from a link layer -- stage two, in one call.

    The layer is normalised first (:func:`valma_bike_and_walk.links.normalise`),
    so lengths and speeds are recomputed from what is actually in the table and
    hand-drawn links get endpoints. Pass a layer straight from
    :func:`valma_bike_and_walk.links.read_links`; it does not have to be pristine.
    """
    validate_mode(mode)
    normalised = links_module.normalise(links, mode)
    network = _assemble(mode, [edge_arrays(normalised, mode)])
    logger.info("Network ready: %d nodes, %d edges", network.n_nodes, network.n_edges)
    return network


def build_links(
    settings: Settings,
    mode: str,
    area: str | None = None,
    bbox: Sequence[float] | None = None,
    force_reload: bool = False,
    links_path: Path | None = None,
) -> tuple[Path, gpd.GeoDataFrame]:
    """
    Stage one: read the PBF and write the editable link layer, or reuse it.

    Returns the GeoPackage's path alongside the layer, so a caller that only
    wanted the file can ignore the frame and one that wants to keep building can
    skip reading it back.
    """
    validate_mode(mode)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    clip, extent_key = resolve_clip(settings, area, bbox, force_reload)
    path = (
        Path(links_path) if links_path else settings.links_cache_path(mode, extent_key)
    )

    if path.exists() and not force_reload:
        logger.info("Loading links from %s", path)
        return path, links_module.read_links(path)

    links = links_module.annotate(read_links(settings, mode, clip), mode)
    links_module.write_links(links, path)
    return path, links


def build_network(
    settings: Settings,
    mode: str,
    area: str | None = None,
    bbox: Sequence[float] | None = None,
    force_reload: bool = False,
    links_path: Path | None = None,
) -> RoutableNetwork:
    """
    Build (or load from cache) a routable network for one mode, PBF to graph.

    Both stages are cached: the link GeoPackage from stage one and the ``.npz``
    from stage two. Edit the GeoPackage between the two -- or point
    ``links_path`` at your own edited copy -- and delete the ``.npz`` (or pass
    ``force_reload``) to have the edits taken up.
    """
    validate_mode(mode)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    _, extent_key = resolve_clip(settings, area, bbox, force_reload)
    cache_path = settings.network_cache_path(mode, extent_key)

    if cache_path.exists() and not force_reload:
        logger.info("Loading network from cache: %s", cache_path)
        return RoutableNetwork.load(cache_path)

    _, links = build_links(
        settings, mode, area, bbox, force_reload, links_path=links_path
    )
    network = network_from_links(links, mode)
    network.save(cache_path)
    return network


def _subset(network: RoutableNetwork, keep: np.ndarray) -> RoutableNetwork:
    """Restrict a network to a boolean node mask, renumbering the CSR."""
    if keep.all():
        return network

    remap = np.full(network.n_nodes, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)

    rows = np.repeat(np.arange(network.n_nodes), np.diff(network.indptr))
    cols = network.indices
    edge_keep = keep[rows] & keep[cols]

    return _from_coo(
        mode=network.mode,
        node_ids=network.node_ids[keep],
        x=network.x[keep],
        y=network.y[keep],
        u_idx=remap[rows[edge_keep]],
        v_idx=remap[cols[edge_keep]],
        travel_time=network.travel_time[edge_keep],
        length=network.length[edge_keep],
        link_id=network.link_id[edge_keep],
        direction=network.direction[edge_keep],
        crs=network.crs,
    )


def _from_coo(
    mode: str,
    node_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u_idx: np.ndarray,
    v_idx: np.ndarray,
    travel_time: np.ndarray,
    length: np.ndarray,
    link_id: np.ndarray,
    direction: np.ndarray,
    crs: str,
) -> RoutableNetwork:
    indptr, indices, travel_time, length, link_id, direction = _build_csr(
        u_idx, v_idx, travel_time, length, link_id, direction, node_ids.shape[0]
    )
    return RoutableNetwork(
        mode=mode,
        node_ids=node_ids,
        x=x,
        y=y,
        indptr=indptr,
        indices=indices,
        travel_time=travel_time,
        length=length,
        link_id=link_id,
        direction=direction,
        crs=crs,
    )


def load_network(path: Path) -> RoutableNetwork:
    return RoutableNetwork.load(path)


def to_networkx(network: RoutableNetwork):  # pragma: no cover - convenience only
    """
    Convert to a NetworkX DiGraph, for plotting or interop.

    Only sensible for city-sized extracts; this is exactly the representation
    that does not scale to the whole country.
    """
    import networkx as nx

    G = nx.DiGraph()
    for i, node_id in enumerate(network.node_ids):
        G.add_node(int(node_id), x=float(network.x[i]), y=float(network.y[i]))

    rows = np.repeat(np.arange(network.n_nodes), np.diff(network.indptr))
    for u, v, t, length in zip(
        rows, network.indices, network.travel_time, network.length
    ):
        G.add_edge(
            int(network.node_ids[u]),
            int(network.node_ids[v]),
            travel_time=float(t),
            length=float(length),
        )
    G.graph["crs"] = network.crs
    return G
