"""A compact, routable representation of a travel network.

A NetworkX ``MultiDiGraph`` stores a Python dict per node and per edge. That is
convenient at city scale and hopeless at country scale: Finland's walking
network runs to millions of nodes, which costs tens of gigabytes as dicts and
makes every shortest-path call slow.

This module keeps the same information in a handful of numpy arrays in CSR
layout -- roughly two orders of magnitude smaller, and routable with the
compiled kernels in :mod:`valma_bike_and_walk.routing` instead of NetworkX's
pure-Python Dijkstra.

Nodes are numbered along a Hilbert curve over their coordinates, so nodes that
are close on the map are close in memory too. A search expands outwards through
space, and this keeps the part of the arrays it is working on in cache. The
graph is an internal format: the version number in each saved file says which
layout it holds, and a file from another version is refused rather than
reinterpreted.

This is stage two of the pipeline. Its input is the link layer
(:mod:`valma_bike_and_walk.links`), not the PBF: geometry, tags and every
editable decision stay in the GeoPackage, and what lands here is the minimum
needed to route. Each directed edge keeps its ``link_id``, so a result computed
here maps straight back onto the rows you edited.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from valma_bike_and_walk import links as links_module
from valma_bike_and_walk import routing
from valma_bike_and_walk.cache import (
    code_digest,
    dump_key,
    file_stamp,
    load_key,
    make_key,
    read_sidecar,
    remove_sidecar,
    stale_reason,
    write_sidecar,
)
from valma_bike_and_walk.config import (
    PROJECTED_CRS,
    WGS84,
    Settings,
    validate_mode,
)
from valma_bike_and_walk.osm import read_links, resolve_clip

logger = logging.getLogger(__name__)

#: Layout of a saved graph. Bump it whenever what the arrays mean changes --
#: node order, edge order, which arrays exist -- so that a file written by
#: another version is refused (see :class:`NetworkFormatError`) and a cached one
#: is rebuilt, rather than either being read as if it were this one.
FORMAT_VERSION = 2

#: Bits per axis of the Hilbert key nodes are sorted by. 2**16 cells across
#: Finland is about 20 m a cell -- far finer than it needs to be to keep
#: neighbouring streets together.
_HILBERT_BITS = 16


class NetworkFormatError(ValueError):
    """A saved graph is from a different version of this package."""


#: Modules whose code decides what a graph built from a link layer looks like.
#: A change to any of them makes every cached graph stale.
BUILD_MODULES = (
    "valma_bike_and_walk.config",
    "valma_bike_and_walk.links",
    "valma_bike_and_walk.network",
    "valma_bike_and_walk.speeds",
)

#: The same, for a link layer extracted from a PBF.
EXTRACT_MODULES = (
    "valma_bike_and_walk.config",
    "valma_bike_and_walk.links",
    "valma_bike_and_walk.osm",
    "valma_bike_and_walk.speeds",
)


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
    #: What this graph was built from, as a JSON cache key (see
    #: :mod:`valma_bike_and_walk.cache`); empty if nobody recorded it.
    source_key: str = ""

    def __post_init__(self) -> None:
        self._tree: cKDTree | None = None
        self._graph: routing.Graph | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.indices.shape[0])

    def routing_graph(self) -> routing.Graph:
        """The arrays the routing kernels search, weighted by travel time. Built once."""
        if self._graph is None:
            self._graph = routing.Graph.from_csr(
                self.indptr, self.indices, self.travel_time
            )
        return self._graph

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

    def save(self, path: Path, compress: bool = True) -> Path:
        """
        Persist as an .npz -- far smaller and faster than GraphML.

        ``compress=False`` trades disk for load time: a country-sized graph
        takes seconds to decompress on every load, which is worth avoiding for a
        cache that is read on every run and can be rebuilt at will.

        Written to a temporary file and moved into place, so a run that dies
        mid-write leaves the previous file (or none), never a truncated one.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(handle, "wb") as out:
                (np.savez_compressed if compress else np.savez)(
                    out,
                    format_version=np.array(FORMAT_VERSION),
                    mode=np.array(self.mode),
                    crs=np.array(self.crs),
                    source_key=np.array(self.source_key),
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
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        logger.info("Saved network to %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @classmethod
    def load(cls, path: Path) -> "RoutableNetwork":
        path = Path(path)
        with np.load(Path(path), allow_pickle=False) as data:
            version = (
                int(data["format_version"]) if "format_version" in data.files else 1
            )
            if version != FORMAT_VERSION:
                raise NetworkFormatError(
                    f"{path} is graph format {version}; this version of "
                    f"valma_bike_and_walk needs format {FORMAT_VERSION}. Rebuild "
                    "it with `valma build --links <links.gpkg> --mode "
                    f"{str(data['mode'])}`."
                )
            return cls(
                mode=str(data["mode"]),
                crs=str(data["crs"]),
                source_key=(
                    str(data["source_key"]) if "source_key" in data.files else ""
                ),
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

    # int32 wherever it fits: a country-sized network has millions of edges, and
    # every array here is saved, loaded and read by every routing thread. The
    # routing kernels take the dtypes they are compiled for from routing.Graph.
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
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    # Number nodes along a Hilbert curve, so that map neighbours are memory
    # neighbours (see the module docstring).
    order = hilbert_order(x, y)
    node_ids, x, y = node_ids[order], x[order], y[order]
    new_index = np.empty_like(order)
    new_index[order] = np.arange(order.shape[0])
    u_idx, v_idx = new_index[u_idx], new_index[v_idx]

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
            x=x,
            y=y,
            indptr=indptr,
            indices=indices,
            travel_time=travel_time,
            length=length,
            link_id=link_id,
            direction=direction,
        ),
        keep,
    )


def _hilbert_keys(ix: np.ndarray, iy: np.ndarray, bits: int) -> np.ndarray:
    """
    Distance along a Hilbert curve of integer grid cells ``(ix, iy)``.

    The classic bit-twiddling form, vectorised over every point at once: one
    pass per bit level, rotating each quadrant into the orientation the curve
    enters it in.
    """
    x = ix.astype(np.int64).copy()
    y = iy.astype(np.int64).copy()
    key = np.zeros(x.shape[0], dtype=np.int64)
    side = 1 << bits
    s = side >> 1
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        key += s * s * ((3 * rx) ^ ry)
        # Rotate the lower quadrants so the curve inside them lines up.
        flip = ~ry & rx
        x = np.where(flip, side - 1 - x, x)
        y = np.where(flip, side - 1 - y, y)
        swap = ~ry
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s >>= 1
    return key


def hilbert_order(
    x: np.ndarray, y: np.ndarray, bits: int = _HILBERT_BITS
) -> np.ndarray:
    """Permutation that sorts points along a Hilbert curve over their bounding box."""
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    cells = (1 << bits) - 1
    span = max(float(np.ptp(x)), float(np.ptp(y)), 1e-9)
    ix = np.floor((x - x.min()) / span * cells).astype(np.int64)
    iy = np.floor((y - y.min()) / span * cells).astype(np.int64)
    return np.argsort(_hilbert_keys(ix, iy, bits), kind="stable")


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


def stored_network_key(path: Path | str) -> dict[str, Any] | None:
    """
    The cache key a saved network was built with, without loading the network.

    An ``.npz`` is a zip of separately compressed arrays, so this decompresses
    one short string, not the graph. A file that cannot be read at all -- or one
    saved before keys were recorded -- has no key, which makes it stale.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            if "source_key" not in data.files:
                return None
            return load_key(str(data["source_key"]))
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None


def links_network_key(
    links_path: Path | str,
    mode: str,
    connectors_path: Path | str | None = None,
) -> dict[str, Any]:
    """What a graph built from ``links_path`` depends on, as it stands now."""
    parts: dict[str, Any] = {}
    if connectors_path is not None:
        # Only present when given, so graphs without connectors keep their key.
        parts["connectors"] = file_stamp(connectors_path)
    return make_key(
        "network",
        format=FORMAT_VERSION,
        mode=mode,
        links=file_stamp(links_path),
        code=code_digest(*BUILD_MODULES),
        **parts,
    )


def network_for_links(
    links_path: Path | str,
    mode: str,
    network_path: Path | str,
    force_reload: bool = False,
    links: gpd.GeoDataFrame | None = None,
    compress: bool = False,
    connectors_path: Path | str | None = None,
) -> RoutableNetwork:
    """
    The graph for a link layer: from ``network_path`` if it is current, else built.

    ``connectors_path`` is a connector layer appended to the link layer before
    the build; it is part of what the graph is current against.

    ``compress`` is for a graph meant to be kept (``valma build``); a cache is
    left uncompressed, because it is loaded on every run and is disposable.

    "Current" means built from this very version of the link layer, by this
    version of the build code -- see :mod:`valma_bike_and_walk.cache`. Edit the
    layer in QGIS, add elevation or car volumes to it, or change a speed profile,
    and the next call rebuilds rather than routing on the graph from before.

    ``links`` may hand over the layer already in memory, when the caller has
    just written it to ``links_path``, to save reading it straight back.
    """
    validate_mode(mode)
    links_path = Path(links_path)
    network_path = Path(network_path)
    # Stamp before reading, so an edit saved mid-build leaves us stale, not wrong.
    key = links_network_key(links_path, mode, connectors_path)

    if network_path.exists() and not force_reload:
        reason = stale_reason(stored_network_key(network_path), key)
        if reason is None:
            logger.info("Loading network from %s", network_path)
            return RoutableNetwork.load(network_path)
        logger.info("Rebuilding %s: %s", network_path, reason)

    if links is None:
        links = links_module.read_links(links_path)
    if connectors_path is not None:
        from valma_bike_and_walk.connectors import merge_connectors

        links = merge_connectors(links, links_module.read_links(connectors_path))
    network = network_from_links(links, mode)
    network.source_key = dump_key(key)
    network.save(network_path, compress=compress)
    return network


def _extract_key(
    settings: Settings,
    mode: str,
    clip: shapely.Geometry | None,
    extent_key: str,
) -> dict[str, Any]:
    """What a link layer extracted from the PBF depends on."""
    return make_key(
        "links",
        mode=mode,
        extent=extent_key,
        # A named area resolves to whatever boundary the PBF holds, so the key
        # follows the geometry rather than just the name.
        clip=(
            None
            if clip is None
            else hashlib.blake2b(shapely.to_wkb(clip), digest_size=16).hexdigest()
        ),
        pbf=file_stamp(settings.require_pbf()),
        code=code_digest(*EXTRACT_MODULES),
    )


def _ensure_links(
    settings: Settings,
    mode: str,
    area: str | None,
    bbox: Sequence[float] | None,
    force_reload: bool,
    links_path: Path | None,
) -> tuple[Path, gpd.GeoDataFrame | None]:
    """
    Make sure the link layer exists and is current, without reading it if so.

    Returns its path, and the layer itself when this call had to extract it --
    None when what was on disk could be reused, and so has not been read.

    Two kinds of layer are treated differently on purpose. One under
    ``--cache-dir`` is ours: keyed on the PBF, the extent and the extract code,
    and re-extracted whenever any of them changes. One at an explicit
    ``links_path`` is the user's -- possibly hours of QGIS edits -- so an
    existing file there is never overwritten without ``force_reload``.
    """
    clip, extent_key = resolve_clip(settings, area, bbox, force_reload)

    if links_path is not None:
        path = Path(links_path)
        if path.exists() and not force_reload:
            return path, None
        links = links_module.annotate(read_links(settings, mode, clip), mode)
        links_module.write_links(links, path)
        return path, links

    path = settings.links_cache_path(mode, extent_key)
    if path.exists() and not force_reload:
        if not settings.require_pbf().exists():
            logger.warning(
                "%s is gone; reusing %s without being able to check it is current",
                settings.pbf_path,
                path,
            )
            return path, None
        key = _extract_key(settings, mode, clip, extent_key)
        reason = stale_reason(read_sidecar(path), key)
        if reason is None:
            return path, None
        logger.info("Re-extracting %s: %s", path, reason)
    else:
        key = _extract_key(settings, mode, clip, extent_key)

    remove_sidecar(path)
    links = links_module.annotate(read_links(settings, mode, clip), mode)
    links_module.write_links(links, path)
    write_sidecar(path, key)
    return path, links


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

    Without ``links_path`` the layer is a cache under ``settings.cache_dir``,
    re-extracted whenever the PBF, the extent or the extract code changes. With
    one, an existing file there is the user's and is reused as it stands unless
    ``force_reload`` is given.
    """
    validate_mode(mode)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    path, links = _ensure_links(settings, mode, area, bbox, force_reload, links_path)
    if links is None:
        logger.info("Loading links from %s", path)
        links = links_module.read_links(path)
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

    Both stages are cached -- the link GeoPackage from stage one and the
    ``.npz`` from stage two -- and each is rebuilt by itself once what it was
    built from changes: a new PBF re-extracts the layer, and a changed layer
    (re-extracted, or edited at ``links_path``) rebuilds the graph. Nothing has
    to be deleted by hand; ``force_reload`` rebuilds both regardless.
    """
    validate_mode(mode)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    path, links = _ensure_links(settings, mode, area, bbox, force_reload, links_path)
    _, extent_key = resolve_clip(settings, area, bbox)
    return network_for_links(
        path,
        mode,
        settings.network_cache_path(mode, extent_key),
        force_reload=force_reload,
        links=links,
    )


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
