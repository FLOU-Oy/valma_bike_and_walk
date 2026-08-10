"""A compact, routable representation of a travel network.

A NetworkX ``MultiDiGraph`` stores a Python dict per node and per edge. That is
convenient at city scale and hopeless at country scale: Finland's walking
network runs to millions of nodes, which costs tens of gigabytes as dicts and
makes every shortest-path call slow.

This module keeps the same information in a handful of numpy arrays plus a SciPy
CSR matrix -- roughly two orders of magnitude smaller, and routable with SciPy's
compiled Dijkstra instead of NetworkX's pure-Python one.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import shapely
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import box

from valma_bike_and_walk.config import (
    FINLAND_BOUNDS,
    PROJECTED_CRS,
    WGS84,
    Settings,
    validate_mode,
)
from valma_bike_and_walk.extract import (
    build_directed_edges,
    read_network_tables,
    resolve_clip,
)
from valma_bike_and_walk.speeds import profile_for

logger = logging.getLogger(__name__)

TRAVEL_TIME_COL = "travel_time"


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
    crs: str = PROJECTED_CRS
    #: object array of WKB bytes, one per directed edge, aligned with
    #: travel_time/length -- i.e. edges for row i live at
    #: geometry[indptr[i]:indptr[i+1]], same as every other per-edge array.
    #: WGS84 (the CRS the geometry was captured in, before node x/y are
    #: projected). None unless the network was built with keep_geometry=True.
    geometry: np.ndarray | None = None

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
        payload = dict(
            mode=np.array(self.mode),
            crs=np.array(self.crs),
            node_ids=self.node_ids,
            x=self.x,
            y=self.y,
            indptr=self.indptr,
            indices=self.indices,
            travel_time=self.travel_time,
            length=self.length,
        )
        if self.geometry is not None:
            payload["geometry"] = self.geometry
        np.savez_compressed(path, **payload)  # type: ignore[arg-type]
        logger.info("Saved network to %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @classmethod
    def load(cls, path: Path) -> "RoutableNetwork":
        path = Path(path)
        # The geometry array is object-dtype WKB and needs pickling to read;
        # every other field doesn't, so pickling only turns on for files that
        # actually carry geometry -- a plain build keeps the stricter default.
        with np.load(path, allow_pickle=False) as probe:
            has_geometry = "geometry" in probe.files
        with np.load(path, allow_pickle=has_geometry) as data:
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
                geometry=data["geometry"] if has_geometry else None,
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
    n_nodes: int,
    geometry: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Build CSR arrays, keeping only the fastest edge for any repeated node pair.

    Parallel edges between the same two nodes are common in OSM, and a shortest
    path can only ever use the quickest one.

    ``geometry``, if given, is an object array of per-edge WKB bytes aligned
    with ``travel_time``/``length``; it is carried through the same sort and
    dedup so the returned array stays aligned with the returned CSR edges.
    """
    if u_idx.shape[0] == 0:
        return (
            np.zeros(n_nodes + 1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float32),
            None if geometry is None else np.empty(0, dtype=object),
        )

    order = np.lexsort((travel_time, v_idx, u_idx))
    u_idx, v_idx = u_idx[order], v_idx[order]
    travel_time, length = travel_time[order], length[order]
    if geometry is not None:
        geometry = geometry[order]

    first = np.empty(u_idx.shape[0], dtype=bool)
    first[0] = True
    first[1:] = (u_idx[1:] != u_idx[:-1]) | (v_idx[1:] != v_idx[:-1])

    u_idx, v_idx = u_idx[first], v_idx[first]
    travel_time, length = travel_time[first], length[first]
    if geometry is not None:
        geometry = geometry[first]

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
        geometry,
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
    #: object array of per-edge WKB bytes, aligned with u/v/travel_time/length.
    #: None unless keep_geometry was requested.
    geometry: np.ndarray | None = None


def _extract_arrays(
    settings: Settings, mode: str, clip, keep_geometry: bool = False
) -> _EdgeArrays:
    """
    Read one extent and reduce it to plain arrays keyed by OSM node id.

    Returning compact arrays rather than DataFrames lets the caller drop the
    pandas objects immediately, which is what makes tiled building affordable.
    """
    nodes, edges = read_network_tables(settings, mode, clip=clip)

    # Speed first, simplification second: pyrosm sums the columns named in
    # length_cols along each collapsed chain, so travel time stays exact while
    # the tags that produced it can safely be discarded.
    profile = profile_for(mode)
    edges[TRAVEL_TIME_COL] = profile.travel_times_seconds(
        edges["length"].to_numpy(dtype=float),
        edges["highway"],
        edges["surface"] if "surface" in edges.columns else None,
    )

    nodes, directed = build_directed_edges(nodes, edges, mode, TRAVEL_TIME_COL)
    logger.info(
        "Simplified to %d nodes and %d directed edges", len(nodes), len(directed)
    )

    # simplify_graph keeps the full merged road geometry per collapsed chain
    # (still WGS84, as read from the PBF); only materialise it into WKB when
    # asked, since it roughly doubles this step's memory otherwise.
    geometry = shapely.to_wkb(directed.geometry.to_numpy()) if keep_geometry else None

    return _EdgeArrays(
        node_ids=nodes["id"].to_numpy(dtype=np.int64),
        lon=nodes["lon"].to_numpy(dtype=float),
        lat=nodes["lat"].to_numpy(dtype=float),
        u=directed["u"].to_numpy(dtype=np.int64),
        v=directed["v"].to_numpy(dtype=np.int64),
        travel_time=directed[TRAVEL_TIME_COL].to_numpy(dtype=float),
        length=directed["length"].to_numpy(dtype=np.float32),
        geometry=geometry,
    )


def _assemble(mode: str, parts: list[_EdgeArrays]) -> RoutableNetwork:
    """
    Turn one or more extents into a single routable network.

    Tiles are stitched purely on OSM node id: a node on a tile boundary carries
    the same id in both tiles, so concatenating and de-duplicating reconnects
    them. Ways duplicated across overlapping tiles collapse in ``_build_csr``,
    which keeps only the fastest edge per node pair -- and a chain that one tile
    simplified further than another still costs the same total time, so shortest
    paths are unaffected.
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
    geometry = (
        np.concatenate([p.geometry for p in parts])
        if parts[0].geometry is not None
        else None
    )

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
    if geometry is not None:
        geometry = geometry[valid]

    transformer = Transformer.from_crs(WGS84, PROJECTED_CRS, always_xy=True)
    x, y = transformer.transform(lon, lat)

    indptr, indices, travel_time, length, geometry = _build_csr(
        u_idx, v_idx, travel_time, length, len(node_ids), geometry=geometry
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
            geometry=geometry,
        ),
        keep,
    )


def build_network(
    settings: Settings,
    mode: str,
    area: str | None = None,
    bbox: Sequence[float] | None = None,
    search_bbox: Sequence[float] | None = None,
    force_reload: bool = False,
    keep_geometry: bool = False,
) -> RoutableNetwork:
    """
    Build (or load from cache) a routable network for one mode.

    Speeds come from this project's walk/bike profiles, applied per way *before*
    simplification so each collapsed chain sums real per-segment travel times.

    Reading a whole country in one call needs more memory than most machines
    have; use :func:`build_network_tiled` for that.

    ``keep_geometry`` additionally carries each edge's road geometry through to
    the result (see :attr:`RoutableNetwork.geometry`), for exporting the
    network to a GeoPackage. It costs extra memory and gets its own cache file,
    so a plain build never loads (or is shadowed by) a --gpkg one.
    """
    validate_mode(mode)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    clip, extent_key = resolve_clip(settings, area, bbox, search_bbox, force_reload)
    cache_path = settings.network_cache_path(mode, extent_key, geometry=keep_geometry)

    if cache_path.exists() and not force_reload:
        logger.info("Loading network from cache: %s", cache_path)
        return RoutableNetwork.load(cache_path)

    network = _assemble(mode, [_extract_arrays(settings, mode, clip, keep_geometry)])

    logger.info("Network ready: %d nodes, %d edges", network.n_nodes, network.n_edges)
    network.save(cache_path)
    return network


def iter_tiles(
    bounds: Sequence[float],
    tile_degrees: float,
    overlap_degrees: float = 0.02,
) -> list[list[float]]:
    """
    Split a lon/lat box into overlapping tiles.

    The overlap matters: pyrosm clips ways at the tile edge, and without a seam
    of shared geometry two tiles can meet without sharing a node, leaving a cut
    in the merged network exactly where a road crosses the boundary.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    tiles = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            tiles.append(
                [
                    max(min_lon, lon - overlap_degrees),
                    max(min_lat, lat - overlap_degrees),
                    min(max_lon, lon + tile_degrees + overlap_degrees),
                    min(max_lat, lat + tile_degrees + overlap_degrees),
                ]
            )
            lon += tile_degrees
        lat += tile_degrees
    return tiles


def _tile_cache_path(
    settings: Settings, mode: str, tile: Sequence[float], geometry: bool = False
) -> Path:
    key = "_".join(f"{c:.4f}" for c in tile)
    suffix = "_geom" if geometry else ""
    return (
        settings.cache_dir
        / "tiles"
        / f"{settings.pbf_path.stem}_{key}_{mode}{suffix}.npz"
    )


def _build_one_tile(
    settings: Settings,
    mode: str,
    tile: Sequence[float],
    force_reload: bool,
    keep_geometry: bool = False,
) -> Path | None:
    """
    Extract one tile to its own .npz and return the path (None if empty).

    Tiles are cached individually so that a country-sized build, which can run
    for hours, resumes instead of starting over after an interruption. Workers
    hand back a path rather than the arrays themselves, so a big tile is not
    pickled across the process boundary.
    """
    path = _tile_cache_path(settings, mode, tile, geometry=keep_geometry)
    if path.exists() and not force_reload:
        return path

    try:
        part = _extract_arrays(
            settings, mode, box(tile[0], tile[1], tile[2], tile[3]), keep_geometry
        )
    except ValueError:
        # Most of Finland's area is forest, lake and sea; empty tiles are normal.
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        node_ids=part.node_ids,
        lon=part.lon,
        lat=part.lat,
        u=part.u,
        v=part.v,
        travel_time=part.travel_time,
        length=part.length,
    )
    if part.geometry is not None:
        payload["geometry"] = part.geometry
    np.savez_compressed(path, **payload)  # type: ignore[arg-type]
    return path


def _load_tile(path: Path) -> _EdgeArrays:
    with np.load(path, allow_pickle=False) as probe:
        has_geometry = "geometry" in probe.files
    with np.load(path, allow_pickle=has_geometry) as data:
        return _EdgeArrays(
            node_ids=data["node_ids"],
            lon=data["lon"],
            lat=data["lat"],
            u=data["u"],
            v=data["v"],
            travel_time=data["travel_time"],
            length=data["length"],
            geometry=data["geometry"] if has_geometry else None,
        )


def build_network_tiled(
    settings: Settings,
    mode: str,
    bounds: Sequence[float] = FINLAND_BOUNDS,
    tile_degrees: float = 2.0,
    overlap_degrees: float = 0.02,
    force_reload: bool = False,
    cache_key: str = "tiled",
    tile_workers: int = 1,
    keep_geometry: bool = False,
) -> RoutableNetwork:
    """
    Build a network over a large area by reading it one tile at a time.

    Peak memory is set by the busiest single tile rather than by the whole
    extent, which is what makes a country-sized build possible on an ordinary
    machine.

    The cost is one full pass over the PBF per tile: pyrosm rescans the entire
    file whatever bounding box you give it, so tiling trades I/O and CPU for
    memory. Two ways to buy that back -- raise ``tile_degrees`` so there are
    fewer passes, or raise ``tile_workers`` so passes happen concurrently.
    Memory is roughly ``tile_workers`` times the busiest tile, so raise them
    together with care.

    Individual tiles are cached, so an interrupted build resumes.

    ``keep_geometry`` -- see :func:`build_network`. It also gets its own tile
    cache files, so tiles built with and without it never collide.
    """
    validate_mode(mode)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = settings.network_cache_path(mode, cache_key, geometry=keep_geometry)

    if cache_path.exists() and not force_reload:
        logger.info("Loading network from cache: %s", cache_path)
        return RoutableNetwork.load(cache_path)

    tiles = iter_tiles(bounds, tile_degrees, overlap_degrees)
    logger.info(
        "Building %s network from %d tiles of %g deg, %d worker(s)",
        mode,
        len(tiles),
        tile_degrees,
        tile_workers,
    )

    if tile_workers > 1:
        # Don't nest pools. pyrosm shards the PBF into one contiguous blob range
        # per worker with no work stealing, so a per-tile bounding box leaves
        # nearly every worker with nothing to do -- a brief flare of activity and
        # then one core grinding. Reading each tile serially and running several
        # tiles at once uses the machine far better.
        #
        # And read in memory unless told otherwise: out_of_core spills the whole
        # decoded file to temporary shards on every read, so N concurrent tiles
        # means N concurrent spills and the disk, not the CPU, sets the pace.
        engine = "in_memory" if settings.engine == "auto" else settings.engine
        settings = replace(settings, osm_workers=1, engine=engine)
        logger.info("Parallel tiles: reading each tile serially with engine=%s", engine)

    paths: list[Path] = []
    if tile_workers <= 1:
        for i, tile in enumerate(tiles, start=1):
            logger.info("Tile %d/%d: %s", i, len(tiles), [round(c, 3) for c in tile])
            path = _build_one_tile(settings, mode, tile, force_reload, keep_geometry)
            if path is not None:
                paths.append(path)
    else:
        with ProcessPoolExecutor(max_workers=tile_workers) as pool:
            futures = {
                pool.submit(
                    _build_one_tile, settings, mode, tile, force_reload, keep_geometry
                ): tile
                for tile in tiles
            }
            for done, future in enumerate(as_completed(futures), start=1):
                path = future.result()
                logger.info(
                    "Tile %d/%d done%s", done, len(tiles), "" if path else " (empty)"
                )
                if path is not None:
                    paths.append(path)

    if not paths:
        raise ValueError(f"No {mode} network found anywhere in {bounds}.")

    logger.info("Merging %d non-empty tiles", len(paths))
    parts = [_load_tile(p) for p in sorted(paths)]

    network = _assemble(mode, parts)
    logger.info("Network ready: %d nodes, %d edges", network.n_nodes, network.n_edges)
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
        crs=network.crs,
        geometry=network.geometry[edge_keep] if network.geometry is not None else None,
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
    crs: str,
    geometry: np.ndarray | None = None,
) -> RoutableNetwork:
    indptr, indices, travel_time, length, geometry = _build_csr(
        u_idx, v_idx, travel_time, length, node_ids.shape[0], geometry=geometry
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
        crs=crs,
        geometry=geometry,
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
