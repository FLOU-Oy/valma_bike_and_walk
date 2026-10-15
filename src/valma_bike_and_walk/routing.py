"""Shortest-path kernels, compiled with Numba.

Everything this project routes -- a point-to-point matrix, a zone matrix
aggregated over weighted access points, an all-or-nothing assignment, the
viewer's routes and select-link pruning (:class:`Searcher`) -- is a
sequence of shortest-path searches followed by something cheap
done with each search's result. This module is those searches, and that
something, compiled together into one loop per operation.

Three properties are what make it fast on a country-sized network, and all
three are things a general-purpose library call cannot offer:

- **A search costs what it explores, not what the network holds.** The
  distance array is allocated once per thread and only the nodes a search
  actually touched are reset afterwards, so a 15-minute search on a network of
  millions of nodes costs a few thousand nodes' worth of work. (A dense
  ``(sources, n_nodes)`` result block, as SciPy returns, costs ``n_nodes`` per
  source whatever the cutoff.)
- **A search stops as soon as it has nothing left to find**: at ``limit``, or
  once every target it was asked for has been settled.
- **The work after a search happens in the same compiled loop.** A zone row is
  accumulated straight into its numerator and denominator; an assignment loads
  its demand onto the shortest-path tree in one pass over the settled nodes,
  farthest first, pushing each node's flow onto the edge it was reached by.
  That is O(settled nodes) however many destinations there are and however long
  their paths.

The kernels release the GIL, so parallelism is plain threads sharing one copy
of the graph and the demand. Each thread owns a :class:`_Workspace`; tasks are
handed out from a bounded window so a run's memory does not grow with how
finely the work was divided.

The searches are Dijkstra with a binary heap and lazy deletion: an improved
node is pushed again rather than decreased in place, and stale entries are
skipped when popped. Edge weights must be non-negative.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)

#: Initial heap capacity per thread. The heap doubles whenever it fills, so this
#: only sets how soon that first happens.
_INITIAL_HEAP = 1 << 14

#: How many tasks are kept queued per worker. Enough that no thread waits for
#: work; few enough that pending tasks are not a memory cost of their own.
_TASKS_IN_FLIGHT_PER_WORKER = 2

#: How long the driver waits for a task before checking for Ctrl-C again.
_POLL_SECONDS = 0.5

T = TypeVar("T")


@dataclass(frozen=True)
class Graph:
    """
    The arrays a search needs, in the dtypes the kernels are compiled for.

    ``weight`` is whatever the searches minimise, one value per directed edge in
    CSR order. ``edge_tail`` is each edge's source node, which the assignment's
    loading pass needs to move flow from a node onto its predecessor.
    """

    indptr: np.ndarray  # int64, (n_nodes + 1,)
    indices: np.ndarray  # int32, (n_edges,)
    weight: np.ndarray  # float64, (n_edges,)
    edge_tail: np.ndarray  # int32, (n_edges,)

    @classmethod
    def from_csr(
        cls, indptr: np.ndarray, indices: np.ndarray, weight: np.ndarray
    ) -> "Graph":
        indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        n_nodes = indptr.shape[0] - 1
        if n_nodes >= np.iinfo(np.int32).max:
            raise ValueError("Networks of 2**31 nodes or more are not supported.")
        weight = np.ascontiguousarray(weight, dtype=np.float64)
        if weight.size and float(weight.min()) < 0:
            raise ValueError("Edge weights must be non-negative for Dijkstra.")
        edge_tail = np.repeat(
            np.arange(n_nodes, dtype=np.int32), np.diff(indptr)
        ).astype(np.int32, copy=False)
        return cls(
            indptr=indptr,
            indices=np.ascontiguousarray(indices, dtype=np.int32),
            weight=weight,
            edge_tail=edge_tail,
        )

    def reversed(self) -> "Graph":
        """
        The same graph with every edge turned around, for searches *to* a node.

        Edge ``k`` here is the edge ``order[k]`` of this graph, where ``order``
        sorts the edges by head; weights travel with their edges.
        """
        order = np.argsort(self.indices, kind="stable")
        indptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.indices, minlength=self.n_nodes), out=indptr[1:])
        return Graph(
            indptr=indptr,
            indices=np.ascontiguousarray(self.edge_tail[order]),
            weight=np.ascontiguousarray(self.weight[order]),
            edge_tail=np.ascontiguousarray(self.indices[order]),
        )

    @property
    def n_nodes(self) -> int:
        return int(self.indptr.shape[0] - 1)

    @property
    def n_edges(self) -> int:
        return int(self.indices.shape[0])


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------


@njit(nogil=True, cache=True)
def _search(
    indptr,
    indices,
    weight,
    sources,
    limit,
    stamp,
    serial,
    n_targets,
    dist,
    pred_edge,
    touched,
    settled,
    heap_key,
    heap_node,
):
    """
    One Dijkstra search from ``sources``, leaving its answer in the workspace.

    Usually there is one source; with several, every one starts at cost 0 and
    the search finds the least cost from any of them.

    On return ``dist`` holds the cost to every node the search touched (``inf``
    elsewhere), ``pred_edge`` the edge each settled node was reached by, and
    ``settled[:n_settled]`` the settled nodes in the order they were settled.
    ``touched[:n_touched]`` lists every node whose ``dist`` was written, which
    is what the caller resets afterwards.

    Nothing costlier than ``limit`` is ever entered, so every finite ``dist`` is
    within it. Nodes with ``stamp[v] == serial`` are targets: once all
    ``n_targets`` of them are settled the search stops, and from that point
    on only the targets' ``dist`` values are final.

    Returns the possibly regrown heap arrays, so the workspace keeps them.
    """
    if sources.shape[0] > heap_key.shape[0]:
        heap_key = np.empty(sources.shape[0], dtype=heap_key.dtype)
        heap_node = np.empty(sources.shape[0], dtype=heap_node.dtype)
    n_touched = 0
    n_settled = 0
    remaining = n_targets

    # Every source enters at cost 0, so the heap is valid in any order.
    size = 0
    for k in range(sources.shape[0]):
        source = sources[k]
        if dist[source] == np.inf:
            dist[source] = 0.0
            pred_edge[source] = -1
            touched[n_touched] = source
            n_touched += 1
            heap_key[size] = 0.0
            heap_node[size] = source
            size += 1

    while size > 0:
        key = heap_key[0]
        node = heap_node[0]
        size -= 1
        if size > 0:
            last_key = heap_key[size]
            last_node = heap_node[size]
            i = 0
            while True:
                child = 2 * i + 1
                if child >= size:
                    break
                if child + 1 < size and heap_key[child + 1] < heap_key[child]:
                    child += 1
                if heap_key[child] >= last_key:
                    break
                heap_key[i] = heap_key[child]
                heap_node[i] = heap_node[child]
                i = child
            heap_key[i] = last_key
            heap_node[i] = last_node

        if key > dist[node]:
            continue  # stale: this node was improved after being pushed

        settled[n_settled] = node
        n_settled += 1
        if stamp[node] == serial:
            remaining -= 1
            if remaining == 0:
                break

        for edge in range(indptr[node], indptr[node + 1]):
            head = indices[edge]
            candidate = key + weight[edge]
            if candidate < dist[head] and candidate <= limit:
                if dist[head] == np.inf:
                    touched[n_touched] = head
                    n_touched += 1
                dist[head] = candidate
                pred_edge[head] = edge

                if size == heap_key.shape[0]:
                    grown_key = np.empty(2 * size, dtype=heap_key.dtype)
                    grown_node = np.empty(2 * size, dtype=heap_node.dtype)
                    grown_key[:size] = heap_key[:size]
                    grown_node[:size] = heap_node[:size]
                    heap_key = grown_key
                    heap_node = grown_node
                i = size
                size += 1
                while i > 0:
                    parent = (i - 1) // 2
                    if heap_key[parent] <= candidate:
                        break
                    heap_key[i] = heap_key[parent]
                    heap_node[i] = heap_node[parent]
                    i = parent
                heap_key[i] = candidate
                heap_node[i] = head

    return n_touched, n_settled, heap_key, heap_node


@njit(nogil=True, cache=True)
def _reset(dist, touched, n_touched):
    for k in range(n_touched):
        dist[touched[k]] = np.inf


@njit(nogil=True, cache=True)
def _mark(stamp, serial, nodes):
    """Stamp ``nodes`` as this search's targets; return how many are distinct."""
    count = 0
    for j in range(nodes.shape[0]):
        node = nodes[j]
        if stamp[node] != serial:
            stamp[node] = serial
            count += 1
    return count


@njit(nogil=True, cache=True)
def _point_rows(
    indptr,
    indices,
    weight,
    sources,
    targets,
    limit,
    out,
    stamp,
    serial,
    dist,
    pred_edge,
    touched,
    settled,
    heap_key,
    heap_node,
):
    """``out[i, j]`` = cost from ``sources[i]`` to ``targets[j]``, inf if beyond reach."""
    for i in range(sources.shape[0]):
        serial += 1
        n_targets = _mark(stamp, serial, targets)
        n_touched, _, heap_key, heap_node = _search(
            indptr,
            indices,
            weight,
            sources[i : i + 1],
            limit,
            stamp,
            serial,
            n_targets,
            dist,
            pred_edge,
            touched,
            settled,
            heap_key,
            heap_node,
        )
        for j in range(targets.shape[0]):
            out[i, j] = dist[targets[j]]
        _reset(dist, touched, n_touched)
    return serial, heap_key, heap_node


@njit(nogil=True, cache=True)
def _zone_rows(
    indptr,
    indices,
    weight,
    source_nodes,
    source_access,
    source_weight,
    source_row,
    target_nodes,
    target_access,
    target_weight,
    target_zone,
    limit,
    decay_mu,
    numerator,
    denominator,
    stamp,
    serial,
    dist,
    pred_edge,
    touched,
    settled,
    heap_key,
    heap_node,
):
    """
    Accumulate the weighted-mean terms for a run of source access points.

    Point ``a`` adds ``w_a * w_b * value(t_ab)`` to ``numerator[row_a, zone_b]``
    and ``w_a * w_b`` to ``denominator[row_a, zone_b]`` for every target point
    ``b`` it reached, where ``t_ab`` includes both access legs and ``value`` is
    ``t`` itself or ``exp(-decay_mu * t)`` when ``decay_mu`` is positive.
    """
    use_decay = decay_mu > 0.0
    for a in range(source_nodes.shape[0]):
        serial += 1
        n_targets = _mark(stamp, serial, target_nodes)
        n_touched, _, heap_key, heap_node = _search(
            indptr,
            indices,
            weight,
            source_nodes[a : a + 1],
            limit,
            stamp,
            serial,
            n_targets,
            dist,
            pred_edge,
            touched,
            settled,
            heap_key,
            heap_node,
        )
        row = source_row[a]
        weight_a = source_weight[a]
        access_a = source_access[a]
        for b in range(target_nodes.shape[0]):
            d = dist[target_nodes[b]]
            if d < np.inf:
                t = d + access_a + target_access[b]
                value = np.exp(-decay_mu * t) if use_decay else t
                w = weight_a * target_weight[b]
                zone = target_zone[b]
                numerator[row, zone] += w * value
                denominator[row, zone] += w
        _reset(dist, touched, n_touched)
    return serial, heap_key, heap_node


@njit(nogil=True, cache=True)
def _zone_rows_settled(
    indptr,
    indices,
    weight,
    source_nodes,
    source_access,
    source_weight,
    source_row,
    points_at_node_ptr,
    points_at_node,
    target_access,
    target_weight,
    target_zone,
    limit,
    decay_mu,
    numerator,
    denominator,
    stamp,
    serial,
    dist,
    pred_edge,
    touched,
    settled,
    heap_key,
    heap_node,
):
    """
    :func:`_zone_rows` for a bounded search: visit what the search settled, not every target.

    With a bound, the search stops by itself, so it runs without targets and its settled
    nodes are read through a node -> target points index. That costs at most the search
    again, where marking and scanning every target point costs O(target points) per source
    however little a short search reached.
    """
    use_decay = decay_mu > 0.0
    for a in range(source_nodes.shape[0]):
        serial += 1  # no node carries this stamp, so the search runs to the bound
        n_touched, n_settled, heap_key, heap_node = _search(
            indptr,
            indices,
            weight,
            source_nodes[a : a + 1],
            limit,
            stamp,
            serial,
            0,
            dist,
            pred_edge,
            touched,
            settled,
            heap_key,
            heap_node,
        )
        row = source_row[a]
        weight_a = source_weight[a]
        access_a = source_access[a]
        for s in range(n_settled):
            node = settled[s]
            d = dist[node]
            for k in range(points_at_node_ptr[node], points_at_node_ptr[node + 1]):
                b = points_at_node[k]
                t = d + access_a + target_access[b]
                value = np.exp(-decay_mu * t) if use_decay else t
                w = weight_a * target_weight[b]
                zone = target_zone[b]
                numerator[row, zone] += w * value
                denominator[row, zone] += w
        _reset(dist, touched, n_touched)
    return serial, heap_key, heap_node


@njit(nogil=True, cache=True)
def _aon_rows(
    indptr,
    indices,
    weight,
    edge_tail,
    sources,
    targets,
    demand_indptr,
    demand_indices,
    demand_data,
    row_offset,
    limit,
    volume,
    missed,
    flow,
    edge_length,
    od_length,
    tree_length,
    stamp,
    serial,
    dist,
    pred_edge,
    touched,
    settled,
    heap_key,
    heap_node,
):
    """
    All-or-nothing load of a run of demand rows onto ``volume``.

    Row ``r`` is source ``sources[r]`` and demand row ``row_offset + r``. Its
    demand is put on the destinations, then pushed up the shortest-path tree in
    reverse settling order: a node settles after its predecessor, so by the time
    a node is visited every descendant has already handed it their flow, and
    one pass loads every path at once.

    Demand to an unreached destination sets ``missed`` at its position in the
    demand arrays. Demand to the source itself is a zero-length trip: neither
    loaded nor missed.

    When ``od_length`` is not empty, each demand entry's position in it gets the
    length of the path its demand was loaded on (``edge_length`` summed along
    the tree, settling order, so a node's predecessor is always done first),
    NaN if unreached and 0 to the source itself. ``tree_length`` is scratch.
    """
    lengths = od_length.shape[0] > 0
    for r in range(sources.shape[0]):
        row = row_offset + r
        lo = demand_indptr[row]
        hi = demand_indptr[row + 1]
        if lo == hi:
            continue
        source = sources[r]

        serial += 1
        n_targets = 0
        for k in range(lo, hi):
            node = targets[demand_indices[k]]
            if stamp[node] != serial:
                stamp[node] = serial
                n_targets += 1

        n_touched, n_settled, heap_key, heap_node = _search(
            indptr,
            indices,
            weight,
            sources[r : r + 1],
            limit,
            stamp,
            serial,
            n_targets,
            dist,
            pred_edge,
            touched,
            settled,
            heap_key,
            heap_node,
        )

        if lengths:
            tree_length[source] = 0.0
            for s in range(1, n_settled):
                node = settled[s]
                edge = pred_edge[node]
                tree_length[node] = tree_length[edge_tail[edge]] + edge_length[edge]

        for k in range(lo, hi):
            node = targets[demand_indices[k]]
            if node == source:
                if lengths:
                    od_length[k] = 0.0
                continue
            if dist[node] == np.inf:
                missed[k] = True
                if lengths:
                    od_length[k] = np.nan
            else:
                flow[node] += demand_data[k]
                if lengths:
                    od_length[k] = tree_length[node]

        # settled[0] is the source, which has no edge to push flow onto.
        for s in range(n_settled - 1, 0, -1):
            node = settled[s]
            f = flow[node]
            if f != 0.0:
                edge = pred_edge[node]
                volume[edge] += f
                flow[edge_tail[edge]] += f
                flow[node] = 0.0
        flow[source] = 0.0

        _reset(dist, touched, n_touched)
    return serial, heap_key, heap_node


@njit(nogil=True, cache=True)
def _aon_select_rows(
    indptr,
    indices,
    weight,
    edge_tail,
    selected,
    sources,
    targets,
    demand_indptr,
    demand_indices,
    demand_data,
    row_offset,
    limit,
    volume,
    missed,
    hit,
    flow,
    select_volume,
    select_flow,
    via,
    stamp,
    serial,
    dist,
    pred_edge,
    touched,
    settled,
    heap_key,
    heap_node,
):
    """
    :func:`_aon_rows`, plus a select-link analysis of the edges in ``selected``.

    After each search, one forward pass in settling order marks the nodes whose
    shortest path from the source crosses a selected edge -- a node is settled
    after its predecessor, so ``via[node]`` is ``via[predecessor]`` or whether
    the edge between them is selected. Demand to such a node sets ``hit`` at
    its position in the demand arrays and is also pushed up the tree into
    ``select_volume``: the trips that use the selected edges, on every edge
    they use, in the same pass that loads everything else.
    """
    for r in range(sources.shape[0]):
        row = row_offset + r
        lo = demand_indptr[row]
        hi = demand_indptr[row + 1]
        if lo == hi:
            continue
        source = sources[r]

        serial += 1
        n_targets = 0
        for k in range(lo, hi):
            node = targets[demand_indices[k]]
            if stamp[node] != serial:
                stamp[node] = serial
                n_targets += 1

        n_touched, n_settled, heap_key, heap_node = _search(
            indptr,
            indices,
            weight,
            sources[r : r + 1],
            limit,
            stamp,
            serial,
            n_targets,
            dist,
            pred_edge,
            touched,
            settled,
            heap_key,
            heap_node,
        )

        via[source] = False
        for s in range(1, n_settled):
            node = settled[s]
            edge = pred_edge[node]
            via[node] = via[edge_tail[edge]] or selected[edge]

        for k in range(lo, hi):
            node = targets[demand_indices[k]]
            if node == source:
                continue
            if dist[node] == np.inf:
                missed[k] = True
            else:
                flow[node] += demand_data[k]
                if via[node]:
                    hit[k] = True
                    select_flow[node] += demand_data[k]

        for s in range(n_settled - 1, 0, -1):
            node = settled[s]
            edge = pred_edge[node]
            f = flow[node]
            if f != 0.0:
                volume[edge] += f
                flow[edge_tail[edge]] += f
                flow[node] = 0.0
            g = select_flow[node]
            if g != 0.0:
                select_volume[edge] += g
                select_flow[edge_tail[edge]] += g
                select_flow[node] = 0.0
        flow[source] = 0.0
        select_flow[source] = 0.0

        _reset(dist, touched, n_touched)
    return serial, heap_key, heap_node


@njit(nogil=True, cache=True)
def _costs(
    indptr,
    indices,
    weight,
    sources,
    targets,
    limit,
    out,
    stamp,
    serial,
    dist,
    pred_edge,
    touched,
    settled,
    heap_key,
    heap_node,
):
    """``out[j]`` = least cost from any of ``sources`` to ``targets[j]``, inf if beyond reach."""
    serial += 1
    n_targets = _mark(stamp, serial, targets)
    n_touched, _, heap_key, heap_node = _search(
        indptr,
        indices,
        weight,
        sources,
        limit,
        stamp,
        serial,
        n_targets,
        dist,
        pred_edge,
        touched,
        settled,
        heap_key,
        heap_node,
    )
    for j in range(targets.shape[0]):
        out[j] = dist[targets[j]]
    _reset(dist, touched, n_touched)
    return serial, heap_key, heap_node


@njit(nogil=True, cache=True)
def _path(
    indptr,
    indices,
    weight,
    edge_tail,
    sources,
    target,
    limit,
    stamp,
    serial,
    dist,
    pred_edge,
    touched,
    settled,
    heap_key,
    heap_node,
):
    """
    The edges of the least-cost path from ``sources[0]`` to ``target``, in order.

    The search stops as soon as ``target`` is settled. Returns an empty path
    and a cost of inf when the target is beyond reach.
    """
    serial += 1
    stamp[target] = serial
    n_touched, _, heap_key, heap_node = _search(
        indptr,
        indices,
        weight,
        sources,
        limit,
        stamp,
        serial,
        1,
        dist,
        pred_edge,
        touched,
        settled,
        heap_key,
        heap_node,
    )
    cost = dist[target]
    n_edges = 0
    if cost < np.inf:
        node = target
        while pred_edge[node] >= 0:
            n_edges += 1
            node = edge_tail[pred_edge[node]]
    edges = np.empty(n_edges, dtype=np.int64)
    node = target
    for k in range(n_edges - 1, -1, -1):
        edge = pred_edge[node]
        edges[k] = edge
        node = edge_tail[edge]
    _reset(dist, touched, n_touched)
    return edges, cost, serial, heap_key, heap_node


# --------------------------------------------------------------------------
# Per-thread state and the driver
# --------------------------------------------------------------------------


class _Workspace:
    """One thread's scratch arrays. About 32 bytes per node, plus a heap."""

    def __init__(
        self,
        n_nodes: int,
        flow: bool = False,
        n_edges: int = 0,
        select: bool = False,
        lengths: bool = False,
    ) -> None:
        self.dist = np.full(n_nodes, np.inf, dtype=np.float64)
        self.pred_edge = np.empty(n_nodes, dtype=np.int64)
        self.touched = np.empty(n_nodes, dtype=np.int32)
        self.settled = np.empty(n_nodes, dtype=np.int32)
        self.stamp = np.full(n_nodes, -1, dtype=np.int64)
        self.serial = 0
        self.heap_key = np.empty(_INITIAL_HEAP, dtype=np.float64)
        self.heap_node = np.empty(_INITIAL_HEAP, dtype=np.int32)
        self.flow = np.zeros(n_nodes if flow else 0, dtype=np.float64)
        self.volume = np.zeros(n_edges if flow else 0, dtype=np.float64)
        # Select-link scratch: only allocated when a run asks for it.
        self.via = np.zeros(n_nodes if select else 0, dtype=np.bool_)
        self.select_flow = np.zeros(n_nodes if select else 0, dtype=np.float64)
        self.select_volume = np.zeros(n_edges if select else 0, dtype=np.float64)
        # Path lengths along the tree: only allocated when a run asks for them.
        self.tree_length = np.zeros(n_nodes if lengths else 0, dtype=np.float64)


def workspace_bytes(n_nodes: int, n_edges: int, flow: bool = False) -> int:
    """Roughly what one thread's :class:`_Workspace` costs, for logging."""
    per_node = 8 + 8 + 4 + 4 + 8 + (8 if flow else 0)
    return n_nodes * per_node + (n_edges * 8 if flow else 0) + _INITIAL_HEAP * 12


def _run(
    tasks: Sequence[T],
    solve: Callable[[T, _Workspace], None],
    make_workspace: Callable[[], _Workspace],
    workers: int,
    progress_every: int,
) -> list[_Workspace]:
    """
    Run ``solve(task, workspace)`` for every task, on up to ``workers`` threads.

    Each thread creates its workspace on first use and keeps it for the run.
    Returns every workspace created, for callers that accumulate into them.
    """
    workspaces: list[_Workspace] = []
    lock = threading.Lock()
    local = threading.local()

    def workspace() -> _Workspace:
        ws = getattr(local, "ws", None)
        if ws is None:
            ws = make_workspace()
            local.ws = ws
            with lock:
                workspaces.append(ws)
        return ws

    def report(done: int) -> None:
        if progress_every and done % progress_every == 0:
            logger.info("  %d/%d chunks", done, len(tasks))

    if workers <= 1 or len(tasks) <= 1:
        for done, task in enumerate(tasks, start=1):
            solve(task, workspace())
            report(done)
        return workspaces

    with ThreadPoolExecutor(max_workers=workers) as pool:
        queue = iter(tasks)
        pending: set[Future[None]] = set()

        def job(task: T) -> None:
            solve(task, workspace())

        def submit_next() -> None:
            task = next(queue, None)
            if task is not None:
                pending.add(pool.submit(job, task))

        for _ in range(min(len(tasks), _TASKS_IN_FLIGHT_PER_WORKER * workers)):
            submit_next()

        done = 0
        try:
            while pending:
                # A timeout rather than blocking outright keeps Ctrl-C working.
                finished, _ = wait(
                    pending, timeout=_POLL_SECONDS, return_when=FIRST_COMPLETED
                )
                for future in finished:
                    pending.discard(future)
                    future.result()
                    done += 1
                    report(done)
                    submit_next()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return workspaces


def task_size(n_items: int, workers: int, chunk_size: int | None) -> int:
    """Sources per task: enough tasks to balance threads, few enough to amortise."""
    if chunk_size is not None:
        return max(1, int(chunk_size))
    return int(np.clip(np.ceil(n_items / (16 * max(1, workers))), 1, 256))


def _limit(limit: float | None) -> float:
    return np.inf if limit is None else float(limit)


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


def point_matrix(
    graph: Graph,
    sources: np.ndarray,
    targets: np.ndarray,
    out: np.ndarray,
    limit: float | None = None,
    workers: int = 1,
    chunk_size: int | None = None,
    progress_every: int = 10,
) -> None:
    """Fill ``out[i, j]`` with the least cost from ``sources[i]`` to ``targets[j]``."""
    sources = np.ascontiguousarray(sources, dtype=np.int32)
    targets = np.ascontiguousarray(targets, dtype=np.int32)
    size = task_size(sources.shape[0], workers, chunk_size)
    starts = list(range(0, sources.shape[0], size))
    bound = _limit(limit)

    def solve(start: int, ws: _Workspace) -> None:
        ws.serial, ws.heap_key, ws.heap_node = _point_rows(
            graph.indptr,
            graph.indices,
            graph.weight,
            sources[start : start + size],
            targets,
            bound,
            out[start : start + size],
            ws.stamp,
            ws.serial,
            ws.dist,
            ws.pred_edge,
            ws.touched,
            ws.settled,
            ws.heap_key,
            ws.heap_node,
        )

    _run(starts, solve, lambda: _Workspace(graph.n_nodes), workers, progress_every)


@dataclass(frozen=True)
class AccessPoints:
    """
    Weighted access points grouped by zone, as :func:`zone_sums` consumes them.

    Points are sorted by zone and ``indptr`` delimits each zone's block, the
    same layout as :class:`valma_bike_and_walk.zones.Zones`.
    """

    indptr: np.ndarray  # (n_zones + 1,)
    node: np.ndarray  # (n_points,)
    access: np.ndarray  # (n_points,) seconds
    weight: np.ndarray  # (n_points,)

    @property
    def n_zones(self) -> int:
        return int(self.indptr.shape[0] - 1)


def zone_sums(
    graph: Graph,
    points: AccessPoints,
    zone_ranges: Sequence[tuple[int, int]],
    numerator: np.ndarray,
    denominator: np.ndarray,
    limit: float | None = None,
    decay_mu: float | None = None,
    workers: int = 1,
    progress_every: int = 10,
) -> None:
    """
    Add every access point's weighted terms into zone-level ``numerator``/``denominator``.

    Each task is one ``[lo, hi)`` run of whole zones and writes only those
    zones' rows, so tasks never write the same memory.
    """
    indptr = np.asarray(points.indptr, dtype=np.int64)
    node = np.ascontiguousarray(points.node, dtype=np.int32)
    access = np.ascontiguousarray(points.access, dtype=np.float64)
    weight = np.ascontiguousarray(points.weight, dtype=np.float64)
    zone = np.repeat(np.arange(points.n_zones, dtype=np.int32), np.diff(indptr))
    bound = _limit(limit)
    mu = 0.0 if decay_mu is None else float(decay_mu)

    # A bounded search is read through its settled nodes (see _zone_rows_settled), which
    # needs the points at each node; an unbounded one stops at its last target instead.
    by_settled = np.isfinite(bound)
    if by_settled:
        points_at_node = np.argsort(node, kind="stable").astype(np.int32)
        points_at_node_ptr = np.zeros(graph.n_nodes + 1, dtype=np.int64)
        np.cumsum(
            np.bincount(node, minlength=graph.n_nodes), out=points_at_node_ptr[1:]
        )

    def solve(zone_range: tuple[int, int], ws: _Workspace) -> None:
        lo, hi = zone_range
        first, last = int(indptr[lo]), int(indptr[hi])
        if by_settled:
            ws.serial, ws.heap_key, ws.heap_node = _zone_rows_settled(
                graph.indptr,
                graph.indices,
                graph.weight,
                node[first:last],
                access[first:last],
                weight[first:last],
                zone[first:last] - lo,
                points_at_node_ptr,
                points_at_node,
                access,
                weight,
                zone,
                bound,
                mu,
                numerator[lo:hi],
                denominator[lo:hi],
                ws.stamp,
                ws.serial,
                ws.dist,
                ws.pred_edge,
                ws.touched,
                ws.settled,
                ws.heap_key,
                ws.heap_node,
            )
            return
        ws.serial, ws.heap_key, ws.heap_node = _zone_rows(
            graph.indptr,
            graph.indices,
            graph.weight,
            node[first:last],
            access[first:last],
            weight[first:last],
            zone[first:last] - lo,
            node,
            access,
            weight,
            zone,
            bound,
            mu,
            numerator[lo:hi],
            denominator[lo:hi],
            ws.stamp,
            ws.serial,
            ws.dist,
            ws.pred_edge,
            ws.touched,
            ws.settled,
            ws.heap_key,
            ws.heap_node,
        )

    _run(
        list(zone_ranges),
        solve,
        lambda: _Workspace(graph.n_nodes),
        workers,
        progress_every,
    )


def aon(
    graph: Graph,
    sources: np.ndarray,
    targets: np.ndarray,
    demand_indptr: np.ndarray,
    demand_indices: np.ndarray,
    demand_data: np.ndarray,
    limit: float | None = None,
    workers: int = 1,
    chunk_size: int | None = None,
    progress_every: int = 10,
    edge_length: np.ndarray | None = None,
    od_length: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    All-or-nothing assignment of a CSR demand matrix (rows = sources).

    Returns ``(volume, missed)``: volume per directed edge, and a mask over the
    demand's nonzeros marking the ones with no path within ``limit``.

    Given ``edge_length`` (per edge) and ``od_length`` (float64, one per demand
    nonzero), each nonzero's path length is written into ``od_length`` from the
    same searches: NaN where unreached, 0 from a source to itself.
    """
    lengths = od_length is not None
    if lengths:
        if edge_length is None or od_length.shape != demand_data.shape:
            raise ValueError("od_length needs edge_length and one slot per demand nonzero.")
        edge_length = np.ascontiguousarray(edge_length, dtype=np.float64)
    else:
        edge_length = np.zeros(0, dtype=np.float64)
        od_length = np.zeros(0, dtype=np.float64)
    sources = np.ascontiguousarray(sources, dtype=np.int32)
    targets = np.ascontiguousarray(targets, dtype=np.int32)
    demand_indptr = np.ascontiguousarray(demand_indptr, dtype=np.int64)
    demand_indices = np.ascontiguousarray(demand_indices, dtype=np.int32)
    demand_data = np.ascontiguousarray(demand_data, dtype=np.float64)
    missed = np.zeros(demand_data.shape[0], dtype=np.bool_)

    # Skip sources with no demand before dividing the work, so the tasks are
    # balanced over the rows that cost anything.
    busy = np.flatnonzero(np.diff(demand_indptr) > 0)
    size = task_size(busy.shape[0], workers, chunk_size)
    bound = _limit(limit)

    # Runs of busy rows, cut into tasks of `size` rows that may span idle ones.
    tasks: list[tuple[int, int]] = []
    for k in range(0, busy.shape[0], size):
        rows = busy[k : k + size]
        tasks.append((int(rows[0]), int(rows[-1]) + 1))

    def solve(task: tuple[int, int], ws: _Workspace) -> None:
        lo, hi = task
        ws.serial, ws.heap_key, ws.heap_node = _aon_rows(
            graph.indptr,
            graph.indices,
            graph.weight,
            graph.edge_tail,
            sources[lo:hi],
            targets,
            demand_indptr,
            demand_indices,
            demand_data,
            lo,
            bound,
            ws.volume,
            missed,
            ws.flow,
            edge_length,
            od_length,
            ws.tree_length,
            ws.stamp,
            ws.serial,
            ws.dist,
            ws.pred_edge,
            ws.touched,
            ws.settled,
            ws.heap_key,
            ws.heap_node,
        )

    workspaces = _run(
        tasks,
        solve,
        lambda: _Workspace(
            graph.n_nodes, flow=True, n_edges=graph.n_edges, lengths=lengths
        ),
        workers,
        progress_every,
    )

    volume = np.zeros(graph.n_edges, dtype=np.float64)
    for ws in workspaces:
        volume += ws.volume
    return volume, missed


def aon_select(
    graph: Graph,
    sources: np.ndarray,
    targets: np.ndarray,
    demand_indptr: np.ndarray,
    demand_indices: np.ndarray,
    demand_data: np.ndarray,
    selected: np.ndarray,
    limit: float | None = None,
    workers: int = 1,
    chunk_size: int | None = None,
    progress_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :func:`aon` with a select-link analysis of the edges ``selected`` marks.

    Returns ``(volume, missed, hit, select_volume)``: as :func:`aon`, plus a
    mask over the demand's nonzeros marking the pairs whose path crosses a
    selected edge, and the volume those pairs alone put on every edge.
    """
    sources = np.ascontiguousarray(sources, dtype=np.int32)
    targets = np.ascontiguousarray(targets, dtype=np.int32)
    demand_indptr = np.ascontiguousarray(demand_indptr, dtype=np.int64)
    demand_indices = np.ascontiguousarray(demand_indices, dtype=np.int32)
    demand_data = np.ascontiguousarray(demand_data, dtype=np.float64)
    selected = np.ascontiguousarray(selected, dtype=np.bool_)
    if selected.shape[0] != graph.n_edges:
        raise ValueError("selected needs one flag per edge.")
    missed = np.zeros(demand_data.shape[0], dtype=np.bool_)
    hit = np.zeros(demand_data.shape[0], dtype=np.bool_)

    busy = np.flatnonzero(np.diff(demand_indptr) > 0)
    size = task_size(busy.shape[0], workers, chunk_size)
    bound = _limit(limit)
    tasks: list[tuple[int, int]] = []
    for k in range(0, busy.shape[0], size):
        rows = busy[k : k + size]
        tasks.append((int(rows[0]), int(rows[-1]) + 1))

    def solve(task: tuple[int, int], ws: _Workspace) -> None:
        lo, hi = task
        ws.serial, ws.heap_key, ws.heap_node = _aon_select_rows(
            graph.indptr,
            graph.indices,
            graph.weight,
            graph.edge_tail,
            selected,
            sources[lo:hi],
            targets,
            demand_indptr,
            demand_indices,
            demand_data,
            lo,
            bound,
            ws.volume,
            missed,
            hit,
            ws.flow,
            ws.select_volume,
            ws.select_flow,
            ws.via,
            ws.stamp,
            ws.serial,
            ws.dist,
            ws.pred_edge,
            ws.touched,
            ws.settled,
            ws.heap_key,
            ws.heap_node,
        )

    workspaces = _run(
        tasks,
        solve,
        lambda: _Workspace(graph.n_nodes, flow=True, n_edges=graph.n_edges, select=True),
        workers,
        progress_every,
    )
    volume = np.zeros(graph.n_edges, dtype=np.float64)
    select_volume = np.zeros(graph.n_edges, dtype=np.float64)
    for ws in workspaces:
        volume += ws.volume
        select_volume += ws.select_volume
    return volume, missed, hit, select_volume


class Searcher:
    """
    Single searches on demand, for callers that ask one question at a time.

    The viewer routes a click, or prunes a select-link job, while a server
    waits; each of those is one search. Allocating a workspace for each would
    cost O(n_nodes) however little the search explores, so finished workspaces
    are kept and handed to the next call, from whichever thread it comes.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._free: list[_Workspace] = []
        self._lock = threading.Lock()

    def _acquire(self) -> _Workspace:
        with self._lock:
            if self._free:
                return self._free.pop()
        return _Workspace(self.graph.n_nodes)

    def _release(self, ws: _Workspace) -> None:
        with self._lock:
            self._free.append(ws)

    def _nodes(self, nodes, what: str) -> np.ndarray:
        nodes = np.ascontiguousarray(np.atleast_1d(nodes), dtype=np.int32)
        if nodes.size and (nodes.min() < 0 or nodes.max() >= self.graph.n_nodes):
            raise ValueError(f"{what} must be node indices in [0, {self.graph.n_nodes}).")
        return nodes

    def path(
        self, source: int, target: int, limit: float | None = None
    ) -> tuple[np.ndarray, float]:
        """
        ``(edges, cost)`` of the least-cost path from ``source`` to ``target``.

        ``edges`` are edge indices in travel order; an unreachable target gives
        no edges and a cost of inf, and ``source == target`` no edges at cost 0.
        """
        sources = self._nodes(source, "source")
        target = int(self._nodes(target, "target")[0])
        ws = self._acquire()
        edges, cost, ws.serial, ws.heap_key, ws.heap_node = _path(
            self.graph.indptr,
            self.graph.indices,
            self.graph.weight,
            self.graph.edge_tail,
            sources,
            target,
            _limit(limit),
            ws.stamp,
            ws.serial,
            ws.dist,
            ws.pred_edge,
            ws.touched,
            ws.settled,
            ws.heap_key,
            ws.heap_node,
        )
        self._release(ws)  # only a search that finished leaves its workspace clean
        return edges, float(cost)

    def costs(
        self, sources: np.ndarray, targets: np.ndarray, limit: float | None = None
    ) -> np.ndarray:
        """
        Least cost from *any* of ``sources`` to each of ``targets``, inf if beyond reach.

        One search, which stops once every target is settled or at ``limit``.
        """
        sources = self._nodes(sources, "sources")
        targets = self._nodes(targets, "targets")
        out = np.full(targets.shape[0], np.inf, dtype=np.float64)
        if sources.size == 0 or targets.size == 0:
            return out
        ws = self._acquire()
        ws.serial, ws.heap_key, ws.heap_node = _costs(
            self.graph.indptr,
            self.graph.indices,
            self.graph.weight,
            sources,
            targets,
            _limit(limit),
            out,
            ws.stamp,
            ws.serial,
            ws.dist,
            ws.pred_edge,
            ws.touched,
            ws.settled,
            ws.heap_key,
            ws.heap_node,
        )
        self._release(ws)
        return out
