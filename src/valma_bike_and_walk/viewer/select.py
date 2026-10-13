"""Select-link maps on demand: the flows that pass a clicked counter.

A full select-link assignment costs as much as ``assign`` itself -- minutes
for the whole country -- because every origin needs its own shortest-path
tree. But only a fraction of the OD pairs could possibly use any one count
line, and two searches find them cheaply:

1. one search *backwards* from the counter's links gives every node's time to
   reach the counter, and one *forwards* gives the time from the counter on;
2. a pair whose origin-to-counter plus counter-to-destination time already
   exceeds ``max_seconds`` cannot be routed through the counter, so it is
   dropped. (Without a cutoff, only pairs that cannot reach the counter go.)

The survivors get the real select-link assignment
(:func:`~valma_bike_and_walk.assignment.select_link_assignment`), so the
answer is exactly the one ``assign --select-link`` gives, and the same pass
loads those trips to draw where they run. Results are cached on disk per
counter, keyed on the assignment settings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from valma_bike_and_walk import routing
from valma_bike_and_walk.assignment import select_link_assignment
from valma_bike_and_walk.network import RoutableNetwork

logger = logging.getLogger(__name__)


@dataclass
class SelectSetup:
    """What the run was assigned with: graph, zones, demand, cutoff."""

    network: RoutableNetwork
    zone_ids: np.ndarray
    zone_nodes: np.ndarray  # node index per zone, -1 where unsnapped
    demand: csr_matrix  # zones x zones, as assign saw it
    max_seconds: float | None
    workers: int = 1
    chunk_size: int | None = None
    #: Text describing the settings; hashed into the cache key.
    description: str = ""


@dataclass
class FlowResult:
    """One counter's select-link map, sparse over link rows."""

    rows: np.ndarray  # link-row indices with flow
    forward: np.ndarray  # float32, along the digitised direction
    reverse: np.ndarray
    summary: dict = field(default_factory=dict)

    def dense(self, n_rows: int, which: str = "total") -> np.ndarray:
        out = np.zeros(n_rows, dtype=np.float32)
        values = {
            "total": self.forward + self.reverse,
            "forward": self.forward,
            "reverse": self.reverse,
        }[which]
        out[self.rows] = values
        return out


def _edge_rows(network: RoutableNetwork, link_ids: np.ndarray) -> np.ndarray:
    """Link-layer row of every directed edge, -1 where the viewer lacks it."""
    return pd.Index(link_ids).get_indexer(network.link_id)


def candidate_demand(
    forward: routing.Searcher,
    backward: routing.Searcher,
    selected: np.ndarray,
    zone_nodes: np.ndarray,
    demand: csr_matrix,
    max_seconds: float | None,
) -> csr_matrix:
    """
    The part of ``demand`` that could be routed over a selected edge.

    Keeps a pair when (origin to the nearest selected edge's tail) plus (the
    nearest selected edge's head to the destination) fits within the cutoff,
    which no pair routed through the counter can fail. A loose bound -- the
    edge itself and the choice of edge are ignored -- but a safe one.

    ``forward`` searches the network and ``backward`` the same network
    reversed. Each is one search from all of the counter's edge ends at once,
    which stops once every zone is settled or at the cutoff.
    """
    graph = forward.graph
    tails = np.unique(graph.edge_tail[selected])
    heads = np.unique(graph.indices[selected])
    limit = np.inf if max_seconds is None else max_seconds

    snapped = zone_nodes >= 0
    a = np.full(zone_nodes.shape[0], np.inf)
    b = np.full(zone_nodes.shape[0], np.inf)
    a[snapped] = backward.costs(tails, zone_nodes[snapped], max_seconds)
    b[snapped] = forward.costs(heads, zone_nodes[snapped], max_seconds)

    coo = demand.tocoo()
    via = a[coo.row] + b[coo.col]
    keep = np.isfinite(via) & (via <= limit) & (coo.row != coo.col)
    return csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=demand.shape
    )


class CounterFlows:
    """Runs select-link jobs one at a time and caches what they find."""

    def __init__(self, setup: SelectSetup, link_ids: np.ndarray, cache_dir: Path) -> None:
        self.setup = setup
        self.n_rows = int(link_ids.shape[0])
        self.link_ids = np.asarray(link_ids)
        net = setup.network
        # The graph select_link_assignment routes on, so the pruning sees the same costs.
        graph = net.routing_graph()
        self.forward = routing.Searcher(graph)
        self.backward = routing.Searcher(graph.reversed())
        self.edge_rows = _edge_rows(net, self.link_ids)
        key = hashlib.sha1(setup.description.encode()).hexdigest()[:12]
        self.cache_dir = Path(cache_dir) / key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._worker_lock = threading.Lock()

    # -- cache -----------------------------------------------------------

    def _path(self, name: str) -> Path:
        slug = hashlib.sha1(name.encode()).hexdigest()[:16]
        return self.cache_dir / f"{slug}.npz"

    def cached(self, name: str) -> FlowResult | None:
        path = self._path(name)
        if not path.exists():
            return None
        with np.load(path) as data:
            return FlowResult(
                rows=data["rows"],
                forward=data["forward"],
                reverse=data["reverse"],
                summary=json.loads(str(data["summary"])),
            )

    # -- jobs ------------------------------------------------------------

    def status(self, name: str) -> dict:
        if self._path(name).exists():
            result = self.cached(name)
            return {"state": "done", "summary": result.summary if result else {}}
        with self._lock:
            return dict(self.jobs.get(name, {"state": "none"}))

    def request(self, name: str, link_rows: list[int]) -> dict:
        """Start (or report) the job for a counter crossing ``link_rows``."""
        current = self.status(name)
        if current["state"] in ("done", "queued", "running"):
            return current
        with self._lock:
            self.jobs[name] = {"state": "queued", "started": time.time()}
        thread = threading.Thread(
            target=self._run, args=(name, list(link_rows)), daemon=True
        )
        thread.start()
        return self.status(name)

    def _run(self, name: str, link_rows: list[int]) -> None:
        with self._worker_lock:  # one heavy job at a time
            self._set(name, state="running", step="pruning OD pairs")
            try:
                result = self.compute(link_rows, progress=lambda step: self._set(name, step=step))
                np.savez_compressed(
                    self._path(name),
                    rows=result.rows,
                    forward=result.forward,
                    reverse=result.reverse,
                    summary=json.dumps(result.summary),
                )
                self._set(name, state="done", summary=result.summary)
            except Exception as error:  # reported to the browser, not swallowed
                logger.exception("Select-link for %s failed", name)
                self._set(name, state="error", error=str(error))

    def _set(self, name: str, **fields) -> None:
        with self._lock:
            self.jobs.setdefault(name, {}).update(fields)

    # -- the computation -------------------------------------------------

    def compute(self, link_rows: list[int], progress=lambda step: None) -> FlowResult:
        setup = self.setup
        net = setup.network
        started = time.time()
        link_ids = self.link_ids[np.asarray(link_rows, dtype=np.int64)]
        selected = np.isin(net.link_id, link_ids)
        if not selected.any():
            raise ValueError("None of the counter's links are in the routable network.")

        snapped = setup.zone_nodes >= 0
        demand = setup.demand
        if not snapped.all():
            keep = np.flatnonzero(snapped)
            demand = demand[keep][:, keep]
        nodes = setup.zone_nodes[snapped]
        ids = setup.zone_ids[snapped]

        candidates = candidate_demand(
            self.forward, self.backward, selected, nodes, demand, setup.max_seconds,
        )
        origins = np.flatnonzero(np.diff(candidates.indptr))
        logger.info(
            "Select-link: %d of %d OD pairs from %d of %d origins could use the counter",
            candidates.nnz, demand.nnz, origins.size, nodes.size,
        )
        empty = FlowResult(np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32))
        if origins.size == 0:
            empty.summary = {"trips": 0.0, "od_pairs": 0, "origins_searched": 0}
            return empty

        progress(f"select-link over {origins.size:,} origins")
        # One pass finds the trips through the counter and loads them.
        _, picked, flow = select_link_assignment(
            net, nodes[origins], candidates[origins], link_ids, targets=nodes,
            max_seconds=setup.max_seconds, workers=setup.workers,
            chunk_size=setup.chunk_size,
        )
        users = np.flatnonzero(np.diff(picked.indptr))
        if users.size == 0:
            empty.summary = {"trips": 0.0, "od_pairs": 0, "origins_searched": int(origins.size)}
            return empty

        rows = self.edge_rows
        ok = (flow > 0) & (rows >= 0)
        forward = np.zeros(self.n_rows, np.float64)
        reverse = np.zeros(self.n_rows, np.float64)
        along = net.direction >= 0
        np.add.at(forward, rows[ok & along], flow[ok & along])
        np.add.at(reverse, rows[ok & ~along], flow[ok & ~along])
        touched = np.flatnonzero((forward > 0) | (reverse > 0))

        as_origin = np.asarray(picked.sum(axis=1)).ravel()
        as_destination = np.asarray(picked.sum(axis=0)).ravel()
        origin_ids = ids[origins]
        top_o = np.argsort(as_origin)[::-1][:10]
        top_d = np.argsort(as_destination)[::-1][:10]
        summary = {
            "trips": float(picked.sum()),
            "od_pairs": int(picked.nnz),
            "origins_searched": int(origins.size),
            "origins_total": int(nodes.size),
            "seconds": round(time.time() - started, 1),
            "top_origins": [[_plain_id(origin_ids[i]), float(as_origin[i])] for i in top_o if as_origin[i] > 0],
            "top_destinations": [[_plain_id(ids[i]), float(as_destination[i])] for i in top_d if as_destination[i] > 0],
        }
        return FlowResult(
            rows=touched,
            forward=forward[touched].astype(np.float32),
            reverse=reverse[touched].astype(np.float32),
            summary=summary,
        )


def _plain_id(value):
    return value.item() if hasattr(value, "item") else value
