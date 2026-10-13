"""Benchmark the Numba routing against the SciPy implementation it replaced.

Runs the same scenarios through three implementations on one network:

- ``old``     -- the frozen SciPy code in ``tests/reference``, process pool,
                 nodes in the old order (sorted by OSM id).
- ``new``     -- :mod:`valma_bike_and_walk.routing`, threads, nodes in Hilbert
                 order as ``network.py`` now builds them.
- ``new-osm`` -- the new kernels on the old node order, to separate what the
                 kernels buy from what the node order buys.

Every scenario's results are compared between implementations, so a speed-up
that changed the answer would show up in the same table.

    python scripts/bench_routing.py --links output/bike_links.gpkg --mode bike \\
        --zones data/zones.gpkg --population-column population --workers 14

Needs the ``dev`` extra (psutil, for memory across worker processes). Run from
the repository root so ``tests.reference`` imports.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import psutil
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.reference import scipy_assignment, scipy_matrix  # noqa: E402
from valma_bike_and_walk import assignment, matrix, routing  # noqa: E402
from valma_bike_and_walk.network import (  # noqa: E402
    RoutableNetwork,
    _from_coo,
    network_for_links,
)
from valma_bike_and_walk.zones import Zones, load_zones  # noqa: E402

logger = logging.getLogger("bench")


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


class PeakMemory:
    """Peak resident memory of this process and all its children, sampled."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _rss(self) -> int:
        total = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakMemory":
        self.baseline = self._rss()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._rss())


def timed(fn: Callable[[], object]) -> tuple[object, float, float, float]:
    """(result, seconds, peak RSS GB, peak above baseline GB)."""
    with PeakMemory() as memory:
        start = time.perf_counter()
        result = fn()
        seconds = time.perf_counter() - start
    return result, seconds, memory.peak / 1e9, (memory.peak - memory.baseline) / 1e9


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def reorder(
    network: RoutableNetwork, order: np.ndarray
) -> tuple[RoutableNetwork, np.ndarray]:
    """``network`` with node ``i`` of the result being ``order[i]`` of the input."""
    new_index = np.empty_like(order)
    new_index[order] = np.arange(order.shape[0])
    rows = np.repeat(np.arange(network.n_nodes), np.diff(network.indptr))
    reordered = _from_coo(
        mode=network.mode,
        node_ids=network.node_ids[order],
        x=network.x[order],
        y=network.y[order],
        u_idx=new_index[rows],
        v_idx=new_index[network.indices],
        travel_time=network.travel_time,
        length=network.length,
        link_id=network.link_id,
        direction=network.direction,
        crs=network.crs,
    )
    return reordered, new_index


def remap_zones(zones: Zones, new_index: np.ndarray) -> Zones:
    rep = dataclasses.replace(
        zones.representative, node_index=new_index[zones.representative.node_index]
    )
    return dataclasses.replace(
        zones, node_index=new_index[zones.node_index], representative=rep
    )


def gravity_demand(
    zones: Zones,
    population: np.ndarray,
    reach_m: float,
    scale_m: float,
    trips_per_person: float,
) -> csr_matrix:
    """Sparse synthetic demand: P_i P_j exp(-d/scale) within ``reach_m``."""
    xy = np.column_stack([zones.representative.x, zones.representative.y])
    pairs = cKDTree(xy).query_pairs(reach_m, output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(zones.n_zones)])
    j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(zones.n_zones)])
    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    value = population[i] * population[j] * np.exp(-d / scale_m)
    value *= trips_per_person * population.sum() / value.sum()
    return coo_matrix((value, (i, j)), shape=(zones.n_zones,) * 2).tocsr()


def edge_order(network: RoutableNetwork) -> np.ndarray:
    """Permutation putting edges in (OSM u, OSM v) order, for comparing volumes."""
    rows = np.repeat(np.arange(network.n_nodes), np.diff(network.indptr))
    return np.lexsort((network.node_ids[network.indices], network.node_ids[rows]))


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def compare_matrices(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    diff = np.abs(a[both].astype(np.float64) - b[both].astype(np.float64))
    return {
        "cells": float(a.size),
        "reachability_mismatches": float((fa != fb).sum()),
        "max_abs_diff_s": float(diff.max()) if diff.size else 0.0,
        "cells_differing_over_0.01s": float((diff > 0.01).sum()),
    }


def compare_volumes(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a - b)
    return {
        "total_volume": float(a.sum()),
        "total_abs_diff": float(diff.sum()),
        "relative_abs_diff": float(diff.sum() / max(a.sum(), 1e-12)),
        "edges_differing_over_1e-6": float((diff > 1e-6).sum()),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--mode", default="bike")
    parser.add_argument("--zones", type=Path, required=True)
    parser.add_argument("--population-column", default="population")
    parser.add_argument("--points-per-zone", type=int, default=8)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/bench"))
    parser.add_argument("--sample-sources", type=int, default=400)
    parser.add_argument("--near-minutes", type=float, default=15.0)
    parser.add_argument(
        "--demand-reach-km",
        type=float,
        default=10.0,
        help="Synthetic demand only between zones this close; sets the OD pair count.",
    )
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--impl", default="old,new,new-osm")
    parser.add_argument(
        "--scenarios",
        default="matrix-unbounded,matrix-max,zone-matrix,assign,zone-assign",
    )
    parser.add_argument("--out", type=Path, default=Path("output/bench_routing.json"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # The implementations log every few chunks; the benchmark only wants its own.
    for name in ("valma_bike_and_walk", "tests"):
        logging.getLogger(name).setLevel(logging.WARNING)

    impls = args.impl.split(",")
    scenarios = args.scenarios.split(",")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"args": {k: str(v) for k, v in vars(args).items()}}

    # ---- network --------------------------------------------------------
    cache = args.cache_dir / f"{args.mode}_network.npz"
    network, seconds, _, _ = timed(
        lambda: network_for_links(args.links, args.mode, cache)
    )
    logger.info(
        "Network: %d nodes, %d edges (%.0f s to build or load)",
        network.n_nodes,
        network.n_edges,
        seconds,
    )
    report["network"] = {"nodes": network.n_nodes, "edges": network.n_edges}

    load_times = {}
    for compress in (True, False):
        path = args.cache_dir / f"load_test_{'z' if compress else 'raw'}.npz"
        network.save(path, compress=compress)
        start = time.perf_counter()
        RoutableNetwork.load(path)
        load_times["compressed" if compress else "uncompressed"] = {
            "seconds": time.perf_counter() - start,
            "MB": path.stat().st_size / 1e6,
        }
        path.unlink()
    report["load"] = load_times
    logger.info("Load: %s", load_times)

    osm_network, osm_index = reorder(
        network, np.argsort(network.node_ids, kind="stable")
    )
    nets = {"old": osm_network, "new": network, "new-osm": osm_network}
    order = {
        key: edge_order(net)
        for key, net in (("hilbert", network), ("osm", osm_network))
    }
    edge_key = {"old": "osm", "new": "hilbert", "new-osm": "osm"}

    # ---- zones and demand -----------------------------------------------
    zones, seconds, _, _ = timed(
        lambda: load_zones(network, args.zones, points_per_zone=args.points_per_zone)
    )
    logger.info(
        "Zones: %d zones, %d access points (%.0f s)",
        zones.n_zones,
        zones.n_points,
        seconds,
    )
    zone_sets = {"new": zones, "old": remap_zones(zones, osm_index)}
    zone_sets["new-osm"] = zone_sets["old"]

    frame = gpd.read_file(args.zones, columns=[args.population_column])
    population = frame[args.population_column].to_numpy(dtype=float)[zones.ids]
    demand = gravity_demand(
        zones, population, args.demand_reach_km * 1000.0, 3_000.0, 0.5
    )
    logger.info("Demand: %d OD pairs, %.0f trips", demand.nnz, demand.sum())
    report["zones"] = {
        "zones": zones.n_zones,
        "points": zones.n_points,
        "od_pairs": demand.nnz,
    }

    # ---- warm-up: compile (or load cached) kernels ----------------------
    start = time.perf_counter()
    small = np.arange(0, network.n_nodes, max(1, network.n_nodes // 4))[:3]
    matrix.travel_time_matrix(network, small, max_seconds=1.0)
    assignment.assign_traffic(network, small, np.ones((3, 3)), max_seconds=1.0)
    routing.zone_sums(
        network.routing_graph(),
        routing.AccessPoints(np.array([0, 3]), small, np.zeros(3), np.ones(3) / 3),
        [(0, 1)],
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        limit=1.0,
    )
    report["warmup_seconds"] = time.perf_counter() - start
    logger.info("Kernel warm-up %.1f s", report["warmup_seconds"])

    W = args.workers
    max_s = args.max_minutes * 60
    near_s = args.near_minutes * 60
    rep_nodes = {k: z.representative.node_index for k, z in zone_sets.items()}
    step = max(1, zones.n_zones // args.sample_sources)
    sample = np.arange(0, zones.n_zones, step)[: args.sample_sources]

    def run(impl: str, scenario: str):
        net, zs, nodes = nets[impl], zone_sets[impl], rep_nodes[impl]
        ref = impl == "old"
        mat = scipy_matrix if ref else matrix
        asg = scipy_assignment if ref else assignment
        if scenario == "matrix-unbounded":
            return mat.travel_time_matrix(net, nodes[sample], nodes, workers=W)
        if scenario == "matrix-max":
            return mat.travel_time_matrix(
                net, nodes, nodes, max_seconds=max_s, workers=W
            )
        if scenario == "zone-matrix":
            return mat.zone_travel_time_matrix(
                net, zs, max_seconds=max_s, near_seconds=near_s, workers=W
            )
        if scenario == "assign":
            return asg.assign_traffic(net, nodes, demand, max_seconds=max_s, workers=W)
        if scenario == "zone-assign":
            return asg.assign_zone_traffic(
                net, zs, demand, max_seconds=max_s, near_seconds=near_s, workers=W
            )
        raise ValueError(scenario)

    results: dict[str, dict] = {}
    for scenario in scenarios:
        results[scenario] = {}
        outputs = {}
        for impl in impls:
            logger.info("%s / %s ...", scenario, impl)
            out, seconds, peak, above = timed(lambda: run(impl, scenario))
            results[scenario][impl] = {
                "seconds": round(seconds, 2),
                "peak_rss_gb": round(peak, 2),
                "peak_above_baseline_gb": round(above, 2),
            }
            if "assign" in scenario:
                out = np.asarray(out)[order[edge_key[impl]]]
            outputs[impl] = out
            logger.info("%s / %s: %.1f s, peak %.2f GB", scenario, impl, seconds, peak)
        if "old" in outputs:
            for impl in impls:
                if impl == "old":
                    continue
                compare = compare_volumes if "assign" in scenario else compare_matrices
                results[scenario][impl]["vs_old"] = compare(
                    outputs["old"], outputs[impl]
                )
                results[scenario][impl]["speedup_vs_old"] = round(
                    results[scenario]["old"]["seconds"]
                    / results[scenario][impl]["seconds"],
                    1,
                )
        if scenario == "matrix-unbounded":
            results[scenario]["sources"] = int(sample.size)
        del outputs
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report["results"] = results
        args.out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
