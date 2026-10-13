# Design: Numba routing core, generalised cost, and AequilibraE assignment

Status: phases 0–2 implemented; Numba is the routing engine (see sections 9 and 13) · Branch: `feat/distributed_taz_routing` · 2026-10-08

## 1. Summary

- **Routing moves onto purpose-written Numba kernels**, and Numba becomes a core
  dependency. There is one routing implementation, with no backend switch and no
  fallback. The SciPy Dijkstra code is removed from `src/`. A copy is kept under
  `tests/reference/` as a test oracle and a benchmark baseline.
- **Routes minimise a generalised cost** computed from link attributes. Travel
  time, length and other attributes are measured *along* the least-cost path as
  skims. Today's perceived-cost factors (Tarkkala 2024 betas, the grade and
  signal penalties) currently live inside speed and travel time; they move into
  an explicit cost model.
- **AequilibraE is an optional extra** for route choice, and later for
  equilibrium assignment. It routes on the same generalised cost.
- **The graph format changes without backwards compatibility.** Old cached
  graphs and `valma build` files are refused with a "rebuild" message.

| Concern | Engine | Install | Python |
|---|---|---|---|
| OD matrices, zone aggregation, all-or-nothing assignment | Numba kernels | core | 3.10+ |
| Sparse demand, KD-tree snapping, connected components | SciPy (unchanged, not routing) | core | 3.10+ |
| Route choice, equilibrium assignment | AequilibraE | `aequilibrae` extra | 3.11+ |
| Old Dijkstra implementation | SciPy, in `tests/reference/` | dev | — |

## 2. Why

### 2.1 Speed

On a country-sized network the time goes to the code around SciPy's Dijkstra,
not the search itself:

1. **Every source costs O(n_nodes), whatever the cutoff.** SciPy returns a dense
   `(sources, n_nodes)` float64 block. A 15-minute near-tier search touches a
   few thousand nodes, but still allocates, fills and slices a row of a few
   million.
2. **Path loading is a Python loop per hop.** `_load_source` walks every
   destination back one hop per iteration, doing a `searchsorted` and an
   `np.add.at` each time.
3. **Parallelism has to use processes.** SciPy holds the GIL, so the code needs
   temporary `.npy` files, memory mapping, pickled results and the Windows
   `__main__` caveat.

A Numba kernel removes all three. Its per-source cost depends only on the area
it explores, it loads demand in one pass up the tree, and it releases the GIL,
so plain threads share one copy of everything. Numba also makes it cheap to
measure several attributes along the chosen path, which generalised cost needs
(section 5).

### 2.2 Why Numba can be a core dependency

Checked on PyPI against Numba 0.68 / llvmlite 0.50 and SciPy 1.18. There are
wheels for CPython 3.10–3.15 on Windows x64 and ARM64, Linux x86_64 and aarch64
(glibc), and macOS on Apple Silicon. Numba's only dependencies are llvmlite and
numpy. Wherever a typical user can install SciPy, they can install Numba.

The gaps are narrow:

| Environment | SciPy | Numba |
|---|---|---|
| Intel Mac | wheels | wheels only up to 0.62.1, for Python ≤ 3.13. The resolver picks that version on its own. Python 3.14+ on an Intel Mac would need a source build. |
| Alpine / musl Linux | wheels | none (llvmlite needs a source build) |
| Free-threaded CPython (`3.14t`) | wheels | none |
| A brand-new CPython | wheels at release | usually weeks to a few months behind |

None of these is a realistic platform for a country-scale routing run. The
lower bound stays `numba>=0.60`, so the Intel-Mac path keeps working. Numba
pins `numpy<2.N`, and the resolver handles that by choosing a compatible numpy.

### 2.3 Why SciPy stays in core, just not for routing

SciPy does more here than Dijkstra:

- `scipy.sparse` is the demand format in the public API. `demand_matrix`
  returns a `csr_matrix`, the README examples pass one to `assign_*`, and
  `expand_demand` computes `W @ D @ W.T` as a sparse product.
- `cKDTree` snaps nodes (`network.nearest_nodes`) and hand-drawn link ends
  (`links.py`).
- `connected_components` keeps the largest component.

Each of these could be replaced: `shapely.STRtree.query_nearest` for snapping,
a Numba component search, and own CSR arrays for demand. But removing SciPy
would change the public demand type and add work, and it gains nothing for
installability, since SciPy installs everywhere Numba does. The proposal keeps
SciPy in core for these uses only, and removes `scipy.sparse.csgraph.dijkstra`
from `src/`. Dropping SciPy entirely can be revisited later (section 12).

### 2.4 AequilibraE

AequilibraE is for growth rather than speed. Path-size-logit route choice is the
most relevant next step for cycling, and volume-delay equilibrium follows if
capacities become meaningful. Both are mature in AequilibraE and not worth
reimplementing.

## 3. Constraints

- **No compiler needed** for the core install or the AequilibraE extra on
  Windows and Linux. AequilibraE has no macOS wheels (section 8.4).
- **Python 3.10+ stays supported** for the core. AequilibraE needs 3.11+ and
  pandas 3, so it stays behind an extra with a version marker.
- **The public API stays the same.** That covers CLI flags and the Python
  function signatures (cutoff parameters change meaning, section 5.4). The
  output file names stay, and results still join back on `link_id`. The matrix
  `.npz` gains new arrays (section 5.5).
- **Memory flat in the worker count.** Today that comes from memory mapping.
  With threads it comes for free, but per-thread workspaces must be sized
  deliberately (section 7.3).

## 4. Module layout

```
src/valma_bike_and_walk/
  costs.py           # NEW: cost profiles -> per-link generalised cost (section 5)
  speeds.py          # physical speed only; perceived factors move to costs.py
  links.py           # derives travel_time_* and cost_* columns
  network.py         # graph format v2 (section 6)
  routing.py         # NEW: Numba kernels + thread-pool driver (section 7)
  matrix.py          # public API unchanged; tiering, aggregation, logging
  assignment.py      # public API unchanged; demand expansion, tiers
  aequilibrae.py     # optional engine adapter, imported lazily (section 8)
tests/reference/
  scipy_matrix.py      # the old matrix routing, frozen, for oracles
  scipy_assignment.py  # the old assignment routing, frozen
scripts/
  bench_routing.py   # benchmark: Numba vs reference, wall time and peak RSS
```

`matrix.py` and `assignment.py` keep everything *about the model*: demand
expansion, two-tier stitching, self-pair exclusion, intrazonal fallback, and
unplaced-demand bookkeeping. `routing.py` only does the work per source.

## 5. Generalised cost

### 5.1 Separate what the cyclist experiences from how long it takes

Today `travel_time_s` mixes two things:

- **Physical time:** length over speed from `highway`/`surface`, plus signal
  delay.
- **Perceived disutility:** the `beta_lane`/`beta_mixed` ratio (Tarkkala 2024)
  scaling speed on major roads, and the grade penalty for climbs.

The first is what a travel-time matrix should report. The second is what route
choice should respond to. Folding preferences into speed makes the reported
times wrong, so they are split:

| Link column | Meaning | Used for |
|---|---|---|
| `travel_time_s`, `travel_time_reverse_s` | Physical seconds: length / speed, plus signal delay, plus climbing time | skims, the reported travel-time matrix |
| `cost_s`, `cost_reverse_s` | Generalised cost in **equivalent seconds** | the routing weight |
| `cost_factor_override` | **Empty, for you**: multiplies the computed cost | hand edits in QGIS, like `speed_override_kmh` |

Cost is in equivalent seconds so that it compares directly with time, a
`decay_mu` keeps its units, and the cutoffs (5.4) stay intuitive.

### 5.2 Cost model

A `CostProfile` per mode lives in `costs.py`. It is a frozen dataclass in the
same style as `SpeedProfile`, explicit and reviewable:

```
cost = travel_time
       × infrastructure_factor(highway, cycleway, segregated)   # Tarkkala betas
       × traffic_factor(car_volume, highway)                    # mixed traffic exposure
       × surface_comfort_factor(surface)                        # beyond the speed effect
       + grade_penalty(ascent, length)                          # perceived climbing
       + signal_penalty                                         # perceived waiting
       + turn/crossing penalties                                # later, needs turn data
cost *= cost_factor_override  (where set)
cost  = max(cost, travel_time)                                  # never "cheaper than free"
```

- **Multiplicative factors** are where the Tarkkala-type distance and time
  multipliers go. `beta_lane`/`beta_mixed` moves out of
  `SpeedProfile.speeds_kph` into here.
- **Additive penalties** are for point events (signals) and climbs.
- **Walking** starts with factors of 1 except steps and surfaces. Its profile
  exists so the code path is the same for both modes.
- The **directional terms** (grade, and oneway-specific factors) give separate
  forward and reverse costs, exactly as `travel_time_reverse_s` does today.
- The **lower bound** `cost ≥ travel_time` is a modelling choice. It keeps
  every factor a penalty, so a cost cutoff is also a time cutoff (5.4). If a
  profile genuinely needs a bonus (a factor below 1, such as the best
  cycleways), the bound becomes `cost ≥ ε · travel_time` and section 5.4 shows
  what changes.

Parameter values are a modelling decision outside this design. The design only
fixes the shape: factors and penalties per link, computed at the link-layer
stage, editable and overridable.

### 5.3 What the graph and the kernels carry

Per directed edge, the graph stores:
- the **routing weight**, `cost` (float64);
- the **skim attributes**, as one `(n_edges, n_skims)` float64 array. The first
  set is `time_s`, `length_m` and `ascent_m`. More are added by name without
  touching the kernels.

Dijkstra settles on `cost`. When an edge is relaxed it also sets
`skim[v, :] = skim[u, :] + edge_skim[e, :]`. The attributes are therefore
measured **along the least-cost path**, not minimised separately, which is how
a generalised-cost skim is defined.

### 5.4 Cutoffs are on cost

A bounded Dijkstra can only stop exactly on the quantity it settles on.
`max_seconds`/`--max-minutes` and `near_seconds`/`--near-minutes` therefore
bound **generalised cost** in equivalent seconds.

- With `cost ≥ travel_time`, every pair found within cost L also has time ≤ L.
  Pairs with time ≤ L but cost > L are not found. For the near tier this doesn't
  matter: the tier exists to cover short distances, and the far tier catches the
  rest. For the outer `--max-minutes` it means "reachable within L
  equivalent-minutes", and the docs say so.
- The CLI flag names stay the same. Their help text and the docstrings change.
- If a strict *time* cutoff is wanted on top, it's a post-filter on the time
  skim (`--max-travel-minutes`). It's cheap, but it isn't a search bound.

### 5.5 Outputs

- **Matrix `.npz`.** It stays `ids` + `seconds`, where `seconds` is now
  physical travel time along the least-cost path. New arrays: `cost` (always)
  and `metres`. `--skims` chooses extra ones such as `ascent_m`. OMX writes one
  matrix per skim, named `<mode>_<skim>`; `--omx-matrix-name` still names the
  time matrix.
- **Zone aggregation.** The weighted mean is computed per skim. `decay_mu`
  applies to **cost**, since gravity and logit models respond to generalised
  cost, and it's reported in the `cost` matrix. The reachability fraction uses
  the cost cutoff.
- **Assignment.** All-or-nothing loads onto least-cost paths. The volume output
  is unchanged.
- **Link GeoPackage.** It gains `cost_s` / `cost_reverse_s` for styling, next
  to `travel_time_s`.

## 6. Graph format v2 (breaking)

| Change | Why |
|---|---|
| **Nodes in Hilbert-curve order** of `(x, y)`, applied in `_assemble` before `_build_csr` | Nodes that are close on the map become close in memory, so a Dijkstra frontier touches nearby cache lines. Typically 1.5–2×. |
| `cost` (f64) per edge, and `skims` `(n_edges, n_skims)` f64 with a `skim_names` array | Section 5.3. `travel_time` and `length` become columns of `skims`. |
| **Parallel edges deduplicated on `cost`**, not `travel_time` | A shortest path picks the cheapest. |
| `edge_tail` (int32), derived on load, not stored | Needed by the loading pass. Replaces `_edge_keys` and the `searchsorted` lookup. |
| `csr()` / `distance_csr()` removed | Nothing in `src/` builds a SciPy graph any more. The reference oracle builds its own. |
| Cache graphs saved **uncompressed**, still atomic; `valma build` files stay compressed | Decompressing a country graph takes seconds on every run, and the cache is disposable. |
| **`FORMAT_VERSION = 2`**, stored in the file and in the cache key | See below. |

**No compatibility.** `RoutableNetwork.load` reads `format_version`, and anything
other than 2 raises:

```
NetworkFormatError: output/bike.npz is graph format 1; this version needs 2.
Rebuild it with: valma build --links <links.gpkg> --mode bike
```

Cached graphs never get that far, because the version is part of their cache
key, so they are rebuilt automatically. The link GeoPackage gains columns (5.1)
and is re-derived by `normalise` on every read, so existing layers keep working
without re-extracting.

`link_volumes_<mode>.npz` rows follow the new edge order. Its rows are
identified by `u`/`v`, and by `link_id` in the GeoPackage.

## 7. Numba routing

### 7.1 Kernel

There is one search primitive, `@njit(nogil=True, cache=True)`:

```python
def _search(indptr, indices, cost, edge_skims, source, limit,
            target_stamp, serial, n_targets,           # early exit
            dist, skim, pred_edge, settled, n_settled, # workspace
            heap_key, heap_node) -> n_settled
```

- **Binary heap with lazy deletion.** Duplicates are pushed instead of using
  decrease-key, and stale entries are skipped on pop. The heap arrays live in
  the workspace and double in size when full.
- **Touched-list reset.** `dist` starts as all-inf once per thread. Each search
  records the nodes it touched and resets only those. Per-source cost is
  proportional to the area searched, never to `n_nodes`.
- **Settle order** is kept in `settled`. The loading pass relies on it.
- **Early exit** at `limit` (cost), or once every marked target has settled.
  `target_stamp[v] == serial` marks targets without O(n_nodes) clearing.
- `pred_edge[v]` holds the CSR position of the edge that reached `v`, and
  `skim[v, :]` the attributes along the path (5.3).

### 7.2 The three operations built on it

**`point_matrix(sources, targets, limit) -> {skim_name: (S, T)}`.** Search, then
copy `dist[targets]` (cost) and `skim[targets, :]` into float32 output rows,
then reset.

**`zone_sums(zone_range, ...)`.** A task is a range of zones, so every row it
writes belongs to it alone and writes don't race. For each access point `a` of
zone `i`, it adds the weighted terms for cost, each skim, and the reachable
weight into `numerator[i, zone_of_target, k]` and `denominator[i, zone]` in one
loop over targets, which replaces the two `reduceat` passes. Access legs add
`access_seconds` to the time skim and to cost (at the access walk's cost
factor of 1), and their length to `metres`.

**`aon(sources, targets, demand_csr, limit) -> (volume, missed_mask)`.** Load
the whole tree in one pass instead of walking each path:

```
search(source, targets = this row's demand columns)
flow[target] += demand  for each nonzero; missed if not settled (and != source)
for v in reversed(settled):
    if flow[v] > 0:
        e = pred_edge[v]
        volume[e] += flow[v]
        flow[edge_tail[e]] += flow[v]
    flow[v] = 0
```

Each settled node is visited once, so the cost is O(settled) however many
destinations there are and however long their paths. This is exact
all-or-nothing loading, the same result as today's lockstep walk. The missed
mask is per nonzero; `assignment._assign` groups it to zone level as
`_Unplaced` does today.

### 7.3 Parallelism and memory

- **Driver.** A `ThreadPoolExecutor(workers)` submits tasks in a bounded window,
  as `_assign` does now. Each task calls a nogil kernel with its thread's
  workspace from a `threading.local`. Progress logging and Ctrl-C keep working.
  This is deliberately not `prange`: there's no threading layer (OpenMP, TBB or
  workqueue) to configure, work is balanced dynamically across uneven search
  sizes, and progress reporting stays as it is.
- **Per-thread workspace.** `dist`, `pred_edge`, `settled`, `target_stamp`,
  `flow`, plus `skim` at 8 B per skim per node. For about 3 M nodes and 3 skims
  that's about 170 MB. AON adds a thread-local f64 volume per edge (about 60 MB
  at 7 M edges), summed once at the end. 16 threads come to about 3.5 GB. The
  total is logged before the run, with a warning above half of physical RAM.
  Matrix-only runs can drop skims nobody asked for.
- **Shared, read-only.** The graph and the demand CSR are shared between
  threads, so there are no temporary files, no memory mapping and no
  `__main__` caveat. That README note is removed.
- `--workers` means threads. `--chunk-size` becomes "sources per task", with a
  default of about 64.

### 7.4 Compilation

- `cache=True` means only the first run compiles (about 5–15 s). No cache
  directory needs configuring: when the package directory is not writable
  (a site-packages install), Numba falls back to a per-user cache by itself.
- Kernels are typed on concrete dtypes (int32/int64 indices, float64 weights).
  Inputs are normalised on the Python side so compiled variants don't multiply.

## 8. AequilibraE adapter

### 8.1 Scope

AequilibraE is used only for what the project doesn't do itself:

1. **Route choice.** Path-size logit over a generated choice set (BFSLE or link
   penalisation), with utility on `cost`. This is the first target: cyclists
   split between near-equal routes, and all-or-nothing loads all of them onto
   one.
2. **Equilibrium assignment** (BFW, CFW or MSA with BPR/conical volume-delay
   functions). This only becomes useful with a capacity source, so it is in
   scope for the adapter but not the first delivery.
3. **Select-link analysis**, which comes with the assignment results.

Matrices and plain all-or-nothing assignment stay with Numba, which handles
multi-point zones correctly (8.3).

### 8.2 Mapping a `RoutableNetwork` onto an AequilibraE `Graph`

The adapter builds the graph in memory only, with no AequilibraE project and so
no SpatiaLite:

- **One row per directed edge.** `link_id = edge_index + 1`, `direction = 1`,
  `a_node`/`b_node` = node index + offset, then `cost` as the graph field, with
  `time_s`, `length_m` and the other skims as skim fields. Results map back by
  `link_id - 1` to our edge order, so `link_volume_frame` and the GeoPackage
  writer stay unchanged.
- **Centroids** are virtual nodes `1..n_zones`, which is AequilibraE's
  convention. Network nodes are offset past them.
- **Connectors** run from each centroid to each of its zone's access points, and
  back, costed at `access_seconds`. Their link ids start after the edges, and
  their results are dropped when mapping back.
- **Demand** is the zone-level matrix as an in-memory `AequilibraeMatrix`, with
  its index set to the centroid ids.
- **Cores.** `workers` maps to AequilibraE's `set_cores`.

The exact API calls (`Graph.network = df`, `prepare_graph(centroids)`,
`set_graph`, `set_skimming`, `TrafficClass`, `RouteChoice`, the results column
names) are **to be confirmed in the spike** (phase 5). This design deliberately
depends only on "build a graph from a links DataFrame, assign a matrix, get
per-link flows back by link_id".

### 8.3 Zones lose some fidelity

AequilibraE is centroid-based. Connecting a centroid to its K access points
lets the router pick the *best* connector for each OD pair, not distribute
demand over all of them by weight as `expand_demand` does. Two modes:

- **`connectors`** (default). Classic centroid connectors, which scale to the
  whole country.
- **`points`.** Every access point becomes its own centroid, and demand is
  expanded with `expand_demand`. This keeps full fidelity, but AequilibraE
  matrices are dense: n_points² × 8 B per class, so 20 k points need 3.2 GB.
  That's practical for a city or region, not for Finland at K=8. Refuse with a
  clear message above a configurable size.

Intrazonal demand is only placed in `points` mode. `connectors` mode drops it
and logs it.

### 8.4 Packaging and versions

```toml
dependencies = [
    # ...existing...
    "numba>=0.60",
    "scipy>=1.11",          # sparse demand, KD-tree, components -- not routing
]

[project.optional-dependencies]
aequilibrae = ["aequilibrae>=1.7; python_version >= '3.11'"]
```

- **Python.** On 3.10 the extra installs nothing, and the adapter raises
  `AequilibraEUnavailable("needs Python 3.11+ and pip install valma_bike_and_walk[aequilibrae]")`.
- **pandas.** AequilibraE 1.7 requires pandas>=3.0. The core keeps
  `pandas>=2.0` and is tested against both (section 10).
- **macOS.** There are no AequilibraE wheels, so installing the extra there
  needs build tools. Documented, not blocked.
- **Weight.** Most of its dependencies overlap with ours. The new ones are
  pyarrow, rtree, pyaml, requests and tqdm.
- AequilibraE is imported only inside `aequilibrae.py`'s functions.

### 8.5 CLI

`valma assign --method {aon,route-choice,equilibrium}`, defaulting to `aon`.
The last two require the extra and accept `--zone-connection {connectors,points}`.
Method-specific options (`--max-routes`, `--penalty`, `--beta`, and later
`--vdf`/`--capacity-column`) are only accepted with their method.

## 9. Phases

| # | Deliverable | Done when |
|---|---|---|
| 0 | ✅ **Baseline.** Today's routing frozen into `tests/reference/scipy_matrix.py` and `scipy_assignment.py`. `scripts/bench_routing.py` runs the centroid matrix, zone matrix, assignment and zone assignment on a real network. | Numbers in section 13 |
| 1 | ✅ **Numba routing on travel time.** `routing.py` with `point_matrix`, `zone_sums` and `aon`. `matrix.py`/`assignment.py` switch over, SciPy Dijkstra is removed from `src/`, numba moves into core deps. Weight is still `travel_time`. | Equal to the reference on random graphs and on the Finland network |
| 2 | ✅ **Graph v2.** Hilbert order, `edge_tail` (in `routing.Graph`), uncompressed cache, `FORMAT_VERSION`. The `skims` array moves to phase 3, where generalised cost first needs it. | Old files refused with a message |
| 3 | **Generalised cost.** `costs.py`, `cost_s`/`cost_reverse_s` and override columns, betas moved out of `speeds.py`. Graph `cost` becomes the routing weight, plus the `skims` array (5.3) and skims in outputs. Cutoff docs updated. | Matrices report physical time along least-cost paths, README cost section written |
| 4 | **Cost parameters.** Calibrate the factors (car volume, infrastructure, grade, surface). | Modelling sign-off (outside this design) |
| 5 | **AequilibraE spike.** Build an AequilibraE graph from a cached Helsinki network, run AON on `cost`, check volumes equal Numba AON. Measure `prepare_graph` time and memory at country scale. | Go/no-go plus the confirmed API recorded here |
| 6 | **Route choice** via the adapter, `--method route-choice`. | Demand conserved, documented in README |
| 7 | Equilibrium, once there is a capacity source. | (later) |

Phase 1 keeps the routing weight as `travel_time`, so it can be checked against
the frozen reference *exactly*. Generalised cost comes after the engine is
proven, which keeps "the engine changed" and "the model changed" in separate,
reviewable steps. Phase 5 can run in parallel with phases 2–4.

## 10. Testing

- **Reference oracle.** `tests/reference/scipy_routing.py` is today's
  implementation, frozen, using SciPy (already a dependency). Random
  continuous weights make shortest paths unique, and on those graphs Numba's
  `point_matrix`, `zone_sums` and `aon` must match the reference with
  `allclose`. The NetworkX oracle in `test_routing.py` stays.
- **Skims.** On a graph where the least-cost and least-time paths differ by
  construction, the time skim equals the sum of edge times along the
  least-*cost* path (computed by NetworkX on `cost`), and is greater than the
  least-time distance.
- **AON invariants.** Total loaded Σ volume × cost equals Σ demand × OD cost.
  Missed plus placed equals total demand.
- **Cost model.** Per-factor unit tests in `test_costs.py`, mirroring
  `test_speeds.py`: lane vs mixed on a primary road, car volume bands, grade
  direction, override, and the `cost ≥ time` bound.
- **Format.** Loading a v1 file raises `NetworkFormatError`, and a cached v1
  graph is rebuilt.
- **CI matrix.** Python 3.10 with pandas 2, core only. 3.12 with the
  `aequilibrae` extra (pandas 3). 3.13 core with pandas 2.
- **AequilibraE tests** use `pytest.importorskip("aequilibrae")`, and on a small
  network compare `aon` through the adapter with Numba `aon`.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Numba lags a new Python release | Wheels usually follow within weeks. Document the supported range. Numba's numpy pin is handled by the resolver. |
| Cost cutoff misses pairs within the time cutoff | Documented (5.4). Optional `--max-travel-minutes` post-filter. Raise `--max-minutes` if the far tier needs to reach further. |
| Moving the betas out of speed changes every result | Intended, and isolated in phase 3 with before/after matrices and volumes in the PR. |
| Ties between equal-cost paths differ from the reference | Documented. Equivalence tests use tie-free weights. |
| Per-thread memory at high `--workers` | Logged up front, with a warning above half of RAM. Unrequested skims are dropped. |
| AequilibraE API differs from what's assumed | Phase 5 confirms it before any CLI surface is built. The adapter is one module. |
| AequilibraE `prepare_graph` is slow or heavy at country scale | Measured in phase 5. Fallbacks: cache the prepared graph, or restrict route choice to a region. |
| JIT compilation on first run surprises users | Log "compiling routing kernels (first run only)". Cache it under `--cache-dir`. |

## 12. Open questions

1. **Drop SciPy entirely?** That would need `shapely.STRtree` for snapping, a
   Numba component search, and a demand type of our own instead of
   `csr_matrix`, which is a public API change. Recommendation: not now. It gains
   nothing for installability (2.3).
2. **Cost-factor bonuses.** Does any profile need a factor below 1? If so, the
   `cost ≥ time` bound loosens, and the cutoff semantics in 5.4 need restating.
3. **Turn and crossing penalties.** Left turns across traffic and unsignalled
   major-road crossings matter for cycling. They need either a turn-expanded
   graph or node penalties. The kernel can take a per-node penalty cheaply; true
   turn costs are a larger change, deliberately out of scope here.
4. **Future: contraction hierarchies / PHAST** for repeated full-country
   matrices. Out of scope. `routing.point_matrix` is the one function such an
   engine would replace.

## 13. Results and decisions

### 13.1 Measured on the Finland bike network

3.38 M nodes, 8.15 M directed edges, 9 273 zones with 8 access points each,
synthetic gravity demand. 14 workers on a 16-core desktop with 32 GB. Run with
`scripts/bench_routing.py`; the old implementation is the frozen SciPy code in
`tests/reference/`. Every result agreed with the old one exactly: no
reachability mismatches, no cell differing by more than 0.01 s, no link volume
differing by more than 1e-6.

| Scenario | SciPy (old) | Numba (new) | Speed-up | Peak memory, old → new |
|---|---|---|---|---|
| 400 origins → all zones, no cutoff | 38.3 s | 27.4 s | 1.4× | 8.2 → 2.7 GB |
| 9 273 × 9 273 matrix, 60 min | 21.6 s | 8.3 s | 2.6× | 8.4 → 2.8 GB |
| Zone matrix, 15 min near / 60 min far tier | 155.0 s | 15.8 s | 9.8× | 9.3 → 5.4 GB |
| All-or-nothing, 1.5 M zone pairs, 60 min | 65.3 s | 7.9 s | 8.2× | 10.2 → 2.9 GB |
| Zone all-or-nothing, 579 k zone pairs (5 km) | 304.0 s | 12.4 s | 24.6× | 11.1 → 3.7 GB |

Where the time went:

- **Bounded searches** cost what they explore rather than `n_nodes` per source.
- **Zone near tiers** read a bounded search through its settled nodes and a
  node → access-point index, instead of marking and scanning every access
  point after every search (`routing._zone_rows_settled`). On its own this
  took the 15-minute near tier from 24 s to 5 s.
- **Assignment** stops each search at its last destination and loads the whole
  tree in one pass.
- **Hilbert node order** gives another 1.15–1.4× across the board.
- An unbounded search covering the whole country is the one case that gains
  little: there is nothing to skip, and Dijkstra costs what it costs.

### 13.2 flougraph: evaluated, not adopted

flougraph (FLOU's own C++ routing library) gained the general-purpose features
this design needs: cost cutoffs, early exit in all-or-nothing loading, an
unassigned-demand report, path skims, grouped matrices and sparse matrices. On
the same network it is 1.2–1.4× faster than the Numba kernels (60-minute
matrix 6.9 s, zone near tier 3.7 s, assignment 5.8 s). It cannot ship with
this package and needs a compiler, so it could only be an optional plugin.

**Decision:** Numba is the only routing engine. A 1.2–1.4× gain does not pay
for a second engine, a plugin interface and a private package that every
feature then has to be implemented and tested against. `routing.py` exposes
three functions over plain arrays, so a flougraph adapter remains a small,
self-contained addition if a concrete need arises. The two that would justify
it: **turn costs** (flougraph supports exact turn costs; the kernels do not) and
**repeated runs with changing costs**, such as calibrating the cost factors,
where flougraph's customizable contraction hierarchies are built for exactly
that workload. Either should be benchmarked at country scale before committing.

### 13.3 Still to do

Phase 3 (generalised cost) waits on the cost-model decisions in section 12:
whether any factor may make a link cheaper than its travel time, whether turn
and crossing penalties are in scope, and who sets the factor values. The
AequilibraE phases (5–7) are unchanged.
