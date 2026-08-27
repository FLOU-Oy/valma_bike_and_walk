# valma_bike_and_walk

Walking and cycling travel-time analysis built on a **local OpenStreetMap extract**.
No Overpass calls, no network access at run time — you point it at a `.osm.pbf`
and it does the rest.

Three things, from one cached network: an origin–destination **travel-time
matrix**, an **all-or-nothing assignment** of OD demand onto per-link volumes,
and a **GeoPackage** of either for GIS. Designed for the country-scale case —
~10 000 centroids across all of Finland.

The pipeline is deliberately two stages, with an **editable GeoPackage in the
middle**:

```
finland.osm.pbf ──valma extract──▶ walk_links.gpkg ──matrix · assign──▶ results
                                          ▲
                                    edit in QGIS
```

`matrix` and `assign` build the routable graph from the link layer themselves
and cache it under `--cache-dir` — there is no separate build step to run, and
nothing named `walk.npz` to know about. `valma build` still exists for when a
graph is worth keeping as a named file, to route against repeatedly with
`--network` without re-reading the link layer each time.

---

## Why it is built this way

Three decisions drive the whole design.

### 1. The network is a GeoPackage you can edit

Stage one turns the PBF into a **link layer**: one row per stretch of way
between two junctions, with its geometry, its tags and its derived speed. Open
it in QGIS, fix what OSM got wrong — a missing underpass, a barrier that is not
mapped, a path that is closed for the winter — save, and build. Stage two reads
whatever is on disk.

Nothing is hidden from that table. Length is recomputed from the geometry, speed
from the tags, so an edit takes effect without there being a second place to
change. And because every row keeps its `link_id` all the way through the graph,
an assignment's volumes come back **on the rows you edited**.

See the module docstring in [`links.py`](src/valma_bike_and_walk/links.py) for
the column-by-column contract.

### 2. Speeds come from the way, not from `maxspeed`

Speed is not imputed from the OSM `maxspeed` tag — that records the **motor
traffic speed limit**, and applying it to a bike or pedestrian network has a
cyclist doing 50 km/h down an arterial while ignoring the surface entirely, so
travel times come out far too optimistic.

Instead, speed is derived from the two tags that actually govern how fast a
person moves — `highway` (what kind of way it is) and `surface` (what it is made
of) — via explicit, reviewable profiles in [`speeds.py`](src/valma_bike_and_walk/speeds.py):

| | walking | cycling |
|---|---|---|
| baseline | 4.8 km/h | 14 km/h |
| segregated cycleway | 4.8 km/h | 18 km/h |
| forest path | 4.3 km/h | 11 km/h |
| steps | 1.2 km/h | 1.2 km/h (pushing) |
| mud vs asphalt | ×0.70 | ×0.40 |
| clamped to | 0.8–6 km/h | 1–30 km/h |

The profiles are plain dataclasses; edit the tables to match your own survey
data. For a one-off exception there is no need to touch them at all — put a
number in the link layer's `speed_override_kmh` column and that link uses it.

### 3. Routing uses SciPy CSR, not NetworkX

A NetworkX `MultiDiGraph` stores a Python dict per node and per edge. That is
fine for one city and hopeless for a country: Finland's walking network is
millions of nodes, which costs tens of GB as dicts, and its Dijkstra is pure
Python.

The same information lives here in a handful of numpy arrays plus a SciPy CSR
matrix, routed with compiled `scipy.sparse.csgraph.dijkstra`. The cached network
is a compressed `.npz` rather than GraphML — for the Helsinki capital region
that is 21 MB against 128 MB.

---

## Install

Python ≥ 3.10.

```bash
uv sync                 # core
uv sync --extra dev     # + pytest/ruff/black/mypy
uv sync --extra viz     # + networkx/matplotlib, only for to_networkx()/plotting
uv sync --extra dem     # + rasterio, only for reading cached elevation tiles
```

You also need an OpenStreetMap extract of your own.
[Geofabrik](https://download.geofabrik.de/europe/finland.html) publishes a daily
`finland-latest.osm.pbf` (~740 MB); everything below assumes a local copy.

`valma dem` is the one command that goes to the network, and only to fill its
tile cache; nothing else here downloads anything at run time.

## Use

```bash
# Stage 1: PBF -> editable link layer
valma extract --pbf finland-260805.osm.pbf --mode bike --bbox 24.5 60.1 25.1 60.4
#   -> output/bike_links.gpkg

#   ... open it in QGIS, fix what needs fixing, save ...

# Optional: attach elevation, fetching only the DEM tiles these links cross
export MML_API_KEY=...          # free, see "Elevation" below
valma dem --links output/bike_links.gpkg
#   -> z_u, z_v, ascent_m, descent_m added in place; tiles kept in .cache/dem/

# Route against the link layer directly -- the routable graph is built and
# cached under .cache/ the first time, and reused after that
valma matrix --links output/bike_links.gpkg --mode bike \
             --centroids centroids.csv --id-column id \
             --workers 8 --max-minutes 120

# Assign an OD demand matrix, and draw the result back on the links you edited
valma assign --links output/bike_links.gpkg --mode bike \
             --centroids centroids.csv --id-column id \
             --demand od.csv --gpkg
```

Each command prints the path it wrote and the next command to run.

Everything a command names in its output — the link layer, the matrix, the
volumes — goes to `--output-dir` (default `output/`). Everything a command
builds for itself along the way and would rebuild identically next time — the
graph built from `--links`, and anything a `--pbf` run extracts — is cached
under `--cache-dir` (default `.cache/`) instead, keyed on its inputs, and that
directory is always safe to delete. If a graph is worth keeping as a named
file — routing against it many times over, or sharing it — run
`valma build --links output/bike_links.gpkg --mode bike` explicitly; that
writes `output/bike.npz`, which any later command can load directly with
`--network output/bike.npz`, skipping the link layer entirely.

If there is nothing you want to edit, `build`, `matrix` and `assign` all take
`--pbf` directly and run stage one themselves too:

```bash
valma matrix --pbf finland-260805.osm.pbf --mode walk \
             --centroids centroids.csv --id-column id
```

`centroids.csv` needs `lon`/`lat` columns by default (`--x-column`/`--y-column`
and `--centroid-crs` if not). Any vector format GeoPandas reads works too.
Points that snap no closer than `--max-snap-distance` (default 1000 m) are
dropped, not silently routed from somewhere else. `valma matrix` writes an
`.npz` holding `ids` and a float32 `seconds` matrix, rows and columns in `ids`
order. Add `--omx` to also write `<output-dir>/travel_times_<mode>.omx` (or
`--omx path/to/file.omx` to name it yourself), in the same matrix+lookup
layout `valma assign --demand-matrix` reads (needs integer centroid ids); use
`--omx-matrix-name`/`--omx-mapping-name` to change the matrix/lookup names
inside it (defaults: `--mode`, `zone_number`).

### Editing the link layer

`output/<mode>_links.gpkg` has one layer, `links`, with one row per link:

| column | meaning |
|---|---|
| `link_id` | this layer's identity — results join back on it |
| `u`, `v` | OSM node ids of the two ends; the graph's topology |
| `osm_way_id` | the way it came from, for tracing back to OSM |
| `highway`, `surface` | what sets the speed |
| `oneway`, `oneway_bicycle`, `junction` | what sets the direction |
| `length_m`, `speed_kmh`, `travel_time_s` | derived — recomputed on every build |
| `speed_override_kmh` | **empty, for you**: a number here wins over the tags |

What the build does with your edits:

- **Change `highway` or `surface`** → the speed follows.
- **Set `speed_override_kmh`** → that link uses it regardless of its tags.
- **Move or reshape a geometry** → `length_m`, and so the cost, follows.
- **Delete a row** → the link is gone. (If that severs the network, only the
  largest remaining component is kept — check the build's log line.)
- **Draw a new row** and leave `u`/`v` empty → each end snaps to the nearest
  existing link end within 2 m and joins there; an end near nothing gets a fresh
  negative node id, as JOSM does.

Reprojecting the layer is fine — lengths are measured in `EPSG:3067` metres
whatever CRS the file is in.

### Assignment

`valma assign` loads an origin–destination demand matrix onto the network's
shortest paths and sums the result per link — **all-or-nothing** assignment.
It ignores congestion, which for footpaths and bike lanes is the right
trade-off: they essentially never reach the capacity that makes traffic
reroute. See the module docstring in
[`assignment.py`](src/valma_bike_and_walk/assignment.py) for when that stops
holding and what the standard fixes are.

`--demand` takes any of three shapes, picked by file extension:

| `--demand` | what it is | relevant options |
|---|---|---|
| `.csv` / `.tsv` / anything else | long format: one `origin_id, destination_id, demand` row per **nonzero** pair | `--origin-column`, `--destination-column`, `--demand-column` |
| `.npz` | dense matrix, as `ids` + `demand` arrays | — |
| `.omx` | [Open Matrix](https://github.com/osPlanning/omx), the HDF5 format transport models exchange | `--demand-matrix` (required — which matrix in the file), `--demand-mapping` (default `zone_number`) |

Ids are matched against the centroid ids, in any order; anything unmatched is
dropped with a warning rather than failing the run. Repeated pairs are summed,
so demand assembled per time period or trip purpose can just be concatenated.
The demand is kept sparse end to end — a dense 10 000 × 10 000 float64 matrix
would be 800 MB, and almost all of it structural zero.

Output is `<output-dir>/link_volumes_<mode>.npz`, one row per **directed** edge
(`u`, `v`, `length_m`, `travel_time_s`, `volume`), aligned with the network's
own edge arrays. Add `--gpkg` (with `--links`) to also write
`<output-dir>/<mode>_volumes.gpkg`: the same rows, drawn on the link layer's own
geometry, with `link_id` and a `direction` of +1 or −1. A two-way link that
carries traffic both ways gets two rows sharing one shape.

### From Python

The CLI is a thin wrapper; everything is callable directly.

```python
from valma_bike_and_walk import Settings, build_links, network_from_links
from valma_bike_and_walk.assignment import assign_traffic
from valma_bike_and_walk.centroids import load_centroids
from valma_bike_and_walk.demand import demand_matrix, read_demand_long
from valma_bike_and_walk.links import read_links
from valma_bike_and_walk.matrix import travel_time_matrix

settings = Settings(pbf_path="finland-260805.osm.pbf")

# Stage 1, then edit links however you like -- it is a plain GeoDataFrame.
path, links = build_links(settings, mode="bike", bbox=[24.5, 60.1, 25.1, 60.4])
links.loc[links["surface"] == "gravel", "speed_override_kmh"] = 8.0

# Stage 2. (Or: network_from_links(read_links(path), "bike") after a QGIS round trip.)
network = network_from_links(links, mode="bike")

centroids = load_centroids(network, "centroids.csv", id_column="id").drop_unsnapped()
seconds = travel_time_matrix(network, centroids.node_index, workers=8)

demand = demand_matrix(read_demand_long("od.csv"), centroids.ids)
volume = assign_traffic(network, centroids.node_index, demand, workers=8)
```

`centroids.ids` and `centroids.node_index` line up positionally, and that is the
contract everything else relies on: the matrix rows, the demand matrix rows and
columns, and the assignment's sources are all in that order. `volume` is one
value per directed edge, in the same order as `network.travel_time`,
`network.indices` and `network.link_id`.

---

## Selecting an extent

| | cost on a country PBF | notes |
|---|---|---|
| `--bbox` | one pass | cheap, recommended |
| `--area Espoo` | one pass **+ a multipolygon assembly pass** | exact municipal polygon; cached after the first run |
| neither | whole country | fine — see below |

`--bbox` keeps any way with at least one node inside the box, whole. `--area`
finds the administrative boundary inside the PBF (so it stays offline) by
assembling every candidate `boundary=administrative` relation, then keeps the
links that intersect it — again whole, never cut, because `u`/`v` have to stay
real OSM node ids. The boundary is cached as GeoJSON, so that pass is paid once
per area.

Name matching is by substring, so `--area Espoo` also sees `Pohjois-Espoo`; an
exact name wins, then the largest match.

---

## Scaling to the whole of Finland

```bash
# 1. Strip the PBF to highways once -- much faster to read, identical network
osmium tags-filter finland-latest.osm.pbf w/highway -o finland-highways.osm.pbf

# 2. Extract the country's links in one pass
valma extract --pbf finland-highways.osm.pbf --mode walk

# 3. Build, route and assign against what you have on disk
valma build  --links output/walk_links.gpkg --mode walk
valma matrix --network output/walk.npz --mode walk \
             --centroids centroids.csv --id-column id \
             --workers 14 --max-minutes 60
```

A country needs no special handling — no tiling, no splitting the extent, no
extra flags. One pass of `valma extract` reads the whole thing.

### Memory, and the node index

osmium reads the file as a stream. It holds a node-coordinate index plus the
ways that pass the filter, so peak memory tracks the size of the *network*
rather than the size of the *file* — the whole of Finland peaks at 5.5 GB (see
the table below). The reader is single-threaded and does not spill to disk.

`--index-storage` picks how osmium keeps those node coordinates. The default
`flex_mem` holds them in memory and is right up to a country; for a planet-sized
file, hand it a file-backed index instead:

```bash
valma extract --pbf planet.osm.pbf --mode walk \
              --index-storage "sparse_file_array,nodes.idx"
```

### Shrink the file first

Every read scans the whole PBF, so the fastest thing you can do is give it less
to scan. A country extract is mostly buildings, landuse and coastline that a
routing network never looks at. Strip it once with
[osmium](https://osmcode.org/osmium-tool/):

```bash
osmium tags-filter finland-260805.osm.pbf w/highway -o finland-highways.osm.pbf
```

737 MB becomes ~190 MB. Cutting by area works too, and composes:

```bash
osmium extract -b 22,60,25,63 finland-highways.osm.pbf -o region.osm.pbf
```

### Measured numbers

Walking networks from the 190 MB highway-filtered Finland extract, on a 16-core
desktop:

| | capital region<br>`--bbox 24.40 60.05 25.30 60.45` | **whole of Finland**<br>no clip |
|---|---|---|
| `valma extract` | 101 s | **173 s** |
| links / GeoPackage | 553 139 · 138 MB | 3 746 910 · 1 153 MB |
| `valma build` | 15 s | **44 s** |
| nodes / edges | 410 848 / 984 708 | 2 966 208 / 6 851 200 |
| network `.npz` | 21 MB | 148 MB |
| peak memory | — | **5.5 GB** |

The whole country, both stages, is under four minutes and fits comfortably in a
desktop's memory.

Extract time is dominated by the pass over the whole file, so it barely depends
on how small the bbox is; build and route times depend only on the network. A
`valma assign` over that capital-region network — 297 centroids, 60-minute
cutoff, 8 workers — takes 20 s including writing a 219 MB volumes GeoPackage.

### Is the network right?

It agrees with an independently built one — [pyrosm](https://pyrosm.readthedocs.io/),
a different OSM reader with its own graph construction — to within rounding on
almost every pair. 150 random capital-region centroids, 22 348 reachable OD
pairs, walking:

| | |
|---|---|
| median relative difference | **+0.26 %** |
| within 1 % | 96.0 % |
| within 5 % | 99.0 % |
| within 10 % | 99.6 % |

The residual comes from one deliberate difference. pyrosm collapses every
degree-2 chain, including where two differently-tagged ways simply meet; this
version splits only at genuine junctions and keeps those joins as separate rows,
because a row you can select in QGIS is the whole point of the middle stage. So
the graph carries ~10–15 % more nodes and edges than pyrosm's (nationally:
2.97 M nodes against 2.67 M), and a handful of pairs snap to a slightly
different nearest node.

### Computing the matrix

Cost is dominated by one number: **one Dijkstra run per origin**. 10 000 origins
means 10 000 runs, so plan with the measured per-source time your run logs.

Levers, in order of effect:

1. **`--max-minutes`** — a cutoff stops the search expanding across the entire
   country and is by far the biggest win if your analysis has one anyway. Use
   one. Walking between distant centroids is not a meaningful number — Helsinki
   to Utsjoki on foot is some 270 hours. In the capital-region benchmark below a
   60-minute cutoff left only 3.8 % of pairs reachable, and it is **20× faster**.
2. **`--workers`** — origins are embarrassingly parallel. The network is shared
   with the workers by memory-mapping, not pickling, so N workers do not cost N
   copies of the graph.
3. **`--chunk-size`** — SciPy returns a dense `(chunk, n_nodes)` float64 block,
   so this is a memory dial, defaulting to a 256 MB block. Lower it if you are
   tight on RAM.

A 10 000 × 10 000 matrix is 100 million values; at float32 that is 400 MB in
memory, which the `.npz` output compresses.

### Assigning demand

`valma assign` runs the same one-Dijkstra-per-origin as the matrix, so the
levers above apply unchanged — `--max-minutes` first, then `--workers`. Two
things differ:

- **A source is routed whether or not it has demand.** The Dijkstra is per
  origin, not per OD pair, so the run cost tracks the centroid count. On top of
  it, each origin's nonzero destinations are walked back up the predecessor
  tree — all of them in lockstep and vectorised, sharing the tail where their
  paths converge — which is cheap next to the search itself. So `--min-demand`
  trims the loading step, not the search; reach for `--max-minutes` first.
- **Chunks hold fewer sources for the same memory.** Walking paths back needs
  Dijkstra's int32 predecessor block on top of the float64 distances, so a
  source costs 12 bytes per node instead of 8, and the same 256 MB
  `--chunk-size` budget buys two-thirds as many sources per block. Nothing to
  set — just don't expect the two commands to chunk identically.

Unreachable pairs — including everything beyond `--max-minutes` — are dropped
with a logged count rather than quietly assigned somewhere.

---

## Elevation

The National Land Survey publishes no whole-country GeoTIFF to download. What it
publishes is a [WCS query
service](https://www.maanmittauslaitos.fi/ortokuvien-ja-korkeusmallien-kyselypalvelu/tekninen-kuvaus)
that cuts an arbitrary rectangle out of the 2 m laser-scanned elevation model,
capped at 10 km × 10 km and 5000 px a request. So `valma dem` is a tiling client:
it cuts the area into 8 km tiles on a fixed grid aligned to the EPSG:3067 origin,
fetches the ones that are missing, and keeps them under `.cache/dem/`.

The grid is fixed on purpose. An extent-shaped request is unshareable — shift the
bbox by a metre and none of it can be reused — whereas a fixed tile is the same
file whichever run asked for it. The cache is additive: every run can only add to
what the next one finds already there. Which tiles get fetched is decided from
the **links**, not their bounding box, so a sparse rural network pays for the
ground its roads actually cross.

```bash
export MML_API_KEY=...                       # or pass --api-key

valma dem --links output/bike_links.gpkg     # attach elevation, fetch as needed
valma dem --bbox 24.5 60.1 25.1 60.4         # just pre-fill the cache
valma dem --links output/bike_links.gpkg --dem-resolution 10
```

An API key is free from
[maanmittauslaitos.fi](https://www.maanmittauslaitos.fi/rajapinnat/api-avaimen-ohje).
The data is CC BY 4.0 — attribute the National Land Survey of Finland in anything
you publish from it.

**Columns added** to the link layer, alongside the existing `speed_override_kmh`
convention:

| column | meaning |
|---|---|
| `z_u`, `z_v` | height at the link's first and last vertex, metres |
| `ascent_m` | metres climbed following the digitised direction |
| `descent_m` | metres dropped following it |
| `grade_override` | empty, for you to fill in where the DEM is wrong |

Ascent and descent are kept apart rather than netted off: a link that climbs ten
metres and drops them again costs nothing like a flat one, and the two swap
places when it is traversed backwards.

For routing, uphill grades below 2.5% receive no penalty. Grades from 2.5% to
5% add `2.32 * gradient * length_m` seconds, and grades above 5% add
`2.50 * gradient * length_m` seconds, where `gradient` is the decimal grade
(`ascent_m / length_m`). Downhill travel receives no surcharge. In the reverse
direction, the original `descent_m` is treated as ascent because that direction
is uphill on the same link.
is uphill on the same link. Grades of 20% or more are treated as unrealistic DEM
artefacts and receive no elevation surcharge in that direction.

Two things are deliberate and worth knowing. Profiles are read at a fixed 25 m
spacing rather than at the geometry's own vertices, so a long two-vertex link
cannot hide a hill between its ends. And steps below 0.5 m are read as noise and
dropped — without that dead band, sampling a flat road picks up the kerb and the
ditch either side of it and reports a climb that is not there. Links tagged
`bridge` or `tunnel` are forced flat, because MML's model is a *terrain* model:
it reports the valley floor under a bridge and the hilltop over a tunnel. A
profile change above 15% on a link immediately next to one of these structures is
also discarded, as the same terrain artefact can spill over the structure's
endpoint. Gentler adjacent grades are retained. The resulting ascent and
descent values are used directionally by routing; downhill travel receives no
surcharge.

## Layout

```
src/valma_bike_and_walk/
├── config.py      coordinate systems, mode names, cache paths
├── speeds.py      walk/bike speed profiles
├── osm.py         the only module that talks to pyosmium   [stage 1]
├── links.py       the editable link layer: schema, repair, direction
├── network.py     RoutableNetwork: CSR + coords + KD-tree  [stage 2]
├── centroids.py   reading points and snapping them to nodes
├── matrix.py      chunked / parallel OD travel-time matrices
├── demand.py      reading OD demand (long / .npz / OMX) as a sparse matrix, writing OMX
├── assignment.py  all-or-nothing assignment: demand -> per-link volumes
├── elevation.py   DEM tiles from the NLS: fetch, cache, height profiles
├── gpkg.py        draw a per-edge result back onto the link layer
└── cli.py         dem / extract / build / matrix / assign
```

Data flows one way: `osm` → `links` → `network` → (`centroids` +) `matrix` /
`assignment` → `gpkg`. Nothing downstream reaches back.

## Development

```bash
uv run pytest
uv run ruff check . && uv run black . && uv run mypy src/
```

The test suite needs no downloaded data. pyosmium can *write* a `.osm.pbf` as
well as read one, so `tests/conftest.py` builds tiny extracts by hand — a
five-node grid, a lollipop way, a boundary relation — and the extract stage is
tested against real parsing rather than a mock. Everything else is small
hand-built link layers and demand matrices, so the whole suite runs in seconds.

| file | what it holds down |
|---|---|
| `test_osm.py` | way filters, junction splitting, clipping, boundary assembly |
| `test_links.py` | the GeoPackage round trip and every editing rule above |
| `test_assembly.py` | stitching extents on OSM node id, dedup, largest component |
| `test_gpkg.py` | results joining back onto the right rows |
| `test_pipeline.py` | end to end, including the CLI and an edit in the middle |
| `test_routing.py` | SciPy routing, checked against NetworkX as an oracle |

Five invariants hold the design together. Breaking one is what the tests are
mostly there to catch:

- **All pyosmium calls live in `osm.py`.** Everything else sees plain
  GeoDataFrames and numpy arrays, so an osmium upgrade can only break in one
  module.
- **The link layer is the only configuration surface.** Length and speed are
  derived from the table on every build, never trusted from a stale column, so
  there is no second place an edit has to be repeated.
- **`link_id` survives to the result.** Every directed edge names the link layer
  row it came from, which is what lets an assignment be drawn back on the rows
  you edited. A new per-edge quantity must keep that alignment.
- **Speeds are applied per link, before anything is merged.** Both profiles are
  plain dataclasses; change the tables, don't add special cases downstream.
- **Big intermediates stay chunked and sparse.** SciPy's Dijkstra returns a
  dense `(sources, n_nodes)` block, which is why both `matrix` and `assignment`
  size their chunks against a byte budget and merge results as they complete.
  Anything new that scales with zones² should be sparse (see `demand.py`).

Workers get the network by memory-mapping it from a temporary directory, not by
pickling — so adding a parallel step means writing the arrays out once, not
handing the graph to each process.

## Licence

EUROPEAN UNION PUBLIC LICENCE v. 1.2 (EUPL-1.2)
