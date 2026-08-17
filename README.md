# valma_bike_and_walk

Walking and cycling travel-time analysis built on a **local OpenStreetMap extract**.
No Overpass calls, no network access at run time — you point it at a `.osm.pbf`
and it does the rest.

Three things, from one cached network: an origin–destination **travel-time
matrix**, an **all-or-nothing assignment** of OD demand onto per-link volumes,
and a **GeoPackage** of either for GIS. Designed for the country-scale case —
~10 000 zones across all of Finland.

Zones can be **points or polygons**. Given polygons, trips start and end all
over a zone rather than at one centroid node, which keeps volumes off a single
node, gets short trips right and lets intrazonal demand load onto the network at
all — see [Zones as polygons](#zones-as-polygons).

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

# Or give it zone *polygons* instead of centroids, and each zone's trips start
# and end all over it rather than at one node -- see "Zones as polygons" below
valma matrix --links output/bike_links.gpkg --mode bike \
             --zones taz.gpkg --id-column zone_id --near-minutes 15

valma assign --links output/bike_links.gpkg --mode bike \
             --zones taz.gpkg --id-column zone_id \
             --demand od.csv --near-minutes 15 --gpkg
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

`--centroids` and `--zones` are alternatives, and every routing command takes
either. `--centroids` is the classic one point per zone; `--zones` takes
polygons and is documented in its own section below.

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

Trips *within* a zone have no path to load with one centroid per zone, so they
are dropped. Give `assign` polygons instead and they are assigned like any
others — see the next section.

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

The polygon path is the same three calls with `Zones` in place of `Centroids`:

```python
from valma_bike_and_walk.assignment import assign_zone_traffic
from valma_bike_and_walk.matrix import zone_travel_time_matrix
from valma_bike_and_walk.zones import load_zones

zones = load_zones(
    network,
    "taz.gpkg",
    id_column="zone_id",
    weights_path="population_grid.gpkg",   # omit to use network node density
    weight_column="population",
    points_per_zone=8,
)

seconds = zone_travel_time_matrix(network, zones, near_seconds=15 * 60, workers=8)
demand = demand_matrix(read_demand_long("od.csv"), zones.ids)
volume = assign_zone_traffic(network, zones, demand, near_seconds=15 * 60, workers=8)
```

`zones.ids` indexes the result exactly as `centroids.ids` does, so everything
downstream — the `.npz`, the OMX writer, the volumes GeoPackage — is unchanged.
Inside, `zones.indptr` delimits each zone's block of access points the way a CSR
matrix delimits its rows, and `zones.weight` sums to 1 within every zone.

> **On Windows, put your script's work under `if __name__ == "__main__":`.**
> Anything with `workers > 1` starts worker processes by *spawning* fresh
> interpreters, which re-import your module — so unguarded top-level code runs
> again in every worker. With a country-sized network that means each worker
> loading its own copy of it before doing any work, and the run using several
> times the memory it should for no benefit. The `valma` CLI is already guarded;
> this only bites scripts that call the library directly.

## Zones as polygons

Routing every trip from one centroid node per zone is the standard
simplification, and it costs three things:

- **Volumes pile up** on whatever links happen to meet the centroid node, which
  makes the assignment as much a picture of where you put the centroids as of
  where people cycle.
- **Short trips come out wrong**, by however far the centroid sits from where
  people actually are.
- **Intrazonal trips come out as exactly zero** and are dropped from the
  assignment entirely — for walking and cycling that is a large share of all
  travel, and precisely the share that belongs on local streets.

Give `--zones` a polygon layer instead and each zone is represented by several
weighted **access points** inside it. Trips are routed point to point, and the
result is aggregated back to zone level with those weights.

```bash
valma matrix --links output/bike_links.gpkg --mode bike \
             --zones taz.gpkg --id-column zone_id \
             --points-per-zone 8 --near-minutes 15

valma assign --links output/bike_links.gpkg --mode bike \
             --zones taz.gpkg --id-column zone_id \
             --demand od.csv --gpkg
```

Points are accepted too, so an existing centroid file works unchanged; a
one-point zone is just the single-centroid method with the snap distance
charged for.

### How wrong is one point, and where

Write a trip as running from `c_i + u` to `c_j + v`, with `c` the zone centroids
and `u`/`v` the within-zone offsets of the real origin and destination. Expand
`|d + v − u|` around `d = |c_i − c_j|`: the linear term averages to zero by
symmetry, and what survives is

```
E|a − b| ≈ d + E[perpendicular offset²] / 2d  =  d + O(R² / d)
```

for a zone of characteristic radius `R`. So a single centroid always
**underestimates** mean travel time, and its relative error falls as `R²/2d²` —
second order, not first:

| separation | relative error |
|---|---|
| `d = 2R` | 12 % |
| `d = 5R` | 2 % |
| `d = 10R` | 0.5 % |
| `d = 0` (intrazonal) | unbounded |

That is the whole design in one table. The error lives in short trips, which is
exactly where a walking or cycling model spends its attention — and it is why
paying for multiple points over long distances buys nothing.

### Where the points come from

| `--weights` | what it uses | when |
|---|---|---|
| *not given* | **network node density** — every node in the zone, weighted by the length of link meeting it | the default; needs no extra data at all |
| a point or polygon layer | those points, optionally sized by `--weight-column` | buildings, address points, a population or workplace grid as points |

Street density is a fair proxy for where people are, which is what makes the
no-extra-data default usable rather than a placeholder. Weighting by *length*
rather than by node count is deliberate: this project splits links at tag
changes as well as at junctions, so a node count would partly measure how finely
OSM has been tagged.

However many candidates a zone has, they are reduced to `--points-per-zone` by
laying a grid over the zone, coarsening it until at most that many cells are
occupied, and taking each occupied cell's weighted centroid. That is stratified
rather than random sampling: deterministic (no seed to record), spread over the
zone by construction, and it preserves the weighted mean position exactly — so
`--points-per-zone 1` gives back the weighted centroid, and a zone's
representative point does not move when you change K.

Accuracy saturates quickly in K — as `1/√K` for random placement and faster for
this one — so 8 is normally indistinguishable from 50. Write the points out with
`--zone-points-gpkg` and look at them once for any new zone or weight layer; bad
placement shows up there and nowhere else.

### Two tiers, so it doesn't cost K times as much

Every access point is a Dijkstra source, so K points per zone means K times the
searches. But a Dijkstra tree's settled node count grows with the **square** of
its cutoff — a 15-minute tree is roughly a sixteenth of a 60-minute one — and
the table above says multiple points only earn their keep at short range. So
`--near-minutes` splits the run in two:

- **near tier** — every access point, bounded at `--near-minutes`. Accurate
  exactly where the single-point error lives, including the whole diagonal.
- **far tier** — one representative point per zone, bounded at `--max-minutes`.
  The classic single-centroid matrix.

Near values win wherever they exist. Eight points explored to 15 minutes cost
roughly *half* of one unbounded run, so the two tiers together land close to the
price of the single-point matrix. Set `--near-minutes` from the zone system
rather than by feel — five or so zone radii, where the error is already down to
about 2 %.

A cell whose near tier only partly reached would average over exactly its
closest point pairs, biasing it low; such cells fall through to the far tier
instead rather than being reported. That also means a zone pair is resolved only
when its *furthest* access-point pair is inside the cutoff, so the effective
reach is roughly `--near-minutes` less the time to cross two zones — budget for
that rather than being surprised by it. The run logs what share of pairs the
near tier resolved, which is the number to tune against.

`valma assign` takes `--near-minutes` too, and splits the same way — but it
cannot use a distance rule to decide what goes where, because a pair sorted into
the near tier and then not reached within the cutoff would be demand silently
lost. So the split is decided by what actually happened:

1. Every access point, bounded at `--near-minutes`. This places all the demand
   whose trips are short enough for the distribution to matter, including
   everything intrazonal.
2. Whatever tier 1 **reported** it could not reach — per zone pair, not guessed
   at — assigned from one representative point per zone at `--max-minutes`.

Total demand is conserved across the two by construction. A pair is either
loaded distributed, loaded from representative points, or genuinely unreachable
and dropped with a warning, exactly as in a single-tier run.

### What else changes

- **Snapping is no longer free.** A point up to `--max-snap-distance` (1000 m)
  from the network used to be teleported onto it at no cost — twelve minutes of
  unpriced walking. Each access point now carries the time to cover its own snap
  distance, added at both ends of every trip. `--access-speed-kmh` sets the pace
  (default: the mode's base speed); `--no-access-time` restores the old
  behaviour.
- **Intrazonal demand is assigned.** Zone `i`'s own demand is spread over the
  ordered pairs of its access points, excluding the pair from a point to itself
  and renormalising so the total is preserved. A zone with only one access point
  still has nowhere to put it; that is reported, not silently dropped.
- **The intrazonal travel time is a real number**, averaged over distinct point
  pairs. Where routing cannot produce one — a single-point zone, or a zone the
  near tier did not cover — it falls back to the equal-area circle,
  `0.9054 · R / v` for `R = √(area/π)`. That is a crow-flies estimate, not a
  routed answer, and it is only ever used to *fill* a missing value, never to
  overrule one: a zone whose demand really does sit in one corner has genuinely
  shorter internal trips than a uniform disc of the same area, and finding that
  out is the whole reason for weighting the points.

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

With `--zones`, the source count is the *access point* count, so
`--points-per-zone` multiplies the searches directly — and the demand grows by
K² for having been spread over those points. Two levers keep that in hand:

- **`--near-minutes`** bounds the multi-point searches and assigns the rest from
  representative points, the same two-tier trade the matrix makes. Reach for it
  first.
- **`--min-demand`** thresholds the zone matrix *before* it is expanded, so
  every pair it drops saves K² point pairs rather than one. On a large, mostly
  noise demand matrix this is worth more here than it is for `--centroids`.

#### Memory, and which dial moves which part of it

A parallel run has two independent costs, and they respond to different flags.

**Per worker**, and nothing to do with how many zones there are:

```
chunk_size x n_nodes x 12 bytes   the Dijkstra distance + predecessor block
      + n_edges x  8 bytes        the volume accumulator
```

On the 3.4 M-node national bike network that is 308 MB per worker at the default
chunk size, or 105 MB at `--chunk-size 1`. The graph, the edge-key lookup and the
demand are **memory-mapped**, so they cost once for the machine rather than once
per worker, and do not enter this. Multiply by `--workers` and budget
accordingly: 16 workers wants ~5 GB before the demand is considered.

**Once, shared:** the expanded demand, which the run logs as `Expanded N zone OD
pair(s) ... M point pair(s)`. It grows with zones² × K², so it is the term that
runs away on a big zone system — roughly 0.8 GB at 1000 zones with K=8, 3 GB at
2000. Watch that log line.

So, in order:

| symptom | reach for |
|---|---|
| too slow | `--near-minutes`, then `--max-minutes` |
| out of memory, big zone system | `--min-demand` (it cuts K² point pairs per zone pair dropped), then `--points-per-zone` |
| out of memory, big *network* | `--chunk-size`, then `--workers` |

`--chunk-size` only ever moves the per-worker Dijkstra block. Lowering it also
makes chunks *more numerous*, so it does nothing for demand-side memory — that
was worth saying because reaching for it first is the natural instinct and the
wrong one when the zone system is what is large.

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

Two things are deliberate and worth knowing. Profiles are read at a fixed 25 m
spacing rather than at the geometry's own vertices, so a long two-vertex link
cannot hide a hill between its ends. And steps below 0.5 m are read as noise and
dropped — without that dead band, sampling a flat road picks up the kerb and the
ditch either side of it and reports a climb that is not there. Links tagged
`bridge` or `tunnel` are forced flat, because MML's model is a *terrain* model:
it reports the valley floor under a bridge and the hilltop over a tunnel.

> **Not yet wired into routing.** `valma build` does not read these columns.
> Travel time is still one scalar per link, applied to both directions, so a hill
> currently costs the same up as down. Making slope bite means splitting
> `travel_time_s` per direction in `links.directed_edges`.

## Layout

```
src/valma_bike_and_walk/
├── config.py      coordinate systems, mode names, cache paths
├── speeds.py      walk/bike speed profiles
├── osm.py         the only module that talks to pyosmium   [stage 1]
├── links.py       the editable link layer: schema, repair, direction
├── network.py     RoutableNetwork: CSR + coords + KD-tree  [stage 2]
├── centroids.py   reading points and snapping them to nodes
├── zones.py       zone polygons -> weighted access points inside them
├── matrix.py      chunked / parallel OD travel-time matrices, point or zone level
├── demand.py      reading OD demand (long / .npz / OMX) as a sparse matrix, writing OMX
├── assignment.py  all-or-nothing assignment: demand -> per-link volumes
├── elevation.py   DEM tiles from the NLS: fetch, cache, height profiles
├── gpkg.py        draw a per-edge result back onto the link layer
└── cli.py         dem / extract / build / matrix / assign
```

Data flows one way: `osm` → `links` → `network` → (`centroids` or `zones` +)
`matrix` / `assignment` → `gpkg`. Nothing downstream reaches back. `zones`
builds on `centroids` rather than replacing it: a zone's representative point
*is* a `Centroids`, which is what lets the far tier of a two-tier matrix and the
plain single-point method be the same code.

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
| `test_zones.py` | where access points land, and how they are weighted |
| `test_zone_matrix.py` | the point-pair-to-zone-pair aggregation arithmetic |

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
