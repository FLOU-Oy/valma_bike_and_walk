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
| cycleway (shared or untagged) | 4.8 km/h | 18 km/h |
| segregated cycleway (`segregated=yes`) | 4.8 km/h | 18.9 km/h (×1.05) |
| forest path | 4.3 km/h | 11 km/h |
| steps | 1.2 km/h | 1.2 km/h (pushing) |
| mud vs asphalt | ×0.70 | ×0.40 |
| clamped to | 0.8–6 km/h | 1–30 km/h |

The profiles are plain dataclasses; edit the tables to match your own survey
data. For a one-off exception there is no need to touch them at all — put a
number in the link layer's `speed_override_kmh` column and that link uses it.

### 3. Routing is compiled kernels over flat arrays, not NetworkX

A NetworkX `MultiDiGraph` stores a Python dict per node and per edge. That is
fine for one city and hopeless for a country: Finland's walking network is
millions of nodes, which costs tens of GB as dicts, and its Dijkstra is pure
Python.

The same information lives here in a handful of numpy arrays in CSR layout,
searched by Dijkstra kernels compiled with [Numba](https://numba.pydata.org/)
([`routing.py`](src/valma_bike_and_walk/routing.py)). A search costs what it
explores rather than what the network holds, stops once it has found everything
it was asked for, and does its follow-up work — aggregating a zone row, loading
demand onto the shortest-path tree — in the same compiled loop. Nodes are
numbered along a Hilbert curve so that map neighbours are memory neighbours.
The graph is a plain `.npz` rather than GraphML — for the Helsinki capital
region that is 21 MB against 128 MB.

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

# Optional: carry car traffic volumes over from another network's links
valma traffic --links output/bike_links.gpkg --volumes data/link_volumes.gpkg
#   -> car_volume added in place, on the roads that carry cars

# Optional: keep islands in the graph -- see "Islands" below
valma connect --links output/bike_links.gpkg
#   -> output/bike_connectors.gpkg; add --connectors it to matrix/assign/build

# Route against the link layer directly -- the routable graph is built and
# cached under .cache/ the first time, and reused until the layer changes
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
under `--cache-dir` (default `.cache/`) instead, keyed on its inputs and rebuilt
when the link layer is newer than it, and that directory is always safe to
delete. If a graph is worth keeping as a named
file — routing against it many times over, or sharing it — run
`valma build --links output/bike_links.gpkg --mode bike` explicitly; that
writes `output/bike.npz`, which any later command can load directly with
`--network output/bike.npz`, skipping the link layer entirely.

**A cache is reused only while its inputs are unchanged.** Each graph records
what it was built from — the link layer's path, size and modification time
(its SQLite `-wal` file included, where QGIS keeps edits it has not yet
checkpointed), the mode, and a digest of the code that builds it. A link layer
cached from a `--pbf` records the PBF, the extent and the extract code. When
any of that differs, the next run rebuilds and logs why. So saving an edit in
QGIS, running `valma dem` or `valma traffic` (both write into the layer in
place), or changing a speed profile all take effect on the next `matrix` or
`assign`, with no `--force-reload` needed. `valma build` refreshes its named
`.npz` the same way. Two things are left alone on purpose: a `--network` file
is used as given, though a warning says so when `--links` has changed since it
was built; and an existing `valma extract` output is never overwritten without
`--force-reload`, because it may hold your edits.

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
order. It also writes `<output-dir>/travel_times_<mode>.omx` (or
`--omx path/to/file.omx` to name it yourself) — in **minutes**, not seconds,
with a `units` attribute on the matrix saying so — in the same matrix+lookup
layout `valma assign --demand-matrix` reads — skipped with a log line if the
centroid ids are not integers, which OMX cannot hold; use
`--omx-matrix-name`/`--omx-mapping-name` to change the matrix/lookup names
inside it (defaults: `--mode`, `zone_number`).

### Filling a full zone system from a baseline

`valma matrix` only has times for zones that snapped to the network and pairs
within `--max-minutes`. `valma overlay` writes those over a baseline OMX
matrix, so the result has the baseline's zones and a time in every cell:

```bash
# Bike: Valma's own bike times (minutes) as the baseline
valma overlay --mode bike --baseline data/time_aht.omx --baseline-matrix bike               --max-minutes 120 --out output/kapy_time_vrk.omx
# Walk: Valma's bike distances (km) at the model's walking pace, 4.8 km/h
valma overlay --mode walk --baseline data/dist_aht.omx --baseline-matrix bike               --baseline-speed-kmh --max-minutes 120 --out output/kapy_time_vrk.omx
```

`--baseline-speed-kmh` given bare converts at the mode's own base speed, so a
baseline cell and a routed one are at the same pace; give a number to override.
Both write into one file, a matrix per mode, in minutes. A cell takes the
model's time when both zones were routed and the time is finite (and, with
`--max-minutes`, no longer than that); otherwise it keeps the baseline. The
model's times are read from `<output-dir>/travel_times_<mode>.npz` (`--model`
to name another). Zones only in the model are ignored.

### Editing the link layer

`output/<mode>_links.gpkg` has one layer, `links`, with one row per link:

| column | meaning |
|---|---|
| `link_id` | this layer's identity — results join back on it |
| `u`, `v` | OSM node ids of the two ends; the graph's topology |
| `osm_way_id` | the way it came from, for tracing back to OSM |
| `highway`, `surface` | what sets the speed |
| `oneway`, `oneway_bicycle`, `junction` | what sets the direction |
| `under_construction` | the way is under roadworks in OSM (`highway=construction`) and was read as what it is being built into, e.g. `construction=cycleway` as a cycleway. On by default; `valma extract --no-include-construction` drops roadworks instead. Ways with no `construction=` type are always dropped. The tag also covers new ways not yet open. |
| `length_m`, `speed_kmh`, `travel_time_s` | derived — recomputed on every build |
| `speed_reverse_kmh`, `travel_time_reverse_s` | the same against the digitised direction; for bikes a cycle lane or track on one side (`cycleway:right` along, `cycleway:left` against) only speeds up its own direction |
| `signal_wait_s`, `signal_wait_reverse_s` | derived, informational: the seconds of each direction's travel time spent waiting at traffic lights. `--gpkg` volumes carry it per edge as `signal_wait_s` |
| `crossing_entry` links | zero-length connectors written by `extract`. Where a path meets one of a signalled junction's roads at an unsignalled crossing, the paths get their own (negative-id) node, joined to the road node by one of these: path to path across the road stays free, path onto the road pays the junction's wait here, road onto path is free. Without it, stepping off the cycleway onto the road at such a crossing rode through the junction without paying |
| `bicycle`, `bicycle_forward`, `bicycle_backward` | the OSM tags; for bikes, a road with `use_sidepath` -- or a `cycleway*=separate` on that side -- is ridden `SIDEPATH_FACTOR` (1.5) times slower in the direction it applies to, so the path alongside wins. A penalty, not a ban; paths and walking are unaffected |
| `speed_override_kmh` | **empty, for you**: a number here wins over the tags |
| `car_volume` | cars a day sharing this link, after `valma traffic` — see below |

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

### Islands

The build keeps only the largest connected part of the network, which drops
stray fragments but also every island's roads — and with them the zones on
those islands, or it snaps them across the water to the mainland. `valma
connect` writes a separate link layer with one straight connector per island,
from its nearest node to the nearest node of another part of the network:

```bash
valma connect --links output/bike_links.gpkg
#   -> output/bike_connectors.gpkg
valma matrix --links output/bike_links.gpkg --connectors output/bike_connectors.gpkg              --mode bike --centroids centroids.csv --id-column id
```

`--connectors` works the same on `build` and `assign`. `assign --gpkg` leaves
the island crossings off the map; their volumes (normally none) are still in
`link_volumes_<mode>.npz`. The graph built with it is cached apart
from the plain one, as `.cache/bike+bike_connectors.npz`.

- Connectors run at `--speed-kmh` (default 0.1 km/h, so 100 m costs an hour).
  They are there to keep islands routable *within* themselves, not to make the
  water crossable: a trip that needs one lands beyond any `--max-minutes`
  cutoff and stays `inf` or unassigned.
- Connectors shorter than `--gap-m` (default 50 m) are *gaps*, not
  crossings: a path a few metres short of the street it obviously meets. They
  get no speed override, so they run at the mode's normal base speed (14 km/h
  bike, 4.8 km/h walk) and are drawn on the `assign --gpkg` map like any other
  link. The `kind` column says `gap` or `island`. `--gap-m 0` treats every
  connector as a crossing.
- Only parts of at least `--min-nodes` nodes (default 20) are connected. The
  smaller ones are mostly mainland data gaps, and keeping them would pull
  zones off a real street onto a fragment behind a connector.
- Each island reaches for its *nearest* other part, and groups that touch then
  reach again, so a chain of islands is joined island to island, not each one
  straight to the mainland.
- It is an ordinary link layer (`highway = connector`, negative `link_id`s,
  `speed_override_kmh` set, `u_part_nodes`/`v_part_nodes` giving the size of
  the part at each end). Open it in QGIS, delete or redraw connectors as you
  like, and the next build uses your version.
- An island with no roads at all in the PBF has nothing to connect; zones on it
  are still dropped by `--max-snap-distance`.

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

`workers` are threads sharing one copy of the network and the demand, so there
is no `if __name__ == "__main__":` ceremony to observe. The first run on a
machine spends a few seconds compiling the routing kernels; the compiled code is
cached, so later runs start straight away.

## Settings files and scenarios

Any option can be written in a TOML file instead of on the command line, named
as the flag without its dashes (`max_minutes` or `max-minutes`). Top-level keys
go to every command that takes the option and are ignored by the rest, so one
file serves `matrix`, `assign` and `view` alike; a `[command]` section holds
what only one command should see. Files layer: `extends` names a base to read
first, and several `-c` stack in the order given. The command line always wins.

```toml
# scenarios/base.toml
mode = "bike"
links = "../output/{mode}_links.gpkg"
connectors = "../output/{mode}_connectors.gpkg"
centroids = "../data/centroids.csv"
id_column = "zone_id"
max_minutes = 60

[assign]
demand = "../data/demand_2023.omx"
demand_matrix = "bike"
gpkg = true
```

```toml
# scenarios/2040.toml
extends = "base.toml"
scenario = "2040"

[assign]
demand = "../data/demand_2040.omx"
```

```bash
valma assign -c scenarios/2040.toml                  # -> output/2040/
valma assign -c scenarios/2040.toml --mode walk      # {mode} follows along
valma assign -c scenarios/2040.toml --show-config    # what it would run with, and why
```

- **Paths are relative to the file they are written in**, not to where valma is
  run, so a scenario folder works from anywhere.
- **`{mode}` and `{scenario}`** in a value are filled in from the final
  settings, a mode given on the command line included.
- **`scenario`** puts results under `<output_dir>/<scenario>/`, so scenarios
  do not overwrite one another, and `valma view --run output/` finds and
  compares them all.
- **Mistakes are refused, not ignored.** A key no command takes, a section
  that is not a command or a value of the wrong type stops the run and names
  the file, with a suggestion for a likely typo. A later layer that picks
  `zones` replaces a base's `centroids` rather than clashing with it.

`extract`, `build`, `matrix`, `assign` and `overlay` record the settings they
ran with next to their results, as `valma_<command>_<mode>.toml`, with
absolute paths and the `scenario` already folded into `output_dir`. `-c` reads
that file back to repeat the run. Its shared keys suit other commands too:
`valma view -c output/2040/valma_assign_bike.toml` opens that run with the
demand and zones it was assigned with, ready for counter clicks.

Keep the NLS API key in its environment variable rather than in a file, so
settings files can be committed.

### The whole chain, only where something changed

```bash
valma pipeline -c scenarios/bike.toml --dry-run   # what would run, and why
valma pipeline -c scenarios/bike.toml             # extract, dem, traffic, connect, assign
```

`valma pipeline` runs the steps from OSM to volumes with the same settings
files, each exactly as its own command would, and skips any step whose
settings, input files and upstream results are as they were when it last
succeeded. A new OSM extract reruns everything; new traffic volumes rerun
`traffic`, `connect` and `assign`; new demand reruns `assign` alone. What each
step saw is kept in `<output_dir>/pipeline_<mode>.json`.

The link layer is written in place by `extract`, `dem` and `traffic`, so each
one's version of it is recorded rather than trusting its timestamp. A layer
edited by hand afterwards — an override typed in QGIS — reruns `connect` and
`assign`, never `extract`, which would overwrite the edit. `traffic` is left
out when no `volumes` are set; `--skip STEP` leaves out any other.

For results already made by running the commands by hand, `--mark-done`
records them as up to date without running anything, so the first pipeline
run does not redo them all. `--force` reruns everything.

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
2. **`--workers`** — origins are embarrassingly parallel. Workers are threads
   sharing one copy of the graph; each adds only its own scratch space (see
   [Memory](#memory-and-which-dial-moves-which-part-of-it)).

`--chunk-size` is how many origins a worker takes at a time. It is sized
automatically and is not a memory dial; there is no reason to set it.

Measured on the Finland bike network (3.4 M nodes, 9 273 zones with 8 access
points each, 14 workers), against the SciPy routing this project used before:

| | before | now | peak memory |
|---|---|---|---|
| 9 273 × 9 273 matrix, `--max-minutes 60` | 21.6 s | **8.3 s** | 8.4 → 2.8 GB |
| `--zones` matrix, `--near-minutes 15 --max-minutes 60` | 155 s | **15.8 s** | 9.3 → 5.4 GB |
| `assign`, 1.5 M zone pairs, `--max-minutes 60` | 65 s | **7.9 s** | 10.2 → 2.9 GB |
| `assign --zones`, 579 k zone pairs, `--near-minutes 15` | 304 s | **12.4 s** | 11.1 → 3.7 GB |

The answers are identical to the old ones. `scripts/bench_routing.py`
reproduces the comparison; see `docs/design/routing-engines.md` for the rest.

A 10 000 × 10 000 matrix is 100 million values; at float32 that is 400 MB in
memory, which the `.npz` output compresses.

### Assigning demand

`valma assign` runs the same one-Dijkstra-per-origin as the matrix, so the
levers above apply unchanged — `--max-minutes` first, then `--workers`. Two
things differ:

- **A search stops at its farthest destination.** An origin with no demand is
  not searched at all, and one whose demand is all local stops as soon as its
  last destination is settled, so `--min-demand` shortens searches as well as
  trimming the loading step.
- **Loading is one pass up the tree.** After the search, every destination's
  demand is pushed up the shortest-path tree in a single pass over the settled
  nodes, farthest first — the cost of the tree, not of the number of
  destinations times the length of their paths.

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
n_nodes x 32 bytes     search scratch: distances, predecessors, visit lists
n_nodes x  8 bytes     + the flow being pushed up the tree   (assign only)
n_edges x  8 bytes     + the volume accumulator              (assign only)
```

On the 3 M-node national bike network that is about 100 MB per worker for a
matrix and 210 MB for an assignment; the run logs the figure before it starts.
The graph and the demand are shared by every worker thread, so they cost once
for the machine and do not enter this. Multiply by `--workers` and budget
accordingly: 16 workers want ~3.5 GB for an assignment before the demand is
considered.

**Once, shared:** the expanded demand, which the run logs as `Expanded N zone OD
pair(s) ... M point pair(s)`. It grows with zones² × K², so it is the term that
runs away on a big zone system — roughly 0.8 GB at 1000 zones with K=8, 3 GB at
2000. Watch that log line.

So, in order:

| symptom | reach for |
|---|---|
| too slow | `--near-minutes`, then `--max-minutes` |
| out of memory, big zone system | `--min-demand` (it cuts K² point pairs per zone pair dropped), then `--points-per-zone` |
| out of memory, big *network* | `--workers` |

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

**Only the strip along the links is kept.** A tile fetched for a link layer is
stored as a strip 10–20 m either side of every link, in whole decimetres in 16
bits, with everything else left as no-data — about a twentieth of the tile on
disk. That is what makes the native 2 m model affordable: the whole Finnish
network at `--dem-resolution 2` is about 3 GB, where whole tiles would be over
200 GB. (The download is still the whole tile; it is trimmed before it is
stored.) Each tile records the strip it holds, so when links are added or
moved outside it, the next `valma dem` — which the pipeline reruns whenever
the link layer changes — fetches that tile again and keeps the old strip and
the new together. A link nudged a few metres stays inside and costs nothing.
Tiles cached whole, by `--bbox` or before strips existed, are used as they are.

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
| `ascent_override_m` | empty, for you to fill in where the DEM is wrong: metres climbed along the digitised direction |
| `descent_override_m` | the same for metres dropped |

Where an override is filled in, routing uses it instead of the DEM's value on
that link; each one works on its own. A link with **both** filled in is skipped
by `valma dem` entirely -- no tile is fetched for it and its `ascent_m` /
`descent_m` are left as they were -- so a rerun never undoes a hand fix. New
links, with both overrides empty, are measured from the DEM as usual. Older
layers' `grade_override` column is dropped on the next `valma dem`.

`valma dem` only measures links whose `ascent_m` or `descent_m` is empty, so
the link layer can be shared on its own, without `.cache/dem/`: whoever adds
links downloads only the tiles under those. A link moved or reshaped keeps its
old values -- empty its `ascent_m` in QGIS to have it measured again, or run
`valma dem --remeasure-all` to measure every link (still skipping those with
both overrides set), e.g. after changing `--dem-resolution`.

Ascent and descent are kept apart rather than netted off: a link that climbs ten
metres and drops them again costs nothing like a flat one, and the two swap
places when it is traversed backwards.

For routing, uphill grades below 2.5% add `4.587 * gradient * length_m`
seconds, grades from 2.5% to 5% add `17.20 * gradient * length_m` seconds, and
grades above 5% add `23.40 * gradient * length_m` seconds, where `gradient` is
the decimal grade (`ascent_m / length_m`). Downhill travel receives no
surcharge. In the reverse direction, the original `descent_m` is treated as
ascent because that direction is uphill on the same link. Grades of 20% or more are treated as unrealistic DEM
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

Both slow phases — downloading tiles and reading heights back out of them —
report progress with an estimate of the time left. On a terminal that is one
line rewritten in place; redirected to a file it becomes an ordinary log line
every couple of seconds.

**If GDAL complains about PROJ.** On Windows, anything that ships PROJ tends to
set the machine-wide `PROJ_DATA` / `PROJ_LIB` (PostgreSQL with PostGIS is the
usual culprit), and GDAL then reads *that* PROJ database instead of the one
rasterio brings with it. When it is the older of the two, every raster opened
warns twice and the tile's CRS comes back as an unnamed local system rather than
EPSG:3067. `valma dem` compares the two databases and prefers rasterio's when the
one in the environment is too old, logging one line when it does; a newer PROJ,
which may carry grid shifts we do not ship, is left alone. Whatever GDAL still
has to say is logged once rather than once per tile.

## Car traffic volumes

How much traffic shares a road changes what it is like to cycle on, and a
cyclist who would rather not ride beside four lanes of it goes another way. That
number does not come from OSM. It comes from a traffic model, as a set of lines
with a volume on each — and those lines are not our lines. They are a coarser
network, digitised separately, agreeing with OSM to within a few metres.
`valma traffic` is the map-matching step that carries the volume across.

```bash
valma traffic --links output/bike_links.gpkg --volumes data/link_volumes.gpkg
valma traffic --links output/bike_links.gpkg --volumes model.gpkg \
              --volume-column aadt --unmatched-out output/no_match.gpkg
```

Run it after `extract` and before anything routes: it is a link-layer stage, for
the same reason elevation is. The columns then survive a QGIS edit, are re-read
on every build, and can be corrected by hand where the match went wrong.

**Columns added:**

| column | meaning |
|---|---|
| `car_volume` | representative hourly vehicles sharing this link; fallback values are used where no source link matched |
| `car_volume_offset_m` | how far the model's line sat from ours — the number to style by when checking a match |
| `car_volume_override` | **empty, for you**: a number here is the last word |
| `adjacent_car_volume` | on a cycleway, footway or path running beside a road (within 20 m, `--adjacent-distance`, and within 30 degrees of parallel along at least half its length), that road's `car_volume`. A path beside a road is charged for its traffic exactly as the road is: Tarkkala et al. (2024) showed traffic volume with every adjacent cycle path in their survey and none with a separated one |

Three things make it more than a nearest-neighbour join, and each is a real
failure this data produces without it.

**Only a road that carries cars may be given a volume.** A cycleway beside a main
road is often *closer* to the model's centreline than the carriageway is, so
proximity alone loads the traffic onto exactly the link it does not belong on.
Candidates are drawn only from the car-carrying `highway` values; the cycle track
next to a road is never one, whatever the distance.

**A model link need not exist here at all.** Motorways are filtered out of the
bicycle network, so a motorway's volume has nothing legitimate to land on — and
a service road twenty metres away is a very tempting wrong answer. The defence is
*coverage*: a source link is matched as a whole or not at all. Its length is
sampled every 20 m, and unless half those samples find a road, every claim it
made is thrown away. On the Espoo test area this is what separates the two
populations cleanly: 75% of source links score exactly 1.0 and 19% exactly 0.0,
so the threshold has a wide valley to sit in. Without it, Länsiväylä's 37 000
vehicles a day land on the residential street beside it.

**A divided road is two links here and one there.** The two carriageways are
digitised in opposite directions, which is what lets them be recognised: a sample
claims its best road, and then the best road running the *other* way, if both are
one-way. When a source is claimed from both directions, the volume it carries is
the two carriageways' total, so each gets half — otherwise a cyclist on one side
of the central reservation is charged for the traffic on the other.

The check that the whole thing is right is conservation, and it is what the run
prints: **vehicle-kilometres carried across**, matched links against the sources
they came from. On the whole of Finland — 126 139 source links onto 4.3 M bike
links — it is 98%, with 522 313 links given a volume in 67 seconds. Without the
divided-road split it would read 117%: the same traffic counted twice down every
dual carriageway.

```
Volumes attached to 522,313 link(s), 89,799 km of 328,414 km that carry cars (27%);
99,909 of 126,139 source link(s) used, 49,561 link(s) on a divided road took half a
source's volume; vehicle-km carried across 98%, 25,684,696 veh-km on source links
with no road here
```

That last figure is the traffic with nowhere to go — nearly all of it motorways,
which is the correct answer rather than a shortfall. `--unmatched-out` writes
those source links out with the share of each that found a road, so the ones
worth a second look (a busy road at 0.4 coverage, not a motorway at 0.0) are one
sort away.

**The knobs**, if the source network is digitised differently from this one:
`--max-distance` (12 m; genuine matches here sit at a median of 0.9 m and wrong
ones start around 5), `--max-angle` (45°), `--min-coverage` (0.5),
`--sample-spacing` (20 m). `--no-split-divided` turns off the halving, which is
right only if your source layer already holds one directed link per carriageway.

For cycling, every finite `car_volume` changes the speed on links carrying motor
traffic by adding `TRAFFIC_COEFFICIENT * car_volume / 5.6` to the link's per-km road-class + facility disutility, so its whole speed divisor is `(road class + facility + traffic) / (4.96 + 2.61)`: Tarkkala et al. (2024) estimate the volume term once per route, so it is spread over their average 5.6 km route. A link that is not a carriageway -- a cycleway, a path, and the quiet `service` and `unclassified` roads -- counts as a separated path on a local link and starts from the reference 7.57, which is also where Tarkkala coded such paths. A road whose road class + facility scores below the reference (a main street with a track) rides at its nominal speed, not faster. Missing volumes
are not penalised.

## The walking model

Walking runs through exactly the same commands as cycling, with `--mode walk`,
and writes the same four outputs: `travel_times_walk.npz` and `.omx` from
`matrix`, `link_volumes_walk.npz` and `walk_volumes.gpkg` from `assign`. The
Valma demand files carry both modes, so the only change on `assign` is
`--demand-matrix walk`:

```bash
valma extract --pbf finland-highways.osm.pbf --mode walk
valma dem     --links output/walk_links.gpkg              # optional
valma matrix  --links output/walk_links.gpkg --mode walk \
              --centroids centroids.csv --id-column id --max-minutes 60
valma assign  --links output/walk_links.gpkg --mode walk \
              --centroids centroids.csv --id-column id \
              --demand demand_vrk.omx --demand-matrix walk \
              --min-demand 0.1 --max-minutes 45 --gpkg
```

Where it differs from cycling, and why:

- **Every way is two-way.** People ignore one-way signs; modelling them would
  invent detours nobody walks.
- **Shared foot and cycle paths are in.** In Finland these are tagged
  `highway=cycleway`, and they are often the separately mapped sidewalk of the
  road beside them, so the walk network keeps cycleways unless they say
  `foot=no` or `foot=use_sidepath`.
- **No car traffic term, but traffic lights count.** `car_volume` only
  affects cycling. A red light costs a walker the same wait as a cyclist; the
  walking speeds are pace in motion and leave it out. Crossings without lights
  cost only cyclists (`UNSIGNALLED_CROSSING_PENALTY_SECONDS`).
- **A road pays a light once per junction; a path pays it per crossing.** OSM
  maps a signalled junction as many nodes — a light at each stop line, a
  signalled crossing on each arm, one on each carriageway of a divided road —
  so `valma extract` groups signal nodes within 30 m of each other into
  junctions (`signals.py`); a link end within 30 m of a junction's signal
  nodes is inside it. Then, each in the direction of travel that does it:
  - a road pays on the link that *enters* the junction
    (`signal_entry` / `signal_entry_reverse`), however many signal nodes it
    passes inside;
  - a cycleway or footway pays on the link stepping onto a signalled crossing
    (`signal_cross_end` / `signal_cross_start`). A crossing over a divided road
    is one crossing, and stepping off is free. Going round the corner without
    crossing anything is free: a right turn on the cycleway costs nothing;
  - a path, or a service road, that gets onto the junction's roads inside it
    pays as the road does (`signal_entry`), so riding the cycleway into the
    junction and joining the road there does not pass the light for free. A
    path reaching the road at a crossing node is crossing it, not joining it.

  Every one of these is the same wait, `SIGNAL_WAIT_S` (16 s, the average for
  a random arrival with 40 % green over a 90 s cycle), so no way in is cheaper
  than another. A mid-block pedestrian light — signalled crossings with no road
  light and no side road — costs `MIDBLOCK_SIGNAL_WAIT_S` (8 s), once to the
  road passing it and once to someone crossing; a sidewalk running past it is
  free. The constants are read at routing time, so they can be tuned (and are,
  by the viewer's route calibration) without re-extracting.
- **Each unsignalled crossing is charged once per passage.** Every crossing
  node — a `crossing=marked`, `uncontrolled`, `zebra`, `unmarked` or bare
  `highway=crossing` node, but not `crossing=no` — is made a link end by
  `valma extract` and flagged on the links that touch it
  (`crossing_node_at_start`/`_end`). Passing one means riding a link in and a
  link out, so each pays half; only on a cycleway, footway or path — riding
  along the road past a zebra costs nothing. A crossing *way*
  (`footway=crossing` with a `crossing` tag) is split where it meets the road,
  so its pieces share one charge by length (`crossing_way_share`); where the
  same crossing is also mapped as a node, the node is charged and the way is
  not. The `crossing` column keeps the way's own tag as mapped. A layer
  extracted before these columns existed is still read, with a flat wait per
  signal node (`TRAFFIC_LIGHT_PENALTY_SECONDS`).
- **Hills cost the same as for cycling**, and only uphill: after `valma dem`
  the climb is charged in whichever direction climbs, even though every way is
  two-way.

`valma traffic` is not needed for walking. `valma dem` works out the mode from
a `<mode>_links.gpkg` file name (or give it `--mode`); that only affects the
layer's `speed_kmh` column shown in QGIS, not the routing.

Walking demand is dominated by short trips — in `demand_vrk.omx` a third of
all walk trips are intrazonal. With `--centroids` those are dropped (there is
no path from a node to itself); give `assign` zone polygons with `--zones` to
load them onto local streets — see [Zones as polygons](#zones-as-polygons).

## Viewing results in the browser

`valma view` opens a link layer and its assignment results on a WebGL map
(MapLibre + deck.gl). It needs the `viewer` extra:

```bash
pip install -e ".[viewer]"
valma view --links output/bike_links.gpkg --connectors output/bike_connectors.gpkg \
           --run output/link_volumes_bike.npz output/select_viikki
```

- **Volume map**: width and colour by volume, either direction or both, with a
  threshold slider; any two runs can be compared as A − B.
- **Speed map**: the speeds the current code gives each direction, plus the
  links read through roadworks.
- **Link inspector**: click a link for its tags and how its speed and travel
  time are built up — road class and facility, surface, segregation, traffic,
  penalties — in each direction, with its volume in every run.
- **Counters**: count lines and observed averages (`data/count_lines.gpkg`,
  `output/bicycle_counter_annual_averages.csv` by default) coloured by
  model ÷ observed for the selected run; click one to see the links it sums.
- **Routing**: *Plan a route*, then click an origin and a destination; drag
  either marker to re-route. The route is the shortest path by the same
  travel times `assign` and `matrix` use, so it is exactly what an OD pair
  between those points is assigned to, listed street by street with its
  length, model travel time and the road types it uses.
- **Calibration from a dragged route**: with a route drawn, click it to add a
  via point and drag that onto the route you believe people really take
  (double-click a via point to remove it). The panel then shows both routes,
  the extra cost of yours split by cause — base riding time, road class and
  facility, surface, segregation, car traffic, signals, crossings, grade — and,
  for every parameter that tells the two apart, the **break-even value** at
  which your route would cost no more than the model's, all else unchanged,
  smallest change first. *Copy report* gives the same as Markdown; *Save case*
  appends it to `output/route_observations.jsonl`. One case is an anecdote —
  a parameter worth changing is one that many saved cases point the same way
  on. If the link tags no longer reproduce the graph's travel times (the code
  changed since `assign` built it), the panel says so: rebuild with
  `assign --force-reload` first.
- **Trips through a counter**: start `view` with the settings the run was
  assigned with, and clicking a counter maps only the trips that pass it — a
  select-link analysis, with its largest origin and destination zones:

  ```bash
  valma view --links output/bike_links.gpkg --connectors output/bike_connectors.gpkg \
             --run output/link_volumes_bike.npz \
             --demand demand_vrk.omx --demand-matrix bike --centroids centroids.csv \
             --id-column id --min-demand 0.01 --max-minutes 90 --workers 8
  ```

  The answer is the one `assign --select-link` gives, but only origins that
  could reach the counter within `--max-minutes` are searched — about an
  eighth of them for a Helsinki counter, so a first click takes a minute or
  two for the whole country. Each counter is cached under `.cache/viewer/`
  (keyed on the settings and the network's travel times), so later clicks
  are instant.

`--run` takes `link_volumes_<mode>.npz` files or directories to search for
them. The first start for a link layer converts it to GeoParquet under
`.cache/viewer/` (about two minutes for the whole of Finland); later starts take
seconds, and map tiles are cached there as they are first drawn. The map
libraries and the OpenFreeMap base map load from the internet. Design notes and
planned features: [docs/viewer-design.md](docs/viewer-design.md).

## Layout

```
src/valma_bike_and_walk/
├── config.py      coordinate systems, mode names, cache paths
├── cache.py       cache keys: when a cached file is still current
├── speeds.py      walk/bike speed profiles
├── osm.py         the only module that talks to pyosmium   [stage 1]
├── links.py       the editable link layer: schema, repair, direction
├── network.py     RoutableNetwork: CSR + coords + KD-tree  [stage 2]
├── centroids.py   reading points and snapping them to nodes
├── zones.py       zone polygons -> weighted access points inside them
├── routing.py     Numba Dijkstra kernels and the thread pool that drives them
├── matrix.py      OD travel-time matrices, point or zone level
├── demand.py      reading OD demand (long / .npz / OMX) as a sparse matrix, writing OMX
├── assignment.py  all-or-nothing assignment: demand -> per-link volumes
├── elevation.py   DEM tiles from the NLS: fetch, cache, height profiles
├── traffic.py     another network's car volumes, matched onto our links
├── progress.py    a progress line with an ETA, for the slow loops
├── gpkg.py        draw a per-edge result back onto the link layer
├── config_file.py TOML settings files layered under the command line
└── cli.py         extract / dem / traffic / build / matrix / assign
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
| `test_routing.py` | routing, checked against NetworkX as an oracle |
| `test_routing_reference.py` | matrices and volumes identical to the SciPy implementation the kernels replaced (`tests/reference/`) |
| `test_zones.py` | where access points land, and how they are weighted |
| `test_zone_matrix.py` | the point-pair-to-zone-pair aggregation arithmetic |
| `test_traffic.py` | which link a car volume lands on, and which it must not |
| `test_cache.py` | that an edited layer, a new PBF or changed code is never served from cache |

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
- **A search costs what it explores.** The routing kernels never touch a
  per-node array in full per source — they reset only what a search visited —
  and nothing per source is ever `n_nodes` long. Keep it that way in anything
  new, and keep anything that scales with zones² sparse (see `demand.py`).

Parallelism is threads over GIL-free kernels sharing one copy of everything, so
a new parallel step is a new kernel plus a task list for
`routing._run`, not a new way of shipping data to processes.

`scripts/bench_routing.py` times the routing against the frozen SciPy
implementation in `tests/reference/` on a real network and checks that the
answers agree.

## Licence

EUROPEAN UNION PUBLIC LICENCE v. 1.2 (EUPL-1.2)
