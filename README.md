# valma_bike_and_walk

Walking and cycling travel-time analysis built on a **local OpenStreetMap extract**.
No Overpass calls, no network access at run time — you point it at a `.osm.pbf`
and it does the rest.

Designed for the country-scale case: an origin–destination travel-time matrix
between ~10 000 centroids across all of Finland.

---

## Why it is built this way

Two decisions drive the whole design.

### 1. Speeds come from the way, not from `maxspeed`

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
data. Speeds are applied **before** topological simplification, so each
collapsed chain sums real per-segment travel times.

### 2. Routing uses SciPy CSR, not NetworkX

A NetworkX `MultiDiGraph` stores a Python dict per node and per edge. That is
fine for one city and hopeless for a country: Finland's walking network is
millions of nodes, which costs tens of GB as dicts, and its Dijkstra is pure
Python.

The same information lives here in a handful of numpy arrays plus a SciPy CSR
matrix, routed with compiled `scipy.sparse.csgraph.dijkstra`. The cached network
is a compressed `.npz` rather than GraphML — for Espoo that is a few MB against
128 MB.

---

## Install

```bash
uv sync                 # core
uv sync --extra dev     # + pytest/ruff/black/mypy
```

## Use

```bash
# Build and cache a network
valma build --pbf finland-260805.osm.pbf --mode bike --bbox 24.5 60.1 25.1 60.4

# Route with the network you just built -- no PBF parsing at all
valma matrix --network .cache/finland-260805.osm_bbox_24.5000_60.1000_25.1000_60.4000_bike.npz \
             --mode bike --centroids centroids.csv --id-column id

# ...or build and route in one go
valma matrix --pbf finland-260805.osm.pbf --mode walk \
             --centroids centroids.csv --id-column id \
             --workers 8 --max-minutes 120
```

`valma build` prints the path it saved to, along with the ready-made `matrix`
command to reuse it. Add `--gpkg` to also write the network's links —
geometry plus length, travel time and speed — to
`<output-dir>/<mode>_links.gpkg`, for inspecting the network and routing costs
in GIS software.

`centroids.csv` needs `lon`/`lat` columns by default (`--x-column`/`--y-column`
and `--centroid-crs` if not). Any vector format GeoPandas reads works too.
Output is an `.npz` holding `ids` and a float32 `seconds` matrix.

```python
from valma_bike_and_walk import Settings, build_network
from valma_bike_and_walk.centroids import load_centroids
from valma_bike_and_walk.matrix import travel_time_matrix

settings = Settings(pbf_path="finland-260805.osm.pbf")
network = build_network(settings, mode="bike", bbox=[24.5, 60.1, 25.1, 60.4])

centroids = load_centroids(network, "centroids.csv", id_column="id").drop_unsnapped()
seconds = travel_time_matrix(network, centroids.node_index, workers=8)
```

---

## Selecting an extent

| | cost on a country PBF | notes |
|---|---|---|
| `--bbox` | one parse | cheap, recommended |
| `--area Espoo` | one parse **+ ~15 min / >10 GB** relation scan | exact municipal polygon; cached after the first run |
| neither | whole country | see memory note below |

The `--area` lookup finds the boundary inside the PBF, so it stays offline, but
pyrosm must walk every relation to do it. `--search-bbox` narrows that scan —
but it must *fully contain* the area. Finnish municipalities include their sea
territory, so Espoo reaches down to 59.90 °N, far south of its built-up area; too
tight a box silently truncates the boundary.

---

## Scaling to the whole of Finland

### Parse on every core

pyrosm's own default reader (`engine="in_memory"`) is **single-core**, which
leaves most of the machine idle on a large extract. This project defaults to the
streaming reader instead. Same extent, byte-identical output:

| engine | time |
|---|---|
| `in_memory` (pyrosm's default) | 159.9 s |
| `out_of_core`, `workers="auto"` | **64.7 s** |

Override with `--engine` / `--osm-workers` if you need to.

> **Scripts must guard their entry point.** The streaming reader parses in a
> process pool, and on Windows the children re-import the main module. A script
> that calls `build_network()` at import time therefore re-runs itself in every
> child. Put your work behind `if __name__ == "__main__":`. The library detects
> this case and falls back to single-core with a warning rather than failing,
> but you lose the speed-up. The CLI is unaffected.

### Building the country network: use `--tiled`

Reading all of Finland in a single pass does **not** fit in a typical desktop's
memory, and the streaming reader does not change that. Measured on this
project's 737 MB Finland extract with ~15 GB free, both readers exhausted memory
on the walking network before finishing:

| reader | died after |
|---|---|
| `in_memory` | 2.7 min (13.2 GB and climbing) |
| `out_of_core` | 4.9 min |

So a country build reads one tile at a time:

```bash
valma build --pbf finland-260805.osm.pbf --mode walk --tiled --tile-degrees 2
```

Peak memory is then set by the busiest single tile rather than by the whole
country. Tiles overlap slightly and are stitched on OSM node id, so a road
crossing a tile edge stays connected; ways duplicated in the overlap collapse
automatically, because the CSR builder keeps only the fastest edge per node
pair. Verified against a single-pass build of the same area: **identical travel
times on 100 % of sampled pairs**.

Individual tiles are cached, so an interrupted build resumes.

The trade-off is time: pyrosm rescans the entire PBF for every tile whatever
bounding box you give it, so **use the largest tiles your memory allows**.

#### Why a tiled build looks single-threaded, and what to do

Watch a tiled build and you see every core light up for about a second, then one
core grinding for minutes. That is real, and it is worth understanding before
reaching for a knob.

The streaming reader splits the file's data blobs into one **contiguous** range
per worker, statically, with no work stealing. A PBF is roughly ordered by
geography, so once you clip to a tile only the range covering that tile holds
anything: the other workers scan their range, find nothing, and exit. Profiling
one 3° tile of the Finland extract:

| phase | time | share |
|---|---|---|
| `get_network` | 167.0 s | 71.7 % |
| `simplify_graph` | 59.4 s | 25.5 % |
| `get_directed_edges` | 5.9 s | 2.5 % |
| speeds | 0.6 s | 0.2 % |

So ~97 % of a tile sits in two phases that run on one core. The fix is not more
reader workers — it is **more tiles at once**:

```bash
valma build --pbf finland-260805.osm.pbf --mode walk \
            --tiled --tile-degrees 3 --tile-workers 6
```

`--tile-workers > 1` automatically drops each tile's own reader to serial, so
pools do not nest and fight. Budget roughly `tile-workers ×` the busiest tile in
memory.

Expect useful but sub-linear gains. Measured on a four-tile build, verified to
produce a byte-identical network either way:

| | wall time |
|---|---|
| `--tile-workers 1` | 424.0 s |
| `--tile-workers 4` | 281.5 s (**1.51×**) |

Two things cap it: a build finishes no sooner than its single densest tile, and
every worker is still scanning the same whole PBF. More, smaller tiles balance
the first better; only the next section fixes the second.

#### The bigger win: shrink the file first

Every read scans the whole PBF, so the fastest thing you can do is give it less
to scan. A country extract is mostly buildings, landuse and coastline that a
routing network never looks at. Strip it once with
[osmium](https://osmcode.org/osmium-tool/):

```bash
osmium tags-filter finland-260805.osm.pbf w/highway -o finland-highways.osm.pbf
```

737 MB becomes ~190 MB, and reading the same extent from it gives a **byte-for-byte
identical network in a fifth of the time**:

| file | result | read |
|---|---|---|
| `finland-260805.osm.pbf` (737 MB) | 25,523 nodes / 27,658 ways | 190.7 s |
| `finland-highways.osm.pbf` (187 MB) | identical | **35.0 s (5.4×)** |

(`-r`/`--add-referenced` turns out not to be needed here: the nodes the highway
ways reference come through anyway, which is why the two results match exactly.)

**Filtering does not reduce peak memory, only read time.** What it strips —
buildings, landuse, coastline — a routing network never loaded in the first
place, so the data that has to be held is identical. A single-pass whole-Finland
cycling build peaked at **~19.8 GB before running out of room** on the 190 MB
file, the same wall the 737 MB file hits. Tiling stays necessary at country
scale whatever you feed it.

Cutting by area works too, and composes with the above:

```bash
osmium extract -b 22,60,25,63 finland-highways.osm.pbf -o region.osm.pbf
```

#### Watch the disk, not just the CPU

If your disk sits at 100 % while the CPU idles, it is the `out_of_core` reader:
it decodes the entire PBF into temporary shard files on **every** read — about
1.3 GB of spill per read of a 190 MB extract — then reads them back and deletes
them. One read, that is a fair trade for flat memory. A tiled build with
`--tile-workers 16` is sixteen concurrent spills, and on a 143-tile run
(`--tile-degrees 1` over Finland) it adds up to hundreds of gigabytes of pointless
write traffic.

`--engine auto` (the default) avoids this by using `in_memory` for parallel tiled
builds. If you override it, override it deliberately.

Interrupted runs leave their shards behind, since the cleanup never runs. Clear
them when nothing is building:

```bash
rm -rf "$TEMP"/pyrosm_ooc_*
```

### Computing the matrix

Cost is dominated by one number: **one Dijkstra run per origin**. 10 000 origins
means 10 000 runs, so plan with the measured per-source time your run logs.

Levers, in order of effect:

1. **`--max-minutes`** — a cutoff stops the search expanding across the entire
   country and is by far the biggest win if your analysis has one anyway.
2. **`--workers`** — origins are embarrassingly parallel. The network is shared
   with the workers by memory-mapping, not pickling, so N workers do not cost N
   copies of the graph.
3. **`--chunk-size`** — SciPy returns a dense `(chunk, n_nodes)` float64 block,
   so this is a memory dial, defaulting to a 256 MB block. Lower it if you are
   tight on RAM.

A 10 000 × 10 000 matrix is 100 million values; at float32 that is 400 MB in
memory, which the `.npz` output compresses.

### Measured numbers

Helsinki capital region walking network (`--bbox 24.40 60.05 25.30 60.45`),
352 115 nodes / 865 232 edges, on a 16-core desktop:

| | per source | 10 000 × 10 000 |
|---|---|---|
| no cutoff | 67 ms | 11 min (1 worker) · ~1 min (14 workers) |
| 60 min cutoff | 3.3 ms | 0.6 min (1 worker) |

Scaling from a 9 500-node network to a 352 000-node one, cost grew as roughly
`nodes^1.2`, in line with Dijkstra's `O(E log V)`. Extrapolating to a plausible
3–6 M-node national walking network gives on the order of **1 s per source**, so
10 000 origins is a few hours on one core or roughly 15 minutes across workers.
Treat that as an estimate: it has not been measured end to end.

### Two things to know before running a national matrix

**Use a cutoff.** Walking between distant centroids is not a meaningful number —
Helsinki to Utsjoki on foot is some 270 hours. In the capital-region benchmark a
60-minute cutoff left only 3.8 % of pairs reachable, and it is *20× faster*. If
your analysis has a horizon, `--max-minutes` gives it to you almost free.

**Pre-cut the PBF if you can.** pyrosm rescans the whole file for every tile, so
a tiled national build is dominated by that: the capital region alone took ~25
minutes, of which ~24 was parsing. Cutting the country into regional `.osm.pbf`
files first with [osmium](https://osmcode.org/osmium-tool/) —
`osmium extract -b <bbox> finland.osm.pbf -o region.osm.pbf` — turns each pass
into a small local read and takes the build from hours to minutes.

---

## Layout

```
src/valma_bike_and_walk/
├── config.py      coordinate systems, mode names, cache paths
├── speeds.py      walk/bike speed profiles
├── extract.py     the only module that talks to pyrosm
├── network.py     RoutableNetwork: CSR + coords + KD-tree
├── centroids.py   reading points and snapping them
├── matrix.py      chunked / parallel OD matrices
├── gpkg.py        export network links to GeoPackage for GIS
└── cli.py
```

All pyrosm calls are confined to `extract.py`, so a pyrosm upgrade can only
break there.

## Development

```bash
uv run pytest
uv run ruff check . && uv run black . && uv run mypy src/
```

## Licence

EUROPEAN UNION PUBLIC LICENCE v. 1.2 (EUPL-1.2)
