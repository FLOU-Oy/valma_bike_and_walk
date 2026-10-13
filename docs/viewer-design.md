# `valma view` — browser viewer for networks and assignment results

## Summary

A `valma view` subcommand starts a local Python server and opens a browser map.
MapLibre GL draws the base map; deck.gl draws the data layers on the GPU.

The design point that makes a country-sized network (4.3 M links) usable is
that **geometry and values travel separately**. Link geometry is cut into
vector tiles once per link layer and cached. Results — volumes, speeds,
differences — are sent as one compact binary array indexed by link. Switching
run or measure only recolours on the GPU: no re-tiling, no geometry reload.

## Constraints

| Item | Size | Consequence |
|---|---|---|
| `bike_links.gpkg` | 4.3 M links, 1.5 GB | GeoJSON is out; vector tiles, simplified at low zoom |
| Volumes | per directed edge, ~8.5 M values | Float32 ≈ 34 MB raw; gzip or log-quantised uint16 |
| Coordinates | EPSG:3067 | reproject once to Web Mercator, cache |
| Platform | Windows, Python | no PostGIS or tippecanoe at run time |

## Architecture

```
valma view --links output/bike_links.gpkg --run output/ [--compare other/] [--port 8765]
   │
   ├─ prepare (once; cached under .cache/viewer/, keyed on file mtimes)
   │    gpkg → GeoParquet (EPSG:3857, link_id, tags)  + spatial index
   │
   ├─ FastAPI + uvicorn   (optional extra:  valma[viewer])
   │    GET  /tiles/links/{z}/{x}/{y}.mvt    geometry + link index only (disk cache)
   │    GET  /api/runs                       discovered runs (link_volumes_*.npz …)
   │    GET  /api/attr/{run}/{measure}.bin   Float32 array indexed by link
   │    GET  /api/link/{i}                   tags + speed breakdown, both directions
   │    GET  /api/counters                   observed vs modelled (GeoJSON, small)
   │    GET  /api/zones, /api/od/{zone}      zone totals, desire lines      (phase 3)
   │    POST /api/screenline                 volumes crossing a drawn line  (phase 2)
   │    POST /api/tree                       one-origin Dijkstra: isochrone (phase 2)
   │    WS   /api/jobs                       long jobs, e.g. select-link    (phase 3)
   │
   └─ static frontend shipped inside the package
        MapLibre GL (base map) + deck.gl MapboxOverlay
        MVTLayer → getColor/getWidth read attr[link_index]; updateTriggers swap measures
```

`valma view` opens the browser once the server answers; `--no-browser` and
`--host` allow running it on another machine.

Base map: OpenFreeMap Positron (free, no key), or the MML background map.

Directions: at low zoom the width shows both directions together; per-direction
values are shown in the link inspector (offset two-way lines are a later step).

## Features, by value to the analysis

1. **Volume map** — width and colour by volume (log scale), filters for road
   type, `under_construction` and minimum volume.
2. **Link inspector** — click a link: tags and how its speed is built up
   (base → road-class + facility divisor → surface → segregation → traffic →
   signal/crossing/grade penalties), in both directions, plus its volumes.
3. **Counter layer** — points coloured by model ÷ observed, with the links each
   count line sums and a linked scatterplot.
4. **Run comparison** — A − B and A ÷ B on a diverging scale.            (phase 2)
5. **Screenlines** — draw a line, get the volume crossing it link by link. (phase 2)
6. **Select-link / select-zone** — as a background job, or open existing
   `select_link_*` outputs.                                               (phase 3)
7. **Isochrones and routes** from a clicked point.                       (phase 2)
8. **Network QA layers** — connectors, roadworks, unsnapped zones,
   zero-volume counters, speed overrides.                                (phase 2)
9. **Export** — PNG of the view, CSV/GeoJSON of a selection, view state in
   the URL.                                                              (phase 2)

## Performance budget

- Tiles: at low zoom only links that matter (by volume in the active run, else
  by road class); simplified to about a pixel per zoom; full detail from z14.
  Target under ~300 KB per tile. Generated tiles are cached on disk.
- Attributes: fetched once per measure and kept in the browser; changing
  measure only updates accessors.
- Preparation: minutes the first time for the whole country, seconds after.

## Alternatives considered

- **Lonboard in Jupyter** — deck.gl + GeoArrow, draws millions of lines from a
  GeoDataFrame. A quick win for notebooks, not a shared tool.
- **PMTiles instead of a tile server** — build the geometry once into one file
  (tippecanoe, e.g. via WSL) and let the browser range-read it. Planned as
  `valma view --export-pmtiles`; on-the-fly tiles stay the default.
- **Kepler.gl** — good for one-off looks, but cannot keep geometry and values
  apart and does not fit the CLI.

## Phases

1. **MVP**: `valma view`, preparation cache, tiles, volume map, link
   inspector, counter layer.
2. Run comparison, screenlines, isochrones, QA layers, export.
3. Select-link jobs, OD desire lines, PMTiles export, Lonboard helpers.

## Risks

- Vector-tile encoding in Python is slow at detailed zooms: disk cache,
  low-zoom filtering, and the PMTiles route.
- Edge order: value arrays must follow the graph's edge order — the order
  `link_volume_frame` and `write_edges_gpkg` already share. `link_volumes_*.npz`
  now carries `link_id` and `direction`; older files are matched on `u`/`v`.
- Frontend: phase 1 is plain HTML + JavaScript loading MapLibre and deck.gl
  from a CDN, so no Node build is needed (an internet connection is). A bundled
  build can replace it later without changing the server.
