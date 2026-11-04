# Running this package inside EMME's Python

Notes from putting `valma_bike_and_walk` into the same interpreter as
OpenPaths EMME, so the VALMA model system can call it in process instead of
shelling out. Written 2026-11-04.

## Why in process at all

The integration with `valma-model-system` needs skims (time and distance),
multi-centroid zones and a link assignment. Two ways to get them:

- **Subprocess** — VALMA shells out to the `valma` CLI in its own venv and the
  two sides exchange OMX files. No dependency coupling whatsoever.
- **In process** — VALMA imports this package. No file round trip, no
  interpreter discovery, and `overlay()` becomes a numpy call on matrices
  already in memory.

In process is only possible if EMME's Python can host this package's
dependencies. EMME 23 ships Python 3.7, which cannot (this package needs
3.10+, and numba dropped 3.7 in 2021). **EMME 25.00.01 ships Python 3.11.9 and
can.**

## What EMME 25.00.01 already provides

Checked against `EMME 25.00.01\Python311\Lib\site-packages`:

| package | EMME has | this package needs |
|---|---|---|
| numpy | 1.26.2 | >=1.24 |
| pandas | 2.0.2 | >=2.0 |
| shapely | 2.0.1 | >=2.0 |
| pyproj | 3.5.0 | >=3.5 |
| openmatrix | 0.3.5.0 | >=0.3 |
| tables | 3.8.0 | (openmatrix) |
| fiona | 1.9.4.post1 | (geopandas backend) |
| packaging | 23.1 | — |

Missing, and needing installation: `geopandas`, `numba`, `llvmlite`, `scipy`,
`osmium`, `openpyxl`, `xlrd`, `truststore`, `et-xmlfile`, plus `requests`,
`urllib3` and `charset-normalizer` (which `osmium` pulls in). `rasterio` is
only needed for the `dem` extra.

**numpy must not move.** EMME's compiled modules are built against 1.26.2.
A resolution pinned to it succeeds and — contrary to first expectations —
still picks up numba 0.68.1, the same version this project develops against.
Every package in the resolved set has a prebuilt win_amd64 wheel, so no
compiler is needed.

## Install recipe

A venv built from EMME's interpreter with `--system-site-packages`, so `inro`
and EMME's own packages stay visible and nothing under `Program Files` is
written. Removing the venv is a complete undo.

```powershell
$EMME = "C:\Program Files\Bentley\OpenPaths\EMME 25.00.01\Python311\python.exe"
$PY   = "C:\valma-bw\venv\Scripts\python.exe"

& $EMME -m venv --system-site-packages C:\valma-bw\venv
```

Constraints — everything EMME ships, pinned so pip cannot move it:

```
numpy==1.26.2
pandas==2.0.2
shapely==2.0.1
pyproj==3.5.0
openmatrix==0.3.5.0
tables==3.8.0
fiona==1.9.4.post1
packaging==23.1
```

Then install, and check the dry run first — nothing should want to move numpy,
and nothing should need building:

```powershell
& $PY -m pip install --dry-run -c emme25-constraints.txt C:\valma-bw\valma_bike_and_walk
& $PY -m pip install           -c emme25-constraints.txt C:\valma-bw\valma_bike_and_walk
```

Do **not** use `pip install --user`: EMME puts
`AppData\Roaming\Python\Python311\site-packages` first on `sys.path`, and it is
shared with any other Python 3.11 on the machine.

## Known traps

- **`pip install pytest` moves `packaging`.** pytest 9 wants a newer
  `packaging` than EMME's 23.1 and will shadow it in the venv. Harmless in
  itself (pure Python), but pin `packaging==23.1` and use `pytest<9` if you
  want the venv to mirror production exactly.
- **pip installs duplicate `pyproj` and `shapely` into the venv** even though
  EMME provides them at the pinned versions. Two GDAL/PROJ stacks in one
  process is worth avoiding; install `geopandas` with `--no-deps` so it uses
  EMME's fiona rather than pulling `pyogrio`, a third GDAL.
- **numpy 1 vs 2 differences are invisible on a developer machine.** See
  below; this cost a day.

## numpy 1 is the deployment target, and CI does not test it

This package has no CI, and developers run numpy 2. EMME runs numpy 1.26.2,
and the two differ in ways that are silent until they are not.

One such bug was found and fixed in `network._hilbert_keys`: the expression
`s * s * ((3 * rx) ^ ry)`, with `rx`/`ry` boolean arrays, takes the platform's
default integer width for `3 * rx`. That is **int32 on Windows under numpy 1**
and int64 under numpy 2 (NEP 50). At `_HILBERT_BITS = 16` the first level
computes `32768 * 32768 * 3 = 3,221,225,472`, past int32, so it wrapped and
scrambled the node ordering — producing `Node index out of range` errors,
halved route distances, and a Hilbert curve whose consecutive points were 63
cells apart instead of 1. The fix casts to int64 explicitly.

Note that **no test running on numpy 2 can catch this class of bug.** The
oracle test added alongside the fix (`test_hilbert_keys_match_an_arbitrary_precision_reference`)
compares against pure-Python integers and is worth having, but it passes with
the broken line on numpy 2. The only real guard is running the suite against
numpy 1.26 — worth a CI job pinning it.

Expected suite result on either numpy, as of this commit: **553 passed, 2
failed**. The two failures are in `test_elevation.py`
(`test_steep_profile_changes_next_to_structures_are_removed` and
`test_a_steep_but_believable_hill_is_untouched`), pre-existing on the
`routing-signal-fixes` branch and unrelated to deployment.

## Resolved: fiona cannot find proj.db

`python -m tests.emme_only.test_assignment` from `valma-model-system` failed at
the very end, writing its result GeoPackage:

```
fiona.errors.CRSError: The EPSG code is unknown.
PROJ: proj_create_from_database: Cannot find proj.db
```

**Cause: `valma-model-system` itself, not the venv's packages.**
`Scripts/datahandling/resultdata.py` sets, before importing fiona:

```python
os.environ["PROJ_DATA"] = (sys.exec_prefix
                           + "\\Lib\\site-packages\\fiona\\proj_data")
```

That assumes fiona lives under `sys.exec_prefix`. In EMME's own interpreter it
does. In a `--system-site-packages` venv, `sys.exec_prefix` is the venv
(`C:\valma-bw\venv`) while fiona is still EMME's copy, so `PROJ_DATA` points
at a directory that does not exist. `PROJ_DATA` outranks both fiona's bundled
data and `PROJ_LIB`, so PROJ finds nothing.

Verified 2026-11-04: the same line, then a fiona GPKG write with EPSG:3067,
succeeds under `$EMME` and fails under `$PY` with exactly the error above. A
plain fiona write *without* that line succeeds in both, with or without the
machine-wide `PROJ_LIB` (see below) set. The earlier control experiment
(`from_epsg(3067)` alone) passes in both, so it could not tell them apart.

**Fix (in `valma-model-system`):** find fiona's real location without importing
it:

```python
import importlib.util
_fiona = importlib.util.find_spec("fiona")
os.environ["PROJ_DATA"] = os.path.join(
    os.path.dirname(_fiona.origin), "proj_data")
```

It still overrides on purpose, which matters for the next point.

**Unrelated machine setting worth knowing about:** PostgreSQL 15/PostGIS sets
the machine-wide variables `PROJ_LIB` (to
`C:\Program Files\PostgreSQL\15\share\contrib\postgis-3.3\proj`) and
`GDAL_DATA` (to PostgreSQL's `gdal-data`). Every Python on the machine
inherits them. They did not cause this failure, but a different PROJ/GDAL
version's data can break things in subtle ways. That is a good reason to keep
setting `PROJ_DATA` explicitly rather than drop the line.
