"""Browser viewer for link layers and assignment results (``valma view``).

The server lives in :mod:`valma_bike_and_walk.viewer.server` and needs the
``viewer`` extra (FastAPI, uvicorn, mapbox-vector-tile, pyarrow); nothing here
is imported by the routing core. See ``docs/viewer-design.md``.
"""
