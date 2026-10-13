"""The ``valma view`` web server: tiles, value arrays and a link inspector."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import urllib.request
import webbrowser
from importlib import resources
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from valma_bike_and_walk.viewer.data import ViewerData, counter_job, counters_geojson
from valma_bike_and_walk.viewer.tiles import TileCache, render_tile

logger = logging.getLogger(__name__)

MVT = "application/vnd.mapbox-vector-tile"


def create_app(
    data: ViewerData, tile_dir: Path, observations_dir: Path | None = None
) -> FastAPI:
    """
    The FastAPI app over prepared viewer data; tiles cached under ``tile_dir``,
    saved route-calibration cases appended under ``observations_dir``.
    """
    observations_dir = Path(observations_dir or Path(tile_dir).parent)
    app = FastAPI(title="valma view")
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # Tiles depend on the geometry and on which links each zoom keeps.
    key = hashlib.sha1(data.min_zoom.tobytes()).hexdigest()[:12]
    cache = TileCache(Path(tile_dir) / key)
    static = resources.files("valma_bike_and_walk.viewer") / "static"

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(static / "index.html"))

    @app.get("/api/meta")
    def meta() -> dict:
        return {
            "mode": data.mode,
            "links": len(data.links),
            "bounds": data.bounds_wgs84,
            "runs": list(data.runs),
            "counters": bool(data.counters),
            "select": data.flows is not None and bool(data.counters),
            "route": data.router is not None,
            "measures": {
                "run": ["volume", "volume_forward", "volume_reverse"],
                "network": ["speed", "speed_forward", "speed_reverse", "under_construction"],
            },
        }

    @app.get("/tiles/links/{z}/{x}/{y}.mvt")
    def tile(z: int, x: int, y: int) -> Response:
        if not (0 <= z <= 22 and 0 <= x < 2**z and 0 <= y < 2**z):
            raise HTTPException(404)
        body = cache.get(z, x, y)
        if body is None:
            body = render_tile(data.tree, data.geometry, data.min_zoom, z, x, y)
            cache.put(z, x, y, body)
        return Response(body, media_type=MVT, headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/attr/{run:path}/{measure}.bin")
    def attr(run: str, measure: str) -> Response:
        try:
            values = data.measure(run, measure)
        except KeyError:
            raise HTTPException(404, f"No measure {measure!r} for {run!r}") from None
        return Response(values.tobytes(), media_type="application/octet-stream")

    @app.get("/api/link/{index}")
    def link(index: int) -> JSONResponse:
        if not 0 <= index < len(data.links):
            raise HTTPException(404)
        return JSONResponse(data.explain(index))

    def _counter(index: int) -> str:
        if data.flows is None or not data.counters:
            raise HTTPException(404, "Start valma view with --demand and --centroids for counter flows.")
        if not 0 <= index < len(data.counters["records"]):
            raise HTTPException(404)
        return counter_job(data, index)

    @app.get("/api/counter-flow/{index}")
    def counter_flow_status(index: int) -> dict:
        name = _counter(index)
        return data.flows.status(name)

    @app.post("/api/counter-flow/{index}")
    def counter_flow_start(index: int) -> dict:
        name = _counter(index)
        return data.flows.request(name, data.counters["records"][index]["links"])

    @app.get("/api/route")
    def route(origin: str, destination: str) -> JSONResponse:
        """Shortest path between two "lon,lat" points, as a GeoJSON feature."""
        if data.router is None:
            raise HTTPException(404, "No routable network loaded.")
        try:
            a = tuple(float(v) for v in origin.split(","))
            b = tuple(float(v) for v in destination.split(","))
            if len(a) != 2 or len(b) != 2:
                raise ValueError
        except ValueError:
            raise HTTPException(400, 'origin and destination are "lon,lat".') from None
        try:
            return JSONResponse(data.router.route(a, b))
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    def _points(body: dict) -> tuple[list, list]:
        try:
            origin = [float(v) for v in body["origin"]]
            destination = [float(v) for v in body["destination"]]
            vias = [[float(v) for v in p] for p in body.get("vias", [])]
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "Send origin, destination and vias as [lon, lat].") from None
        return [origin, destination], vias

    def _calibrate(body: dict) -> dict:
        if data.router is None:
            raise HTTPException(404, "No routable network loaded.")
        from valma_bike_and_walk.viewer.calibrate import calibrate

        points, vias = _points(body)
        if not vias:
            raise HTTPException(400, "Add at least one via point to the route.")
        try:
            return calibrate(data.router, data.links, data.mode, points, vias)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.post("/api/calibrate")
    def calibrate_route(body: dict = Body(...)) -> JSONResponse:
        """Model route vs the route dragged through via points."""
        return JSONResponse(_calibrate(body))

    @app.post("/api/calibrate/save")
    def save_case(body: dict = Body(...)) -> dict:
        from valma_bike_and_walk.viewer.calibrate import save_observation

        result = _calibrate(body)
        path = save_observation(observations_dir, result, str(body.get("note", "")))
        with path.open(encoding="utf-8") as f:
            count = sum(1 for _ in f)
        return {"saved": str(path), "observations": count}

    @app.get("/api/counters")
    def counters(run: str | None = None) -> JSONResponse:
        return JSONResponse(counters_geojson(data, run))

    app.mount("/static", StaticFiles(directory=str(static)), name="static")
    return app


def serve(
    data: ViewerData,
    tile_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    observations_dir: Path | None = None,
) -> None:
    """Run the server until interrupted, opening the browser once it answers."""
    import uvicorn

    app = create_app(data, tile_dir, observations_dir)
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}/"

    def open_when_ready() -> None:
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "api/meta", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        webbrowser.open(url)

    if open_browser:
        threading.Thread(target=open_when_ready, daemon=True).start()
    print(f"valma view: {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
