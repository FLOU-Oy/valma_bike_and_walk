"""valma view: prepared links, run volumes, tiles and the API."""

import numpy as np
import pytest

pytest.importorskip("fastapi")
mapbox_vector_tile = pytest.importorskip("mapbox_vector_tile")

from fastapi.testclient import TestClient  # noqa: E402

from valma_bike_and_walk import links as links_module  # noqa: E402
from valma_bike_and_walk.viewer.data import (  # noqa: E402
    ViewerData,
    load_run,
    prepare_links,
)
from valma_bike_and_walk.viewer.server import create_app  # noqa: E402
from valma_bike_and_walk.viewer.tiles import lonlat_to_tile, render_tile  # noqa: E402

from .conftest import links_frame  # noqa: E402

ROWS = [
    {"link_id": 10, "u": 1, "v": 2, "coords": [(24.000, 60.0), (24.002, 60.0)],
     "highway": "cycleway"},
    {"link_id": 11, "u": 2, "v": 3, "coords": [(24.002, 60.0), (24.004, 60.0)],
     "highway": "secondary", "cycleway_right": "lane"},
]


@pytest.fixture
def links_path(tmp_path):
    links = links_module.annotate(links_module.normalise(links_frame(ROWS), "bike"), "bike")
    path = tmp_path / "bike_links.gpkg"
    links_module.write_links(links, path)
    return path


def test_preparing_twice_reads_the_cache(links_path, tmp_path):
    first = prepare_links(links_path, tmp_path / "cache")
    assert list((tmp_path / "cache").glob("*.parquet"))
    second = prepare_links(links_path, tmp_path / "cache")
    assert first.crs.to_epsg() == 3857
    assert list(second["link_id"]) == [10, 11]


def test_runs_join_on_link_id_and_on_end_nodes_alike(links_path, tmp_path):
    links = prepare_links(links_path, tmp_path / "cache")
    with_ids = tmp_path / "new" / "link_volumes_bike.npz"
    with_ids.parent.mkdir()
    np.savez(with_ids, u=np.array([1, 3, 2]), v=np.array([2, 2, 1]),
             link_id=np.array([10, 11, 10]), direction=np.array([1, -1, -1]),
             volume=np.array([5.0, 7.0, 2.0]))
    old = tmp_path / "old" / "link_volumes_bike.npz"
    old.parent.mkdir()
    np.savez(old, u=np.array([1, 3, 2]), v=np.array([2, 2, 1]), volume=np.array([5.0, 7.0, 2.0]))

    for path in (with_ids, old):
        run = load_run(path, links)
        assert run.forward.tolist() == [5.0, 0.0]
        assert run.reverse.tolist() == [2.0, 7.0]


def test_a_tile_carries_each_visible_links_row_index(links_path, tmp_path):
    data = ViewerData(prepare_links(links_path, tmp_path / "cache"), "bike")
    x, y = lonlat_to_tile(24.002, 60.0, 15)
    tile = render_tile(data.tree, data.geometry, data.min_zoom, 15, x, y)
    decoded = mapbox_vector_tile.decode(tile)
    assert sorted(f["properties"]["i"] for f in decoded["links"]["features"]) == [0, 1]
    # Far below the links' first zoom, the same place is empty.
    x, y = lonlat_to_tile(24.002, 60.0, 5)
    assert render_tile(data.tree, data.geometry, data.min_zoom, 5, x, y) == b""


def test_the_api_serves_meta_values_tiles_and_the_inspector(links_path, tmp_path):
    links = prepare_links(links_path, tmp_path / "cache")
    run_path = tmp_path / "run" / "link_volumes_bike.npz"
    run_path.parent.mkdir()
    np.savez(run_path, u=np.array([1]), v=np.array([2]), link_id=np.array([10]),
             direction=np.array([1]), volume=np.array([4.0]))
    run = load_run(run_path, links)
    client = TestClient(create_app(ViewerData(links, "bike", {run.name: run}), tmp_path / "tiles"))

    meta = client.get("/api/meta").json()
    assert meta["links"] == 2 and meta["runs"] == [run.name]

    volume = np.frombuffer(client.get(f"/api/attr/{run.name}/volume.bin").content, np.float32)
    assert volume.tolist() == [4.0, 0.0]
    speed = np.frombuffer(client.get("/api/attr/network/speed_forward.bin").content, np.float32)
    # An untagged cycleway rides at the cycleway's own speed.
    assert speed.shape == (2,) and speed[0] == pytest.approx(18.0)
    assert client.get("/api/attr/network/nonsense.bin").status_code == 404

    x, y = lonlat_to_tile(24.002, 60.0, 15)
    response = client.get(f"/tiles/links/15/{x}/{y}.mvt")
    assert response.status_code == 200 and response.content
    assert list((tmp_path / "tiles").rglob("*.mvt"))  # cached for next time

    detail = client.get("/api/link/1").json()
    directions = detail["speed"]["directions"]
    assert directions["forward"]["facility"] == "lane"
    assert directions["reverse"]["facility"] == "mixed traffic"
    assert directions["forward"]["speed_kmh"] > directions["reverse"]["speed_kmh"]
    assert detail["tags"]["link_id"] == 11
    assert client.get("/").status_code == 200


# --------------------------------------------------------------------------
# Counter flows (select-link on click)
# --------------------------------------------------------------------------

from scipy.sparse import csr_matrix  # noqa: E402

from valma_bike_and_walk.assignment import (  # noqa: E402
    assign_traffic,
    select_link_assignment,
)
from valma_bike_and_walk.viewer.select import CounterFlows, SelectSetup  # noqa: E402

from .conftest import make_network  # noqa: E402

# A chain 0-1-3-4 with a shortcut 0-2-1; the counter is the 1<->3 link pair.
EDGES = [(0, 1, 5.0), (1, 0, 5.0), (2, 1, 5.0), (1, 2, 5.0), (1, 3, 5.0), (3, 1, 5.0),
         (0, 2, 1.0), (2, 0, 1.0), (3, 4, 50.0), (4, 3, 50.0)]
COUNTER_ROWS = [4, 5]


def _setup(max_seconds):
    net = make_network(EDGES, 5)
    demand = np.arange(25, dtype=float).reshape(5, 5) + 1.0
    np.fill_diagonal(demand, 0.0)
    return SelectSetup(
        network=net, zone_ids=np.array([100, 101, 102, 103, 104]),
        zone_nodes=np.arange(5), demand=csr_matrix(demand), max_seconds=max_seconds,
        description=f"test {max_seconds}",
    )


@pytest.mark.parametrize("max_seconds", [None, 20.0])
def test_counter_flows_equal_a_full_select_link(tmp_path, max_seconds):
    setup = _setup(max_seconds)
    flows = CounterFlows(setup, np.arange(len(EDGES)), tmp_path)
    result = flows.compute(COUNTER_ROWS)

    # The reference: select-link over every origin, then those trips alone.
    _, picked = select_link_assignment(
        setup.network, setup.zone_nodes, setup.demand, COUNTER_ROWS, max_seconds=max_seconds
    )
    reference = assign_traffic(setup.network, setup.zone_nodes, picked, max_seconds=max_seconds)
    by_row = np.zeros(len(EDGES))
    np.add.at(by_row, setup.network.link_id, reference)

    assert result.summary["trips"] == pytest.approx(picked.sum())
    assert result.dense(len(EDGES)) == pytest.approx(by_row.astype(np.float32))
    if max_seconds is not None:
        # The cutoff keeps node 4 (50 s away) out, so fewer pairs qualify.
        assert result.summary["trips"] < setup.demand.sum()


def test_counter_flows_are_cached_by_name(tmp_path):
    flows = CounterFlows(_setup(None), np.arange(len(EDGES)), tmp_path)
    assert flows.status("c")["state"] == "none"
    flows._run("c", COUNTER_ROWS)
    assert flows.status("c")["state"] == "done"
    assert flows.cached("c").summary["trips"] > 0


def test_counter_flows_say_why_they_are_off(links_path, tmp_path):
    links = prepare_links(links_path, tmp_path / "cache")
    client = TestClient(create_app(ViewerData(links, "bike"), tmp_path / "tiles"))
    assert client.get("/api/meta").json()["select"] is False
    response = client.post("/api/counter-flow/0")
    assert response.status_code == 404 and "--demand" in response.text


# --------------------------------------------------------------------------
# Point-to-point routing
# --------------------------------------------------------------------------

from valma_bike_and_walk.network import network_from_links  # noqa: E402
from valma_bike_and_walk.viewer.route import Router  # noqa: E402


@pytest.fixture
def router_data(links_path, tmp_path):
    links = prepare_links(links_path, tmp_path / "cache")
    network = network_from_links(links_module.read_links(links_path), "bike")
    data = ViewerData(links, "bike")
    data.router = Router(network, links)
    return data


def test_a_route_follows_the_links_end_to_end(router_data):
    route = router_data.router.route((24.0, 60.0), (24.004, 60.0))
    p = route["properties"]
    assert p["links"] == 2
    assert p["length_m"] == pytest.approx(222.6, abs=2.0)
    assert [s["highway"] for s in p["steps"]] == ["cycleway", "secondary"]
    # The time is the graph's own: what assign routes on.
    assert p["time_s"] == pytest.approx(sum(s["time_s"] for s in p["steps"]))
    coords = route["geometry"]["coordinates"]
    assert coords[0] == pytest.approx([24.0, 60.0], abs=1e-6)
    assert coords[-1] == pytest.approx([24.004, 60.0], abs=1e-6)


def test_the_way_back_uses_the_reverse_direction(router_data):
    back = router_data.router.route((24.004, 60.0), (24.0, 60.0))
    coords = back["geometry"]["coordinates"]
    assert coords[0] == pytest.approx([24.004, 60.0], abs=1e-6)
    assert [s["highway"] for s in back["properties"]["steps"]] == ["secondary", "cycleway"]


def test_the_route_api_reports_bad_input_plainly(router_data, tmp_path):
    client = TestClient(create_app(router_data, tmp_path / "tiles"))
    assert client.get("/api/meta").json()["route"] is True
    ok = client.get("/api/route", params={"origin": "24.0,60.0", "destination": "24.004,60.0"})
    assert ok.status_code == 200 and ok.json()["properties"]["links"] == 2
    assert client.get("/api/route", params={"origin": "x", "destination": "1,2"}).status_code == 400
    far = client.get("/api/route", params={"origin": "24.0,60.0", "destination": "25.0,61.0"})
    assert far.status_code == 422 and "from the network" in far.json()["detail"]


# --------------------------------------------------------------------------
# Calibration from a dragged route
# --------------------------------------------------------------------------

from valma_bike_and_walk import speeds as speeds_module  # noqa: E402
from valma_bike_and_walk.viewer.calibrate import calibrate  # noqa: E402

# Two ways from node 1 to node 2: straight along a secondary road (mixed
# traffic, ~60 s), or a cycleway detour through node 3 (~63 s).
TWO_WAYS = [
    {"link_id": 0, "u": 1, "v": 2, "coords": [(24.000, 60.0), (24.004, 60.0)], "highway": "secondary"},
    {"link_id": 1, "u": 1, "v": 3, "coords": [(24.000, 60.0), (24.002, 60.001)], "highway": "cycleway"},
    {"link_id": 2, "u": 3, "v": 2, "coords": [(24.002, 60.001), (24.004, 60.0)], "highway": "cycleway"},
]


@pytest.fixture
def two_ways(tmp_path):
    raw = links_module.annotate(links_module.normalise(links_frame(TWO_WAYS), "bike"), "bike")
    path = tmp_path / "two_links.gpkg"
    links_module.write_links(raw, path)
    links = prepare_links(path, tmp_path / "cache")
    data = ViewerData(links, "bike")
    data.router = Router(network_from_links(links_module.read_links(path), "bike"), links)
    return data


def test_dragging_the_route_explains_the_gap_and_finds_break_evens(two_ways):
    before = speeds_module.MIXED_TRAFFIC_FACTOR
    result = calibrate(two_ways.router, two_ways.links, "bike",
                       [[24.0, 60.0], [24.004, 60.0]], [[24.002, 60.001]])
    m, a = result["model_parts"], result["alternative_parts"]
    assert set(m["by_highway_m"]) == {"secondary"}
    assert set(a["by_highway_m"]) == {"cycleway"}
    assert a["cost_s"] > m["cost_s"]
    # The causes add up to the cost of each route.
    for parts in (m, a):
        assert sum(parts["parts_s"].values()) == pytest.approx(parts["cost_s"], rel=1e-6)

    rows = {r["parameter"]: r for r in result["break_even"]}
    mixed = rows["MIXED_TRAFFIC_FACTOR"]
    assert mixed["break_even"] > mixed["current"]  # traffic must feel worse
    # Applying the break-even value does make the cycleway at least as cheap.
    speeds_module.MIXED_TRAFFIC_FACTOR = mixed["break_even"] * 1.001
    try:
        again = calibrate(two_ways.router, two_ways.links, "bike",
                          [[24.0, 60.0], [24.004, 60.0]], [[24.002, 60.001]])
        assert again["alternative_parts"]["cost_s"] <= again["model_parts"]["cost_s"]
    finally:
        speeds_module.MIXED_TRAFFIC_FACTOR = before
    # Nothing the search patched was left changed.
    assert speeds_module.MIXED_TRAFFIC_FACTOR == before
    assert "## Single-parameter break-even values" in result["report"]


def test_cases_are_saved_through_the_api(two_ways, tmp_path):
    client = TestClient(create_app(two_ways, tmp_path / "tiles", tmp_path / "out"))
    body = {"origin": [24.0, 60.0], "destination": [24.004, 60.0], "vias": [[24.002, 60.001]]}
    response = client.post("/api/calibrate", json=body)
    assert response.status_code == 200 and response.json()["break_even"]
    saved = client.post("/api/calibrate/save", json={**body, "note": "prefer the path"})
    assert saved.json()["observations"] == 1
    record = (tmp_path / "out" / "route_observations.jsonl").read_text(encoding="utf-8")
    assert "prefer the path" in record
    assert client.post("/api/calibrate", json={**body, "vias": []}).status_code == 400


def test_flags_survive_a_merge_with_connectors(tmp_path):
    """A connector layer lacks the signal columns; the cache must keep them flags."""
    raw = links_module.annotate(links_module.normalise(links_frame(ROWS), "bike"), "bike")
    raw["traffic_signal_at_start"] = [False, False]
    raw["traffic_signal_at_end"] = [False, True]
    links_path = tmp_path / "bike_links.gpkg"
    links_module.write_links(raw, links_path)
    connector = raw.iloc[[0]].copy()
    connector = connector[["link_id", "u", "v", "highway", "geometry"]]
    connector["link_id"] = -1
    connector_path = tmp_path / "bike_connectors.gpkg"
    links_module.write_links(connector, connector_path)

    links = prepare_links(links_path, tmp_path / "cache", connector_path)
    flags = links.set_index("link_id")["traffic_signal_at_end"]
    assert flags.loc[10] is False or flags.loc[10] == False  # noqa: E712
    assert bool(flags.loc[11]) is True
    assert links["traffic_signal_at_end"].dtype == bool


def test_calibration_agrees_with_the_graph_it_explains(two_ways):
    result = calibrate(two_ways.router, two_ways.links, "bike",
                       [[24.0, 60.0], [24.004, 60.0]], [[24.002, 60.001]])
    assert result["warning"] is None
    for key in ("model_parts", "alternative_parts"):
        parts = result[key]
        assert parts["cost_s"] == pytest.approx(parts["network_cost_s"], rel=1e-6)
