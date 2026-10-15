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
# Municipalities
# --------------------------------------------------------------------------


@pytest.fixture
def municipality_data(links_path, tmp_path):
    """Link 10 in "Alpha" (three zones' worth), link 11 in "Beta"."""
    import geopandas as gpd
    import shapely

    from valma_bike_and_walk.viewer.municipalities import load_municipalities

    zones = gpd.GeoDataFrame(
        {"municipality": ["Alpha", "Alpha", "Beta"]},
        geometry=[
            shapely.box(23.99, 59.99, 24.0015, 60.0),
            shapely.box(23.99, 60.0, 24.0015, 60.01),
            shapely.box(24.0015, 59.99, 24.01, 60.01),
        ],
        crs="EPSG:4326",
    )
    zones_path = tmp_path / "zones.gpkg"
    zones.to_file(zones_path)

    links = prepare_links(links_path, tmp_path / "cache")
    run_path = tmp_path / "run" / "link_volumes_bike.npz"
    run_path.parent.mkdir()
    np.savez(
        run_path, u=np.array([1, 2, 2]), v=np.array([2, 1, 3]),
        link_id=np.array([10, 10, 11]), direction=np.array([1, -1, 1]),
        volume=np.array([10.0, 30.0, 5.0]),
        # The way back over link 10 is slower: a climb, or a signal.
        travel_time_s=np.array([20.0, 60.0, 15.0]),
    )
    run = load_run(run_path, links)
    data = ViewerData(links, "bike", {run.name: run})
    data.municipalities = load_municipalities(
        links, zones_path, "municipality", tmp_path / "cache"
    )
    return data, run


def test_links_fall_in_the_municipality_of_their_midpoint(municipality_data):
    data, _ = municipality_data
    assert data.municipalities.names == ["Alpha", "Beta"]
    assert data.municipalities.of_link.tolist() == [0, 1]


def test_municipality_speed_is_bike_km_over_bike_hours(municipality_data):
    data, run = municipality_data
    stats = data.municipalities.stats(data.links, run)
    length = data.links["length_m"].to_numpy()
    alpha, beta = stats["rows"]

    km = (10.0 + 30.0) * length[0] / 1000
    hours = (10.0 * 20.0 + 30.0 * 60.0) / 3600
    assert alpha["bike_km"] == pytest.approx(km)
    assert alpha["bike_hours"] == pytest.approx(hours)
    # Weighted by trips each way, with each direction's own time: not the
    # mean of the two directions' speeds.
    assert alpha["speed_kmh"] == pytest.approx(km / hours)
    assert beta["speed_kmh"] == pytest.approx(length[1] / 15.0 * 3.6)
    assert stats["total"]["links"] == 2


def test_the_municipality_api_serves_the_table_and_outlines(municipality_data, tmp_path):
    data, run = municipality_data
    client = TestClient(create_app(data, tmp_path / "tiles"))
    assert client.get("/api/meta").json()["municipalities"] is True

    table = client.get("/api/municipalities", params={"run": run.name}).json()
    assert [r["name"] for r in table["rows"]] == ["Alpha", "Beta"]
    outline = client.get("/api/municipality/0").json()
    assert outline["properties"]["name"] == "Alpha"
    west, south, east, north = outline["properties"]["bounds"]
    assert west == pytest.approx(23.99) and north == pytest.approx(60.01)
    assert client.get("/api/municipality/5").status_code == 404
    assert client.get("/api/municipalities", params={"run": "nope"}).status_code == 404


def test_modal_shares_count_the_trips_leaving_each_municipality(municipality_data, tmp_path):
    omx = pytest.importorskip("openmatrix")
    from valma_bike_and_walk.viewer.municipalities import load_trip_shares

    data, run = municipality_data
    # Zones 1 and 2 are Alpha, zone 3 Beta; zone 9 is in no municipality.
    zones = tmp_path / "zones.gpkg"
    import geopandas as gpd

    frame = gpd.read_file(zones)
    frame["input_zone_id"] = [1, 2, 3]
    frame.to_file(zones)
    path = tmp_path / "demand.omx"
    bike = np.array([[1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [5, 0, 0, 0]], dtype="float32")
    car = np.array([[3, 3, 2, 0], [4, 4, 0, 0], [0, 0, 1, 0], [9, 0, 0, 0]], dtype="float32")
    with omx.open_file(str(path), "w") as f:
        f.create_mapping("zone_number", [1, 2, 3, 9])
        f.create_matrix("bike", obj=bike)
        f.create_matrix("car", obj=car)

    for _ in range(2):  # computed, then read back from the cache
        load_trip_shares(data.municipalities, path, "bike", "zone_number", zones,
                         "input_zone_id", "municipality", tmp_path / "cache")
        stats = data.municipalities.stats(data.links, run)
        alpha, beta = stats["rows"]
        assert alpha["mode_trips"] == 4 and alpha["all_trips"] == 4 + 16
        assert beta["mode_share"] == pytest.approx(1 / 2)
        assert stats["total"]["mode_share"] == pytest.approx(5 / 22)
        # Two municipalities are too few for a fit.
        assert stats["fit"] is None


def test_the_share_fit_is_least_squares_with_both_r2s():
    from valma_bike_and_walk.viewer.municipalities import share_fit

    rows = [
        {"speed_kmh": s, "mode_share": m, "all_trips": t}
        for s, m, t in [(10, 0.05, 100), (12, 0.07, 100), (14, 0.08, 1000), (16, 0.12, 10)]
    ]
    fit = share_fit(rows)
    x = np.array([10, 12, 14, 16.0])
    y = np.array([0.05, 0.07, 0.08, 0.12])
    slope, intercept = np.polyfit(x, y, 1)
    assert fit["slope"] == pytest.approx(slope)
    assert fit["intercept"] == pytest.approx(intercept)
    assert fit["r2"] == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)
    # Weighting by trips lets the 1000-trip municipality dominate, so the two
    # measures differ.
    assert fit["r2_weighted"] != pytest.approx(fit["r2"])
    assert fit["n"] == 4
    rows.append({"speed_kmh": None, "mode_share": 0.5, "all_trips": 1})
    assert share_fit(rows)["n"] == 4


def test_trip_lengths_are_binned_by_origin_with_intrazonal_estimates(tmp_path):
    import geopandas as gpd
    import shapely

    from valma_bike_and_walk.viewer.trip_lengths import (
        DISC_MEAN_DISTANCE,
        TripLengthGroups,
        load_regions,
        load_zone_table,
    )

    # Zone 1 is a 2 km square in Alpha; zones 2 and 3 are in Beta and Gamma.
    zones = gpd.GeoDataFrame(
        {"input_zone_id": [1, 2, 3], "municipality": ["Alpha", "Beta", "Gamma"]},
        geometry=[shapely.box(0, 0, 2000, 2000), shapely.box(2000, 0, 3000, 1000),
                  shapely.box(3000, 0, 4000, 1000)],
        crs="EPSG:3067",
    )
    zones.to_file(tmp_path / "zones.gpkg")
    (tmp_path / "regions.csv").write_text(
        "municipality;region\nAlpha;Seutu\nBeta;Seutu\n", encoding="utf-8"
    )
    table = load_zone_table(tmp_path / "zones.gpkg", "input_zone_id", "municipality")
    groups = TripLengthGroups(table, ["Alpha", "Beta", "Gamma"],
                              load_regions(tmp_path / "regions.csv"))

    path = tmp_path / "trip_lengths_bike.npz"
    np.savez(
        path,
        origin=np.array([1, 1, 1, 2, 3, 3]),
        destination=np.array([1, 2, 3, 1, 1, 2]),
        demand=np.array([10.0, 4.0, 1.0, 5.0, 2.0, 3.0]),
        length_m=np.array([0.0, 1500.0, 25000.0, 4000.0, np.nan, 800.0], dtype=np.float32),
    )
    result = groups.histograms("run", path)
    assert result["labels"][0] == "0–1" and len(result["labels"]) == 8

    alpha = result["municipalities"]["Alpha"]
    # Intrazonal: 0.905 * sqrt(4 km² / pi) = 1.02 km, so the 1-2 km bin.
    estimate = DISC_MEAN_DISTANCE * np.sqrt(2000.0**2 / np.pi)
    assert 1000 < estimate < 2000
    assert alpha["trips"] == [0, 14, 0, 0, 0, 0, 0, 1]
    assert alpha["intrazonal"] == 10
    assert alpha["mean_km"] == pytest.approx((10 * estimate + 4 * 1500 + 25000) / 15 / 1000)

    gamma = result["municipalities"]["Gamma"]
    assert gamma["trips"][0] == 3 and gamma["unreached"] == 2 and gamma["total"] == 3

    region = result["regions"]["Seutu"]
    assert region["municipalities"] == ["Alpha", "Beta"]
    assert region["total"] == 20 and region["trips"][3] == 5  # Beta's 4 km trip
    assert result["country"]["total"] == 23
    assert sum(result["country"]["shares"]) == pytest.approx(1.0)


def test_the_trip_length_api_needs_the_runs_file(municipality_data, tmp_path):
    from valma_bike_and_walk.viewer.trip_lengths import TripLengthGroups, load_zone_table

    data, run = municipality_data
    zones = tmp_path / "zones.gpkg"
    import geopandas as gpd

    frame = gpd.read_file(zones)
    frame["input_zone_id"] = [1, 2, 3]
    frame.to_file(zones)
    data.trip_lengths = TripLengthGroups(
        load_zone_table(zones, "input_zone_id", "municipality"), data.municipalities.names
    )
    client = TestClient(create_app(data, tmp_path / "tiles"))
    assert client.get("/api/meta").json()["trip_lengths"] is True
    assert client.get("/api/trip-lengths", params={"run": run.name}).status_code == 409

    np.savez(run.path.with_name("trip_lengths_bike.npz"), origin=np.array([1]),
             destination=np.array([3]), demand=np.array([2.0]),
             length_m=np.array([2500.0], dtype=np.float32))
    body = client.get("/api/trip-lengths", params={"run": run.name}).json()
    assert body["municipalities"]["Alpha"]["trips"][2] == 2.0
    assert body["regions"] == {}


def test_a_run_without_travel_times_says_so(links_path, tmp_path):
    links = prepare_links(links_path, tmp_path / "cache")
    run_path = tmp_path / "old" / "link_volumes_bike.npz"
    run_path.parent.mkdir()
    np.savez(run_path, u=np.array([1]), v=np.array([2]), link_id=np.array([10]),
             direction=np.array([1]), volume=np.array([4.0]))
    run = load_run(run_path, links)
    from valma_bike_and_walk.viewer.municipalities import Municipalities

    table = Municipalities(["A"], np.array([0, 0], dtype=np.int32), [{}], [[0, 0, 0, 0]])
    with pytest.raises(ValueError, match="without travel times"):
        table.stats(links, run)


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
    _, picked, _ = select_link_assignment(
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
