"""Cached files are reused only while what they were built from is unchanged.

The failure these guard against is silent: edit the link layer in QGIS (or run
`valma dem` / `valma traffic`, which write into it in place), route again, and
get the graph from before the edit -- with nothing in the output to say so.
"""

import logging
import os

import numpy as np
import pytest

from valma_bike_and_walk import network as network_module
from valma_bike_and_walk.cache import (
    file_stamp,
    make_key,
    read_sidecar,
    sidecar_path,
    stale_reason,
)
from valma_bike_and_walk.cli import main
from valma_bike_and_walk.config import Settings
from valma_bike_and_walk.links import read_links, write_links
from valma_bike_and_walk.network import (
    RoutableNetwork,
    build_links,
    build_network,
    network_for_links,
    stored_network_key,
)

from .conftest import write_pbf

# Three ~111 m footway links in a chain, as in test_pipeline.
NODES = {
    1: (24.000, 60.000),
    2: (24.002, 60.000),
    3: (24.004, 60.000),
    4: (24.006, 60.000),
}
FOOTWAY = {"highway": "footway", "surface": "asphalt"}
WAYS = [
    (100, [1, 2], FOOTWAY),
    (101, [2, 3], FOOTWAY),
    (102, [3, 4], FOOTWAY),
]


@pytest.fixture
def chain_pbf(tmp_path):
    return write_pbf(tmp_path / "chain.osm.pbf", NODES, WAYS)


@pytest.fixture
def builds(monkeypatch):
    """Count how many times a graph is actually built rather than loaded."""
    calls = []
    real = network_module.network_from_links

    def counting(links, mode):
        calls.append(mode)
        return real(links, mode)

    monkeypatch.setattr(network_module, "network_from_links", counting)
    return calls


def settings_for(pbf, tmp_path):
    return Settings(
        pbf_path=pbf, cache_dir=tmp_path / "cache", output_dir=tmp_path / "out"
    )


def later(path):
    """
    Push a file's mtime a second on, as a save a moment later would.

    Filesystems with coarse timestamps can give two writes inside one test the
    same mtime; a real edit in QGIS is never that quick.
    """
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def edit_link(path, link_id, speed_kmh):
    """Set one link's speed override, the way an edit in QGIS would."""
    links = read_links(path)
    links.loc[links["link_id"] == link_id, "speed_override_kmh"] = speed_kmh
    write_links(links, path)
    later(path)


@pytest.fixture
def links_path(chain_pbf, tmp_path):
    path, _ = build_links(
        settings_for(chain_pbf, tmp_path),
        "walk",
        links_path=tmp_path / "out" / "walk_links.gpkg",
    )
    return path


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------


def test_a_stamp_follows_the_file_contents(tmp_path):
    path = tmp_path / "layer.gpkg"
    path.write_bytes(b"one")
    before = file_stamp(path)

    path.write_bytes(b"three")
    assert file_stamp(path) != before

    same_size = file_stamp(path)
    later(path)
    assert file_stamp(path) != same_size


def test_a_stamp_sees_edits_still_sitting_in_the_sqlite_wal(tmp_path):
    # QGIS edits a GeoPackage in WAL mode: committed changes can live in the
    # -wal file while the main file's size and mtime stay put.
    path = tmp_path / "layer.gpkg"
    path.write_bytes(b"pages")
    before = file_stamp(path)

    (tmp_path / "layer.gpkg-wal").write_bytes(b"an edit")
    assert file_stamp(path) != before


def test_stale_reason_names_what_changed():
    key = make_key("network", mode="walk", links={"size": 1}, code="abc")

    assert stale_reason(key, key) is None
    assert stale_reason(None, key) == "it records no inputs"
    moved = make_key("network", mode="walk", links={"size": 2}, code="abc")
    assert stale_reason(key, moved) == "links has changed"
    both = make_key("network", mode="walk", links={"size": 2}, code="def")
    assert stale_reason(key, both) == "code, links have changed"


# --------------------------------------------------------------------------
# A graph built from a link layer
# --------------------------------------------------------------------------


def test_an_unchanged_layer_reuses_its_graph(links_path, tmp_path, builds):
    graph = tmp_path / "cache" / "walk.npz"

    network_for_links(links_path, "walk", graph)
    network_for_links(links_path, "walk", graph)

    assert builds == ["walk"]
    assert stored_network_key(graph)["links"] == file_stamp(links_path)


def test_an_edited_layer_rebuilds_its_graph(links_path, tmp_path, builds):
    graph = tmp_path / "cache" / "walk.npz"
    before = network_for_links(links_path, "walk", graph)

    edit_link(links_path, 0, 1.0)
    after = network_for_links(links_path, "walk", graph)

    assert builds == ["walk", "walk"]
    # 111 m at 1 km/h is ~400 s; before the edit it was well under 300.
    assert before.travel_time[before.link_id == 0].max() < 300
    assert after.travel_time[after.link_id == 0].min() > 300
    # And the rebuilt file on disk is the new graph, not just the return value.
    reloaded = RoutableNetwork.load(graph)
    assert reloaded.travel_time[reloaded.link_id == 0].min() > 300


def test_a_change_to_the_build_code_rebuilds_the_graph(
    links_path, tmp_path, builds, monkeypatch
):
    graph = tmp_path / "cache" / "walk.npz"
    network_for_links(links_path, "walk", graph)

    # Stand-in for editing a speed profile or a penalty in the source.
    monkeypatch.setattr(network_module, "code_digest", lambda *modules: "new code")
    network_for_links(links_path, "walk", graph)

    assert builds == ["walk", "walk"]


def test_a_graph_saved_without_a_key_is_rebuilt(links_path, tmp_path, builds):
    graph = tmp_path / "cache" / "walk.npz"
    legacy = network_for_links(links_path, "walk", graph)
    legacy.source_key = ""  # as written before keys existed
    legacy.save(graph)

    network_for_links(links_path, "walk", graph)
    assert builds == ["walk", "walk"]
    assert stored_network_key(graph) is not None


def test_an_unreadable_graph_is_rebuilt_not_fatal(links_path, tmp_path, builds):
    graph = tmp_path / "cache" / "walk.npz"
    graph.parent.mkdir(parents=True, exist_ok=True)
    graph.write_bytes(b"half a zip file")

    network = network_for_links(links_path, "walk", graph)
    assert builds == ["walk"]
    assert network.n_nodes == 4


def test_force_reload_rebuilds_a_current_graph(links_path, tmp_path, builds):
    graph = tmp_path / "cache" / "walk.npz"
    network_for_links(links_path, "walk", graph)
    network_for_links(links_path, "walk", graph, force_reload=True)
    assert builds == ["walk", "walk"]


def test_saving_leaves_no_temporary_files_behind(links_path, tmp_path):
    graph = tmp_path / "cache" / "walk.npz"
    network_for_links(links_path, "walk", graph)
    assert sorted(p.name for p in graph.parent.iterdir()) == ["walk.npz"]


# --------------------------------------------------------------------------
# A link layer extracted from a PBF
# --------------------------------------------------------------------------


def test_a_changed_pbf_is_re_extracted_and_rebuilt(chain_pbf, tmp_path, builds):
    settings = settings_for(chain_pbf, tmp_path)
    assert build_network(settings, "walk").n_nodes == 4
    assert build_network(settings, "walk").n_nodes == 4
    assert builds == ["walk"]

    cached_links = settings.links_cache_path("walk", "full")
    assert read_sidecar(cached_links) is not None

    # A newer extract of the same area, with the chain one link longer.
    os.remove(chain_pbf)
    write_pbf(
        chain_pbf,
        {**NODES, 5: (24.008, 60.000)},
        WAYS + [(103, [4, 5], FOOTWAY)],
    )
    later(chain_pbf)

    assert build_network(settings, "walk").n_nodes == 5
    assert builds == ["walk", "walk"]


def test_an_explicit_link_layer_is_never_overwritten_by_extract(
    chain_pbf, links_path, tmp_path
):
    # Hand edits live here; a newer PBF must not silently replace them.
    edit_link(links_path, 0, 1.0)
    os.remove(chain_pbf)
    write_pbf(
        chain_pbf, {**NODES, 5: (24.008, 60.000)}, WAYS + [(103, [4, 5], FOOTWAY)]
    )

    _, links = build_links(
        settings_for(chain_pbf, tmp_path), "walk", links_path=links_path
    )
    assert len(links) == 3
    assert (links["speed_override_kmh"] == 1.0).sum() == 1
    assert not sidecar_path(links_path).exists()


# --------------------------------------------------------------------------
# Through the CLI: the workflow the README promises
# --------------------------------------------------------------------------


def _matrix(links_path, centroids, tmp_path):
    assert (
        main(
            [
                "matrix",
                "--links",
                str(links_path),
                "--mode",
                "walk",
                "--centroids",
                str(centroids),
                "--id-column",
                "id",
                "--output-dir",
                str(tmp_path / "out"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--workers",
                "1",
            ]
        )
        == 0
    )
    with np.load(tmp_path / "out" / "travel_times_walk.npz") as data:
        return float(data["seconds"][0, 1])


def test_cli_matrix_routes_on_the_layer_as_edited(links_path, tmp_path, builds):
    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\na,24.000,60.000\nb,24.006,60.000\n")

    before = _matrix(links_path, centroids, tmp_path)
    again = _matrix(links_path, centroids, tmp_path)
    assert again == before
    assert builds == ["walk"]

    # Slow one link to a crawl in "QGIS", with no --force-reload.
    edit_link(links_path, 0, 1.0)
    after = _matrix(links_path, centroids, tmp_path)

    assert builds == ["walk", "walk"]
    assert after > before + 300


def test_cli_warns_when_an_explicit_network_predates_the_layer(
    links_path, tmp_path, caplog
):
    built = tmp_path / "out" / "walk.npz"
    assert (
        main(
            ["build", "--links", str(links_path), "--mode", "walk", "--out", str(built)]
        )
        == 0
    )
    edit_link(links_path, 0, 1.0)

    with caplog.at_level(logging.WARNING, logger="valma_bike_and_walk"):
        main(
            [
                "build",
                "--network",
                str(built),
                "--links",
                str(links_path),
                "--mode",
                "walk",
            ]
        )
    assert "was not built from" in caplog.text


def test_cli_build_refreshes_its_named_graph_after_an_edit(links_path, tmp_path):
    built = tmp_path / "out" / "walk.npz"
    args = ["build", "--links", str(links_path), "--mode", "walk", "--out", str(built)]

    main(args)
    before = RoutableNetwork.load(built)
    edit_link(links_path, 0, 1.0)
    main(args)
    after = RoutableNetwork.load(built)

    assert after.travel_time[after.link_id == 0].min() > (
        before.travel_time[before.link_id == 0].max() + 200
    )
