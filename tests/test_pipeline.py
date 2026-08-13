"""End to end: PBF -> GeoPackage -> (edit) -> routable graph -> results.

These are the tests that would catch the two stages drifting apart.
"""

import geopandas as gpd
import numpy as np
import pytest

from valma_bike_and_walk.cli import main
from valma_bike_and_walk.config import Settings
from valma_bike_and_walk.demand import read_demand_omx
from valma_bike_and_walk.links import LINKS_LAYER, read_links, write_links
from valma_bike_and_walk.matrix import travel_time_matrix
from valma_bike_and_walk.network import (
    RoutableNetwork,
    build_links,
    build_network,
    network_from_links,
)

from .conftest import write_pbf

# A straight 4-node chain, ~111 m between neighbours. Three separate ways, so
# the shared nodes 2 and 3 are junctions and the chain is three links -- one per
# row you could select in QGIS.
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


def settings_for(pbf, tmp_path):
    return Settings(
        pbf_path=pbf, cache_dir=tmp_path / "cache", output_dir=tmp_path / "out"
    )


def test_the_whole_pipeline_runs_from_pbf_to_network(chain_pbf, tmp_path):
    network = build_network(settings_for(chain_pbf, tmp_path), "walk")

    assert isinstance(network, RoutableNetwork)
    assert network.n_nodes == 4
    # Walking is bidirectional: 3 links -> 6 edges.
    assert network.n_edges == 6


def test_stage_one_writes_a_geopackage_qgis_can_open(chain_pbf, tmp_path):
    path, links = build_links(settings_for(chain_pbf, tmp_path), "walk")

    assert path.exists() and path.suffix == ".gpkg"
    layers = gpd.list_layers(path)
    assert LINKS_LAYER in set(layers["name"])

    assert len(links) == 3
    assert links["length_m"].min() > 100


def test_stage_one_is_cached_and_stage_two_reads_what_is_on_disk(chain_pbf, tmp_path):
    settings = settings_for(chain_pbf, tmp_path)
    path, _ = build_links(settings, "walk")

    # Edit the layer the way QGIS would, then build.
    edited = read_links(path)
    edited.loc[edited["link_id"] == 0, "speed_override_kmh"] = 1.0
    write_links(edited, path)

    network = network_from_links(read_links(path), "walk")
    slowed = network.travel_time[network.link_id == 0]
    assert slowed.size > 0
    # 111 m at 1 km/h is about 400 s; the untouched links are far quicker.
    assert slowed.min() > 300
    assert network.travel_time[network.link_id != 0].max() < 300


def test_deleting_a_link_in_the_geopackage_cuts_the_route(chain_pbf, tmp_path):
    settings = settings_for(chain_pbf, tmp_path)
    path, links = build_links(settings, "walk")

    whole = network_from_links(links, "walk")
    assert whole.n_nodes == 4

    # Remove the middle link (2 -> 3) and the chain falls in two; the build
    # keeps only the largest component.
    cut = links[~((links["u"] == 2) & (links["v"] == 3))]
    severed = network_from_links(cut, "walk")
    assert severed.n_nodes == 2


def test_adding_a_link_in_the_geopackage_creates_a_shortcut(chain_pbf, tmp_path):
    import shapely

    settings = settings_for(chain_pbf, tmp_path)
    _, links = build_links(settings, "walk")

    def seconds(net):
        i = int(np.searchsorted(net.node_ids, 1))
        j = int(np.searchsorted(net.node_ids, 4))
        return travel_time_matrix(net, [i], [j])[0, 0]

    before = seconds(network_from_links(links, "walk"))

    # A new row drawn end to end between nodes 1 and 4, with no ids filled in --
    # exactly what comes out of QGIS. It must snap onto both and be quicker.
    drawn = gpd.GeoDataFrame(
        [
            {
                "link_id": None,
                "u": None,
                "v": None,
                "highway": "footway",
                "surface": "asphalt",
                "speed_override_kmh": 30.0,
            }
        ],
        geometry=[shapely.LineString([(24.000, 60.000), (24.006, 60.000)])],
        crs=links.crs,
    )
    with_shortcut = gpd.GeoDataFrame(
        gpd.pd.concat([links, drawn], ignore_index=True), crs=links.crs
    )

    after = seconds(network_from_links(with_shortcut, "walk"))
    assert after < before


def test_link_ids_survive_the_whole_way_to_the_result(chain_pbf, tmp_path):
    _, links = build_links(settings_for(chain_pbf, tmp_path), "walk")
    network = network_from_links(links, "walk")

    assert set(network.link_id) <= set(links["link_id"])
    assert network.link_id.shape[0] == network.n_edges


def test_a_network_round_trips_through_its_npz(chain_pbf, tmp_path):
    network = build_network(settings_for(chain_pbf, tmp_path), "walk")
    path = network.save(tmp_path / "net.npz")
    loaded = RoutableNetwork.load(path)

    assert loaded.mode == network.mode
    for name in (
        "node_ids",
        "indptr",
        "indices",
        "travel_time",
        "length",
        "link_id",
        "direction",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(network, name))


# --------------------------------------------------------------------------
# The CLI, which is what most of this is actually driven through
# --------------------------------------------------------------------------


def test_cli_extract_then_build_then_matrix(chain_pbf, tmp_path, capsys):
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    links_path = out_dir / "walk_links.gpkg"

    assert (
        main(
            [
                "extract",
                "--pbf",
                chain_pbf,
                "--mode",
                "walk",
                "--output-dir",
                str(out_dir),
                "--cache-dir",
                str(cache),
            ]
        )
        == 0
    )
    assert links_path.exists()

    assert (
        main(
            [
                "build",
                "--links",
                str(links_path),
                "--mode",
                "walk",
                "--output-dir",
                str(out_dir),
                "--cache-dir",
                str(cache),
            ]
        )
        == 0
    )
    # The graph is a deliverable, not a cache: it lands beside the link layer
    # it was built from, named after it.
    assert (out_dir / "walk.npz").exists()
    assert not list(cache.glob("*.npz"))

    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\na,24.000,60.000\nb,24.006,60.000\n")

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
                str(out_dir),
                "--cache-dir",
                str(cache),
                "--workers",
                "1",
            ]
        )
        == 0
    )
    result = np.load(out_dir / "travel_times_walk.npz")
    assert result["seconds"].shape == (2, 2)
    assert result["seconds"][0, 1] > 0


def test_cli_matrix_omx_round_trips_through_read_demand_omx(chain_pbf, tmp_path):
    pytest.importorskip("openmatrix")
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    links_path = out_dir / "walk_links.gpkg"
    assert (
        main(
            [
                "extract",
                "--pbf",
                chain_pbf,
                "--mode",
                "walk",
                "--output-dir",
                str(out_dir),
                "--cache-dir",
                str(cache),
            ]
        )
        == 0
    )

    # Unlike the CSV/npz outputs, OMX zone mappings need integer ids.
    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\n1,24.000,60.000\n2,24.006,60.000\n")

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
                str(out_dir),
                "--cache-dir",
                str(cache),
                "--workers",
                "1",
                "--omx",
            ]
        )
        == 0
    )

    omx_path = out_dir / "travel_times_walk.omx"
    assert omx_path.exists()

    npz_result = np.load(out_dir / "travel_times_walk.npz")
    ids, matrix = read_demand_omx(omx_path, matrix_name="walk")
    np.testing.assert_array_equal(ids, npz_result["ids"])
    np.testing.assert_allclose(
        matrix.toarray(), npz_result["seconds"], rtol=1e-5, atol=1e-2
    )


def test_cli_matrix_omx_accepts_an_explicit_path(chain_pbf, tmp_path):
    pytest.importorskip("openmatrix")
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    links_path = out_dir / "walk_links.gpkg"
    assert (
        main(
            [
                "extract",
                "--pbf",
                chain_pbf,
                "--mode",
                "walk",
                "--output-dir",
                str(out_dir),
                "--cache-dir",
                str(cache),
            ]
        )
        == 0
    )

    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\n1,24.000,60.000\n2,24.006,60.000\n")

    # A path in a directory that doesn't exist yet -- it should be created,
    # and the default <output-dir>/travel_times_<mode>.omx name skipped.
    omx_path = tmp_path / "elsewhere" / "walk_times.omx"

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
                str(out_dir),
                "--cache-dir",
                str(cache),
                "--workers",
                "1",
                "--omx",
                str(omx_path),
            ]
        )
        == 0
    )

    assert omx_path.exists()
    assert not (out_dir / "travel_times_walk.omx").exists()
    ids, _ = read_demand_omx(omx_path, matrix_name="walk")
    np.testing.assert_array_equal(ids, np.array([1, 2]))


def test_cli_assign_draws_volumes_back_onto_the_link_layer(chain_pbf, tmp_path):
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    links_path = out_dir / "walk_links.gpkg"
    main(
        [
            "extract",
            "--pbf",
            chain_pbf,
            "--mode",
            "walk",
            "--output-dir",
            str(out_dir),
            "--cache-dir",
            str(cache),
        ]
    )

    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\na,24.000,60.000\nb,24.006,60.000\n")
    demand = tmp_path / "od.csv"
    demand.write_text("origin_id,destination_id,demand\na,b,7\n")

    assert (
        main(
            [
                "assign",
                "--links",
                str(links_path),
                "--mode",
                "walk",
                "--centroids",
                str(centroids),
                "--id-column",
                "id",
                "--demand",
                str(demand),
                "--gpkg",
                "--output-dir",
                str(out_dir),
                "--cache-dir",
                str(cache),
                "--workers",
                "1",
            ]
        )
        == 0
    )

    volumes = gpd.read_file(out_dir / "walk_volumes.gpkg", layer=LINKS_LAYER)
    assert "volume" in volumes.columns
    # All seven trips run the length of the chain, so every link carries them.
    loaded = volumes[volumes["volume"] > 0]
    assert set(loaded["link_id"]) == {0, 1, 2}
    assert (loaded["volume"] == 7).all()


def test_cli_assign_gpkg_without_a_link_layer_explains_itself(chain_pbf, tmp_path):
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    network = build_network(settings_for(chain_pbf, tmp_path), "walk")
    net_path = network.save(tmp_path / "net.npz")

    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\na,24.000,60.000\nb,24.006,60.000\n")
    demand = tmp_path / "od.csv"
    demand.write_text("origin_id,destination_id,demand\na,b,7\n")

    with pytest.raises(SystemExit, match="needs --links"):
        main(
            [
                "assign",
                "--network",
                str(net_path),
                "--mode",
                "walk",
                "--centroids",
                str(centroids),
                "--id-column",
                "id",
                "--demand",
                str(demand),
                "--gpkg",
                "--output-dir",
                str(out_dir),
                "--cache-dir",
                str(cache),
                "--workers",
                "1",
            ]
        )


def test_the_graph_is_named_after_its_link_layer(tmp_path):
    settings = Settings(output_dir=tmp_path / "out")

    # `valma extract` names the layer by mode, so the graph needs no repetition.
    assert settings.network_path("walk", "out/walk_links.gpkg").name == "walk.npz"
    # A hand-named layer gets the mode, so two modes cannot overwrite each other.
    assert settings.network_path("walk", "espoo.gpkg").name == "espoo_walk.npz"
    assert settings.network_path("bike", "espoo.gpkg").name == "espoo_bike.npz"

    # The implicit graph `matrix`/`assign` build from `--links` is the same
    # name, just parked under --cache-dir instead of --output-dir.
    settings = Settings(cache_dir=tmp_path / "cache", output_dir=tmp_path / "out")
    assert (
        settings.network_cache_path_for_links("walk", "out/walk_links.gpkg").name
        == "walk.npz"
    )
    assert (
        settings.network_cache_path_for_links("walk", "out/walk_links.gpkg").parent
        == tmp_path / "cache"
    )


def test_cli_matrix_with_links_caches_the_graph_instead_of_writing_it_out(
    chain_pbf, tmp_path
):
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    links_path = out_dir / "walk_links.gpkg"
    main(
        [
            "extract",
            "--pbf",
            chain_pbf,
            "--mode",
            "walk",
            "--output-dir",
            str(out_dir),
            "--cache-dir",
            str(cache),
        ]
    )

    centroids = tmp_path / "points.csv"
    centroids.write_text("id,lon,lat\na,24.000,60.000\nb,24.006,60.000\n")

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
                str(out_dir),
                "--cache-dir",
                str(cache),
                "--workers",
                "1",
            ]
        )
        == 0
    )

    # No explicit `valma build` ran, so the graph is a cache, not a named
    # output sitting next to the results.
    assert (cache / "walk.npz").exists()
    assert not (out_dir / "walk.npz").exists()


def test_cli_build_takes_an_explicit_out_path(chain_pbf, tmp_path, capsys):
    out_dir = tmp_path / "out"
    links_path = out_dir / "walk_links.gpkg"
    build_links(settings_for(chain_pbf, tmp_path), "walk", links_path=links_path)

    target = tmp_path / "elsewhere" / "graph.npz"
    assert (
        main(
            [
                "build",
                "--links",
                str(links_path),
                "--mode",
                "walk",
                "--out",
                str(target),
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    assert target.exists()
    assert not (out_dir / "walk.npz").exists()


def test_cli_refuses_a_network_built_for_another_mode(chain_pbf, tmp_path):
    network = build_network(settings_for(chain_pbf, tmp_path), "walk")
    net_path = network.save(tmp_path / "net.npz")

    with pytest.raises(SystemExit, match="but --mode is 'bike'"):
        main(["build", "--network", str(net_path), "--mode", "bike"])


def test_cli_without_any_source_says_what_it_wants():
    with pytest.raises(SystemExit, match="--network"):
        main(["build", "--mode", "walk"])
