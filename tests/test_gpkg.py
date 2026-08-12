"""Drawing a per-edge result back onto the link layer it was built from."""

import geopandas as gpd
import numpy as np
import pytest

from valma_bike_and_walk.gpkg import edges_to_geodataframe, write_edges_gpkg
from valma_bike_and_walk.links import normalise, write_links
from valma_bike_and_walk.network import network_from_links

from .conftest import links_frame

CHAIN = [
    {"link_id": 0, "u": 1, "v": 2, "coords": [(24.000, 60.0), (24.002, 60.0)]},
    {"link_id": 1, "u": 2, "v": 3, "coords": [(24.002, 60.0), (24.004, 60.0)]},
]


def built(rows, mode="walk"):
    links = normalise(links_frame(rows), mode)
    return network_from_links(links, mode), links


def test_every_edge_gets_the_geometry_of_its_own_link():
    network, links = built(CHAIN)
    gdf = edges_to_geodataframe(network, links)

    assert len(gdf) == network.n_edges
    for _, row in gdf.iterrows():
        expected = links.loc[links["link_id"] == row["link_id"], "geometry"].iloc[0]
        assert row.geometry.equals(expected)


def test_a_two_way_link_produces_one_row_per_direction():
    network, links = built(CHAIN)
    gdf = edges_to_geodataframe(network, links)

    both = gdf[gdf["link_id"] == 0]
    assert sorted(both["direction"]) == [-1, 1]
    # Same shape, opposite endpoints.
    assert sorted(zip(both["u"], both["v"])) == [(1, 2), (2, 1)]


def test_extra_columns_land_on_the_right_edges():
    network, links = built(CHAIN)
    volume = np.arange(network.n_edges, dtype=float)
    gdf = edges_to_geodataframe(network, links, extra_columns={"volume": volume})
    np.testing.assert_array_equal(gdf["volume"].to_numpy(), volume)


def test_speed_is_derived_from_the_edge_cost():
    rows = [{"link_id": 0, "u": 1, "v": 2, "coords": [(24.0, 60.0), (24.002, 60.0)]}]
    network, links = built(rows)
    gdf = edges_to_geodataframe(network, links)

    expected = gdf["length_m"] / gdf["travel_time_s"] * 3.6
    np.testing.assert_allclose(gdf["speed_kmh"], expected)


def test_the_wrong_link_layer_is_refused_rather_than_mis_joined():
    """Silently drawing volumes on someone else's geometry would be worse."""
    network, _ = built(CHAIN)
    other = normalise(
        links_frame(
            [{"link_id": 99, "u": 7, "v": 8, "coords": [(25.0, 61.0), (25.1, 61.0)]}]
        ),
        "walk",
    )
    with pytest.raises(ValueError, match="not the layer this network was built from"):
        edges_to_geodataframe(network, other)


def test_write_edges_gpkg_round_trips_through_a_real_file(tmp_path):
    network, links = built(CHAIN)
    links_path = write_links(links, tmp_path / "links.gpkg")

    out = write_edges_gpkg(
        network,
        links_path,
        tmp_path / "volumes.gpkg",
        extra_columns={"volume": np.full(network.n_edges, 2.5)},
    )

    gdf = gpd.read_file(out, layer="links")
    assert len(gdf) == network.n_edges
    assert set(gdf.columns) >= {
        "link_id",
        "direction",
        "u",
        "v",
        "length_m",
        "travel_time_s",
        "speed_kmh",
        "volume",
        "geometry",
    }
    assert gdf.crs.to_epsg() == 4326
    assert (gdf["volume"] == 2.5).all()


def test_results_join_back_onto_the_rows_you_edited(tmp_path):
    """The point of the two-stage split: edit a row, get results on that row."""
    edited = [dict(CHAIN[0], speed_override_kmh=1.0), CHAIN[1]]
    network, links = built(edited)

    gdf = edges_to_geodataframe(network, links)
    slowed = gdf[gdf["link_id"] == 0]["speed_kmh"].to_numpy()
    np.testing.assert_allclose(slowed, 1.0)
    assert (gdf[gdf["link_id"] == 1]["speed_kmh"] > 1.0).all()
