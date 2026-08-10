"""Geometry threading through the build pipeline, and the GeoPackage export."""

import geopandas as gpd
import numpy as np
import pytest
import shapely

from valma_bike_and_walk.gpkg import edges_to_geodataframe, write_edges_gpkg
from valma_bike_and_walk.network import RoutableNetwork, _assemble, _EdgeArrays


def part(node_ids, lons, edges):
    """edges: list of (u_osmid, v_osmid, seconds). One straight LineString per edge."""
    lon_by_id = dict(zip(node_ids, lons))
    geoms = [
        shapely.LineString([(lon_by_id[u], 60.0), (lon_by_id[v], 60.0)])
        for u, v, _ in edges
    ]
    return _EdgeArrays(
        node_ids=np.array(node_ids, dtype=np.int64),
        lon=np.array(lons, dtype=float),
        lat=np.full(len(node_ids), 60.0),
        u=np.array([e[0] for e in edges], dtype=np.int64),
        v=np.array([e[1] for e in edges], dtype=np.int64),
        travel_time=np.array([e[2] for e in edges], dtype=float),
        length=np.array([e[2] for e in edges], dtype=np.float32),
        geometry=shapely.to_wkb(np.array(geoms, dtype=object)),
    )


def test_geometry_survives_assembly_aligned_with_edges():
    net = _assemble(
        "walk", [part([1, 2, 3], [24.0, 24.1, 24.2], [(1, 2, 10.0), (2, 3, 10.0)])]
    )
    assert net.geometry is not None
    assert net.geometry.shape[0] == net.n_edges
    for wkb in net.geometry:
        assert isinstance(shapely.from_wkb(wkb), shapely.LineString)


def test_geometry_deduped_alongside_the_fastest_parallel_edge():
    """The surviving edge's geometry must match the surviving (fastest) edge, not a discard."""
    fast = _EdgeArrays(
        node_ids=np.array([1, 2], dtype=np.int64),
        lon=np.array([24.0, 24.1]),
        lat=np.full(2, 60.0),
        u=np.array([1], dtype=np.int64),
        v=np.array([2], dtype=np.int64),
        travel_time=np.array([5.0]),
        length=np.array([5.0], dtype=np.float32),
        geometry=shapely.to_wkb(
            np.array([shapely.LineString([(24.0, 60.0), (24.1, 60.0)])], dtype=object)
        ),
    )
    slow = _EdgeArrays(
        node_ids=np.array([1, 2], dtype=np.int64),
        lon=np.array([24.0, 24.1]),
        lat=np.full(2, 60.0),
        u=np.array([1], dtype=np.int64),
        v=np.array([2], dtype=np.int64),
        travel_time=np.array([50.0]),
        # A deliberately different (detoured) shape, so a wrong pairing is detectable.
        geometry=shapely.to_wkb(
            np.array(
                [shapely.LineString([(24.0, 60.0), (24.05, 60.5), (24.1, 60.0)])],
                dtype=object,
            )
        ),
        length=np.array([50.0], dtype=np.float32),
    )

    net = _assemble("walk", [fast, slow])
    assert net.n_edges == 1
    assert net.travel_time[0] == pytest.approx(5.0)
    kept = shapely.from_wkb(net.geometry[0])
    assert kept.equals(shapely.LineString([(24.0, 60.0), (24.1, 60.0)]))


def test_save_and_load_round_trip_preserves_geometry(tmp_path):
    net = _assemble(
        "walk", [part([1, 2, 3], [24.0, 24.1, 24.2], [(1, 2, 10.0), (2, 3, 10.0)])]
    )
    path = net.save(tmp_path / "net.npz")
    loaded = RoutableNetwork.load(path)

    assert loaded.geometry is not None
    np.testing.assert_array_equal(loaded.geometry, net.geometry)


def test_load_without_geometry_leaves_it_none(tmp_path):
    net = _assemble(
        "walk",
        [
            _EdgeArrays(
                node_ids=np.array([1, 2], dtype=np.int64),
                lon=np.array([24.0, 24.1]),
                lat=np.full(2, 60.0),
                u=np.array([1], dtype=np.int64),
                v=np.array([2], dtype=np.int64),
                travel_time=np.array([10.0]),
                length=np.array([10.0], dtype=np.float32),
            )
        ],
    )
    path = net.save(tmp_path / "net.npz")
    loaded = RoutableNetwork.load(path)
    assert loaded.geometry is None


def test_edges_to_geodataframe_requires_geometry():
    net = _assemble(
        "walk",
        [
            _EdgeArrays(
                node_ids=np.array([1, 2], dtype=np.int64),
                lon=np.array([24.0, 24.1]),
                lat=np.full(2, 60.0),
                u=np.array([1], dtype=np.int64),
                v=np.array([2], dtype=np.int64),
                travel_time=np.array([10.0]),
                length=np.array([10.0], dtype=np.float32),
            )
        ],
    )
    with pytest.raises(ValueError, match="without geometry"):
        edges_to_geodataframe(net)


def test_write_edges_gpkg_round_trips_through_a_real_file(tmp_path):
    net = _assemble(
        "walk", [part([1, 2, 3], [24.0, 24.1, 24.2], [(1, 2, 10.0), (2, 3, 20.0)])]
    )
    path = write_edges_gpkg(net, tmp_path / "links.gpkg")

    gdf = gpd.read_file(path, layer="links")
    assert len(gdf) == net.n_edges
    assert set(gdf.columns) >= {
        "u",
        "v",
        "length_m",
        "travel_time_s",
        "speed_kmh",
        "geometry",
    }
    assert gdf.crs.to_epsg() == 4326
    # length 10 over 10 seconds -> 3.6 km/h.
    row = gdf[(gdf["u"] == 1) & (gdf["v"] == 2)].iloc[0]
    assert row["speed_kmh"] == pytest.approx(3.6)
