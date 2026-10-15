"""Vector reads and writes: Arrow when it can, the plain path when it cannot."""

import geopandas as gpd
import pandas as pd
import pytest
import shapely

from valma_bike_and_walk import vector_io


def frame(**columns):
    return gpd.GeoDataFrame(
        columns,
        geometry=[shapely.LineString([(0, 0), (1, 1)]), shapely.LineString([(1, 1), (2, 0)])],
        crs="EPSG:3067",
    )


def test_a_round_trip_keeps_columns_types_and_values(tmp_path):
    original = frame(link_id=[1, 2], name=["a", None], speed=[1.5, float("nan")])
    path = vector_io.write_vector(original, tmp_path / "x.gpkg", driver="GPKG", layer="links")
    back = vector_io.read_vector(path, layer="links")
    pd.testing.assert_frame_equal(
        pd.DataFrame(back.drop(columns="geometry")),
        pd.DataFrame(original.drop(columns="geometry")),
    )
    assert back.geometry.geom_equals(original.geometry).all()


def test_a_column_arrow_cannot_take_is_written_plainly(tmp_path):
    pytest.importorskip("pyarrow")
    # One int and one string in an object column: Arrow refuses, GDAL does not.
    mixed = frame(link_id=[1, 2], tag=[3, "yes"])
    path = vector_io.write_vector(mixed, tmp_path / "mixed.gpkg", driver="GPKG", layer="links")
    back = vector_io.read_vector(path, layer="links")
    assert back["tag"].astype(str).tolist() == ["3", "yes"]


def test_without_pyarrow_the_plain_path_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_io, "arrow_available", lambda: False)
    calls = []
    real = gpd.GeoDataFrame.to_file

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("use_arrow"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", spy)
    path = vector_io.write_vector(frame(link_id=[1, 2]), tmp_path / "p.gpkg", driver="GPKG")
    assert calls == [None]
    assert len(vector_io.read_vector(path)) == 2
