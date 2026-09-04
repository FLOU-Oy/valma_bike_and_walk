"""Conflating another network's car volumes onto the link layer.

Everything here is built in EPSG:3067, so a coordinate is a metre and every
tolerance in the module can be written into a test as the distance it is.
"""

import geopandas as gpd
import numpy as np
import pytest
import shapely

from valma_bike_and_walk.cli import main
from valma_bike_and_walk.config import PROJECTED_CRS
from valma_bike_and_walk.links import LINKS_LAYER, write_links
from valma_bike_and_walk.traffic import (
    OFFSET_COLUMN,
    OVERRIDE_COLUMN,
    VOLUME_COLUMN,
    MatchSettings,
    add_volumes,
    carries_cars,
    match_volumes,
    read_volumes,
    unmatched_sources,
)

# Somewhere in the Finnish grid; the offsets from it are what the tests are about.
EAST, NORTH = 300_000.0, 6_700_000.0


def link_layer(rows) -> gpd.GeoDataFrame:
    """A link layer from ``(highway, coords)`` dicts, in projected metres."""
    records = []
    geometries = []
    for i, row in enumerate(rows):
        records.append(
            {
                "link_id": i,
                "u": 2 * i + 1,
                "v": 2 * i + 2,
                "highway": row.get("highway", "residential"),
                "oneway": row.get("oneway", None),
                "junction": row.get("junction", None),
            }
        )
        geometries.append(
            shapely.LineString([(EAST + x, NORTH + y) for x, y in row["coords"]])
        )
    links = gpd.GeoDataFrame(records, geometry=geometries, crs=PROJECTED_CRS)
    links["length_m"] = links.geometry.length
    return links


def volume_layer(rows) -> gpd.GeoDataFrame:
    """A source layer from ``(volume, coords)`` dicts, in the same metres."""
    return gpd.GeoDataFrame(
        {"volume": [row["volume"] for row in rows]},
        geometry=[
            shapely.LineString([(EAST + x, NORTH + y) for x, y in row["coords"]])
            for row in rows
        ],
        crs=PROJECTED_CRS,
    )


def volume_of(links, link_id):
    """The volume attached to one row, as a float or nan."""
    return float(links.set_index("link_id").loc[link_id, VOLUME_COLUMN])


# --------------------------------------------------------------------------
# The rule the whole thing exists for: cars go on roads, not on cycleways
# --------------------------------------------------------------------------


def test_the_road_takes_the_volume_and_the_cycleway_beside_it_does_not():
    links = link_layer(
        [
            {"highway": "residential", "coords": [(0, 0), (200, 0)]},
            {"highway": "cycleway", "coords": [(0, 3), (200, 3)]},
        ]
    )
    # Deliberately closer to the cycle track than to the road it belongs on.
    volumes = volume_layer([{"volume": 5000, "coords": [(0, 2), (200, 2)]}])

    matched, _ = add_volumes(links, volumes)

    assert volume_of(matched, 0) == 5000
    assert np.isnan(volume_of(matched, 1))


def test_a_source_over_nothing_but_a_cycleway_matches_nothing():
    links = link_layer([{"highway": "cycleway", "coords": [(0, 0), (200, 0)]}])
    volumes = volume_layer([{"volume": 5000, "coords": [(0, 1), (200, 1)]}])

    with pytest.raises(ValueError, match="carries motor traffic"):
        match_volumes(links, volumes)


def test_carries_cars_names_the_candidate_links():
    links = link_layer(
        [
            {"highway": "primary", "coords": [(0, 0), (10, 0)]},
            {"highway": "cycleway", "coords": [(0, 5), (10, 5)]},
            {"highway": "footway", "coords": [(0, 9), (10, 9)]},
            {"highway": "service", "coords": [(0, 20), (10, 20)]},
        ]
    )
    assert list(carries_cars(links)) == [True, False, False, True]


# --------------------------------------------------------------------------
# A source link that is not in this network at all
# --------------------------------------------------------------------------


def test_a_motorway_missing_from_the_network_does_not_load_the_road_beside_it():
    """The case the coverage rule is for: only a quarter of the source is here."""
    links = link_layer(
        [{"highway": "service", "coords": [(300, 8), (400, 8)]}],
    )
    volumes = volume_layer([{"volume": 30000, "coords": [(0, 0), (400, 0)]}])

    matched, match = add_volumes(links, volumes)

    assert np.isnan(volume_of(matched, 0))
    assert not match.source_used.any()
    assert match.source_coverage[0] == pytest.approx(0.25, abs=0.06)


def test_lowering_the_coverage_threshold_lets_that_partial_match_through():
    links = link_layer([{"highway": "service", "coords": [(300, 8), (400, 8)]}])
    volumes = volume_layer([{"volume": 30000, "coords": [(0, 0), (400, 0)]}])

    matched, _ = add_volumes(links, volumes, MatchSettings(min_coverage=0.1))

    assert volume_of(matched, 0) == 30000


def test_unmatched_sources_come_back_with_the_share_that_found_a_road():
    links = link_layer([{"highway": "service", "coords": [(300, 8), (400, 8)]}])
    volumes = volume_layer([{"volume": 30000, "coords": [(0, 0), (400, 0)]}])

    _, match = add_volumes(links, volumes)
    missed = unmatched_sources(volumes, match)

    assert len(missed) == 1
    assert missed["volume"].iloc[0] == 30000
    assert 0 < missed["coverage"].iloc[0] < 0.5


# --------------------------------------------------------------------------
# Geometry that does not line up
# --------------------------------------------------------------------------


def test_one_source_link_covers_every_link_of_the_road_it_follows():
    """The usual shape of the problem: their network is coarser than ours."""
    links = link_layer(
        [
            {"highway": "tertiary", "coords": [(0, 0), (100, 0)]},
            {"highway": "tertiary", "coords": [(100, 0), (250, 0)]},
            {"highway": "tertiary", "coords": [(250, 0), (400, 0)]},
        ]
    )
    volumes = volume_layer([{"volume": 4200, "coords": [(0, 2), (400, 2)]}])

    matched, _ = add_volumes(links, volumes)

    assert [volume_of(matched, i) for i in range(3)] == [4200, 4200, 4200]


def test_a_road_crossing_at_a_right_angle_takes_none_of_the_volume():
    links = link_layer(
        [
            {"highway": "secondary", "coords": [(0, 0), (200, 0)]},
            {"highway": "residential", "coords": [(100, -60), (100, 60)]},
        ]
    )
    volumes = volume_layer([{"volume": 9000, "coords": [(0, 1), (200, 1)]}])

    matched, _ = add_volumes(links, volumes)

    assert volume_of(matched, 0) == 9000
    assert np.isnan(volume_of(matched, 1))


def test_a_source_further_off_than_the_tolerance_matches_nothing():
    links = link_layer(
        [
            {"highway": "primary", "coords": [(0, 0), (200, 0)]},
            # Puts the source well inside the layer's extent, so this is a test
            # about the distance tolerance and not about the extent filter.
            {"highway": "primary", "coords": [(0, 400), (200, 400)]},
        ]
    )
    volumes = volume_layer([{"volume": 9000, "coords": [(0, 25), (200, 25)]}])

    matched, match = add_volumes(links, volumes)

    assert match.source_in_extent.all()
    assert matched[VOLUME_COLUMN].isna().all()


def test_a_source_layer_covering_the_whole_country_is_not_all_a_failed_match():
    """A national volume layer against one city's links is the normal case."""
    links = link_layer([{"highway": "primary", "coords": [(0, 0), (200, 0)]}])
    volumes = volume_layer(
        [
            {"volume": 9000, "coords": [(0, 2), (200, 2)]},
            {"volume": 5000, "coords": [(0, 300_000), (200, 300_000)]},
        ]
    )

    matched, match = add_volumes(links, volumes)

    assert volume_of(matched, 0) == 9000
    assert list(match.source_in_extent) == [True, False]
    assert np.isnan(match.source_coverage[1])
    # The far-away source is somewhere else, not a match that failed.
    assert len(unmatched_sources(volumes, match)) == 0


def test_the_nearer_of_two_sources_wins_the_link_outright():
    """Winner takes all: a link's volume is one source's, never a blend."""
    links = link_layer([{"highway": "residential", "coords": [(0, 0), (200, 0)]}])
    volumes = volume_layer(
        [
            {"volume": 1000, "coords": [(0, 1), (200, 1)]},
            {"volume": 9000, "coords": [(0, 9), (200, 9)]},
        ]
    )

    matched, _ = add_volumes(links, volumes)

    assert volume_of(matched, 0) == 1000


# --------------------------------------------------------------------------
# Divided roads
# --------------------------------------------------------------------------


def divided_road():
    """Two one-way carriageways 12 m apart, digitised against each other."""
    links = link_layer(
        [
            {"highway": "primary", "oneway": "yes", "coords": [(0, 6), (200, 6)]},
            {"highway": "primary", "oneway": "yes", "coords": [(200, -6), (0, -6)]},
        ]
    )
    volumes = volume_layer([{"volume": 8000, "coords": [(0, 0), (200, 0)]}])
    return links, volumes


def test_a_divided_roads_carriageways_get_half_the_volume_each():
    links, volumes = divided_road()

    matched, match = add_volumes(links, volumes)

    assert volume_of(matched, 0) == 4000
    assert volume_of(matched, 1) == 4000
    assert match.divided.all()


def test_the_split_can_be_turned_off_for_an_already_directed_source():
    links, volumes = divided_road()

    matched, match = add_volumes(links, volumes, MatchSettings(split_divided=False))

    assert volume_of(matched, 0) == 8000
    assert volume_of(matched, 1) == 8000
    assert not match.divided.any()


def test_two_way_roads_are_never_read_as_a_divided_pair():
    """Only one-way links pair up: a parallel two-way street is a different road."""
    links = link_layer(
        [
            {"highway": "residential", "coords": [(0, 6), (200, 6)]},
            {"highway": "residential", "coords": [(200, -6), (0, -6)]},
        ]
    )
    volumes = volume_layer([{"volume": 8000, "coords": [(0, 0), (200, 0)]}])

    matched, match = add_volumes(links, volumes)

    assert not match.divided.any()
    assert matched[VOLUME_COLUMN].max() == 8000


def test_a_roundabout_is_not_mistaken_for_a_divided_road():
    """A roundabout is one-way and faces every direction; it must not pair with itself."""
    circle = shapely.Point(EAST + 100, NORTH).buffer(20).exterior
    links = gpd.GeoDataFrame(
        {
            "link_id": [0],
            "u": [1],
            "v": [1],
            "highway": ["tertiary"],
            "oneway": ["yes"],
            "junction": ["roundabout"],
        },
        geometry=[shapely.LineString(circle.coords)],
        crs=PROJECTED_CRS,
    )
    links["length_m"] = links.geometry.length
    volumes = volume_layer([{"volume": 6000, "coords": [(75, 0), (125, 0)]}])

    _, match = add_volumes(links, volumes, MatchSettings(min_coverage=0.1))

    assert not match.divided.any()


# --------------------------------------------------------------------------
# Reading the source layer
# --------------------------------------------------------------------------


def test_read_volumes_says_which_columns_it_found(tmp_path):
    path = tmp_path / "src.gpkg"
    volume_layer([{"volume": 1, "coords": [(0, 0), (10, 0)]}]).rename(
        columns={"volume": "aadt"}
    ).to_file(path, driver="GPKG", layer="links")

    with pytest.raises(ValueError, match="no 'volume' column"):
        read_volumes(path)

    assert len(read_volumes(path, column="aadt")) == 1


def test_read_volumes_explodes_multi_part_lines_and_drops_empty_ones(tmp_path):
    frame = gpd.GeoDataFrame(
        {"volume": [100, 200]},
        geometry=[
            shapely.MultiLineString(
                [
                    [(EAST, NORTH), (EAST + 10, NORTH)],
                    [(EAST + 20, NORTH), (EAST + 30, NORTH)],
                ]
            ),
            shapely.LineString([(EAST, NORTH + 50), (EAST + 10, NORTH + 50)]),
        ],
        crs=PROJECTED_CRS,
    )
    path = tmp_path / "multi.gpkg"
    frame.to_file(path, driver="GPKG", layer="links")

    volumes = read_volumes(path)

    assert len(volumes) == 3
    assert volumes.geometry.geom_type.unique().tolist() == ["LineString"]


def test_a_source_in_another_crs_is_matched_in_metres_all_the_same():
    links = link_layer([{"highway": "primary", "coords": [(0, 0), (200, 0)]}])
    volumes = volume_layer([{"volume": 7000, "coords": [(0, 1), (200, 1)]}]).to_crs(
        "EPSG:4326"
    )

    matched, _ = add_volumes(links, volumes)

    assert volume_of(matched, 0) == 7000


# --------------------------------------------------------------------------
# The columns, and the command
# --------------------------------------------------------------------------


def test_the_layer_comes_back_with_the_offset_and_an_empty_override():
    links = link_layer([{"highway": "primary", "coords": [(0, 0), (200, 0)]}])
    volumes = volume_layer([{"volume": 7000, "coords": [(0, 3), (200, 3)]}])

    matched, _ = add_volumes(links, volumes)

    assert matched[OFFSET_COLUMN].iloc[0] == pytest.approx(3.0, abs=0.1)
    assert matched[OVERRIDE_COLUMN].isna().all()


def test_bad_settings_are_refused_before_any_work_is_done():
    with pytest.raises(ValueError, match="min_coverage"):
        MatchSettings(min_coverage=1.5)
    with pytest.raises(ValueError, match="max_angle_deg"):
        MatchSettings(max_angle_deg=120)


def test_the_traffic_command_writes_the_volumes_back_into_the_link_layer(tmp_path):
    links = link_layer(
        [
            {"highway": "primary", "coords": [(0, 0), (200, 0)]},
            {"highway": "cycleway", "coords": [(0, 4), (200, 4)]},
        ]
    ).to_crs("EPSG:4326")
    links_path = write_links(links, tmp_path / "bike_links.gpkg")

    volumes_path = tmp_path / "model.gpkg"
    volume_layer([{"volume": 12000, "coords": [(0, 2), (200, 2)]}]).to_file(
        volumes_path, driver="GPKG", layer="net"
    )
    missed_path = tmp_path / "unmatched.gpkg"

    assert (
        main(
            [
                "traffic",
                "--links",
                str(links_path),
                "--volumes",
                str(volumes_path),
                "--unmatched-out",
                str(missed_path),
            ]
        )
        == 0
    )

    back = gpd.read_file(links_path, layer=LINKS_LAYER)
    by_id = back.set_index("link_id")[VOLUME_COLUMN]
    assert by_id.loc[0] == 12000
    assert np.isnan(by_id.loc[1])
    assert missed_path.exists()
