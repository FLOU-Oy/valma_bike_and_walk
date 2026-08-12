"""Stage 1: PBF -> link layer. Filtering, junction splitting, boundaries."""

import numpy as np
import pytest
import shapely

from valma_bike_and_walk.config import Settings
from valma_bike_and_walk.osm import (
    BIKE_FILTER,
    WALK_FILTER,
    find_boundary,
    read_links,
    resolve_clip,
)

from .conftest import write_pbf


def settings_for(pbf, tmp_path):
    return Settings(pbf_path=pbf, cache_dir=tmp_path / "cache")


# --------------------------------------------------------------------------
# Way filters
# --------------------------------------------------------------------------


def test_a_way_without_highway_is_never_a_link():
    assert not WALK_FILTER.keeps({"building": "yes"})
    assert not BIKE_FILTER.keeps({"waterway": "river"})


def test_modes_disagree_about_steps_and_cycleways():
    """The whole point of having two filters."""
    assert WALK_FILTER.keeps({"highway": "steps"})
    assert not BIKE_FILTER.keeps({"highway": "steps"})

    assert BIKE_FILTER.keeps({"highway": "cycleway"})
    assert not WALK_FILTER.keeps({"highway": "cycleway"})


def test_mode_specific_access_tags_are_honoured():
    assert not WALK_FILTER.keeps({"highway": "path", "foot": "no"})
    assert WALK_FILTER.keeps({"highway": "path", "bicycle": "no"})
    assert not BIKE_FILTER.keeps({"highway": "path", "bicycle": "no"})


def test_a_semicolon_tag_matches_any_of_its_values():
    """OSM writes access=private;delivery, and that is still private."""
    assert not WALK_FILTER.keeps({"highway": "service", "access": "delivery;private"})
    assert WALK_FILTER.keeps({"highway": "service", "access": "delivery;customers"})


def test_a_mapped_sidewalk_does_not_double_count_the_road():
    assert not WALK_FILTER.keeps({"highway": "residential", "sidewalk": "separate"})
    assert WALK_FILTER.keeps({"highway": "residential", "sidewalk": "both"})


def test_motorways_and_unbuilt_ways_are_excluded_for_both_modes():
    for tags in ({"highway": "motorway"}, {"highway": "proposed"}):
        assert not WALK_FILTER.keeps(tags)
        assert not BIKE_FILTER.keeps(tags)


# --------------------------------------------------------------------------
# Reading and splitting
# --------------------------------------------------------------------------


def test_ways_split_into_links_at_shared_nodes(grid_pbf, tmp_path):
    links = read_links(settings_for(grid_pbf, tmp_path), "walk")

    # Way 100 (1-2-3) is cut at node 2, which way 101 also touches; way 101
    # itself is one link; way 102 (steps) is walkable and one link.
    assert sorted(zip(links["u"], links["v"])) == [(1, 2), (2, 3), (2, 4), (4, 5)]
    assert set(links["osm_way_id"]) == {100, 101, 102}


def test_a_link_keeps_the_full_shape_between_its_junctions(tmp_path):
    """Interstitial nodes leave the graph but must stay in the geometry."""
    nodes = {
        1: (24.000, 60.000),
        2: (24.001, 60.001),  # a bend, referenced by nothing else
        3: (24.002, 60.000),
    }
    pbf = write_pbf(
        tmp_path / "bend.osm.pbf", nodes, [(1, [1, 2, 3], {"highway": "path"})]
    )
    links = read_links(settings_for(pbf, tmp_path), "walk")

    assert len(links) == 1
    assert (links.iloc[0]["u"], links.iloc[0]["v"]) == (1, 3)
    assert shapely.get_num_coordinates(links.geometry.iloc[0]) == 3


def test_the_cycling_filter_drops_steps(grid_pbf, tmp_path):
    links = read_links(settings_for(grid_pbf, tmp_path), "bike")
    assert set(links["osm_way_id"]) == {100, 101}
    assert 5 not in set(links["u"]) | set(links["v"])


def test_link_ids_are_unique_and_dense(grid_pbf, tmp_path):
    links = read_links(settings_for(grid_pbf, tmp_path), "walk")
    np.testing.assert_array_equal(
        np.sort(links["link_id"].to_numpy()), np.arange(len(links))
    )


def test_length_is_measured_in_metres(grid_pbf, tmp_path):
    links = read_links(settings_for(grid_pbf, tmp_path), "walk")
    row = links[(links["u"] == 1) & (links["v"] == 2)].iloc[0]
    # 0.002 degrees of longitude at 60 N is ~111 m.
    assert row["length_m"] == pytest.approx(111.0, rel=0.05)


def test_tags_are_carried_onto_every_link_of_their_way(grid_pbf, tmp_path):
    links = read_links(settings_for(grid_pbf, tmp_path), "walk")
    way_100 = links[links["osm_way_id"] == 100]
    assert len(way_100) == 2
    assert set(way_100["highway"]) == {"residential"}
    assert set(way_100["surface"]) == {"asphalt"}
    assert links[links["osm_way_id"] == 101].iloc[0]["oneway"] == "yes"


def test_a_bbox_keeps_only_ways_that_reach_into_it(grid_pbf, tmp_path):
    links = read_links(
        settings_for(grid_pbf, tmp_path),
        "walk",
        clip=shapely.box(23.999, 59.9995, 24.0025, 60.0005),
    )
    # Ways 101 and 102 run north, out of the box, and neither has a node in it
    # except the shared junction 2 -- which way 101 does have.
    assert set(links["osm_way_id"]) == {100, 101}


def test_an_empty_extent_is_an_error_not_an_empty_frame(grid_pbf, tmp_path):
    with pytest.raises(ValueError, match="No walk network"):
        read_links(
            settings_for(grid_pbf, tmp_path),
            "walk",
            clip=shapely.box(10.0, 50.0, 10.1, 50.1),
        )


def test_a_way_referencing_a_node_outside_the_extract_is_dropped(tmp_path):
    """Regional extracts cut ways; a link with no coordinates has no length."""
    nodes = {1: (24.0, 60.0), 2: (24.001, 60.0)}
    ways = [
        (1, [1, 2], {"highway": "residential"}),
        (2, [2, 999], {"highway": "residential"}),  # node 999 is not in the file
    ]
    pbf = write_pbf(tmp_path / "cut.osm.pbf", nodes, ways)

    links = read_links(settings_for(pbf, tmp_path), "walk")
    assert set(links["osm_way_id"]) == {1}


def test_a_self_touching_way_splits_at_its_own_repeat(tmp_path):
    """A lollipop: the stem and the loop must not become one link."""
    nodes = {
        1: (24.000, 60.000),
        2: (24.001, 60.000),
        3: (24.002, 60.001),
        4: (24.002, 59.999),
    }
    pbf = write_pbf(
        tmp_path / "loop.osm.pbf", nodes, [(1, [1, 2, 3, 4, 2], {"highway": "path"})]
    )
    links = read_links(settings_for(pbf, tmp_path), "walk")

    assert sorted(zip(links["u"], links["v"])) == [(1, 2), (2, 2)]


# --------------------------------------------------------------------------
# Boundaries
# --------------------------------------------------------------------------


def test_find_boundary_assembles_the_relation_and_caches_it(tmp_path):
    nodes = {
        10: (23.99, 59.99),
        11: (24.03, 59.99),
        12: (24.03, 60.02),
        13: (23.99, 60.02),
    }
    ways = [(200, [10, 11, 12, 13, 10], {})]
    relations = [
        (
            300,
            [("w", 200, "outer")],
            {
                "type": "boundary",
                "boundary": "administrative",
                "admin_level": "8",
                "name": "Testila",
            },
        )
    ]
    pbf = write_pbf(tmp_path / "area.osm.pbf", nodes, ways, relations)
    settings = settings_for(pbf, tmp_path)

    boundary = find_boundary(settings, "Testila")
    assert boundary.iloc[0]["name"] == "Testila"
    assert boundary.geometry.iloc[0].contains(shapely.Point(24.0, 60.0))

    cache = settings.boundary_cache_path("Testila")
    assert cache.exists()
    # Second call comes from the cache; deleting the PBF must not matter.
    assert len(find_boundary(settings, "Testila")) == 1


def test_an_exact_name_beats_a_substring_match(tmp_path):
    """'Espoo' must not silently return 'Pohjois-Espoo'."""
    nodes = {
        10: (23.99, 59.99),
        11: (24.03, 59.99),
        12: (24.03, 60.02),
        13: (23.99, 60.02),
        20: (25.00, 61.00),
        21: (25.50, 61.00),
        22: (25.50, 61.50),
        23: (25.00, 61.50),
    }
    ways = [(200, [10, 11, 12, 13, 10], {}), (201, [20, 21, 22, 23, 20], {})]
    admin = {"type": "boundary", "boundary": "administrative", "admin_level": "8"}
    relations = [
        (300, [("w", 200, "outer")], {**admin, "name": "Espoo"}),
        # Deliberately the larger of the two, so size alone would pick it.
        (301, [("w", 201, "outer")], {**admin, "name": "Pohjois-Espoo"}),
    ]
    pbf = write_pbf(tmp_path / "two.osm.pbf", nodes, ways, relations)

    boundary = find_boundary(settings_for(pbf, tmp_path), "Espoo")
    assert boundary.iloc[0]["name"] == "Espoo"


def test_a_missing_area_is_an_error(grid_pbf, tmp_path):
    with pytest.raises(ValueError, match="No administrative boundary"):
        find_boundary(settings_for(grid_pbf, tmp_path), "Atlantis")


def test_resolve_clip_prefers_bbox_and_keys_the_cache_on_it(grid_pbf, tmp_path):
    settings = settings_for(grid_pbf, tmp_path)
    clip, key = resolve_clip(settings, area="Testila", bbox=[24.0, 60.0, 24.1, 60.1])
    assert clip.bounds == (24.0, 60.0, 24.1, 60.1)
    assert key == "bbox_24.0000_60.0000_24.1000_60.1000"

    clip, key = resolve_clip(settings, area=None, bbox=None)
    assert clip is None and key == "full"
