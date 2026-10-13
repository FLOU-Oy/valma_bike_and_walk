"""Settings files: layered under the command line, refused when mistyped.

Most of these read the settings back through ``--show-config``, which prints
them as TOML and runs nothing -- the same text a run records next to its
results, so what is checked is what a later ``-c`` would read.
"""

import sys

import numpy as np
import pytest

from valma_bike_and_walk.cli import main
from valma_bike_and_walk.config import Settings
from valma_bike_and_walk.network import build_links

from .conftest import write_pbf

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def show(capsys, *argv):
    """The settings a command would run with, top level and section merged."""
    assert main([*argv, "--show-config"]) == 0
    settings = tomllib.loads(capsys.readouterr().out)
    command = next(a for a in argv if not a.startswith("-") and "." not in a)
    return {**settings, **settings.pop(command, {})}


def refused(capsys, *argv):
    with pytest.raises(SystemExit) as exit:
        main([*argv, "--show-config"])
    assert exit.value.code == 2
    return capsys.readouterr().err


def posix(path):
    return path.as_posix()


# --------------------------------------------------------------------------
# Layering
# --------------------------------------------------------------------------


def test_a_file_supplies_what_the_command_line_would_have_to(tmp_path, capsys):
    config = write(
        tmp_path / "run.toml",
        'mode = "walk"\nlinks = "walk_links.gpkg"\ncentroids = "points.csv"\n'
        'demand = "od.csv"\n',
    )
    # --mode, --demand and one of --centroids/--zones are required otherwise.
    settings = show(capsys, "assign", "-c", str(config))
    assert settings["mode"] == "walk"
    assert settings["links"] == posix(tmp_path / "walk_links.gpkg")
    assert settings["demand"] == posix(tmp_path / "od.csv")


def test_top_level_keys_go_to_whichever_commands_take_them(tmp_path, capsys):
    config = write(
        tmp_path / "run.toml",
        'mode = "bike"\ncentroids = "p.csv"\ndemand = "od.csv"\nmax_minutes = 30\n',
    )
    matrix = show(capsys, "matrix", "-c", str(config))
    assign = show(capsys, "assign", "-c", str(config))
    assert "demand" not in matrix  # matrix has no --demand; not an error
    assert assign["demand"] == posix(tmp_path / "od.csv")
    assert matrix["max_minutes"] == assign["max_minutes"] == 30.0


def test_a_section_wins_within_its_file_and_a_later_file_wins_over_both(
    tmp_path, capsys
):
    base = write(
        tmp_path / "base.toml",
        'mode = "bike"\ncentroids = "p.csv"\nmax_minutes = 60\nworkers = 2\n'
        "[matrix]\nmax_minutes = 45\nworkers = 3\n",
    )
    scenario = write(tmp_path / "s.toml", "workers = 5\n")

    assert show(capsys, "matrix", "-c", str(base))["max_minutes"] == 45.0
    assert show(capsys, "matrix", "-c", str(base), "-c", str(scenario))["workers"] == 5
    assert show(capsys, "assign", "-c", str(base), "--demand", "x")["max_minutes"] == 60


def test_the_command_line_wins_over_every_file(tmp_path, capsys):
    config = write(
        tmp_path / "run.toml",
        'mode = "bike"\ncentroids = "p.csv"\n[matrix]\nworkers = 3\n',
    )
    settings = show(
        capsys, "matrix", "-c", str(config), "--workers", "7", "--mode", "walk"
    )
    assert settings["workers"] == 7
    assert settings["mode"] == "walk"


def test_extends_reads_the_base_first_from_beside_the_file(tmp_path, capsys):
    write(tmp_path / "common" / "base.toml", 'mode = "bike"\ncentroids = "p.csv"\n')
    scenario = write(
        tmp_path / "scenarios" / "2040.toml",
        'extends = "../common/base.toml"\nmode = "walk"\n',
    )
    settings = show(capsys, "matrix", "-c", str(scenario))
    assert settings["mode"] == "walk"
    # Relative to base.toml, where it is written, not to 2040.toml.
    assert settings["centroids"] == posix(tmp_path / "common" / "p.csv")


def test_files_that_extend_each_other_in_a_loop_are_refused(tmp_path, capsys):
    write(tmp_path / "a.toml", 'extends = "b.toml"\n')
    write(tmp_path / "b.toml", 'extends = "a.toml"\n')
    assert "loop" in refused(capsys, "matrix", "-c", str(tmp_path / "a.toml"))


def test_a_later_layer_choosing_zones_replaces_the_bases_centroids(tmp_path, capsys):
    base = write(tmp_path / "base.toml", 'mode = "bike"\ncentroids = "p.csv"\n')
    zones = write(tmp_path / "zones.toml", 'zones = "taz.gpkg"\n')
    settings = show(capsys, "matrix", "-c", str(base), "-c", str(zones))
    assert settings["zones"] == posix(tmp_path / "taz.gpkg")
    assert "centroids" not in settings


def test_zones_on_the_command_line_replace_centroids_from_a_file(tmp_path, capsys):
    base = write(tmp_path / "base.toml", 'mode = "bike"\ncentroids = "p.csv"\n')
    settings = show(capsys, "matrix", "-c", str(base), "--zones", "taz.gpkg")
    assert "centroids" not in settings
    assert settings["zones"].endswith("taz.gpkg")


def test_centroids_and_zones_in_one_file_are_refused(tmp_path, capsys):
    both = write(
        tmp_path / "both.toml", 'mode = "bike"\ncentroids = "p.csv"\nzones = "z.gpkg"\n'
    )
    assert "exclude each other" in refused(capsys, "matrix", "-c", str(both))


# --------------------------------------------------------------------------
# Spelling and types
# --------------------------------------------------------------------------


def test_keys_may_be_spelled_like_the_flags(tmp_path, capsys):
    config = write(
        tmp_path / "run.toml",
        'mode = "bike"\ncentroids = "p.csv"\nmax-snap-distance = 50\n',
    )
    assert show(capsys, "matrix", "-c", str(config))["max_snap_distance"] == 50.0


def test_a_key_spelled_both_ways_is_refused(tmp_path, capsys):
    config = write(tmp_path / "run.toml", "max_minutes = 5\nmax-minutes = 6\n")
    assert "given twice" in refused(capsys, "matrix", "-c", str(config))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("max_minute = 30\n", "Did you mean 'max_minutes'?"),
        ("[assgn]\nworkers = 2\n", "Did you mean 'assign'?"),
        ("[matrix]\ndemand = 'od.csv'\n", "valma matrix takes no 'demand'"),
        ("workers = '4'\n", "workers should be a whole number"),
        ("max_minutes = true\n", "max_minutes should be a number"),
        ("mode = 'car'\n", "one of 'walk', 'bike'"),
        ("bbox = [24.0, 60.0, 25.0]\n", "a list of 4"),
        ("gpkg = 'yes'\n", "true or false"),
        ("links = 3\n", "links should be a path"),
        ("extends = 3\n", "extends must be a file name"),
        ("mode = \n", "Invalid"),
    ],
)
def test_mistakes_are_refused_not_ignored(tmp_path, capsys, text, message):
    config = write(tmp_path / "bad.toml", text)
    error = refused(
        capsys, "matrix", "-c", str(config), "--mode", "bike", "--centroids", "p"
    )
    assert message in error
    assert "bad.toml" in error


def test_a_mistake_in_another_commands_section_is_caught_too(tmp_path, capsys):
    config = write(tmp_path / "run.toml", "[assign]\nmin_demand = 'lots'\n")
    assert "min_demand" in refused(
        capsys, "matrix", "-c", str(config), "--mode", "bike", "--centroids", "p"
    )


def test_a_missing_file_is_named(tmp_path, capsys):
    assert "No such settings file" in refused(
        capsys, "matrix", "-c", str(tmp_path / "nope.toml")
    )


def test_an_abbreviated_config_flag_is_refused_rather_than_ignored(tmp_path, capsys):
    config = write(tmp_path / "run.toml", 'mode = "bike"\n')
    with pytest.raises(SystemExit):
        main(["matrix", "--conf", str(config), "--centroids", "p", "--mode", "bike"])
    assert "in full" in capsys.readouterr().err


def test_a_flag_taking_an_optional_value_reads_true_as_the_bare_flag(tmp_path, capsys):
    bare = write(
        tmp_path / "bare.toml", 'mode = "bike"\ncentroids = "p.csv"\nomx = true\n'
    )
    named = write(tmp_path / "named.toml", 'extends = "bare.toml"\nomx = "out/t.omx"\n')
    off = write(tmp_path / "off.toml", 'extends = "bare.toml"\nomx = false\n')
    assert show(capsys, "matrix", "-c", str(bare))["omx"] is True
    assert show(capsys, "matrix", "-c", str(named))["omx"] == posix(
        tmp_path / "out" / "t.omx"
    )
    assert "omx" not in show(capsys, "matrix", "-c", str(off))


# --------------------------------------------------------------------------
# Placeholders and scenarios
# --------------------------------------------------------------------------


def test_mode_and_scenario_are_filled_in(tmp_path, capsys):
    config = write(
        tmp_path / "run.toml",
        'mode = "walk"\nscenario = "{mode}_2040"\ncentroids = "p.csv"\n'
        'links = "{scenario}/{mode}_links.gpkg"\n',
    )
    settings = show(capsys, "matrix", "-c", str(config))
    assert settings["links"] == posix(tmp_path / "walk_2040" / "walk_links.gpkg")


def test_a_placeholder_follows_a_mode_given_on_the_command_line(tmp_path, capsys):
    config = write(
        tmp_path / "run.toml",
        'mode = "bike"\ncentroids = "p.csv"\nlinks = "{mode}_links.gpkg"\n',
    )
    settings = show(capsys, "matrix", "-c", str(config), "--mode", "walk")
    assert settings["links"] == posix(tmp_path / "walk_links.gpkg")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("{area}_links.gpkg", "only {mode} and {scenario}"),
        ("{scenario}.gpkg", "no scenario"),
    ],
)
def test_a_placeholder_that_cannot_be_filled_is_refused(
    tmp_path, capsys, value, message
):
    config = write(
        tmp_path / "run.toml",
        f'mode = "bike"\ncentroids = "p.csv"\nlinks = "{value}"\n',
    )
    assert message in refused(capsys, "matrix", "-c", str(config))


def test_a_scenario_keeps_its_results_apart(tmp_path, capsys):
    config = write(
        tmp_path / "run.toml",
        'mode = "bike"\ncentroids = "p.csv"\noutput_dir = "out"\nscenario = "2040"\n',
    )
    settings = show(capsys, "matrix", "-c", str(config))
    assert settings["output_dir"] == posix(tmp_path / "out" / "2040")
    assert "scenario" not in settings  # folded in; reading it back must not nest twice


# --------------------------------------------------------------------------
# A real run, recorded and repeated
# --------------------------------------------------------------------------

NODES = {1: (24.000, 60.000), 2: (24.002, 60.000), 3: (24.004, 60.000)}
FOOTWAY = {"highway": "footway", "surface": "asphalt"}
WAYS = [(100, [1, 2], FOOTWAY), (101, [2, 3], FOOTWAY)]


@pytest.fixture
def project(tmp_path):
    """A link layer, zone points and demand, with a settings file naming them."""
    pbf = write_pbf(tmp_path / "chain.osm.pbf", NODES, WAYS)
    settings = Settings(pbf_path=pbf, cache_dir=tmp_path / "cache", output_dir=tmp_path)
    build_links(settings, "walk", links_path=tmp_path / "data" / "walk_links.gpkg")
    write(
        tmp_path / "data" / "points.csv",
        "id,lon,lat\n1,24.000,60.000\n2,24.004,60.000\n",
    )
    write(tmp_path / "data" / "od.csv", "origin_id,destination_id,demand\n1,2,10\n")
    return write(
        tmp_path / "scenarios" / "base.toml",
        'mode = "walk"\n'
        'links = "../data/{mode}_links.gpkg"\n'
        'centroids = "../data/points.csv"\n'
        'id_column = "id"\n'
        'demand = "../data/od.csv"\n'
        'cache_dir = "../cache"\n'
        'output_dir = "../output"\n'
        'scenario = "base"\n'
        "workers = 1\n",
    )


def test_a_run_from_a_file_records_settings_that_repeat_it(project, tmp_path):
    assert main(["matrix", "-c", str(project)]) == 0
    out = tmp_path / "output" / "base"
    first = np.load(out / "travel_times_walk.npz")["seconds"]

    record = out / "valma_matrix_walk.toml"
    recorded = tomllib.loads(record.read_text(encoding="utf-8"))
    assert recorded["links"] == posix(tmp_path / "data" / "walk_links.gpkg")
    assert recorded["output_dir"] == posix(out)

    # The record alone, from somewhere else entirely, does the same run.
    (out / "travel_times_walk.npz").unlink()
    assert main(["matrix", "-c", str(record)]) == 0
    np.testing.assert_array_equal(
        np.load(out / "travel_times_walk.npz")["seconds"], first
    )
    assert first[0, 1] > 0


def test_an_assign_record_opens_in_view_with_the_runs_inputs(project, tmp_path, capsys):
    assert main(["assign", "-c", str(project)]) == 0
    capsys.readouterr()
    record = tmp_path / "output" / "base" / "valma_assign_walk.toml"

    settings = show(capsys, "view", "-c", str(record))
    assert settings["demand"] == posix(tmp_path / "data" / "od.csv")
    assert settings["centroids"] == posix(tmp_path / "data" / "points.csv")
    assert settings["output_dir"] == posix(tmp_path / "output" / "base")


def test_a_run_without_a_file_is_recorded_too(project, tmp_path):
    data = tmp_path / "data"
    argv = [
        "matrix", "--mode", "walk", "--links", str(data / "walk_links.gpkg"),
        "--centroids", str(data / "points.csv"), "--id-column", "id",
        "--output-dir", str(tmp_path / "plain"), "--cache-dir", str(tmp_path / "cache"),
        "--workers", "1",
    ]  # fmt: skip
    assert main(argv) == 0
    assert (tmp_path / "plain" / "valma_matrix_walk.toml").exists()
