"""valma pipeline: which steps run, and which are skipped as up to date."""

import argparse
import os
from pathlib import Path

import pytest

from valma_bike_and_walk.pipeline import build_plan, run_pipeline


class World:
    """Fake commands over real files, so stamps and in-place writes are real."""

    def __init__(self, root: Path, volumes: bool = True):
        self.root = root
        self.pbf = root / "finland.osm.pbf"
        self.volumes = root / "volumes.gpkg"
        self.demand = root / "demand.omx"
        self.links = root / "out" / "bike_links.gpkg"
        for path in (self.pbf, self.volumes, self.demand):
            path.write_text("v1")
        self.links.parent.mkdir()
        self.settings = {"dem_resolution": 10.0, "volumes": volumes}
        self.ran: list[str] = []
        self.fail: str | None = None

    def resolve(self, argv):
        command = argv[0]
        args = argparse.Namespace(
            command=command, links=self.links, mode="bike", connectors=None,
            output_dir=self.root / "out", workers=4,
        )
        if command == "extract":
            args.pbf = self.pbf
        if command == "dem":
            args.dem_resolution = self.settings["dem_resolution"]
        if command == "traffic":
            if not self.settings["volumes"]:
                return None
            args.volumes = self.volumes
        if command == "assign":
            args.centroids, args.demand = None, self.demand
        args.argv = list(argv)
        return args

    def run(self, argv):
        command = argv[0]
        self.ran.append(command)
        if command == self.fail:
            return 1
        if command == "extract":
            self.links.write_text("links")
        elif command in ("dem", "traffic"):
            with self.links.open("a") as f:
                f.write(f"+{command}")
        elif command == "connect":
            Path(argv[argv.index("--out") + 1]).write_text("connectors")
        elif command == "assign":
            (self.root / "out" / "link_volumes_bike.npz").write_text("volumes")
        return 0

    def go(self, **kwargs):
        self.ran = []
        plan = build_plan(self.resolve, skip=set(kwargs.pop("skip", ())))
        code = run_pipeline(plan, self.run, "test", report=lambda line: None, **kwargs)
        return code, self.ran


def touch(path: Path, text: str) -> None:
    path.write_text(text)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))


ALL = ["extract", "dem", "traffic", "connect", "assign"]


def test_a_second_run_with_nothing_changed_runs_nothing(tmp_path):
    world = World(tmp_path)
    assert world.go() == (0, ALL)
    assert world.go() == (0, [])


def test_new_osm_data_reruns_everything(tmp_path):
    world = World(tmp_path)
    world.go()
    touch(world.pbf, "v2")
    assert world.go()[1] == ALL


def test_new_volumes_rerun_traffic_and_what_follows_only(tmp_path):
    world = World(tmp_path)
    world.go()
    touch(world.volumes, "v2")
    assert world.go()[1] == ["traffic", "connect", "assign"]


def test_a_changed_setting_reruns_its_step_onwards(tmp_path):
    world = World(tmp_path)
    world.go()
    world.settings["dem_resolution"] = 2.0
    assert world.go()[1] == ["dem", "traffic", "connect", "assign"]


def test_new_demand_reruns_assign_only(tmp_path):
    world = World(tmp_path)
    world.go()
    touch(world.demand, "v2")
    assert world.go()[1] == ["assign"]


def test_a_hand_edit_of_the_links_reruns_what_reads_them_after(tmp_path):
    """An override typed in QGIS: connect and assign, never extract over it."""
    world = World(tmp_path)
    world.go()
    touch(world.links, world.links.read_text() + "+edit")
    assert world.go()[1] == ["connect", "assign"]
    assert world.go()[1] == []


def test_a_missing_output_reruns_its_step(tmp_path):
    world = World(tmp_path)
    world.go()
    (tmp_path / "out" / "link_volumes_bike.npz").unlink()
    assert world.go()[1] == ["assign"]
    world.links.unlink()
    assert world.go()[1] == ALL


def test_a_failure_stops_and_is_retried_next_time(tmp_path):
    world = World(tmp_path)
    world.fail = "traffic"
    assert world.go() == (1, ["extract", "dem", "traffic"])
    world.fail = None
    assert world.go()[1] == ["traffic", "connect", "assign"]


def test_unconfigured_traffic_is_left_out(tmp_path):
    world = World(tmp_path, volumes=False)
    assert world.go()[1] == ["extract", "dem", "connect", "assign"]
    assert world.go()[1] == []


def test_mark_done_adopts_existing_outputs(tmp_path):
    world = World(tmp_path)
    for command in ALL:  # as if each had been run by hand
        argv = ["connect", "--out", str(tmp_path / "out" / "bike_connectors.gpkg")]
        world.run(argv if command == "connect" else [command])
    assert world.go(mark_done=True) == (0, [])
    assert world.go()[1] == []
    touch(world.demand, "v2")
    assert world.go()[1] == ["assign"]


def test_dry_run_runs_nothing_and_records_nothing(tmp_path):
    world = World(tmp_path)
    assert world.go(dry_run=True) == (0, [])
    assert not (tmp_path / "out" / "pipeline_bike.json").exists()
    assert world.go()[1] == ALL


def test_skipped_steps_are_left_out(tmp_path):
    world = World(tmp_path)
    assert world.go(skip={"dem"})[1] == ["extract", "traffic", "connect", "assign"]


BASE_SETTINGS = """\
mode = "walk"
scenario = "base"
output_dir = "../output"
cache_dir = "../cache"
links = "../output/{scenario}/{mode}_links.gpkg"
centroids = "../data/points.csv"
id_column = "id"
demand = "../data/od.csv"
workers = 1

[extract]
pbf = "../data/chain.osm.pbf"
"""


@pytest.fixture
def real_project(tmp_path, capsys):
    """A tiny OSM file, zones and demand, and `valma pipeline` over them."""
    from valma_bike_and_walk.cli import main

    from .conftest import write_pbf
    from .test_config_file import NODES, WAYS

    (tmp_path / "data").mkdir()
    write_pbf(tmp_path / "data" / "chain.osm.pbf", NODES, WAYS)
    (tmp_path / "data" / "points.csv").write_text(
        "id,lon,lat\n1,24.000,60.000\n2,24.004,60.000\n"
    )
    (tmp_path / "data" / "od.csv").write_text("origin_id,destination_id,demand\n1,2,10\n")
    settings = tmp_path / "scenarios" / "base.toml"
    settings.parent.mkdir()

    def write_settings(extra: str = "") -> None:
        """The base settings, plus ``extra`` -- more lines or sections."""
        settings.write_text(BASE_SETTINGS + extra, encoding="utf-8")

    def pipeline(*flags):
        """Run it; return each step's status line, e.g. {"dem": "skipped (--skip)"}."""
        capsys.readouterr()
        assert main(["pipeline", "-c", str(settings), "--skip", "dem", *flags]) == 0
        lines = capsys.readouterr().out.splitlines()
        return {
            line.split(":")[0].strip(): line.split(":", 1)[1].strip()
            for line in lines
            if line.startswith("  ")
        }

    write_settings()
    return write_settings, pipeline


def test_the_real_commands_run_once_and_then_stay_up_to_date(real_project, tmp_path):
    """extract -> connect -> assign through `valma pipeline`, on a tiny OSM file."""
    _, pipeline = real_project
    first = pipeline()
    assert first["traffic"] == "skipped (not configured)"
    assert all(first[s].startswith("running") for s in ("extract", "connect", "assign"))
    out = tmp_path / "output" / "base"
    assert (out / "walk_links.gpkg").exists() and (out / "link_volumes_walk.npz").exists()
    # The chain is in one piece, so connect leaves no file and assign runs without.
    assert not (out / "walk_connectors.gpkg").exists()

    again = pipeline()
    assert {again[s] for s in ("extract", "connect", "assign")} == {"up to date"}

    touch(tmp_path / "data" / "od.csv", "origin_id,destination_id,demand\n1,2,20\n")
    third = pipeline()
    assert third["extract"] == "up to date" and third["connect"] == "up to date"
    assert third["assign"] == "running (inputs changed)"


def test_an_edited_settings_section_reruns_from_that_step(real_project, tmp_path):
    """Editing [assign], [connect] or [extract] in the file reruns from that step."""
    write_settings, pipeline = real_project
    pipeline()

    write_settings("[assign]\nmax_minutes = 30.0\n")
    run = pipeline()
    assert run["extract"] == run["connect"] == "up to date"
    assert run["assign"] == "running (settings changed: max_minutes None -> 30.0)"

    write_settings("[assign]\nmax_minutes = 30.0\n[connect]\nmin_nodes = 5\n")
    run = pipeline()
    assert run["extract"] == "up to date"
    assert run["connect"].startswith("running (settings changed: min_nodes")
    # assign's own settings are unchanged, but it reads what connect wrote.
    assert run["assign"].startswith("running")

    # The base settings end inside [extract], so its new key comes first.
    sections = "[assign]\nmax_minutes = 30.0\n[connect]\nmin_nodes = 5\n"
    write_settings("include_construction = false\n" + sections)
    links = tmp_path / "output" / "base" / "walk_links.gpkg"
    before = links.stat().st_mtime_ns
    # Marked so a reuse of the old file, rather than a rebuild, would show.
    from valma_bike_and_walk import links as links_module

    layer = links_module.read_links(links)
    layer["speed_override_kmh"] = 1.0
    links_module.write_links(layer, links)
    run = pipeline()
    assert run["extract"].startswith("running (settings changed: include_construction")
    # Really re-extracted from the PBF, not the existing layer reused as it stood.
    rebuilt = links_module.read_links(links)
    assert links.stat().st_mtime_ns != before
    assert rebuilt["speed_override_kmh"].isna().all()
    assert run["connect"].startswith("running") and run["assign"].startswith("running")

    # A comment, or a setting that only changes how fast it runs, reruns nothing.
    write_settings(
        "include_construction = false\n# a note\n"
        "[assign]\nmax_minutes = 30.0\nworkers = 3\n"
        "[connect]\nmin_nodes = 5\n"
    )
    statuses = set(pipeline().values()) - {"skipped (--skip)", "skipped (not configured)"}
    assert statuses == {"up to date"}

