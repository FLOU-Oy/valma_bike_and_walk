"""valma pipeline: which steps run, and which are skipped as up to date."""

import argparse
import os
from pathlib import Path

import numpy as np
import pytest

from valma_bike_and_walk.pipeline import build_plan, run_pipeline


class World:
    """Fake commands over real files, so stamps and in-place writes are real."""

    def __init__(
        self, root: Path, volumes: bool = True, baseline: bool = True, demand: bool = True
    ):
        self.root = root
        self.pbf = root / "finland.osm.pbf"
        self.volumes = root / "volumes.gpkg"
        self.zones = root / "zones.gpkg"
        self.baseline = root / "baseline.omx"
        self.demand = root / "demand.omx"
        self.links = root / "out" / "bike_links.gpkg"
        self.travel_times = root / "out" / "travel_times_bike.npz"
        self.overlaid = root / "out" / "overlaid.omx"
        for path in (self.pbf, self.volumes, self.zones, self.baseline, self.demand):
            path.write_text("v1")
        self.links.parent.mkdir()
        self.settings = {
            "dem_resolution": 10.0, "volumes": volumes, "baseline": baseline, "demand": demand
        }
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
        if command == "matrix":
            args.centroids, args.zones = None, self.zones
        if command == "overlay":
            if not self.settings["baseline"]:
                return None
            args.baseline, args.out = self.baseline, self.overlaid
        if command == "assign":
            if not self.settings["demand"]:
                return None
            args.centroids, args.zones, args.demand = None, self.zones, self.demand
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
        elif command == "matrix":
            self.travel_times.write_text("times")
        elif command == "overlay":
            self.overlaid.write_text("overlaid")
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


ALL = ["extract", "dem", "traffic", "connect", "matrix", "overlay", "assign"]


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
    assert world.go()[1] == ["traffic", "connect", "matrix", "overlay", "assign"]


def test_volumes_traffic_builds_itself_do_not_rerun_it_next_time(tmp_path):
    world = World(tmp_path)
    world.volumes.unlink()
    run = world.run

    def run_and_build(argv):
        if argv[0] == "traffic":
            world.volumes.write_text("built from assignment results")
        return run(argv)

    world.run = run_and_build
    assert world.go() == (0, ALL)
    assert world.go() == (0, [])


def test_rebuilding_volumes_on_every_traffic_run_does_not_loop(tmp_path):
    world = World(tmp_path)
    run = world.run
    builds = iter(range(100))

    def run_and_rebuild(argv):
        if argv[0] == "traffic":
            touch(world.volumes, f"rebuilt {next(builds)}")
        return run(argv)

    world.run = run_and_rebuild
    assert world.go() == (0, ALL)
    assert world.go() == (0, [])


def test_a_changed_setting_reruns_its_step_onwards(tmp_path):
    world = World(tmp_path)
    world.go()
    world.settings["dem_resolution"] = 2.0
    assert world.go()[1] == ["dem", "traffic", "connect", "matrix", "overlay", "assign"]


def test_new_zones_rerun_matrix_overlay_and_assign(tmp_path):
    world = World(tmp_path)
    world.go()
    touch(world.zones, "v2")
    assert world.go()[1] == ["matrix", "overlay", "assign"]


def test_a_new_baseline_reruns_overlay_only(tmp_path):
    world = World(tmp_path)
    world.go()
    touch(world.baseline, "v2")
    assert world.go()[1] == ["overlay"]


def test_new_demand_reruns_assign_only(tmp_path):
    world = World(tmp_path)
    world.go()
    touch(world.demand, "v2")
    assert world.go()[1] == ["assign"]


def test_a_hand_edit_of_the_links_reruns_what_reads_them_after(tmp_path):
    """An override typed in QGIS: connect, matrix and overlay, never extract over it."""
    world = World(tmp_path)
    world.go()
    touch(world.links, world.links.read_text() + "+edit")
    assert world.go()[1] == ["connect", "matrix", "overlay", "assign"]
    assert world.go()[1] == []


def test_a_missing_output_reruns_its_step(tmp_path):
    world = World(tmp_path)
    world.go()
    world.overlaid.unlink()
    assert world.go()[1] == ["overlay"]
    (tmp_path / "out" / "link_volumes_bike.npz").unlink()
    assert world.go()[1] == ["assign"]
    world.travel_times.unlink()
    assert world.go()[1] == ["matrix", "overlay"]
    world.links.unlink()
    assert world.go()[1] == ALL


def test_a_failure_stops_and_is_retried_next_time(tmp_path):
    world = World(tmp_path)
    world.fail = "traffic"
    assert world.go() == (1, ["extract", "dem", "traffic"])
    world.fail = None
    assert world.go()[1] == ["traffic", "connect", "matrix", "overlay", "assign"]


def test_unconfigured_traffic_is_left_out(tmp_path):
    world = World(tmp_path, volumes=False)
    assert world.go()[1] == ["extract", "dem", "connect", "matrix", "overlay", "assign"]
    assert world.go()[1] == []


def test_overlay_without_a_baseline_is_left_out(tmp_path):
    world = World(tmp_path, baseline=False)
    assert world.go()[1] == ["extract", "dem", "traffic", "connect", "matrix", "assign"]
    assert world.go()[1] == []


def test_assign_without_demand_is_left_out(tmp_path):
    world = World(tmp_path, demand=False)
    assert world.go()[1] == ["extract", "dem", "traffic", "connect", "matrix", "overlay"]
    assert world.go()[1] == []


def test_mark_done_adopts_existing_outputs(tmp_path):
    world = World(tmp_path)
    for command in ALL:  # as if each had been run by hand
        argv = ["connect", "--out", str(tmp_path / "out" / "bike_connectors.gpkg")]
        world.run(argv if command == "connect" else [command])
    assert world.go(mark_done=True) == (0, [])
    assert world.go()[1] == []
    touch(world.zones, "v2")
    assert world.go()[1] == ["matrix", "overlay", "assign"]


def test_dry_run_runs_nothing_and_records_nothing(tmp_path):
    world = World(tmp_path)
    assert world.go(dry_run=True) == (0, [])
    assert not (tmp_path / "out" / "pipeline_bike.json").exists()
    assert world.go()[1] == ALL


def test_skipped_steps_are_left_out(tmp_path):
    world = World(tmp_path)
    assert world.go(skip={"dem"})[1] == ["extract", "traffic", "connect", "matrix", "overlay", "assign"]


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
    """extract -> connect -> matrix -> assign through `valma pipeline`, on a tiny OSM file."""
    _, pipeline = real_project
    first = pipeline()
    assert first["traffic"] == "skipped (not configured)"
    assert first["overlay"] == "skipped (not configured)"
    ran = ("extract", "connect", "matrix", "assign")
    assert all(first[s].startswith("running") for s in ran)
    out = tmp_path / "output" / "base"
    assert (out / "walk_links.gpkg").exists() and (out / "travel_times_walk.npz").exists()
    assert (out / "link_volumes_walk.npz").exists()
    # The chain is in one piece, so connect leaves no file and matrix runs without.
    assert not (out / "walk_connectors.gpkg").exists()

    again = pipeline()
    assert {again[s] for s in ran} == {"up to date"}

    touch(tmp_path / "data" / "od.csv", "origin_id,destination_id,demand\n1,2,20\n")
    demand = pipeline()
    assert demand["matrix"] == "up to date"
    assert demand["assign"] == "running (inputs changed)"

    touch(tmp_path / "data" / "points.csv", "id,lon,lat\n1,24.000,60.000\n2,24.003,60.000\n")
    third = pipeline()
    assert third["extract"] == "up to date" and third["connect"] == "up to date"
    assert third["matrix"] == "running (inputs changed)"


def test_the_real_overlay_runs_after_matrix_once_configured(real_project, tmp_path):
    from valma_bike_and_walk.demand import write_omx_matrix
    from valma_bike_and_walk.overlay import read_omx_dense

    write_settings, pipeline = real_project
    write_omx_matrix(
        tmp_path / "data" / "baseline.omx",
        np.array([1, 2, 3]),
        np.full((3, 3), 99.0),
        matrix_name="walk",
    )
    out = tmp_path / "output" / "base" / "overlaid.omx"
    write_settings(
        '[overlay]\nbaseline = "../data/baseline.omx"\nbaseline_matrix = "walk"\n'
        'out = "../output/base/overlaid.omx"\n'
    )
    first = pipeline()
    assert first["matrix"].startswith("running") and first["overlay"].startswith("running")
    ids, minutes = read_omx_dense(out, "walk", "zone_number")
    assert list(ids) == [1, 2, 3]
    # Routed between the two points; zone 3 is not in the model, so its row stays.
    assert minutes[0, 1] < 99.0
    assert minutes[2, 2] == pytest.approx(99.0)
    assert pipeline()["overlay"] == "up to date"


def test_an_edited_settings_section_reruns_from_that_step(real_project, tmp_path):
    """Editing [matrix], [connect] or [extract] in the file reruns from that step."""
    write_settings, pipeline = real_project
    pipeline()

    write_settings("[matrix]\nmax_minutes = 30.0\n")
    run = pipeline()
    assert run["extract"] == run["connect"] == "up to date"
    assert run["matrix"] == "running (settings changed: max_minutes None -> 30.0)"

    write_settings("[matrix]\nmax_minutes = 30.0\n[connect]\nmin_nodes = 5\n")
    run = pipeline()
    assert run["extract"] == "up to date"
    assert run["connect"].startswith("running (settings changed: min_nodes")
    # matrix's own settings are unchanged, but it reads what connect wrote.
    assert run["matrix"].startswith("running")

    # The base settings end inside [extract], so its new key comes first.
    sections = "[matrix]\nmax_minutes = 30.0\n[connect]\nmin_nodes = 5\n"
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
    assert run["connect"].startswith("running") and run["matrix"].startswith("running")

    # A comment, or a setting that only changes how fast it runs, reruns nothing.
    write_settings(
        "include_construction = false\n# a note\n"
        "[matrix]\nmax_minutes = 30.0\nworkers = 3\n"
        "[connect]\nmin_nodes = 5\n"
    )
    statuses = set(pipeline().values()) - {"skipped (--skip)", "skipped (not configured)"}
    assert statuses == {"up to date"}

