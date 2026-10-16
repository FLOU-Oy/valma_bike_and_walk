"""``valma pipeline``: extract, dem, traffic, connect and assign, each only when needed.

Each step is run exactly as its own command would be, with the same settings
files, and is skipped when nothing it depends on has changed since it last
succeeded. What it depends on is recorded, after each success, in
``<output_dir>/pipeline_<mode>.json`` as a signature over:

- its own resolved settings (paths absolute; thread counts and the like left
  out, since they do not change the result) and the package version;
- the size and modification time of every input file from outside the
  pipeline: the OSM extract, the traffic volumes, the zones, the demand;
- the version of every pipeline file it read -- the link layer, the connectors.

The link layer is the subtle one, because three steps write it *in place*:
extract writes it, dem rewrites it, traffic rewrites it again. Its timestamp
alone cannot tell "dem has already run" from "the input dem saw has changed",
so each writer's version of the file is recorded, and a step reading it is
compared with the version the writer before it left. A rerun with nothing
changed then runs nothing, however many steps rewrote the file.

A link layer edited by hand after the pipeline wrote it -- a speed override
typed in QGIS -- is a new input for the steps that read it *after* its last
writer (connect and assign), so they run again. The steps before are not rerun:
extract would overwrite the edit.

A step also runs when one of its outputs is missing, and every step after one
that ran sees its new output and runs too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from valma_bike_and_walk.cache import file_stamp

logger = logging.getLogger(__name__)

#: In the order they run.
STEPS = ("extract", "dem", "traffic", "connect", "assign")

#: Settings that change how fast a step runs or how it reports, not its result.
_NOT_SIGNED = frozenset(
    {
        "workers",
        "dem_workers",
        "chunk_size",
        "verbose",
        "force_reload",
        "config",
        "show_config",
        "command",
        "func",
        "config_files",
    }
)

STATE_VERSION = 1


@dataclass
class Step:
    name: str
    #: The command line it runs with, after the settings files.
    argv: list[str]
    args: argparse.Namespace
    #: Pipeline files it reads, by name ("links", "connectors").
    reads: list[str] = field(default_factory=list)
    #: Pipeline files it writes, by name.
    writes: list[str] = field(default_factory=list)
    #: Input files from outside the pipeline.
    external: list[Path] = field(default_factory=list)
    #: Results that must exist for the step to count as done.
    outputs: list[Path] = field(default_factory=list)
    #: Pipeline files passed with this flag only when they exist when it runs.
    optional: dict[str, str] = field(default_factory=dict)
    #: Pipeline files it writes but may rightly not produce: removed before it
    #: runs, so a stale one from an earlier run is never mistaken for its result.
    fresh: list[str] = field(default_factory=list)


@dataclass
class Plan:
    steps: list[Step]
    files: dict[str, Path]
    state_path: Path
    skipped: dict[str, str]


def settings_of(args: argparse.Namespace) -> dict[str, Any]:
    """A step's settings as plain JSON values, without the ones that only affect speed."""
    out: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if key in _NOT_SIGNED or key.startswith("_") or callable(value):
            continue
        out[key] = _plain(value)
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def build_plan(
    resolve: Callable[[list[str]], argparse.Namespace | None],
    skip: set[str] = frozenset(),
) -> Plan:
    """
    Resolve every step's settings and wire their files together.

    ``resolve(argv)`` returns a command's arguments as the settings files and
    ``argv`` make them, or None when the command cannot run as configured.
    """
    assign = resolve(["assign"])
    if assign is None:
        raise ValueError("valma assign cannot run with these settings; see the error above.")
    links = Path(assign.links).resolve()
    mode = assign.mode
    connectors = (
        Path(assign.connectors).resolve()
        if assign.connectors is not None
        else links.with_name(f"{links.stem.removesuffix('_links')}_connectors.gpkg")
    )
    files = {"links": links, "connectors": connectors}
    skipped: dict[str, str] = {name: "--skip" for name in STEPS if name in skip}
    steps: list[Step] = []

    def add(name: str, argv: list[str], **kwargs: Any) -> argparse.Namespace | None:
        if name in skip:
            return None
        args = resolve(argv)
        if args is None:
            skipped[name] = "not configured"
            return None
        step = Step(name, argv, args, **kwargs)
        steps.append(step)
        return args

    # --force-reload: given an existing --out, extract reuses it as the user's
    # own layer. When the pipeline runs extract it is because the layer must
    # be rebuilt, so it has to say so.
    extract = add(
        "extract", ["extract", "--out", str(links), "--force-reload"], writes=["links"]
    )
    if extract is not None:
        if extract.pbf is None:
            raise ValueError("The extract step needs pbf (an OSM extract) in the settings.")
        steps[-1].external = [Path(extract.pbf)]

    add("dem", ["dem", "--links", str(links)], reads=["links"], writes=["links"])

    traffic = add("traffic", ["traffic", "--links", str(links)], reads=["links"], writes=["links"])
    if traffic is not None:
        steps[-1].external = [Path(traffic.volumes)]

    # A network already in one piece needs no connectors, and connect then
    # writes no file: assign is given them only if they exist when it runs.
    add(
        "connect",
        ["connect", "--links", str(links), "--out", str(connectors)],
        reads=["links"],
        writes=["connectors"],
        fresh=["connectors"],
    )
    final = add(
        "assign",
        ["assign", "--links", str(links)],
        reads=["links", "connectors"],
        optional={"connectors": "--connectors"},
    )
    if final is not None:
        external = [
            Path(p)
            for p in (
                getattr(final, "centroids", None),
                getattr(final, "zones", None),
                getattr(final, "demand", None),
                getattr(final, "weights", None),
            )
            if p is not None
        ]
        steps[-1].external = external
        steps[-1].outputs = [Path(final.output_dir) / f"link_volumes_{final.mode}.npz"]

    state_path = Path(assign.output_dir) / f"pipeline_{mode}.json"
    return Plan(steps, files, state_path, skipped)


def _load_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": STATE_VERSION, "steps": {}, "files": {}}
    if state.get("version") != STATE_VERSION:
        return {"version": STATE_VERSION, "steps": {}, "files": {}}
    state.setdefault("steps", {})
    state.setdefault("files", {})
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _stamp(path: Path) -> dict[str, Any] | None:
    return file_stamp(path) if Path(path).exists() else None


def _remove(path: Path) -> None:
    """Delete a pipeline output and the SQLite side files a GeoPackage may leave."""
    for target in (path, *(path.with_name(path.name + s) for s in ("-wal", "-shm", "-journal"))):
        target.unlink(missing_ok=True)


def _what_changed(before: dict[str, Any] | None, now: dict[str, Any]) -> str:
    """Name the settings that differ from the last run, or say it was an input."""
    if before is None:
        return "inputs or settings changed"
    changed = sorted(k for k in before.keys() | now.keys() if before.get(k) != now.get(k))
    if not changed:
        return "inputs changed"
    shown = ", ".join(f"{k} {before.get(k)!r} -> {now.get(k)!r}" for k in changed[:3])
    more = f" and {len(changed) - 3} more" if len(changed) > 3 else ""
    return f"settings changed: {shown}{more}"


def _signature(step: Step, read_versions: dict[str, Any], version: str) -> str:
    payload = {
        "version": version,
        "settings": settings_of(step.args),
        "external": [_stamp(p) or {"missing": str(p)} for p in step.external],
        "reads": read_versions,
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_pipeline(
    plan: Plan,
    run: Callable[[list[str]], int],
    version: str,
    force: bool = False,
    dry_run: bool = False,
    mark_done: bool = False,
    report: Callable[[str], None] = print,
) -> int:
    """
    Run the steps that need it, in order; returns the first failing exit code, or 0.

    ``mark_done`` records every step as done against the files as they are now,
    running nothing: for outputs made by hand-run commands before the pipeline
    kept track, so its first run does not redo them all.
    """
    state = _load_state(plan.state_path)
    files_state: dict[str, dict[str, Any]] = state["files"]  # name -> {writer: stamp}
    names = [s.name for s in plan.steps]
    order = {name: i for i, name in enumerate(STEPS)}

    def writers(file: str) -> list[str]:
        return [s.name for s in plan.steps if file in s.writes]

    # A file changed since its last writer left it was edited by hand: adopt
    # the edit as that writer's version, so whoever reads it after re-runs.
    for file, path in plan.files.items():
        recorded = files_state.get(file, {})
        last = [w for w in writers(file) if w in recorded]
        if not last or not path.exists():
            continue
        current = file_stamp(path)
        if current != recorded[last[-1]]:
            report(f"  {file}: {path.name} changed since {last[-1]} wrote it; "
                   "steps reading it after that will run")
            recorded[last[-1]] = current

    for name, why in plan.skipped.items():
        report(f"  {name}: skipped ({why})")

    rewritten: dict[str, str] = {}  # file -> step that wrote it in this run
    for step in plan.steps:
        read_versions: dict[str, Any] = {}
        for file in step.reads:
            before = [w for w in writers(file) if order[w] < order[step.name]]
            recorded = files_state.get(file, {})
            if before:
                read_versions[file] = recorded.get(before[-1])
            else:
                # Nothing in this pipeline writes it before this step: an input.
                read_versions[file] = _stamp(plan.files[file])
        signature = _signature(step, read_versions, version)
        previous = state["steps"].get(step.name, {})

        reason = None
        if force:
            reason = "--force"
        elif not previous:
            reason = "never run"
        else:
            for file in step.reads:
                if file in rewritten:
                    reason = f"{file} rewritten by {rewritten[file]}"
                    break
            missing = [p for p in step.outputs if not Path(p).exists()]
            # A file the step recorded producing is missing; one it recorded
            # not producing (no connectors needed) is as it left it.
            missing += [
                plan.files[f]
                for f in step.writes
                if not plan.files[f].exists()
                and files_state.get(f, {}).get(step.name) is not None
            ]
            if reason is None and missing:
                reason = f"{Path(missing[0]).name} missing"
            if reason is None and previous.get("signature") != signature:
                reason = _what_changed(previous.get("settings"), settings_of(step.args))

        if mark_done:
            # As a run would leave it: the files as they are now are this
            # step's output, and what it read is what the step before recorded.
            for file in step.writes:
                if plan.files[file].exists():
                    files_state.setdefault(file, {})[step.name] = file_stamp(plan.files[file])
            state["steps"][step.name] = {"signature": signature, "settings": settings_of(step.args)}
            report(f"  {step.name}: marked done")
            continue
        if reason is None:
            report(f"  {step.name}: up to date")
            continue
        if dry_run:
            report(f"  {step.name}: would run ({reason})")
            for file in step.writes:
                rewritten[file] = step.name
            continue

        report(f"  {step.name}: running ({reason})")
        for file in step.fresh:
            _remove(plan.files[file])
        argv = list(step.argv)
        for file, flag in step.optional.items():
            if plan.files[file].exists():
                argv += [flag, str(plan.files[file])]
        code = run(argv)
        if code != 0:
            report(f"  {step.name}: failed (exit code {code}); stopping")
            return code
        for file in step.writes:
            files_state.setdefault(file, {})[step.name] = _stamp(plan.files[file])
            rewritten[file] = step.name
        # Signed with what it read before it ran, so an edit saved meanwhile
        # still shows up next time.
        state["steps"][step.name] = {"signature": signature, "settings": settings_of(step.args)}
        _save_state(plan.state_path, state)

    if not dry_run:
        _save_state(plan.state_path, state)
    if mark_done:
        report(f"Recorded the current files as up to date in {plan.state_path}.")
        return 0
    done = ", ".join(names) if names else "nothing"
    report(f"Pipeline {'checked' if dry_run else 'done'}: {done}.")
    return 0
