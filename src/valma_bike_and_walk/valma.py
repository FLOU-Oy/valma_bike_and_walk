"""
Running this package from inside VALMA (valma-model-system), in process.

VALMA makes two calls, and everything else stays on this side:

- :func:`skims` once, after EMME has made its own walk and bike skims for the
  ``pt`` period. Our times and distances are written over EMME's wherever we
  routed a pair, and EMME's are kept everywhere else -- external zones, zones
  that did not snap, pairs beyond the cutoff -- so VALMA gets back complete
  matrices in its own zone order and units (minutes, km). The diagonal comes
  from us too: zone polygons give every zone a routed intrazonal distance,
  where EMME leaves 0.
- :func:`assign` once, at the end of the run, on the demand files VALMA has
  just written. It writes one volumes GeoPackage per mode, a column per
  period, and returns where they went and the person-km each period loaded.

Settings come from a directory holding ``walk.toml`` and ``bike.toml`` (and
whatever they ``extends``), exactly as ``valma -c`` reads them. VALMA passes
what it owns -- its zone data file, the submodel column naming its zones, and
where results go -- as command-line overrides, so the two zone systems cannot
drift apart however the settings files are written.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openmatrix as omx

from valma_bike_and_walk.cli import _parse, execute
from valma_bike_and_walk.overlay import overlay, read_model_km, read_model_minutes

logger = logging.getLogger(__name__)

MODES = ("walk", "bike")

# VALMA's matrix types, and how each is read from a `valma matrix --distance` run.
_QUANTITIES = {"time": read_model_minutes, "dist": read_model_km}


@dataclass(frozen=True)
class Setup:
    """What VALMA tells this package about a run."""

    config_dir: Path
    """Holds ``walk.toml`` and ``bike.toml``."""
    zone_data_file: Path
    """VALMA's ``--zone-data-file``: zone polygons, one row per data zone."""
    submodel: str
    """VALMA's ``--submodel``: the column naming each row's assignment zone."""
    output_dir: Path
    """Where this package's results go, e.g. ``<results>/<scenario>/bike_and_walk``."""
    workers: int | None = None
    """Routing threads; ``None`` leaves it to the settings files."""

    def settings(self, mode: str) -> list[Path]:
        path = Path(self.config_dir) / f"{mode}.toml"
        if not path.is_file():
            raise FileNotFoundError(
                f"No settings for {mode}: expected {path} "
                f"(the bike-and-walk config directory needs walk.toml and bike.toml)"
            )
        return [path]

    def overrides(self, mode: str, output_dir: Path | None = None) -> list[str]:
        argv = [
            "--mode", mode,
            "--zones", str(self.zone_data_file),
            "--id-column", self.submodel,
            "--output-dir", str(output_dir or self.output_dir),
        ]  # fmt: skip
        if self.workers is not None:
            argv += ["--workers", str(self.workers)]
        return argv


def _run(command: str, argv: list[str], files: list[Path]):
    """A command run as ``valma <command> -c <files> <argv>`` would, returning its arguments."""
    parsed = _parse([command, *argv], files)
    code = execute(parsed)
    if code != 0:
        raise RuntimeError(
            f"valma {command} failed (exit code {code}); see the log above."
        )
    return parsed.args


def _cutoffs(setup: Setup, mode: str) -> dict[str, float | None]:
    """The ``[overlay]`` cutoffs from the settings files, per VALMA matrix type."""
    # Parsed as the overlay command would see them; the baseline is VALMA's
    # matrices in memory and the result goes back to it, so the files named
    # here are never opened.
    args = _parse(
        [
            "overlay",
            "--mode",
            mode,
            "--baseline",
            "-",
            "--baseline-matrix",
            mode,
            "--out",
            "-",
        ],
        setup.settings(mode),
    ).args
    return {"time": args.max_minutes, "dist": args.max_km}


@functools.cache
def _routed(setup: Setup, mode: str) -> Path:
    """
    Route ``mode`` over VALMA's zones, once per process.

    VALMA asks for skims once per period that has walk and bike LOS -- one
    with EMME, every period with the mock assignment -- and the routing is
    the same each time; only the baseline it is laid over differs.
    """
    args = _run("matrix", [*setup.overrides(mode), "--distance"], setup.settings(mode))
    return args.output_dir / f"travel_times_{mode}.npz"


def _matrices_in(path: Path) -> set[str]:
    with omx.open_file(str(path), "r") as f:
        return set(f.list_matrices())


def skims(
    setup: Setup,
    zone_numbers: Iterable[int],
    baseline: Mapping[str, Mapping[str, np.ndarray]],
    modes: Iterable[str] = MODES,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Walk and bike skims over VALMA's zone system, ours wherever we routed.

    ``baseline[type][mode]`` is VALMA's own matrix -- ``type`` is ``"time"``
    (minutes) or ``"dist"`` (km) -- square over ``zone_numbers``, which is how
    VALMA holds one period's impedance. The result has the same layout, the
    same zone order and the same units, for every ``type`` and mode given in
    ``baseline``; it is new arrays, and ``baseline`` is not modified.

    Distances are measured along the route the time chose, so the two describe
    one journey. Cells keep the baseline where we have no finite value, or one
    beyond the ``[overlay]`` section's ``max_minutes`` / ``max_km``.
    """
    zone_numbers = np.asarray(zone_numbers)
    square = (zone_numbers.size, zone_numbers.size)
    for kind, by_mode in baseline.items():
        for mode, base in by_mode.items():
            if np.shape(base) != square:
                raise ValueError(
                    f"Baseline {kind} {mode} is {np.shape(base)}, "
                    f"not square over {zone_numbers.size} zones."
                )

    result: dict[str, dict[str, np.ndarray]] = {}
    for mode in modes:
        wanted = [kind for kind in _QUANTITIES if mode in baseline.get(kind, {})]
        if not wanted:
            continue
        cutoffs = _cutoffs(setup, mode)
        model_path = _routed(setup, mode)
        for kind in wanted:
            base = np.asarray(baseline[kind][mode])
            model_ids, model = _QUANTITIES[kind](model_path)
            matrix, replaced = overlay(
                zone_numbers, base, model_ids, model, cutoff=cutoffs[kind]
            )
            logger.info(
                "%s %s: %d of %d cells (%.1f %%) from valma_bike_and_walk",
                mode, kind, replaced, matrix.size, 100.0 * replaced / matrix.size,
            )  # fmt: skip
            result.setdefault(kind, {})[mode] = matrix
    return result


@dataclass(frozen=True)
class ModeVolumes:
    """One mode's assignment of every demand file VALMA handed over."""

    gpkg: Path
    """``<mode>_volumes.gpkg``: one row per directed edge, ``volume_<label>`` each."""
    npz: Path
    """``link_volumes_<mode>.npz``: every edge, loaded or not, same columns."""
    kilometres: dict[str, float]
    """Demand times distance on the network, per label: person-km loaded."""
    trips: dict[str, float]
    """The demand read from each label's file, as VALMA wrote it."""


def assign(
    setup: Setup,
    demand: Mapping[str, Path],
    modes: Iterable[str] = MODES,
    all_links: bool = False,
) -> dict[str, ModeVolumes]:
    """
    Assign VALMA's walk and bike demand, and draw the volumes on the links.

    ``demand`` maps a label -- a time period such as ``"aht"`` or ``"vrk"`` --
    to an OMX file holding one matrix per mode, named after it, with a
    ``zone_number`` mapping: VALMA's own ``demand_<period>.omx``. A mode a file
    has no matrix for is skipped there -- VALMA's transit-only period has no
    walk or bike demand.

    Each mode's network and zones are loaded once and every label assigned on
    them, and the results go into one GeoPackage per mode with a
    ``volume_<label>`` column each -- not a file per label, which would re-read
    and re-write a national link layer every time. Only edges that carry
    something in some label are written, unless ``all_links``; the ``.npz``
    beside it always has every edge.
    """
    from valma_bike_and_walk.assignment import (
        assign_traffic,
        assign_zone_traffic,
        link_volume_frame,
    )
    from valma_bike_and_walk.cli import (
        _analysis_points,
        _assign_demand,
        _link_layer,
        _network_and_links,
        _record,
    )
    from valma_bike_and_walk.connectors import island_link_ids
    from valma_bike_and_walk.gpkg import write_edges_gpkg
    from valma_bike_and_walk.links import read_links
    from valma_bike_and_walk.zones import Zones

    present = {label: _matrices_in(path) for label, path in demand.items()}
    result: dict[str, ModeVolumes] = {}
    for mode in modes:
        labels = [label for label in demand if mode in present[label]]
        for label in demand:
            if label not in labels:
                logger.info("No %s demand in %s; not assigned", mode, demand[label])
        if not labels:
            continue

        argv = [
            *setup.overrides(mode),
            "--demand", str(demand[labels[0]]),
            "--demand-matrix", mode,
            "--gpkg",
        ]  # fmt: skip
        parsed = _parse(["assign", *argv], setup.settings(mode))
        args = parsed.args
        network, links_path = _network_and_links(args)
        points = _analysis_points(args, network)
        if len(points) == 0:
            raise RuntimeError(f"No {mode} zone snapped to the network.")
        max_seconds = args.max_minutes * 60 if args.max_minutes else None

        volumes: dict[str, np.ndarray] = {}
        trips: dict[str, float] = {}
        for label in labels:
            args.demand = Path(demand[label])
            matrix = _assign_demand(args, points)
            trips[label] = float(matrix.sum())
            logger.info("Assigning %s %s: %.6g trips", mode, label, trips[label])
            if isinstance(points, Zones):
                volume = assign_zone_traffic(
                    network,
                    points,
                    matrix,
                    max_seconds=max_seconds,
                    near_seconds=args.near_minutes * 60 if args.near_minutes else None,
                    workers=args.workers,
                    chunk_size=args.chunk_size,
                )
            else:
                volume = assign_traffic(
                    network,
                    points.node_index,
                    matrix,
                    max_seconds=max_seconds,
                    workers=args.workers,
                    chunk_size=args.chunk_size,
                )
            volumes[f"volume_{label}"] = volume

        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame = link_volume_frame(network, next(iter(volumes.values())))
        frame.pop("volume", None)
        npz = args.output_dir / f"link_volumes_{mode}.npz"
        np.savez_compressed(npz, **frame, **volumes)

        # Island crossings stand in for nothing anyone travels on, so they are
        # left off the map as `valma assign --gpkg` leaves them.
        layer = read_links(args.links) if args.links else links_path
        edge_mask = np.ones(network.n_edges, dtype=bool)
        if args.connectors is not None:
            layer = _link_layer(args)
            crossings = island_link_ids(read_links(args.connectors))
            edge_mask &= ~np.isin(network.link_id, crossings)
        if not all_links:
            edge_mask &= np.logical_or.reduce([v > 0 for v in volumes.values()])
        gpkg = args.output_dir / f"{mode}_volumes.gpkg"
        write_edges_gpkg(
            network, layer, gpkg, extra_columns=volumes, edge_mask=edge_mask
        )
        _record(args, parsed.parser, "assign", parsed.files)

        length_km = network.length.astype(np.float64) / 1000.0
        kilometres = {
            column.removeprefix("volume_"): float((v * length_km).sum())
            for column, v in volumes.items()
        }
        logger.info(
            "Wrote %s: %d edge(s), %s",
            gpkg,
            int(edge_mask.sum()),
            ", ".join(f"{k} {v:,.0f} km" for k, v in kilometres.items()),
        )
        result[mode] = ModeVolumes(gpkg, npz, kilometres, trips)
    return result
