"""Calibration from a dragged route: why the model prefers its route, and
which single parameter change would make it prefer yours.

Given the model's route M and an alternative A the user believes is right
(the model route dragged through via points), this module

1. splits each route's cost into its parts -- base riding time at the nominal
   speed, then the extra from the road-class + facility cost, surface,
   segregation and car traffic, and the signal, crossing and grade penalties;
2. finds, for every parameter that touches either route, the **break-even
   value**: the value at which A costs the same as M, all else held fixed.

The routes are held fixed while a parameter moves. A break-even value says A
would beat M -- not that A becomes the overall best route; a third one might.
And one route pair is an anecdote: saved observations exist so that several
can be weighed together.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from valma_bike_and_walk import links as links_module
from valma_bike_and_walk import speeds as speeds_module

_PATCH_LOCK = threading.Lock()


# -- what a route costs, part by part ------------------------------------


def _route_frame(links: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """The link rows a route uses, one per traversal, in route order."""
    frame = links.iloc[rows].drop(columns="geometry").reset_index(drop=True)
    return frame


def _times(frame: pd.DataFrame, forward: np.ndarray, mode: str, **drop) -> np.ndarray:
    """Travel time of each traversal, optionally with a penalty switched off."""
    frame = frame.copy()
    if drop.get("grade"):
        for column in ("ascent_m", "descent_m"):
            if column in frame.columns:
                frame[column] = 0.0
    if drop.get("signals"):
        for column in ("traffic_signal_at_start", "traffic_signal_at_end"):
            if column in frame.columns:
                frame[column] = False
        if "crossing" in frame.columns:
            frame.loc[frame["crossing"] == "traffic_signals", "crossing"] = None
    if drop.get("crossings") and "crossing" in frame.columns:
        signal = frame["crossing"] == "traffic_signals"
        frame.loc[~signal, "crossing"] = None
    fwd, rev = links_module.link_speeds_by_direction(frame, mode)
    frame["speed_kmh"], frame["speed_reverse_kmh"] = fwd, rev
    links_module._set_travel_times(frame, mode)
    return np.where(
        forward, frame["travel_time_s"].to_numpy(float), frame["travel_time_reverse_s"].to_numpy(float)
    )


def _factors(frame: pd.DataFrame, forward: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    """The multiplicative speed factors of each traversal (bike only)."""
    n = len(frame)
    profile = speeds_module.profile_for(mode)
    surface = frame.get("surface", pd.Series([None] * n))
    surface_f = surface.map(profile.surface_factor).astype(float).fillna(1.0).to_numpy()
    if mode != "bike":
        return {"surface": 1 / surface_f}
    seg_f = speeds_module.segregation_factor(frame["highway"], frame.get("segregated"))
    vol = pd.to_numeric(frame.get("car_volume", pd.Series([np.nan] * n)), errors="coerce").to_numpy()
    traffic = np.where(
        np.isfinite(vol),
        speeds_module.AVG_BASE_DISUTIL
        / (speeds_module.AVG_BASE_DISUTIL
           + speeds_module.TRAFFIC_COEFFICIENT * np.nan_to_num(vol) / speeds_module.AVG_ROUTE_LENGTH_KM),
        1.0,
    )
    fwd_cw, rev_cw = links_module._cycleway_by_direction(frame)
    text = np.where(
        forward,
        (fwd_cw if fwd_cw is not None else pd.Series([""] * n)).fillna("").to_numpy(str),
        (rev_cw if rev_cw is not None else pd.Series([""] * n)).fillna("").to_numpy(str),
    )
    parts = [set(p.strip() for p in t.split(";")) for t in text]
    facility = np.array([
        speeds_module.CYCLE_TRACK_FACTOR if "track" in p
        else speeds_module.CYCLE_LANE_FACTOR if "lane" in p
        else speeds_module.MIXED_TRAFFIC_FACTOR
        for p in parts
    ])
    road = frame["highway"].map(speeds_module.ROAD_CLASS_BETA).astype(float).to_numpy()
    divisor = np.where(
        np.isfinite(road),
        (np.nan_to_num(road) + speeds_module.BETA_SEPARATED * facility) / speeds_module.REFERENCE_DISUTIL,
        1.0,
    )
    return {"carriageway": divisor, "surface": 1 / surface_f, "segregation": 1 / seg_f,
            "traffic": 1 / traffic}


def decompose(links: pd.DataFrame, rows: np.ndarray, forward: np.ndarray, mode: str) -> dict:
    """Seconds of a route's cost by cause, plus what it is made of."""
    frame = _route_frame(links, rows)
    profile = speeds_module.profile_for(mode)
    length = frame["length_m"].to_numpy(float)
    nominal = frame["highway"].map(profile.highway_kph).astype(float).fillna(profile.base_kph).to_numpy()
    base = length / (nominal / 3.6)

    total = _times(frame, forward, mode)
    no_signals = _times(frame, forward, mode, signals=True)
    no_crossings = _times(frame, forward, mode, crossings=True)
    no_grade = _times(frame, forward, mode, grade=True)
    no_pen = _times(frame, forward, mode, signals=True, crossings=True, grade=True)
    moving = no_pen

    parts = {"base riding time": float(base.sum())}
    factors = _factors(frame, forward, mode)
    extra = moving - base
    logs = {k: np.log(v) for k, v in factors.items()}
    log_total = np.sum(list(logs.values()), axis=0) if logs else np.zeros_like(base)
    explained = np.zeros_like(base)
    for name, lg in logs.items():
        share = np.divide(lg, log_total, out=np.zeros_like(lg), where=np.abs(log_total) > 1e-12)
        part = extra * share
        explained += part
        parts[{"carriageway": "road class + facility", "surface": "surface",
               "segregation": "segregation", "traffic": "car traffic"}[name]] = float(part.sum())
    parts["speed limits / overrides"] = float((extra - explained).sum())
    parts["traffic signals"] = float((total - no_signals).sum())
    parts["unsignalled crossings"] = float((total - no_crossings).sum())
    parts["grade"] = float((total - no_grade).sum())

    by_highway = pd.Series(length, index=frame["highway"].astype(str)).groupby(level=0).sum()
    return {
        "cost_s": float(total.sum()),
        "length_m": float(length.sum()),
        "parts_s": parts,
        "by_highway_m": {k: float(v) for k, v in by_highway.sort_values(ascending=False).items()},
        "signal_links": int(((total - no_signals) > 0).sum()),
        "crossing_links": int(((total - no_crossings) > 0).sum()),
        "_frame": frame,
        "_forward": forward,
    }


# -- parameters and break-even values --------------------------------------


@dataclass
class Parameter:
    name: str
    get: Callable[[], float]
    set: Callable[[float], None]
    lo: float
    hi: float


def _module_param(module, attr: str, lo: float, hi: float, label: str | None = None) -> Parameter:
    def setter(value: float) -> None:
        setattr(module, attr, value)
    return Parameter(label or attr, lambda: float(getattr(module, attr)), setter, lo, hi)


def _road_beta_param(attr: str, highways: list[str]) -> Parameter:
    def setter(value: float) -> None:
        setattr(speeds_module, attr, value)
        for hw in highways:
            speeds_module.ROAD_CLASS_BETA[hw] = value  # type: ignore[index]
        speeds_module.REFERENCE_DISUTIL = speeds_module.BETA_MAIN + speeds_module.BETA_SEPARATED
    return Parameter(attr, lambda: float(getattr(speeds_module, attr)), setter, 0.5, 15.0)


def _dict_param(mapping: dict, key: str, label: str, lo: float, hi: float, default: float) -> Parameter:
    present = key in mapping

    def getter() -> float:
        return float(mapping.get(key, default))

    def setter(value: float) -> None:
        mapping[key] = value

    p = Parameter(label, getter, setter, lo, hi)
    p._restore_absent = not present  # type: ignore[attr-defined]
    p._mapping, p._key = mapping, key  # type: ignore[attr-defined]
    return p


def parameters(mode: str, frames: list[pd.DataFrame]) -> list[Parameter]:
    """The parameters that could matter for these routes."""
    profile = speeds_module.profile_for(mode)
    highways = sorted(set().union(*[set(f["highway"].astype(str)) for f in frames]))
    surfaces = sorted(set().union(*[set(f["surface"].dropna().astype(str)) for f in frames if "surface" in f]))
    light = "TRAFFIC_LIGHT_PENALTY_SECONDS" if mode == "bike" else "WALK_TRAFFIC_LIGHT_PENALTY_SECONDS"
    params = [
        _module_param(links_module, light, 0.0, 120.0),
        _module_param(links_module, "UNSIGNALLED_CROSSING_PENALTY_SECONDS", 0.0, 60.0),
        _module_param(links_module, "GRADE_LOW_FACTOR", 0.0, 60.0),
        _module_param(links_module, "GRADE_MODERATE_FACTOR", 0.0, 60.0),
        _module_param(links_module, "GRADE_STEEP_FACTOR", 0.0, 60.0),
    ]
    if mode == "bike":
        local = [h for h, b in speeds_module.ROAD_CLASS_BETA.items() if b == speeds_module.BETA_LOCAL]
        arterial = [h for h, b in speeds_module.ROAD_CLASS_BETA.items() if b == speeds_module.BETA_ARTERIAL]
        params += [
            _module_param(speeds_module, "MIXED_TRAFFIC_FACTOR", 0.5, 5.0),
            _module_param(speeds_module, "CYCLE_LANE_FACTOR", 0.5, 5.0),
            _module_param(speeds_module, "CYCLE_TRACK_FACTOR", 0.5, 5.0),
            _module_param(speeds_module, "TRAFFIC_COEFFICIENT", 0.0, 0.1),
            _road_beta_param("BETA_LOCAL", local),
            _road_beta_param("BETA_ARTERIAL", arterial),
        ]
    for hw in highways:
        params.append(_dict_param(profile.highway_kph, hw, f"speed of highway={hw} (km/h)", 0.5, 40.0, profile.base_kph))
    for sf in surfaces:
        params.append(_dict_param(profile.surface_factor, sf, f"surface factor {sf}", 0.1, 1.5, 1.0))
    return params


def _restore(param: Parameter, value: float) -> None:
    if getattr(param, "_restore_absent", False):
        param._mapping.pop(param._key, None)  # type: ignore[attr-defined]
    else:
        param.set(value)


def break_even(model: dict, alternative: dict, mode: str) -> list[dict]:
    """For each parameter: the value at which the alternative costs no more."""
    frames = [model["_frame"], alternative["_frame"]]

    def gap() -> float:  # alternative minus model, seconds
        return float(
            _times(alternative["_frame"], alternative["_forward"], mode).sum()
            - _times(model["_frame"], model["_forward"], mode).sum()
        )

    out = []
    with _PATCH_LOCK:
        base_gap = gap()
        for p in parameters(mode, frames):
            current = p.get()
            try:
                probe = current * 1.2 if current else (p.lo + p.hi) / 20 or 1.0
                p.set(min(max(probe, p.lo), p.hi))
                moved = gap()
                if abs(moved - base_gap) < 1e-6:
                    continue  # does not tell the routes apart
                grid = np.unique(np.concatenate([
                    np.linspace(p.lo, p.hi, 25),
                    [current] if p.lo <= current <= p.hi else [],
                ]))
                values = []
                for g in grid:
                    p.set(float(g))
                    values.append(gap())
                values = np.array(values)
                target = None
                order = np.argsort(np.abs(grid - current))  # nearest crossing first
                for i in order:
                    for j in (i - 1, i + 1):
                        if 0 <= j < len(grid) and (values[i] <= 0) != (values[j] <= 0):
                            lo, hi = sorted((i, j))
                            a, b = grid[lo], grid[hi]
                            for _ in range(30):
                                mid = (a + b) / 2
                                p.set(float(mid))
                                if (gap() <= 0) == (values[lo] <= 0):
                                    a = mid
                                else:
                                    b = mid
                            target = (a + b) / 2
                            break
                    if target is not None:
                        break
                out.append({
                    "parameter": p.name,
                    "current": current,
                    "break_even": None if target is None else float(target),
                    "change_pct": None if target is None or current == 0
                    else 100.0 * (target - current) / abs(current),
                    "gap_per_10pct_s": float(moved - base_gap) / 2.0,
                })
            finally:
                _restore(p, current)
    out.sort(key=lambda r: (r["change_pct"] is None, abs(r["change_pct"] or 0) if r["change_pct"] is not None else 0))
    return out


# -- the whole analysis, and a report ------------------------------------


def calibrate(router, links: pd.DataFrame, mode: str, points: list, vias: list) -> dict:
    """Model route between the end points vs the route through ``vias``."""
    model_edges, *_ = router.path_edges([points[0], points[-1]])
    alt_edges, *_ = router.path_edges([points[0], *vias, points[-1]])
    result = {"model": router.route_via([points[0], points[-1]]),
              "alternative": router.route_via([points[0], *vias, points[-1]])}

    def analyse(edges):
        rows = router.edge_rows[edges]
        keep = rows >= 0
        forward = router.network.direction[edges][keep] >= 0
        parts = decompose(links, rows[keep], forward, mode)
        # What the graph itself charges for the same links: if the tags no
        # longer reproduce it (code changed since the graph was built), every
        # number below would describe a different model than the routes do.
        parts["network_cost_s"] = float(router.network.travel_time[edges][keep].sum())
        return parts

    with _PATCH_LOCK:
        m = analyse(model_edges)
        a = analyse(alt_edges)
    result["break_even"] = break_even(m, a, mode)
    for key, d in (("model_parts", m), ("alternative_parts", a)):
        result[key] = {k: v for k, v in d.items() if not k.startswith("_")}
    result["points"] = {"origin": points[0], "destination": points[-1], "vias": vias}
    mismatch = max(
        abs(d["cost_s"] - d["network_cost_s"]) / max(d["network_cost_s"], 1e-9) for d in (m, a)
    )
    result["warning"] = (
        f"The link tags reproduce the routing graph's travel times only to within "
        f"{100 * mismatch:.0f} %: the graph was probably built with older code or "
        "settings. Rebuild it (assign --force-reload) before trusting this analysis."
        if mismatch > 0.01 else None
    )
    result["report"] = report(result, mode)
    return result


def report(result: dict, mode: str) -> str:
    """The analysis as Markdown, to save or to hand to someone (or Claude)."""
    m, a = result["model_parts"], result["alternative_parts"]
    pts = result["points"]
    fmt = lambda p: f"{p[1]:.5f}, {p[0]:.5f}"  # noqa: E731  lat, lon
    lines = [
        f"# Route calibration case ({mode})",
        "",
        *([f"> **Warning:** {result['warning']}", ""] if result.get("warning") else []),
        f"- Origin (lat, lon): {fmt(pts['origin'])}",
        f"- Destination (lat, lon): {fmt(pts['destination'])}",
        f"- Via points of the preferred route: " + ("; ".join(fmt(v) for v in pts["vias"]) or "none"),
        "",
        "| | Model route | Preferred route | Difference |",
        "|---|---:|---:|---:|",
        f"| Length | {m['length_m']/1000:.2f} km | {a['length_m']/1000:.2f} km | {(a['length_m']-m['length_m'])/1000:+.2f} km |",
        f"| Model cost | {m['cost_s']/60:.1f} min | {a['cost_s']/60:.1f} min | {(a['cost_s']-m['cost_s'])/60:+.1f} min |",
        f"| Signalised links | {m['signal_links']} | {a['signal_links']} | {a['signal_links']-m['signal_links']:+d} |",
        f"| Unsignalled crossing links | {m['crossing_links']} | {a['crossing_links']} | {a['crossing_links']-m['crossing_links']:+d} |",
        "",
        "## Cost by cause (seconds)",
        "",
        "| Cause | Model | Preferred | Difference |",
        "|---|---:|---:|---:|",
    ]
    for k in m["parts_s"]:
        mv, av = m["parts_s"][k], a["parts_s"].get(k, 0.0)
        if abs(mv) < 0.05 and abs(av) < 0.05:
            continue
        lines.append(f"| {k} | {mv:.0f} | {av:.0f} | {av - mv:+.0f} |")
    lines += ["", "## Road types (km)", "", "| highway | Model | Preferred |", "|---|---:|---:|"]
    for hw in sorted(set(m["by_highway_m"]) | set(a["by_highway_m"])):
        lines.append(f"| {hw} | {m['by_highway_m'].get(hw, 0)/1000:.2f} | {a['by_highway_m'].get(hw, 0)/1000:.2f} |")
    lines += [
        "",
        "## Single-parameter break-even values",
        "",
        "Value at which the preferred route costs no more than the model route, "
        "every other parameter unchanged (routes held fixed).",
        "",
        "| Parameter | Current | Break-even | Change |",
        "|---|---:|---:|---:|",
    ]
    for r in result["break_even"]:
        be = "not within range" if r["break_even"] is None else f"{r['break_even']:.4g}"
        ch = "" if r["change_pct"] is None else f"{r['change_pct']:+.0f} %"
        lines.append(f"| {r['parameter']} | {r['current']:.4g} | {be} | {ch} |")
    return "\n".join(lines)


def save_observation(directory: Path, result: dict, note: str = "") -> Path:
    """Append a case to ``route_observations.jsonl``, to calibrate many together."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "route_observations.jsonl"
    record = {
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "points": result["points"],
        "model": result["model_parts"],
        "preferred": result["alternative_parts"],
        "break_even": result["break_even"],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if math.isnan(value) else float(value)
    return str(value)
