"""Cyclic load/unload analysis on top of the virtual extensometer.

The CSV protocol (frame_id,time_s,force_N,cycle_id,phase) is validated as a
whole: frames of one cycle are contiguous, a load segment (force
non-decreasing) is followed by an unload segment (force non-increasing),
each segment has at least three frames, and every cycle starts and ends at
zero force with a positive peak.  Any violation rejects the entire request.

Per cycle the module reports the peak stress, the residual (permanent)
strain "last minus first", and the signed hysteresis loop work density from
a trapezoidal integral of sigma*d(eps) over time-adjacent frames (no
sorting, no absolute values; MPa*strain = MJ/m^3).  The unloading modulus is
least-squares fitted over the valid unload points whose stress lies in the
closed [20 %, 80 %] interval of the cycle peak stress; the last load frame
counts as the unload start.  The first cycle with a successful fit is the
modulus-retention baseline.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np

from cyclic_dic226.tensile import (fit_modulus, gauge_grid, measure_frames)
from cyclic_dic226.validation import (AnalysisParams, RequestError,
                                      parse_float)

MIN_FRAMES = 6
MAX_FRAMES = 40
MIN_SEGMENT_FRAMES = 3
PEAK_FRACTION_LO = 0.2
PEAK_FRACTION_HI = 0.8
PHASES = ("load", "unload")


@dataclass(frozen=True)
class CyclicRow:
    frame_id: str
    time_s: float
    force_N: float
    cycle_id: str
    phase: str


@dataclass(frozen=True)
class CyclicGaugeSpec:
    p1_px: tuple
    p2_px: tuple
    area_mm2: float


@dataclass
class CycleResult:
    cycle_id: str
    frames: list           # FrameResult, time order
    peak_stress_MPa: float
    residual_strain: object
    residual_reason: str
    loop_work_MJ_per_m3: object
    loop_work_reason: str
    unload_fit: object
    unload_fit_reason: str
    fit_interval_MPa: tuple
    modulus_retention: object = None


@dataclass
class CyclicResult:
    params: AnalysisParams
    spec: CyclicGaugeSpec
    rows: list
    frames: list           # all FrameResult, time order
    cycles: list           # CycleResult, cycle order
    gauge_length0_mm: float
    retention_baseline_cycle: object
    retention_baseline_E_MPa: object
    retention_reason: str


def parse_cyclic_csv(data: bytes) -> list:
    """Parse and fully validate the cyclic protocol CSV."""
    if not data:
        raise RequestError("curve CSV is empty")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RequestError(f"curve CSV is not valid UTF-8: {exc}") from exc
    rows = [r for r in csv.reader(io.StringIO(text))
            if any(cell.strip() for cell in r)]
    if not rows:
        raise RequestError("curve CSV is empty")
    header = [h.strip() for h in rows[0]]
    required = ("frame_id", "time_s", "force_N", "cycle_id", "phase")
    missing = [c for c in required if c not in header]
    if missing:
        raise RequestError("curve CSV missing columns: " + ", ".join(missing))
    idx = {c: header.index(c) for c in required}

    out = []
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) < len(header):
            raise RequestError(f"curve CSV row {line_no} has too few fields")
        fid = row[idx["frame_id"]].strip()
        if not fid:
            raise RequestError(f"curve CSV row {line_no} has an empty frame_id")
        if fid in (".", "..") or "/" in fid or "\\" in fid:
            raise RequestError(f"frame_id '{fid}' must be a plain file stem")
        t = parse_float(f"time_s (row {line_no})", row[idx["time_s"]].strip())
        force = parse_float(f"force_N (row {line_no})",
                            row[idx["force_N"]].strip())
        if force < 0:
            raise RequestError(f"force_N must be non-negative (row {line_no})")
        cid = row[idx["cycle_id"]].strip()
        if not cid:
            raise RequestError(f"curve CSV row {line_no} has an empty cycle_id")
        phase = row[idx["phase"]].strip().lower()
        if phase not in PHASES:
            raise RequestError(
                f"phase must be 'load' or 'unload' (row {line_no})")
        out.append(CyclicRow(fid, t, force, cid, phase))

    if not (MIN_FRAMES <= len(out) <= MAX_FRAMES):
        raise RequestError(
            f"curve CSV must list {MIN_FRAMES}..{MAX_FRAMES} frames, "
            f"got {len(out)}")
    if len({r.frame_id for r in out}) != len(out):
        raise RequestError("frame_id values must be unique")
    for prev, cur in zip(out, out[1:]):
        if not cur.time_s > prev.time_s:
            raise RequestError("time_s must be strictly increasing")
    _validate_protocol(out)
    return out


def group_cycles(rows: list) -> list:
    """Group time-ordered rows into (cycle_id, rows) contiguous blocks."""
    groups = []
    for row in rows:
        if groups and groups[-1][0] == row.cycle_id:
            groups[-1][1].append(row)
        else:
            groups.append((row.cycle_id, [row]))
    return groups


def _validate_protocol(rows: list) -> None:
    groups = group_cycles(rows)
    ids = [cid for cid, _ in groups]
    if len(set(ids)) != len(ids):
        raise RequestError("rows of the same cycle_id must be contiguous")
    for cid, block in groups:
        phases = [r.phase for r in block]
        n_load = phases.count("load")
        n_unload = phases.count("unload")
        if n_load < MIN_SEGMENT_FRAMES or n_unload < MIN_SEGMENT_FRAMES:
            raise RequestError(
                f"cycle '{cid}' needs at least {MIN_SEGMENT_FRAMES} load and "
                f"{MIN_SEGMENT_FRAMES} unload frames "
                f"(got {n_load} load, {n_unload} unload)")
        if phases != sorted(phases, key=PHASES.index):
            raise RequestError(
                f"cycle '{cid}' must list all load frames before unload frames")
        loads = [r.force_N for r in block if r.phase == "load"]
        unloads = [r.force_N for r in block if r.phase == "unload"]
        if any(b < a for a, b in zip(loads, loads[1:])):
            raise RequestError(
                f"cycle '{cid}' load force must be non-decreasing")
        if any(b > a for a, b in zip(unloads, unloads[1:])):
            raise RequestError(
                f"cycle '{cid}' unload force must be non-increasing")
        if block[0].force_N != 0 or block[-1].force_N != 0:
            raise RequestError(
                f"cycle '{cid}' must start and end at zero force")
        if max(r.force_N for r in block) <= 0:
            raise RequestError(f"cycle '{cid}' peak force must be positive")


def build_cyclic_gauge_spec(form: dict) -> CyclicGaugeSpec:
    required = ("area_mm2", "p1_x", "p1_y", "p2_x", "p2_y")
    missing = [k for k in required if form.get(k) in (None, "")]
    if missing:
        raise RequestError("missing fields: " + ", ".join(missing))
    area = parse_float("area_mm2", form["area_mm2"])
    if area <= 0:
        raise RequestError("area_mm2 must be positive")
    p1 = (parse_float("p1_x", form["p1_x"]), parse_float("p1_y", form["p1_y"]))
    p2 = (parse_float("p2_x", form["p2_x"]), parse_float("p2_y", form["p2_y"]))
    if p1 == p2:
        raise RequestError("extensometer endpoints must be distinct")
    return CyclicGaugeSpec(p1, p2, area)


def loop_work_MJ_per_m3(frames: list) -> float:
    """Signed trapezoidal integral of sigma*d(eps) in time order.

    MPa * dimensionless strain = MJ/m^3.  No sorting, no absolute values.
    """
    work = 0.0
    for a, b in zip(frames, frames[1:]):
        work += 0.5 * (a.stress_MPa + b.stress_MPa) * (b.strain - a.strain)
    return float(work)


def unload_fit_points(cycle_rows: list, frame_by_id: dict, peak_MPa: float):
    """Valid unload-branch points in the closed [20 %, 80 %] peak interval.

    The unload branch starts at the last load frame (the peak), which is a
    fit candidate here but is never integrated twice.
    """
    lo = PEAK_FRACTION_LO * peak_MPa
    hi = PEAK_FRACTION_HI * peak_MPa
    last_load = max(i for i, r in enumerate(cycle_rows) if r.phase == "load")
    pts = []
    for row in cycle_rows[last_load:]:
        fr = frame_by_id[row.frame_id]
        if fr.gauge_valid and lo <= fr.stress_MPa <= hi:
            pts.append(fr)
    return pts, (lo, hi)


def run_cyclic(ref: np.ndarray, frames_img: dict, rows: list,
               params: AnalysisParams, spec: CyclicGaugeSpec) -> CyclicResult:
    ctx = gauge_grid(params, spec.p1_px, spec.p2_px)
    frames = measure_frames(ref, frames_img, rows, params, spec, ctx)
    frame_by_id = {fr.row.frame_id: fr for fr in frames}

    cycles = []
    for cid, block in group_cycles(rows):
        cframes = [frame_by_id[r.frame_id] for r in block]
        peak = max(fr.stress_MPa for fr in cframes)

        first, last = cframes[0], cframes[-1]
        if first.gauge_valid and last.gauge_valid:
            residual = float(last.strain - first.strain)
            residual_reason = ""
        else:
            residual = None
            where = [w for w, ok in (("cycle_start", first.gauge_valid),
                                     ("cycle_end", last.gauge_valid))
                     if not ok]
            residual_reason = "gauge_invalid_at_" + "_and_".join(where)

        n_invalid = sum(1 for fr in cframes if not fr.gauge_valid)
        if n_invalid:
            work = None
            work_reason = (f"cycle has {n_invalid} frame(s) without a valid "
                           "gauge measurement; integral not filled in")
        else:
            work = loop_work_MJ_per_m3(cframes)
            work_reason = ""

        pts, interval = unload_fit_points(block, frame_by_id, peak)
        fit, fit_reason = fit_modulus(
            np.array([fr.strain for fr in pts]),
            np.array([fr.stress_MPa for fr in pts]))
        cycles.append(CycleResult(
            cycle_id=cid,
            frames=cframes,
            peak_stress_MPa=float(peak),
            residual_strain=residual,
            residual_reason=residual_reason,
            loop_work_MJ_per_m3=work,
            loop_work_reason=work_reason,
            unload_fit=fit,
            unload_fit_reason=fit_reason,
            fit_interval_MPa=interval,
        ))

    baseline = next((c for c in cycles if c.unload_fit is not None), None)
    if baseline is None:
        base_cycle = base_E = None
        retention_reason = "no cycle with a successful unload fit"
    else:
        base_cycle = baseline.cycle_id
        base_E = baseline.unload_fit.E_MPa
        retention_reason = ""
        for c in cycles:
            if c.unload_fit is not None:
                c.modulus_retention = float(c.unload_fit.E_MPa / base_E)
    return CyclicResult(params, spec, rows, frames, cycles, ctx.length0_mm,
                        base_cycle, base_E, retention_reason)
