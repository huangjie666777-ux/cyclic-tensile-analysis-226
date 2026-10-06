"""Cyclic load/unload analysis on top of the virtual extensometer.

The CSV protocol adds ``cycle_id`` and ``phase`` (load/unload) columns.  Each
cycle is a contiguous block: a load segment (force non-decreasing, at least
three frames, starting at zero force) followed by an unload segment (force
non-increasing, at least three frames, ending at zero force) with a positive
peak.  Any protocol violation rejects the whole request.

Per frame the same reference image is re-measured independently (no reference
reset, no displacement accumulation, no gap filling).  Per cycle we report
the peak stress, the permanent strain (last minus first frame strain), the
signed loop work density (trapezoidal integral of sigma d-epsilon in time
order, MPa = MJ/m^3, never sorted, never abs-valued) and an unloading
modulus fit over the closed 20-80 % peak-stress band.  The last load frame
is the unload starting point and is integrated exactly once.  The first
cycle with a successful fit is the modulus-retention baseline.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np

from cyclic_dic226.pipeline import run_measurement
from cyclic_dic226.tensile import (_gauge_length_mm, fit_modulus,
                                   load_frames_zip)
from cyclic_dic226.validation import (AnalysisParams, RequestError,
                                      grid_points, parse_float,
                                      validate_pair)

MIN_FRAMES = 6
MAX_FRAMES = 40
MIN_SEGMENT_FRAMES = 3
FIT_LO_FRAC = 0.2
FIT_HI_FRAC = 0.8
PHASES = ("load", "unload")


@dataclass(frozen=True)
class CyclicRow:
    frame_id: str
    time_s: float
    force_N: float
    cycle_id: str
    phase: str


@dataclass(frozen=True)
class CyclicSpec:
    p1_px: tuple
    p2_px: tuple
    area_mm2: float


@dataclass
class CyclicFrame:
    row: CyclicRow
    gauge_valid: bool
    length_mm: float
    strain: float
    stress_MPa: float
    measurement: object


@dataclass
class CycleResult:
    cycle_id: str
    frames: list
    peak_stress_MPa: float
    measured: bool
    measure_reason: str
    permanent_strain: object      # float or None
    loop_work_MJ_m3: object       # float or None
    fit: object                   # FitResult or None
    fit_reason: str
    retention: object             # float or None, filled after baseline


@dataclass
class CyclicResult:
    params: AnalysisParams
    spec: CyclicSpec
    rows: list
    frames: list
    cycles: list
    gauge_length0_mm: float
    baseline: object              # CycleResult or None
    baseline_reason: str


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
                f"phase must be one of {PHASES} (row {line_no}, got "
                f"'{row[idx['phase']].strip()}')")
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


def _validate_protocol(rows: list) -> None:
    """Contiguous cycles, load-then-unload segments, monotone forces."""
    cycles = []
    for row in rows:
        if cycles and cycles[-1][0] == row.cycle_id:
            cycles[-1][1].append(row)
        else:
            cycles.append((row.cycle_id, [row]))
    ids = [c for c, _ in cycles]
    if len(set(ids)) != len(ids):
        raise RequestError("rows of the same cycle_id must be contiguous")

    for cid, crows in cycles:
        label = f"cycle '{cid}'"
        phases = [r.phase for r in crows]
        first_unload = phases.index("unload") if "unload" in phases else len(phases)
        if "load" in phases[first_unload:]:
            raise RequestError(
                f"{label}: load frames must precede unload frames")
        n_load, n_unload = first_unload, len(phases) - first_unload
        if n_load < MIN_SEGMENT_FRAMES or n_unload < MIN_SEGMENT_FRAMES:
            raise RequestError(
                f"{label}: load and unload segments need at least "
                f"{MIN_SEGMENT_FRAMES} frames each (got {n_load}/{n_unload})")
        forces = [r.force_N for r in crows]
        for a, b in zip(forces[:n_load], forces[1:n_load]):
            if b < a:
                raise RequestError(f"{label}: load force must be non-decreasing")
        for a, b in zip(forces[n_load:], forces[n_load + 1:]):
            if b > a:
                raise RequestError(f"{label}: unload force must be non-increasing")
        if forces[0] != 0 or forces[-1] != 0:
            raise RequestError(
                f"{label}: first and last force of a cycle must be zero")
        if max(forces) <= 0:
            raise RequestError(f"{label}: peak force must be positive")


def group_cycles(rows: list) -> list:
    """Consecutive rows grouped as [(cycle_id, [rows])], order preserved."""
    cycles = []
    for row in rows:
        if cycles and cycles[-1][0] == row.cycle_id:
            cycles[-1][1].append(row)
        else:
            cycles.append((row.cycle_id, [row]))
    return cycles


def build_cyclic_spec(form: dict) -> CyclicSpec:
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
    return CyclicSpec(p1, p2, area)


def loop_work_MJ_m3(stresses: np.ndarray, strains: np.ndarray) -> float:
    """Signed trapezoidal integral of sigma d-epsilon in time order.

    MPa times dimensionless strain equals MJ/m^3.  No sorting, no abs.
    """
    work = 0.0
    for i in range(stresses.size - 1):
        work += 0.5 * (stresses[i] + stresses[i + 1]) * (strains[i + 1]
                                                         - strains[i])
    return float(work)


def run_cyclic(ref: np.ndarray, frames_img: dict, rows: list,
               params: AnalysisParams, spec: CyclicSpec) -> CyclicResult:
    first = frames_img[rows[0].frame_id]
    validate_pair(ref, first, params)
    gx, gy = grid_points(params)
    xs, ys = np.unique(gx), np.unique(gy)
    if xs.size < 2 or ys.size < 2:
        raise RequestError("grid is too small for gauge interpolation; "
                           "need at least 2x2 grid points")
    for label, (px, py) in (("p1", spec.p1_px), ("p2", spec.p2_px)):
        if not (xs[0] <= px <= xs[-1] and ys[0] <= py <= ys[-1]):
            raise RequestError(
                f"extensometer endpoint {label}=({px}, {py}) lies outside the "
                f"grid coverage x=[{xs[0]}, {xs[-1]}], y=[{ys[0]}, {ys[-1]}] px")
    index = {(round(float(gx[i]), 6), round(float(gy[i]), 6)): i
             for i in range(gx.size)}
    length0 = float(np.hypot(spec.p2_px[0] - spec.p1_px[0],
                             spec.p2_px[1] - spec.p1_px[1])
                    * params.scale_mm_per_px)

    frames = []
    for row in rows:
        img = frames_img[row.frame_id]
        if img.shape != ref.shape:
            raise RequestError(
                f"frame '{row.frame_id}' size {img.shape[1]}x{img.shape[0]} "
                f"differs from reference {ref.shape[1]}x{ref.shape[0]}")
        m = run_measurement(ref, img, gx, gy, params)
        length = _gauge_length_mm(m, index, xs, ys, spec,
                                  params.scale_mm_per_px)
        strain = length / length0 - 1.0 if length is not None else float("nan")
        frames.append(CyclicFrame(
            row=row,
            gauge_valid=length is not None,
            length_mm=length if length is not None else float("nan"),
            strain=strain,
            stress_MPa=row.force_N / spec.area_mm2,
            measurement=m,
        ))
    by_id = {fr.row.frame_id: fr for fr in frames}

    cycles = []
    for cid, crows in group_cycles(rows):
        cframes = [by_id[r.frame_id] for r in crows]
        peak = max(fr.stress_MPa for fr in cframes)
        bad = [fr.row.frame_id for fr in cframes if not fr.gauge_valid]
        if bad:
            measured, reason = False, "gauge_invalid_frames: " + ", ".join(bad)
            permanent = work = None
        else:
            measured, reason = True, ""
            permanent = float(cframes[-1].strain - cframes[0].strain)
            work = loop_work_MJ_m3(
                np.array([fr.stress_MPa for fr in cframes]),
                np.array([fr.strain for fr in cframes]))
        band = [fr for fr in cframes
                if fr.row.phase == "unload" and fr.gauge_valid
                and FIT_LO_FRAC * peak <= fr.stress_MPa <= FIT_HI_FRAC * peak]
        fit, fit_reason = fit_modulus(
            np.array([fr.strain for fr in band]),
            np.array([fr.stress_MPa for fr in band]))
        cycles.append(CycleResult(cid, cframes, peak, measured, reason,
                                  permanent, work, fit, fit_reason, None))

    baseline = next((c for c in cycles if c.fit is not None), None)
    if baseline is None:
        baseline_reason = "no_successful_unload_fit_in_any_cycle"
    else:
        baseline_reason = ""
        for c in cycles:
            if c.fit is not None:
                c.retention = float(c.fit.E_MPa / baseline.fit.E_MPa)
    return CyclicResult(params, spec, rows, frames, cycles, length0,
                        baseline, baseline_reason)
