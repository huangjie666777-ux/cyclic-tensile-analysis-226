"""Virtual extensometer for tensile tests, built on the subset DIC pipeline.

Each deformed frame is correlated independently against the same reference
image (no displacement accumulation).  Two user-supplied endpoints in
reference pixel coordinates define a gauge; per frame the endpoint
displacements are bilinearly interpolated from the four surrounding grid
points and only used when all four corners are valid.  Engineering strain is
the deformed endpoint distance over the original gauge length minus one;
engineering stress is force over the original cross-section area (MPa).
"""
from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass

import numpy as np

from cyclic_dic226.pipeline import Measurement, run_measurement
from cyclic_dic226.validation import (AnalysisParams, RequestError,
                                       decode_gray_png, grid_points,
                                       parse_float, validate_pair)

MIN_FRAMES = 2
MAX_FRAMES = 12
OFFSET_STRAIN = 0.002          # 0.2 percent offset yield rule
MIN_FIT_STRAINS = 3


@dataclass(frozen=True)
class CurveRow:
    frame_id: str
    time_s: float
    force_N: float


@dataclass(frozen=True)
class GaugeSpec:
    p1_px: tuple
    p2_px: tuple
    area_mm2: float
    fit_lo: float
    fit_hi: float


@dataclass
class FrameResult:
    row: CurveRow
    gauge_valid: bool
    length_mm: float
    strain: float
    stress_MPa: float
    measurement: Measurement


@dataclass
class FitResult:
    E_MPa: float
    b_MPa: float
    r_squared: float
    n_points: int


@dataclass
class YieldPoint:
    strain: float
    stress_MPa: float
    frame_id_before: str
    frame_id_after: str


@dataclass
class TensileResult:
    params: AnalysisParams
    spec: GaugeSpec
    curve: list
    frames: list
    gauge_length0_mm: float
    fit: object
    fit_reason: str
    yield_point: object
    yield_reason: str


def parse_curve_csv(data: bytes) -> list:
    """Parse and fully validate the frame/time/force CSV."""
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
    required = ("frame_id", "time_s", "force_N")
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
        out.append(CurveRow(fid, t, force))

    if not (MIN_FRAMES <= len(out) <= MAX_FRAMES):
        raise RequestError(
            f"curve CSV must list {MIN_FRAMES}..{MAX_FRAMES} frames, got {len(out)}")
    if len({r.frame_id for r in out}) != len(out):
        raise RequestError("frame_id values must be unique")
    for prev, cur in zip(out, out[1:]):
        if not cur.time_s > prev.time_s:
            raise RequestError("time_s must be strictly increasing")
    return out


def load_frames_zip(data: bytes, curve: list) -> dict:
    """Decode the frame PNGs; ZIP stems must match the CSV ids one-to-one."""
    if not data:
        raise RequestError("frames archive is empty")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise RequestError(f"frames archive is not a ZIP: {exc}") from exc
    raw = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        base = name.rsplit("/", 1)[-1]
        if not base.lower().endswith(".png"):
            continue
        stem = base[:-4]
        if stem in raw:
            raise RequestError(f"duplicate frame id in ZIP: '{stem}'")
        raw[stem] = zf.read(name)
    ids = {r.frame_id for r in curve}
    missing = ids - raw.keys()
    extra = raw.keys() - ids
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing PNGs for frame_id: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("PNGs without a CSV row: " + ", ".join(sorted(extra)))
        raise RequestError(
            "frames ZIP and curve CSV are not one-to-one (" + "; ".join(parts) + ")")
    return {fid: decode_gray_png(blob, f"frame '{fid}'")
            for fid, blob in raw.items()}


def build_gauge_spec(form: dict) -> GaugeSpec:
    required = ("area_mm2", "p1_x", "p1_y", "p2_x", "p2_y",
                "fit_strain_min", "fit_strain_max")
    missing = [k for k in required if form.get(k) in (None, "")]
    if missing:
        raise RequestError("missing fields: " + ", ".join(missing))
    area = parse_float("area_mm2", form["area_mm2"])
    if area <= 0:
        raise RequestError("area_mm2 must be positive")
    p1 = (parse_float("p1_x", form["p1_x"]), parse_float("p1_y", form["p1_y"]))
    p2 = (parse_float("p2_x", form["p2_x"]), parse_float("p2_y", form["p2_y"]))
    lo = parse_float("fit_strain_min", form["fit_strain_min"])
    hi = parse_float("fit_strain_max", form["fit_strain_max"])
    if lo < 0 or hi < 0:
        raise RequestError("fit strain bounds must be non-negative")
    if hi < lo:
        raise RequestError("fit_strain_max must be >= fit_strain_min")
    if p1 == p2:
        raise RequestError("extensometer endpoints must be distinct")
    return GaugeSpec(p1, p2, area, lo, hi)


def _cell_weights(axis: np.ndarray, value: float):
    i = int(np.searchsorted(axis, value, side="right")) - 1
    i = min(max(i, 0), axis.size - 2)
    t = (value - axis[i]) / (axis[i + 1] - axis[i])
    return i, float(t)


def _gauge_length_mm(m: Measurement, index: dict, xs: np.ndarray,
                     ys: np.ndarray, spec: GaugeSpec, scale: float):
    """Deformed endpoint distance in mm, or None when a corner is invalid."""
    deformed = []
    for (px, py) in (spec.p1_px, spec.p2_px):
        i, tx = _cell_weights(xs, px)
        j, ty = _cell_weights(ys, py)
        corners = (index[(round(float(xs[i]), 6), round(float(ys[j]), 6))],
                   index[(round(float(xs[i + 1]), 6), round(float(ys[j]), 6))],
                   index[(round(float(xs[i]), 6), round(float(ys[j + 1]), 6))],
                   index[(round(float(xs[i + 1]), 6), round(float(ys[j + 1]), 6))])
        if not all(m.valid[c] for c in corners):
            return None
        w = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)
        u = sum(wk * m.u_mm[c] for wk, c in zip(w, corners))
        v = sum(wk * m.v_mm[c] for wk, c in zip(w, corners))
        deformed.append((px * scale + u, py * scale + v))
    (x1, y1), (x2, y2) = deformed
    return float(np.hypot(x2 - x1, y2 - y1))


def fit_modulus(strains: np.ndarray, stresses: np.ndarray):
    """Least-squares sigma = E*eps + b; needs >= 3 distinct strains, E > 0."""
    if np.unique(strains).size < MIN_FIT_STRAINS:
        return None, ("fewer_than_3_distinct_strains_in_fit_interval "
                      f"(got {np.unique(strains).size})")
    A = np.column_stack([strains, np.ones(strains.size)])
    (E, b), *_ = np.linalg.lstsq(A, stresses, rcond=None)
    if not np.isfinite(E) or E <= 0:
        return None, "non_positive_modulus"
    resid = stresses - (E * strains + b)
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((stresses - stresses.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return FitResult(float(E), float(b), float(r2), int(strains.size)), ""


def find_yield(frames: list, fit: FitResult, fit_hi: float):
    """First + -> <=0 crossing of (measured - offset line) past the fit bound.

    Only adjacent (in sequence) gauge-valid frames with increasing strain are
    considered; invalid frames break adjacency and nothing is extrapolated.
    """
    prev = None
    saw_beyond = False
    for fr in frames:
        if not fr.gauge_valid or fr.strain <= fit_hi:
            prev = None
            continue
        saw_beyond = True
        d = fr.stress_MPa - (fit.E_MPa * (fr.strain - OFFSET_STRAIN) + fit.b_MPa)
        if prev is not None and fr.strain > prev[0].strain:
            d1, d2 = prev[1], d
            if d1 > 0 and d2 <= 0:
                t = d1 / (d1 - d2)
                eps_y = prev[0].strain + t * (fr.strain - prev[0].strain)
                sig_y = fit.E_MPa * (eps_y - OFFSET_STRAIN) + fit.b_MPa
                return YieldPoint(float(eps_y), float(sig_y),
                                  prev[0].row.frame_id, fr.row.frame_id), ""
        prev = (fr, d)
    if not saw_beyond:
        return None, "no_valid_gauge_points_beyond_fit_upper_bound"
    return None, "no_positive_to_nonpositive_crossing"


def run_tensile(ref: np.ndarray, frames_img: dict, curve: list,
                params: AnalysisParams, spec: GaugeSpec) -> TensileResult:
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

    first = frames_img[curve[0].frame_id]
    validate_pair(ref, first, params)
    frames = []
    for row in curve:
        img = frames_img[row.frame_id]
        if img.shape != ref.shape:
            raise RequestError(
                f"frame '{row.frame_id}' size {img.shape[1]}x{img.shape[0]} "
                f"differs from reference {ref.shape[1]}x{ref.shape[0]}")
        m = run_measurement(ref, img, gx, gy, params)
        length = _gauge_length_mm(m, index, xs, ys, spec,
                                  params.scale_mm_per_px)
        strain = length / length0 - 1.0 if length is not None else float("nan")
        frames.append(FrameResult(
            row=row,
            gauge_valid=length is not None,
            length_mm=length if length is not None else float("nan"),
            strain=strain,
            stress_MPa=row.force_N / spec.area_mm2,
            measurement=m,
        ))

    in_range = [fr for fr in frames
                if fr.gauge_valid and spec.fit_lo <= fr.strain <= spec.fit_hi]
    fit, fit_reason = fit_modulus(
        np.array([fr.strain for fr in in_range]),
        np.array([fr.stress_MPa for fr in in_range]))
    if fit is not None:
        yield_point, yield_reason = find_yield(frames, fit, spec.fit_hi)
    else:
        yield_point, yield_reason = None, "fit_failed: " + fit_reason
    return TensileResult(params, spec, curve, frames, length0,
                         fit, fit_reason, yield_point, yield_reason)
