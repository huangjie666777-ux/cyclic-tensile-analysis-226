"""Build the tensile result ZIP: curve CSV, per-frame CSVs, result JSON."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np

from cyclic_dic226.delivery import _fmt, points_csv
from cyclic_dic226.tensile import OFFSET_STRAIN, TensileResult

CURVE_CSV = "curve.csv"
RESULT_JSON = "result.json"
FRAMES_DIR = "frames"


def frame_csv_name(frame_id: str) -> str:
    return f"{FRAMES_DIR}/{frame_id}_points.csv"


def curve_csv(result: TensileResult) -> str:
    fit = result.fit
    header = ("frame_id,time_s,force_N,stress_MPa,gauge_valid,length_mm,"
              "engineering_strain,in_fit_interval,fit_line_MPa,"
              "offset_line_MPa,stress_minus_offset_MPa\n")
    lines = [header]
    for fr in result.frames:
        in_fit = (fr.gauge_valid and fit is not None
                  and result.spec.fit_lo <= fr.strain <= result.spec.fit_hi)
        fit_line = offset_line = diff = ""
        if fit is not None and fr.gauge_valid:
            fit_line = _fmt(fit.E_MPa * fr.strain + fit.b_MPa)
            offset = fit.E_MPa * (fr.strain - OFFSET_STRAIN) + fit.b_MPa
            offset_line = _fmt(offset)
            diff = _fmt(fr.stress_MPa - offset)
        row = [
            fr.row.frame_id,
            _fmt(fr.row.time_s),
            _fmt(fr.row.force_N),
            _fmt(fr.stress_MPa),
            int(fr.gauge_valid),
            _fmt(fr.length_mm),
            _fmt(fr.strain),
            int(in_fit),
            fit_line,
            offset_line,
            diff,
        ]
        lines.append(",".join(str(x) for x in row) + "\n")
    return "".join(lines)


def result_json(result: TensileResult, params: dict) -> str:
    fit = result.fit
    yp = result.yield_point
    payload = {
        "parameters": params,
        "gauge": {
            "p1_px": list(result.spec.p1_px),
            "p2_px": list(result.spec.p2_px),
            "length0_mm": result.gauge_length0_mm,
            "interpolation": "bilinear over the 4 surrounding grid points; "
                             "frame gauge invalid unless all 4 corners valid",
        },
        "units": {"length": "mm", "stress": "MPa (N/mm^2)",
                  "strain": "engineering, dimensionless"},
        "curve_csv": CURVE_CSV,
        "frames": [
            {
                "frame_id": fr.row.frame_id,
                "time_s": fr.row.time_s,
                "force_N": fr.row.force_N,
                "stress_MPa": fr.stress_MPa,
                "gauge_valid": fr.gauge_valid,
                "length_mm": None if not np.isfinite(fr.length_mm) else fr.length_mm,
                "engineering_strain": None if not np.isfinite(fr.strain) else fr.strain,
                "n_valid_displacement": fr.measurement.n_valid,
                "n_points": fr.measurement.n_points,
                "points_csv": frame_csv_name(fr.row.frame_id),
            }
            for fr in result.frames
        ],
        "fit": ({
            "model": "stress_MPa = E_MPa * strain + b_MPa",
            "interval": [result.spec.fit_lo, result.spec.fit_hi],
            "E_MPa": fit.E_MPa,
            "b_MPa": fit.b_MPa,
            "r_squared": fit.r_squared,
            "n_points": fit.n_points,
        } if fit is not None else {
            "interval": [result.spec.fit_lo, result.spec.fit_hi],
            "reason": result.fit_reason,
        }),
        "yield": ({
            "method": "0.2% offset: stress = E*(strain - 0.002) + b",
            "strain": yp.strain,
            "stress_MPa": yp.stress_MPa,
            "frame_id_before": yp.frame_id_before,
            "frame_id_after": yp.frame_id_after,
        } if yp is not None else {
            "method": "0.2% offset: stress = E*(strain - 0.002) + b",
            "value": None,
            "reason": result.yield_reason,
        }),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_tensile_zip(result: TensileResult, params: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(CURVE_CSV, curve_csv(result))
        for fr in result.frames:
            zf.writestr(frame_csv_name(fr.row.frame_id),
                        points_csv(fr.measurement,
                                   result.params.scale_mm_per_px))
        zf.writestr(RESULT_JSON, result_json(result, params))
    return buf.getvalue()
