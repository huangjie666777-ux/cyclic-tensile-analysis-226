"""Build the cyclic result ZIP: curve CSV, per-cycle CSV, frames, JSON."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np

from cyclic_dic226.cyclic import CyclicResult
from cyclic_dic226.delivery import _fmt, points_csv

CURVE_CSV = "curve.csv"
CYCLES_CSV = "cycles.csv"
RESULT_JSON = "result.json"
FRAMES_DIR = "frames"


def frame_csv_name(frame_id: str) -> str:
    return f"{FRAMES_DIR}/{frame_id}_points.csv"


def curve_csv(result: CyclicResult) -> str:
    header = ("frame_id,time_s,force_N,cycle_id,phase,stress_MPa,"
              "gauge_valid,length_mm,engineering_strain\n")
    lines = [header]
    for fr in result.frames:
        row = [
            fr.row.frame_id,
            _fmt(fr.row.time_s),
            _fmt(fr.row.force_N),
            fr.row.cycle_id,
            fr.row.phase,
            _fmt(fr.stress_MPa),
            int(fr.gauge_valid),
            _fmt(fr.length_mm),
            _fmt(fr.strain),
        ]
        lines.append(",".join(str(x) for x in row) + "\n")
    return "".join(lines)


def cycles_csv(result: CyclicResult) -> str:
    header = ("cycle_id,n_frames,peak_stress_MPa,residual_strain,"
              "residual_reason,loop_work_MJ_per_m3,loop_work_reason,"
              "unload_E_MPa,unload_b_MPa,unload_r_squared,unload_n_points,"
              "unload_fit_reason,modulus_retention\n")
    lines = [header]
    for c in result.cycles:
        fit = c.unload_fit
        row = [
            c.cycle_id,
            len(c.frames),
            _fmt(c.peak_stress_MPa),
            _fmt(c.residual_strain),
            c.residual_reason,
            _fmt(c.loop_work_MJ_per_m3),
            c.loop_work_reason.replace(",", ";"),
            _fmt(fit.E_MPa) if fit is not None else "",
            _fmt(fit.b_MPa) if fit is not None else "",
            _fmt(fit.r_squared) if fit is not None else "",
            fit.n_points if fit is not None else "",
            "" if fit is not None else c.unload_fit_reason.replace(",", ";"),
            _fmt(c.modulus_retention),
        ]
        lines.append(",".join(str(x) for x in row) + "\n")
    return "".join(lines)


def _cycle_json(c) -> dict:
    fit = c.unload_fit
    return {
        "cycle_id": c.cycle_id,
        "n_frames": len(c.frames),
        "frame_id_first": c.frames[0].row.frame_id,
        "frame_id_last": c.frames[-1].row.frame_id,
        "peak_stress_MPa": c.peak_stress_MPa,
        "residual_strain": c.residual_strain,
        **({} if c.residual_strain is not None
           else {"residual_reason": c.residual_reason}),
        "loop_work_MJ_per_m3": c.loop_work_MJ_per_m3,
        **({} if c.loop_work_MJ_per_m3 is not None
           else {"loop_work_reason": c.loop_work_reason}),
        "unload_fit": ({
            "model": "stress_MPa = E_MPa * strain + b_MPa",
            "stress_interval_MPa": list(c.fit_interval_MPa),
            "E_MPa": fit.E_MPa,
            "b_MPa": fit.b_MPa,
            "r_squared": fit.r_squared,
            "n_points": fit.n_points,
        } if fit is not None else {
            "stress_interval_MPa": list(c.fit_interval_MPa),
            "reason": c.unload_fit_reason,
        }),
        "modulus_retention": c.modulus_retention,
    }


def result_json(result: CyclicResult, params: dict) -> str:
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
                  "strain": "engineering, dimensionless",
                  "loop_work": "MJ/m^3 (MPa * strain, signed trapezoidal "
                               "integral in time order)"},
        "curve_csv": CURVE_CSV,
        "cycles_csv": CYCLES_CSV,
        "frames": [
            {
                "frame_id": fr.row.frame_id,
                "time_s": fr.row.time_s,
                "force_N": fr.row.force_N,
                "cycle_id": fr.row.cycle_id,
                "phase": fr.row.phase,
                "stress_MPa": fr.stress_MPa,
                "gauge_valid": fr.gauge_valid,
                "length_mm": (None if not np.isfinite(fr.length_mm)
                              else fr.length_mm),
                "engineering_strain": (None if not np.isfinite(fr.strain)
                                       else fr.strain),
                "n_valid_displacement": fr.measurement.n_valid,
                "n_points": fr.measurement.n_points,
                "points_csv": frame_csv_name(fr.row.frame_id),
            }
            for fr in result.frames
        ],
        "cycles": [_cycle_json(c) for c in result.cycles],
        "modulus_retention_baseline": ({
            "cycle_id": result.retention_baseline_cycle,
            "E_MPa": result.retention_baseline_E_MPa,
        } if result.retention_baseline_cycle is not None else {
            "value": None,
            "reason": result.retention_reason,
        }),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_cyclic_zip(result: CyclicResult, params: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(CURVE_CSV, curve_csv(result))
        zf.writestr(CYCLES_CSV, cycles_csv(result))
        for fr in result.frames:
            zf.writestr(frame_csv_name(fr.row.frame_id),
                        points_csv(fr.measurement,
                                   result.params.scale_mm_per_px))
        zf.writestr(RESULT_JSON, result_json(result, params))
    return buf.getvalue()
