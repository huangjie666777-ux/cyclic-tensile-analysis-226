"""FastAPI application: upload two 8-bit grayscale PNGs and analysis form."""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from cyclic_dic226.delivery import build_zip
from cyclic_dic226.pipeline import run_measurement
from cyclic_dic226.tensile import (build_gauge_spec, load_frames_zip,
                                    parse_curve_csv, run_tensile)
from cyclic_dic226.tensile_delivery import build_tensile_zip
from cyclic_dic226.validation import (MAX_EDGE, MAX_GRID_POINTS,
                                          MAX_ITERATIONS, MAX_SEARCH_RADIUS,
                                          RequestError, build_params,
                                          decode_gray_png, grid_points,
                                          validate_pair)

app = FastAPI(
    title="cyclic_dic226",
    description="2D subset DIC displacement and Green-Lagrange strain backend",
    version="0.1.0",
)


LIMITS = {
    "max_edge_px": MAX_EDGE,
    "max_grid_points": MAX_GRID_POINTS,
    "max_search_radius_px": MAX_SEARCH_RADIUS,
    "max_iterations": MAX_ITERATIONS,
}


@app.get("/health")
def health():
    return {"status": "ok", "limits": LIMITS}


async def _prepare(reference: UploadFile, deformed: UploadFile, form: dict):
    try:
        params = build_params(form)
        ref = decode_gray_png(await reference.read(), "reference")
        defo = decode_gray_png(await deformed.read(), "deformed")
        validate_pair(ref, defo, params)
        gx, gy = grid_points(params)
    except RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    measurement = run_measurement(ref, defo, gx, gy, params)
    form_dump = {k: form.get(k) for k in (
        "scale_mm_per_px", "roi_x", "roi_y", "roi_w", "roi_h",
        "subset_size", "grid_step", "search_radius", "max_iterations")}
    return measurement, params, form_dump


@app.post("/analyze", summary="Run DIC and return per-point JSON")
async def analyze(
    reference: UploadFile = File(..., description="reference 8-bit gray PNG"),
    deformed: UploadFile = File(..., description="deformed 8-bit gray PNG"),
    scale_mm_per_px: str = Form(...),
    roi_x: str = Form(...),
    roi_y: str = Form(...),
    roi_w: str = Form(...),
    roi_h: str = Form(...),
    subset_size: str = Form(...),
    grid_step: str = Form(...),
    search_radius: str = Form(...),
    max_iterations: str = Form(...),
):
    form = dict(scale_mm_per_px=scale_mm_per_px, roi_x=roi_x, roi_y=roi_y,
                roi_w=roi_w, roi_h=roi_h, subset_size=subset_size,
                grid_step=grid_step, search_radius=search_radius,
                max_iterations=max_iterations)
    m, params, form_dump = await _prepare(reference, deformed, form)

    def cell(a):
        return None if not np.isfinite(a) else float(a)

    points = []
    for i in range(m.n_points):
        points.append({
            "index": i,
            "grid_x_mm": cell(m.grid_x_px[i] * params.scale_mm_per_px),
            "grid_y_mm": cell(m.grid_y_px[i] * params.scale_mm_per_px),
            "valid": bool(m.valid[i]),
            "u_mm": cell(m.u_mm[i]),
            "v_mm": cell(m.v_mm[i]),
            "zncc": cell(m.correlation[i]),
            "converged": bool(m.converged[i]),
            "iterations": int(m.iterations[i]),
            "exx": cell(m.exx[i]),
            "eyy": cell(m.eyy[i]),
            "exy": cell(m.exy[i]),
            "failure_reason": m.reasons[i],
        })
    return {
        "parameters": form_dump,
        "limits": LIMITS,
        "n_points": m.n_points,
        "n_valid_displacement": m.n_valid,
        "n_valid_strain": int(np.isfinite(m.exx).sum()),
        "points": points,
}


@app.post("/download", summary="Run DIC and download the result ZIP")
async def download(
    reference: UploadFile = File(...),
    deformed: UploadFile = File(...),
    scale_mm_per_px: str = Form(...),
    roi_x: str = Form(...),
    roi_y: str = Form(...),
    roi_w: str = Form(...),
    roi_h: str = Form(...),
    subset_size: str = Form(...),
    grid_step: str = Form(...),
    search_radius: str = Form(...),
    max_iterations: str = Form(...),
):
    form = dict(scale_mm_per_px=scale_mm_per_px, roi_x=roi_x, roi_y=roi_y,
                roi_w=roi_w, roi_h=roi_h, subset_size=subset_size,
                grid_step=grid_step, search_radius=search_radius,
                max_iterations=max_iterations)
    m, params, form_dump = await _prepare(reference, deformed, form)
    archive = build_zip(m, params.scale_mm_per_px, form_dump)
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition":
                 'attachment; filename="cyclic226_result.zip"'},
    )


TENSILE_FIELDS = ("scale_mm_per_px", "roi_x", "roi_y", "roi_w", "roi_h",
                  "subset_size", "grid_step", "search_radius", "max_iterations",
                  "area_mm2", "p1_x", "p1_y", "p2_x", "p2_y",
                  "fit_strain_min", "fit_strain_max")


async def _prepare_tensile(reference: UploadFile, frames: UploadFile,
                           curve: UploadFile, form: dict):
    try:
        params = build_params(form)
        spec = build_gauge_spec(form)
        rows = parse_curve_csv(await curve.read())
        ref = decode_gray_png(await reference.read(), "reference")
        images = load_frames_zip(await frames.read(), rows)
    except RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        result = run_tensile(ref, images, rows, params, spec)
    except RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    form_dump = {k: form.get(k) for k in TENSILE_FIELDS}
    return result, form_dump


def _tensile_form(**kw) -> dict:
    return {k: str(v) for k, v in kw.items()}


@app.post("/tensile/analyze", summary="Virtual extensometer: JSON result")
async def tensile_analyze(
    reference: UploadFile = File(..., description="reference 8-bit gray PNG"),
    frames: UploadFile = File(..., description="ZIP of 2..12 deformed PNGs"),
    curve: UploadFile = File(..., description="CSV: frame_id,time_s,force_N"),
    scale_mm_per_px: str = Form(...),
    roi_x: str = Form(...),
    roi_y: str = Form(...),
    roi_w: str = Form(...),
    roi_h: str = Form(...),
    subset_size: str = Form(...),
    grid_step: str = Form(...),
    search_radius: str = Form(...),
    max_iterations: str = Form(...),
    area_mm2: str = Form(...),
    p1_x: str = Form(...),
    p1_y: str = Form(...),
    p2_x: str = Form(...),
    p2_y: str = Form(...),
    fit_strain_min: str = Form(...),
    fit_strain_max: str = Form(...),
):
    form = _tensile_form(**{k: v for k, v in locals().items()
                            if k in TENSILE_FIELDS})
    result, form_dump = await _prepare_tensile(reference, frames, curve, form)
    return {
        "parameters": form_dump,
        "limits": LIMITS,
        "gauge_length0_mm": result.gauge_length0_mm,
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
            }
            for fr in result.frames
        ],
        "fit": ({
            "E_MPa": result.fit.E_MPa,
            "b_MPa": result.fit.b_MPa,
            "r_squared": result.fit.r_squared,
            "n_points": result.fit.n_points,
        } if result.fit is not None else {"reason": result.fit_reason}),
        "yield": ({
            "strain": result.yield_point.strain,
            "stress_MPa": result.yield_point.stress_MPa,
            "frame_id_before": result.yield_point.frame_id_before,
            "frame_id_after": result.yield_point.frame_id_after,
        } if result.yield_point is not None else
            {"value": None, "reason": result.yield_reason}),
    }


@app.post("/tensile/download", summary="Virtual extensometer: result ZIP")
async def tensile_download(
    reference: UploadFile = File(...),
    frames: UploadFile = File(...),
    curve: UploadFile = File(...),
    scale_mm_per_px: str = Form(...),
    roi_x: str = Form(...),
    roi_y: str = Form(...),
    roi_w: str = Form(...),
    roi_h: str = Form(...),
    subset_size: str = Form(...),
    grid_step: str = Form(...),
    search_radius: str = Form(...),
    max_iterations: str = Form(...),
    area_mm2: str = Form(...),
    p1_x: str = Form(...),
    p1_y: str = Form(...),
    p2_x: str = Form(...),
    p2_y: str = Form(...),
    fit_strain_min: str = Form(...),
    fit_strain_max: str = Form(...),
):
    form = _tensile_form(**{k: v for k, v in locals().items()
                            if k in TENSILE_FIELDS})
    result, form_dump = await _prepare_tensile(reference, frames, curve, form)
    archive = build_tensile_zip(result, form_dump)
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition":
                 'attachment; filename="cyclic226_extensometer.zip"'},
    )
