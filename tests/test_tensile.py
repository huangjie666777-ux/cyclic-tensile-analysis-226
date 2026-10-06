import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cyclic_dic226.app import app
from cyclic_dic226.validation import RequestError, decode_gray_png

client = TestClient(app)

E_TRUE = 70000.0
H_TRUE = 8000.0
AREA = 25.0
STRAINS = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006]

FORM = dict(scale_mm_per_px="0.05", roi_x="32", roi_y="32",
            roi_w="192", roi_h="192", subset_size="31",
            grid_step="16", search_radius="8", max_iterations="50",
            area_mm2=str(AREA), p1_x="56", p1_y="128", p2_x="200", p2_y="128",
            fit_strain_min="0.0002", fit_strain_max="0.002")


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _stress(eps):
    return E_TRUE * eps if eps <= 0.002 else E_TRUE * 0.002 + H_TRUE * (eps - 0.002)


def _tensile_sample(strains=STRAINS, seed=11, size=256):
    rng = np.random.default_rng(seed)
    imp = np.zeros((size, size))
    ys = rng.integers(0, size, 4200)
    xs = rng.integers(0, size, 4200)
    np.add.at(imp, (ys, xs), rng.uniform(0.6, 1.0, 4200))
    field = gaussian_filter(imp, 1.1)
    ref = np.clip(30 + 175 * field / field.max()
                  + rng.normal(0, 1.2, (size, size)), 0, 255).astype(np.uint8)
    x, y = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))
    cx = cy = size / 2.0
    frames = {}
    csv_lines = ["frame_id,time_s,force_N"]
    for k, eps in enumerate(strains):
        fid = f"frame_{k:02d}"
        F = np.array([[1 + eps, 0.0], [0.0, 1 - 0.3 * eps]])
        rel = np.vstack([(x - cx).ravel(), (y - cy).ravel()])
        src = np.linalg.solve(F, rel) + np.array([[cx], [cy]])
        img = map_coordinates(ref.astype(float), [src[1], src[0]],
                              order=3, mode="reflect").reshape(size, size)
        frames[fid] = np.clip(1.03 * img - 4.0, 0, 255).astype(np.uint8)
        csv_lines.append(f"{fid},{(k + 1) * 0.5:.2f},{_stress(eps) * AREA:.3f}")
    return ref, frames, "\n".join(csv_lines) + "\n"


def _files(ref, frames, csv_text):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for fid, img in frames.items():
            zf.writestr(f"{fid}.png", _png(img))
    return {
        "reference": ("ref.png", _png(ref), "image/png"),
        "frames": ("frames.zip", zbuf.getvalue(), "application/zip"),
        "curve": ("curve.csv", csv_text.encode(), "text/csv"),
    }


def test_tensile_analyze_recovers_modulus_and_yield():
    ref, frames, csv_text = _tensile_sample()
    r = client.post("/tensile/analyze", data=FORM, files=_files(ref, frames, csv_text))
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["frames"]) == len(STRAINS)
    assert all(f["gauge_valid"] for f in body["frames"])
    fit = body["fit"]
    assert fit["E_MPa"] == pytest.approx(E_TRUE, rel=0.05)
    assert abs(fit["b_MPa"]) < 5.0
    assert fit["r_squared"] > 0.99
    y = body["yield"]
    assert y["strain"] == pytest.approx(0.00426, abs=5e-4)
    assert y["stress_MPa"] == pytest.approx(E_TRUE * (y["strain"] - 0.002), rel=0.05)


def test_tensile_download_zip_links_files():
    ref, frames, csv_text = _tensile_sample()
    r = client.post("/tensile/download", data=FORM, files=_files(ref, frames, csv_text))
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"curve.csv", "result.json"} <= names
    assert {f"frames/frame_{k:02d}_points.csv" for k in range(len(STRAINS))} <= names
    meta = json.loads(zf.read("result.json"))
    assert meta["fit"]["E_MPa"] == pytest.approx(E_TRUE, rel=0.05)
    assert meta["yield"]["strain"] > 0.002
    for fr in meta["frames"]:
        assert fr["points_csv"] in names
    curve = zf.read("curve.csv").decode().splitlines()
    assert curve[0].startswith("frame_id,time_s,force_N,stress_MPa")
    assert len(curve) == len(STRAINS) + 1


@pytest.mark.parametrize("csv_text,fragment", [
    ("frame_id,time_s,force_N\nf1,0.5,10\nf1,1.0,20\n", "unique"),
    ("frame_id,time_s,force_N\nf1,1.0,10\nf2,0.5,20\n", "increasing"),
    ("frame_id,time_s,force_N\nf1,0.5,-1\nf2,1.0,20\n", "non-negative"),
    ("frame_id,time_s,force_N\nf1,0.5,10\n", "2..12"),
    ("frame_id,time_s\nf1,0.5\nf2,1.0\n", "missing columns"),
])
def test_bad_curve_csv_rejected(csv_text, fragment):
    ref, frames, _ = _tensile_sample(strains=[0.001, 0.002])
    r = client.post("/tensile/analyze", data=FORM, files=_files(ref, frames, csv_text))
    assert r.status_code == 422
    assert fragment in r.json()["detail"]


def test_frames_csv_mismatch_rejected():
    ref, frames, csv_text = _tensile_sample(strains=[0.001, 0.002])
    csv_text = csv_text.replace("frame_01", "frame_99")
    r = client.post("/tensile/analyze", data=FORM, files=_files(ref, frames, csv_text))
    assert r.status_code == 422
    assert "one-to-one" in r.json()["detail"]


def test_endpoint_outside_grid_rejected():
    ref, frames, csv_text = _tensile_sample(strains=[0.001, 0.002])
    form = dict(FORM, p2_x="400")
    r = client.post("/tensile/analyze", data=form, files=_files(ref, frames, csv_text))
    assert r.status_code == 422
    assert "outside the grid coverage" in r.json()["detail"]


def test_fit_failure_keeps_curve():
    ref, frames, csv_text = _tensile_sample()
    form = dict(FORM, fit_strain_min="0.1", fit_strain_max="0.2")
    r = client.post("/tensile/download", data=form, files=_files(ref, frames, csv_text))
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    meta = json.loads(zf.read("result.json"))
    assert "reason" in meta["fit"]
    assert meta["yield"]["value"] is None
    assert "curve.csv" in zf.namelist()


def test_16bit_png_rejected():
    arr = np.arange(256, dtype=np.uint16).reshape(16, 16) * 100
    buf = io.BytesIO()
    Image.fromarray(arr, mode="I;16").save(buf, format="PNG")
    with pytest.raises(RequestError, match="16-bit"):
        decode_gray_png(buf.getvalue(), "reference")
