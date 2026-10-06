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

client = TestClient(app)

E_TRUE = 70000.0
E_SOFT = 63000.0
AREA = 25.0
EPS_P1 = 0.0005
EPS_P2 = 0.001

FORM = dict(scale_mm_per_px="0.05", roi_x="32", roi_y="32",
            roi_w="192", roi_h="192", subset_size="31",
            grid_step="16", search_radius="8", max_iterations="50",
            area_mm2=str(AREA), p1_x="56", p1_y="128", p2_x="200", p2_y="128")

LOAD1 = [0.0, 0.001, 0.002, 0.003]
UNLOAD1 = [0.0028, 0.0024, 0.0020, 0.0016, 0.0012, 0.0008, 0.0005]
LOAD2 = [0.0005, 0.0015, 0.0025, 0.0035]
UNLOAD2 = [0.0033, 0.0029, 0.0025, 0.0021, 0.0017, 0.0013, 0.001]


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _cyclic_sample(seed=7, size=256):
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

    def warp(eps):
        F = np.array([[1 + eps, 0.0], [0.0, 1 - 0.3 * eps]])
        rel = np.vstack([(x - cx).ravel(), (y - cy).ravel()])
        src = np.linalg.solve(F, rel) + np.array([[cx], [cy]])
        img = map_coordinates(ref.astype(float), [src[1], src[0]],
                              order=3, mode="reflect").reshape(size, size)
        return np.clip(1.03 * img - 4.0, 0, 255).astype(np.uint8)

    plan = []  # (cycle, phase, strain, stress_MPa)
    for eps in LOAD1:
        plan.append(("1", "load", eps, E_TRUE * eps))
    for eps in UNLOAD1:
        plan.append(("1", "unload", eps, E_TRUE * (eps - EPS_P1)))
    for eps in LOAD2:
        plan.append(("2", "load", eps, E_TRUE * (eps - EPS_P1)))
    for eps in UNLOAD2:
        plan.append(("2", "unload", eps, E_SOFT * (eps - EPS_P2)))

    frames, lines = {}, ["frame_id,time_s,force_N,cycle_id,phase"]
    for k, (cid, phase, eps, stress) in enumerate(plan):
        fid = f"frame_{k:02d}"
        frames[fid] = warp(eps)
        lines.append(f"{fid},{(k + 1) * 0.5:.2f},{stress * AREA:.3f},{cid},{phase}")
    return ref, frames, "\n".join(lines) + "\n"


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


def test_cyclic_analyze_metrics():
    ref, frames, csv_text = _cyclic_sample()
    r = client.post("/cyclic/analyze", data=FORM, files=_files(ref, frames, csv_text))
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["frames"]) == 22
    assert all(f["gauge_valid"] for f in body["frames"])
    c1, c2 = body["cycles"]
    assert c1["peak_stress_MPa"] == pytest.approx(210.0)
    assert c2["peak_stress_MPa"] == pytest.approx(210.0)
    assert c1["residual_strain"] == pytest.approx(EPS_P1, abs=2e-4)
    assert c2["residual_strain"] == pytest.approx(EPS_P2 - EPS_P1, abs=2e-4)
    # signed loop work: positive hysteresis area, no abs() needed
    assert c1["loop_work_MJ_per_m3"] == pytest.approx(0.09625, rel=0.1)
    assert c2["loop_work_MJ_per_m3"] > 0
    assert c1["unload_fit"]["E_MPa"] == pytest.approx(E_TRUE, rel=0.05)
    assert c2["unload_fit"]["E_MPa"] == pytest.approx(E_SOFT, rel=0.05)
    assert c1["unload_fit"]["r_squared"] > 0.99
    assert c1["modulus_retention"] == pytest.approx(1.0)
    assert c2["modulus_retention"] == pytest.approx(0.9, rel=0.05)
    base = body["modulus_retention_baseline"]
    assert base["cycle_id"] == "1"
    assert base["E_MPa"] == pytest.approx(E_TRUE, rel=0.05)


def test_cyclic_download_zip_links_files():
    ref, frames, csv_text = _cyclic_sample()
    r = client.post("/cyclic/download", data=FORM, files=_files(ref, frames, csv_text))
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"curve.csv", "cycles.csv", "result.json"} <= names
    assert {f"frames/frame_{k:02d}_points.csv" for k in range(22)} <= names
    meta = json.loads(zf.read("result.json"))
    assert len(meta["cycles"]) == 2
    assert meta["cycles"][0]["unload_fit"]["E_MPa"] == pytest.approx(E_TRUE, rel=0.05)
    assert meta["modulus_retention_baseline"]["cycle_id"] == "1"
    for fr in meta["frames"]:
        assert fr["points_csv"] in names
    curve = zf.read("curve.csv").decode().splitlines()
    assert curve[0].startswith("frame_id,time_s,force_N,cycle_id,phase,stress_MPa")
    assert len(curve) == 23
    cycles = zf.read("cycles.csv").decode().splitlines()
    assert cycles[0].startswith("cycle_id,n_frames,peak_stress_MPa")
    assert len(cycles) == 3


GOOD_ROWS = [
    "f00,0.5,0,1,load", "f01,1.0,10,1,load", "f02,1.5,20,1,load",
    "f03,2.0,15,1,unload", "f04,2.5,5,1,unload", "f05,3.0,0,1,unload",
]


@pytest.mark.parametrize("rows,fragment", [
    # load force decreases
    ([GOOD_ROWS[0], "f01,1.0,12,1,load", "f02,1.5,10,1,load"] + GOOD_ROWS[3:],
     "non-decreasing"),
    # unload force increases
    (GOOD_ROWS[:3] + ["f03,2.0,15,1,unload", "f04,2.5,18,1,unload", GOOD_ROWS[5]],
     "non-increasing"),
    # cycle does not start at zero force
    (["f00,0.5,3,1,load"] + GOOD_ROWS[1:], "start and end at zero"),
    # cycle does not end at zero force
    (GOOD_ROWS[:5] + ["f05,3.0,2,1,unload"], "start and end at zero"),
    # zero peak
    (["f00,0.5,0,1,load", "f01,1.0,0,1,load", "f02,1.5,0,1,load",
      "f03,2.0,0,1,unload", "f04,2.5,0,1,unload", "f05,3.0,0,1,unload"],
     "peak force must be positive"),
    # unload before load
    (["f00,0.5,0,1,unload", "f01,1.0,5,1,unload", "f02,1.5,10,1,unload",
      "f03,2.0,20,1,load", "f04,2.5,10,1,load", "f05,3.0,0,1,load"],
     "load frames before unload"),
    # segment shorter than 3 frames
    (["f00,0.5,0,1,load", "f01,1.0,20,1,load",
      "f03,2.0,15,1,unload", "f04,2.5,5,1,unload", "f05,3.0,0,1,unload",
      "f06,3.5,0,1,unload"], "at least 3 load"),
    # same cycle split into two blocks
    (GOOD_ROWS + ["f06,3.5,0,2,load", "f07,4.0,10,2,load", "f08,4.5,20,2,load",
                  "f09,5.0,15,2,unload", "f10,5.5,5,2,unload", "f11,6.0,0,2,unload",
                  "f12,6.5,0,1,load", "f13,7.0,10,1,load", "f14,7.5,20,1,load",
                  "f15,8.0,15,1,unload", "f16,8.5,5,1,unload", "f17,9.0,0,1,unload"],
     "contiguous"),
    # unknown phase token
    ([GOOD_ROWS[0], GOOD_ROWS[1], "f02,1.5,20,1,hold"] + GOOD_ROWS[3:],
     "'load' or 'unload'"),
    # too few frames
    (GOOD_ROWS[:5], "6..40"),
    # non-increasing time
    ([GOOD_ROWS[0], "f01,0.5,10,1,load"] + GOOD_ROWS[2:], "strictly increasing"),
    # negative force
    ([GOOD_ROWS[0], "f01,1.0,-1,1,load"] + GOOD_ROWS[2:], "non-negative"),
])
def test_bad_protocol_rejected(rows, fragment):
    ref, frames, _ = _cyclic_sample()
    csv_text = "frame_id,time_s,force_N,cycle_id,phase\n" + "\n".join(rows) + "\n"
    keep = {r.split(",")[0] for r in rows}
    frames = {k: v for k, v in frames.items() if k in keep}
    r = client.post("/cyclic/analyze", data=FORM, files=_files(ref, frames, csv_text))
    assert r.status_code == 422
    assert fragment in r.json()["detail"]


def test_missing_cycle_column_rejected():
    ref, frames, _ = _cyclic_sample()
    csv_text = ("frame_id,time_s,force_N\n"
                + "\n".join(r.rsplit(",", 2)[0] for r in GOOD_ROWS) + "\n")
    keep = {r.split(",")[0] for r in GOOD_ROWS}
    frames = {k: v for k, v in frames.items() if k in keep}
    r = client.post("/cyclic/analyze", data=FORM, files=_files(ref, frames, csv_text))
    assert r.status_code == 422
    assert "missing columns" in r.json()["detail"]


def test_equal_tensile_fit_bounds_rejected():
    from tests.test_tensile import FORM as T_FORM, _tensile_sample, _files as t_files
    ref, frames, csv_text = _tensile_sample(strains=[0.001, 0.002])
    form = dict(T_FORM, fit_strain_min="0.001", fit_strain_max="0.001")
    r = client.post("/tensile/analyze", data=form, files=t_files(ref, frames, csv_text))
    assert r.status_code == 422
    assert "greater than" in r.json()["detail"]


def test_tiny_roi_rejected_not_500():
    ref, frames, csv_text = _cyclic_sample()
    form = dict(FORM, roi_w="20", roi_h="20")
    r = client.post("/cyclic/analyze", data=form, files=_files(ref, frames, csv_text))
    assert r.status_code == 422
    assert "too small" in r.json()["detail"]
    from tests.test_tensile import FORM as T_FORM, _tensile_sample, _files as t_files
    tref, tframes, tcsv = _tensile_sample(strains=[0.001, 0.002])
    tform = dict(T_FORM, roi_w="20", roi_h="20")
    r = client.post("/tensile/analyze", data=tform,
                    files=t_files(tref, tframes, tcsv))
    assert r.status_code == 422
    assert "too small" in r.json()["detail"]
