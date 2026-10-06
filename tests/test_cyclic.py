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

E_LOAD = 70000.0
E_UNLOAD = [80000.0, 78000.0]
EPS_PEAK = [0.004, 0.005]
AREA = 25.0
SEG = 6

FORM = dict(scale_mm_per_px="0.05", roi_x="32", roi_y="32",
            roi_w="192", roi_h="192", subset_size="31",
            grid_step="16", search_radius="8", max_iterations="50",
            area_mm2=str(AREA), p1_x="56", p1_y="128", p2_x="200", p2_y="128")


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _cyclic_sample(seed=7, size=256, flat_frames=()):
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
    rows = []
    truth = []
    eps0 = 0.0
    k = 0
    for c, (eps_peak, e_u) in enumerate(zip(EPS_PEAK, E_UNLOAD), start=1):
        cid = f"C{c}"
        sig_peak = E_LOAD * (eps_peak - eps0)
        eps_resid = eps_peak - sig_peak / e_u
        eps_load = np.linspace(eps0, eps_peak, SEG)
        eps_unload = np.linspace(eps_peak, eps_resid, SEG)[1:]
        sig_load = E_LOAD * (eps_load - eps0)
        sig_unload = sig_peak - e_u * (eps_peak - eps_unload)
        eps = np.concatenate([eps_load, eps_unload])
        sig = np.concatenate([sig_load, sig_unload])
        sig[0] = sig[-1] = 0.0
        for i, (e, s) in enumerate(zip(eps, sig)):
            fid = f"frame_{k:02d}"
            phase = "load" if i < SEG else "unload"
            F = np.array([[1 + e, 0.0], [0.0, 1 - 0.3 * e]])
            rel = np.vstack([(x - cx).ravel(), (y - cy).ravel()])
            src = np.linalg.solve(F, rel) + np.array([[cx], [cy]])
            img = map_coordinates(ref.astype(float), [src[1], src[0]],
                                  order=3, mode="reflect").reshape(size, size)
            img = np.clip(1.03 * img - 4.0, 0, 255).astype(np.uint8)
            if fid in flat_frames:
                img = np.full_like(img, 128)
            frames[fid] = img
            rows.append((fid, (k + 1) * 0.5, s * AREA, cid, phase))
            k += 1
        work = float(np.sum(0.5 * (sig[:-1] + sig[1:]) * np.diff(eps)))
        truth.append(dict(cycle_id=cid, peak=sig_peak,
                          permanent=eps_resid - eps0, work=work, E=e_u))
        eps0 = eps_resid
    return ref, frames, rows, truth


def _csv(rows):
    lines = ["frame_id,time_s,force_N,cycle_id,phase"]
    lines += [f"{f},{t:.2f},{F:.3f},{c},{p}" for f, t, F, c, p in rows]
    return "\n".join(lines) + "\n"


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


def test_cyclic_analyze_recovers_cycles():
    ref, frames, rows, truth = _cyclic_sample()
    r = client.post("/cyclic/analyze", data=FORM, files=_files(ref, frames, _csv(rows)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["frames"]) == len(rows)
    assert all(f["gauge_valid"] for f in body["frames"])
    assert [c["cycle_id"] for c in body["cycles"]] == ["C1", "C2"]
    for got, want in zip(body["cycles"], truth):
        assert got["peak_stress_MPa"] == pytest.approx(want["peak"], rel=1e-6)
        assert got["permanent_strain"] == pytest.approx(want["permanent"], abs=2e-4)
        assert got["loop_work_MJ_m3"] == pytest.approx(want["work"], rel=0.05)
        assert got["loop_work_MJ_m3"] > 0
        fit = got["unload_fit"]
        assert fit["E_MPa"] == pytest.approx(want["E"], rel=0.05)
        assert fit["r_squared"] > 0.99
    c1, c2 = body["cycles"]
    assert c1["modulus_retention"] == pytest.approx(1.0)
    assert c2["modulus_retention"] == pytest.approx(E_UNLOAD[1] / E_UNLOAD[0],
                                                    rel=0.05)
    base = body["modulus_retention_baseline"]
    assert base["cycle_id"] == "C1"
    assert base["E_MPa"] == pytest.approx(E_UNLOAD[0], rel=0.05)


def test_cyclic_download_zip_links_files():
    ref, frames, rows, truth = _cyclic_sample()
    r = client.post("/cyclic/download", data=FORM, files=_files(ref, frames, _csv(rows)))
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert {"curve.csv", "cycles.csv", "result.json"} <= names
    assert {f"frames/frame_{k:02d}_points.csv" for k in range(len(rows))} <= names
    meta = json.loads(zf.read("result.json"))
    assert meta["parameters"]["area_mm2"] == str(AREA)
    assert len(meta["cycles"]) == 2
    assert meta["cycles"][0]["unload_fit"]["E_MPa"] == pytest.approx(
        E_UNLOAD[0], rel=0.05)
    assert meta["modulus_retention_baseline"]["cycle_id"] == "C1"
    for fr in meta["frames"]:
        assert fr["points_csv"] in names
    curve = zf.read("curve.csv").decode().splitlines()
    assert curve[0].startswith("frame_id,time_s,force_N,cycle_id,phase,stress_MPa")
    assert len(curve) == len(rows) + 1
    cycles = zf.read("cycles.csv").decode().splitlines()
    assert cycles[0].startswith("cycle_id,n_frames,peak_stress_MPa")
    assert len(cycles) == 3


def _mutated(rows, fn):
    rows = [list(r) for r in rows]
    fn(rows)
    return [tuple(r) for r in rows]


@pytest.mark.parametrize("mutate,fragment", [
    (lambda rs: rs.__setitem__(3, (rs[3][0], rs[3][1], rs[3][2], "C2", rs[3][4])),
     "contiguous"),
    (lambda rs: rs.__setitem__(7, (rs[7][0], rs[7][1], rs[7][2], rs[7][3], "load")),
     "load frames must precede unload"),
    (lambda rs: rs.__setitem__(2, (rs[2][0], rs[2][1], 5.0, rs[2][3], rs[2][4])),
     "non-decreasing"),
    (lambda rs: rs.__setitem__(7, (rs[7][0], rs[7][1], 9000.0, rs[7][3], rs[7][4])),
     "non-increasing"),
    (lambda rs: rs.__setitem__(0, (rs[0][0], rs[0][1], 3.0, rs[0][3], rs[0][4])),
     "first and last force"),
    (lambda rs: rs.__setitem__(1, (rs[1][0], rs[1][1], -2.0, rs[1][3], rs[1][4])),
     "non-negative"),
    (lambda rs: rs.__setitem__(2, (rs[2][0], 0.10, rs[2][2], rs[2][3], rs[2][4])),
     "strictly increasing"),
    (lambda rs: rs.__setitem__(4, (rs[4][0], rs[4][1], rs[4][2], rs[4][3], "hold")),
     "phase must be one of"),
])
def test_bad_protocol_rejected(mutate, fragment):
    ref, frames, rows, _ = _cyclic_sample()
    r = client.post("/cyclic/analyze", data=FORM,
                    files=_files(ref, frames, _csv(_mutated(rows, mutate))))
    assert r.status_code == 422
    assert fragment in r.json()["detail"]


def test_short_segment_rejected():
    ref, frames, rows, _ = _cyclic_sample()
    # drop three unload frames of cycle C1 -> unload segment has 2 frames
    drop = {"frame_06", "frame_07", "frame_08"}
    rows = [r for r in rows if r[0] not in drop]
    frames = {k: v for k, v in frames.items() if k not in drop}
    r = client.post("/cyclic/analyze", data=FORM,
                    files=_files(ref, frames, _csv(rows)))
    assert r.status_code == 422
    assert "at least" in r.json()["detail"]


def test_frame_count_bounds():
    ref, frames, rows, _ = _cyclic_sample()
    rows5 = rows[:5]
    frames5 = {k: frames[k] for k, *_ in [(r[0],) for r in rows5]}
    frames5 = {r[0]: frames[r[0]] for r in rows5}
    r = client.post("/cyclic/analyze", data=FORM,
                    files=_files(ref, frames5, _csv(rows5)))
    assert r.status_code == 422
    assert "6..40" in r.json()["detail"]


def test_missing_measurement_cycle_nulls_others_kept():
    flat = {f"frame_{k:02d}" for k in (12, 13)}
    ref, frames, rows, truth = _cyclic_sample(flat_frames=flat)
    r = client.post("/cyclic/analyze", data=FORM, files=_files(ref, frames, _csv(rows)))
    assert r.status_code == 200, r.text
    body = r.json()
    c1, c2 = body["cycles"]
    assert c1["measure"]["ok"] is True
    assert c1["permanent_strain"] == pytest.approx(truth[0]["permanent"], abs=2e-4)
    assert c2["measure"]["ok"] is False
    assert "gauge_invalid_frames" in c2["measure"]["reason"]
    assert c2["permanent_strain"] is None
    assert c2["loop_work_MJ_m3"] is None
    assert c2["peak_stress_MPa"] == pytest.approx(truth[1]["peak"], rel=1e-6)
    assert body["modulus_retention_baseline"]["cycle_id"] == "C1"


def test_frames_csv_mismatch_rejected():
    ref, frames, rows, _ = _cyclic_sample()
    rows = [("frame_99" if r[0] == "frame_01" else r[0], *r[1:]) for r in rows]
    r = client.post("/cyclic/analyze", data=FORM,
                    files=_files(ref, frames, _csv(rows)))
    assert r.status_code == 422
    assert "one-to-one" in r.json()["detail"]


def test_equal_fit_bounds_rejected_tensile():
    from tests.test_tensile import FORM as TFORM, _tensile_sample, _files as tfiles
    ref, frames, csv_text = _tensile_sample(strains=[0.001, 0.002])
    form = dict(TFORM, fit_strain_min="0.001", fit_strain_max="0.001")
    r = client.post("/tensile/analyze", data=form,
                    files=tfiles(ref, frames, csv_text))
    assert r.status_code == 422
    assert "greater than" in r.json()["detail"]


def test_tiny_roi_is_422_not_500():
    from tests.test_tensile import FORM as TFORM, _tensile_sample, _files as tfiles
    ref, frames, csv_text = _tensile_sample(strains=[0.001, 0.002])
    form = dict(TFORM, roi_w="10", roi_h="10")
    r = client.post("/tensile/analyze", data=form,
                    files=tfiles(ref, frames, csv_text))
    assert r.status_code == 422
    assert "too small" in r.json()["detail"]
