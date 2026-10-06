"""Generate a reproducible synthetic cyclic load/unload test.

Writes examples/cyclic/{reference.png, frames.zip, curve.csv, truth.json}.
Two cycles: elastic loading at E_LOAD, stiffer unloading (E_U1, E_U2) that
leaves a permanent set, so the loop work is positive and the modulus
retention of cycle 2 vs cycle 1 is E_U2 / E_U1.
Run: .venv/bin/python examples/generate_cyclic_sample.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

SEED = 226
SIZE = 256
N_DOTS = 4200
DOT_SIGMA = 1.1
AREA_MM2 = 25.0
E_LOAD = 70000.0        # MPa, loading slope
E_UNLOAD = [80000.0, 78000.0]   # MPa, per-cycle unloading slope
EPS_PEAK = [0.004, 0.005]
SEGMENT_FRAMES = 6      # frames per load / unload segment (>= 3)
DT_S = 0.5
POISSON = 0.3
P1 = (56.0, 128.0)
P2 = (200.0, 128.0)


def make_reference(rng: np.random.Generator) -> np.ndarray:
    impulses = np.zeros((SIZE, SIZE))
    ys = rng.integers(0, SIZE, N_DOTS)
    xs = rng.integers(0, SIZE, N_DOTS)
    np.add.at(impulses, (ys, xs), rng.uniform(0.6, 1.0, N_DOTS))
    field = gaussian_filter(impulses, DOT_SIGMA)
    img = 30.0 + 175.0 * field / field.max()
    img += rng.normal(0, 1.2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def warp(ref: np.ndarray, exx: float) -> np.ndarray:
    x, y = np.meshgrid(np.arange(SIZE, dtype=float), np.arange(SIZE, dtype=float))
    cx = cy = SIZE / 2.0
    F = np.array([[1.0 + exx, 0.0], [0.0, 1.0 - POISSON * exx]])
    rel = np.vstack([(x - cx).ravel(), (y - cy).ravel()])
    src = np.linalg.solve(F, rel) + np.array([[cx], [cy]])
    warped = map_coordinates(ref.astype(float), [src[1], src[0]],
                             order=3, mode="reflect")
    return np.clip(1.03 * warped.reshape(SIZE, SIZE) - 4.0, 0, 255).astype(np.uint8)


def cycle_protocol(eps0: float, eps_peak: float, e_unload: float):
    """Strain/stress sequences for one load+unload cycle."""
    sig_peak = E_LOAD * (eps_peak - eps0)
    eps_resid = eps_peak - sig_peak / e_unload
    eps_load = np.linspace(eps0, eps_peak, SEGMENT_FRAMES)
    eps_unload = np.linspace(eps_peak, eps_resid, SEGMENT_FRAMES)[1:]
    sig_load = E_LOAD * (eps_load - eps0)
    sig_unload = sig_peak - e_unload * (eps_peak - eps_unload)
    eps = np.concatenate([eps_load, eps_unload])
    sig = np.concatenate([sig_load, sig_unload])
    sig[0] = sig[-1] = 0.0
    return eps, sig, eps_resid, sig_peak


def loop_work(sig: np.ndarray, eps: np.ndarray) -> float:
    return float(np.sum(0.5 * (sig[:-1] + sig[1:]) * np.diff(eps)))


def main() -> None:
    out = Path(__file__).resolve().parent / "cyclic"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ref = make_reference(rng)
    Image.fromarray(ref, mode="L").save(out / "reference.png")

    csv_lines = ["frame_id,time_s,force_N,cycle_id,phase"]
    truth_cycles = []
    frames = []
    eps0 = 0.0
    k = 0
    for c, (eps_peak, e_u) in enumerate(zip(EPS_PEAK, E_UNLOAD), start=1):
        cid = f"C{c}"
        eps, sig, eps_resid, sig_peak = cycle_protocol(eps0, eps_peak, e_u)
        n_load = SEGMENT_FRAMES
        for e, s, i in zip(eps, sig, range(eps.size)):
            fid = f"frame_{k:02d}"
            phase = "load" if i < n_load else "unload"
            frames.append((fid, e))
            csv_lines.append(
                f"{fid},{(k + 1) * DT_S:.2f},{s * AREA_MM2:.3f},{cid},{phase}")
            k += 1
        truth_cycles.append({
            "cycle_id": cid,
            "peak_stress_MPa": sig_peak,
            "permanent_strain": eps_resid - eps0,
            "loop_work_MJ_m3": loop_work(sig, eps),
            "unload_E_MPa": e_u,
        })
        eps0 = eps_resid

    with zipfile.ZipFile(out / "frames.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for fid, eps in frames:
            buf = io.BytesIO()
            Image.fromarray(warp(ref, eps), mode="L").save(buf, format="PNG")
            zf.writestr(f"{fid}.png", buf.getvalue())
    (out / "curve.csv").write_text("\n".join(csv_lines) + "\n")
    truth = {
        "area_mm2": AREA_MM2,
        "E_load_MPa": E_LOAD,
        "cycles": truth_cycles,
        "modulus_retention": E_UNLOAD[1] / E_UNLOAD[0],
        "form": {
            "scale_mm_per_px": "0.05", "roi_x": "32", "roi_y": "32",
            "roi_w": "192", "roi_h": "192", "subset_size": "31",
            "grid_step": "16", "search_radius": "8", "max_iterations": "50",
            "area_mm2": str(AREA_MM2),
            "p1_x": str(P1[0]), "p1_y": str(P1[1]),
            "p2_x": str(P2[0]), "p2_y": str(P2[1]),
        },
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

