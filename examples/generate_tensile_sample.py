"""Generate a reproducible synthetic tensile test for the virtual extensometer.

Writes examples/tensile/{reference.png, frames.zip, curve.csv, truth.json}.
The material model is bilinear: elastic with E_TRUE up to 0.2 % strain, then
plastic hardening with slope H_TRUE, so the 0.2 % offset yield is well defined.
Run: .venv/bin/python examples/generate_tensile_sample.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

SEED = 224
SIZE = 256
N_DOTS = 4200
DOT_SIGMA = 1.1
SCALE_MM_PER_PX = 0.05
AREA_MM2 = 25.0
E_TRUE = 70000.0        # MPa
H_TRUE = 8000.0         # plastic hardening slope, MPa
EPS_Y_TRUE = 0.002      # elastic limit strain
POISSON = 0.3
STRAINS = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006]
DT_S = 0.5
FIT_LO, FIT_HI = 0.0002, 0.002
P1 = (56.0, 128.0)      # gauge endpoints, reference pixels
P2 = (200.0, 128.0)


def stress_MPa(eps: float) -> float:
    if eps <= EPS_Y_TRUE:
        return E_TRUE * eps
    return E_TRUE * EPS_Y_TRUE + H_TRUE * (eps - EPS_Y_TRUE)


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


def main() -> None:
    out = Path(__file__).resolve().parent / "tensile"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ref = make_reference(rng)
    Image.fromarray(ref, mode="L").save(out / "reference.png")

    csv_lines = ["frame_id,time_s,force_N"]
    with zipfile.ZipFile(out / "frames.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for k, eps in enumerate(STRAINS):
            fid = f"frame_{k:02d}"
            img = warp(ref, eps)
            buf = io.BytesIO()
            Image.fromarray(img, mode="L").save(buf, format="PNG")
            zf.writestr(f"{fid}.png", buf.getvalue())
            csv_lines.append(f"{fid},{(k + 1) * DT_S:.2f},{stress_MPa(eps) * AREA_MM2:.3f}")
    (out / "curve.csv").write_text("\n".join(csv_lines) + "\n")

    truth = {
        "seed": SEED, "size_px": [SIZE, SIZE],
        "E_true_MPa": E_TRUE, "hardening_MPa": H_TRUE,
        "elastic_limit_strain": EPS_Y_TRUE,
        "yield_strain_offset_rule": EPS_Y_TRUE + stress_MPa(EPS_Y_TRUE) / (E_TRUE - H_TRUE),
        "applied_strains": STRAINS,
        "poisson_ratio": POISSON,
        "request": {
            "scale_mm_per_px": SCALE_MM_PER_PX,
            "roi": [32, 32, 192, 192],
            "subset_size": 31, "grid_step": 16,
            "search_radius": 8, "max_iterations": 50,
            "area_mm2": AREA_MM2,
            "p1": P1, "p2": P2,
            "fit_strain_min": FIT_LO, "fit_strain_max": FIT_HI,
        },
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {out}/reference.png, frames.zip, curve.csv, truth.json")


if __name__ == "__main__":
    main()
