"""Generate a reproducible synthetic cyclic load/unload test.

Writes examples/cyclic/{reference.png, frames.zip, curve.csv, truth.json}.
Two cycles; each loads elastically (E_TRUE) to a peak, then unloads along a
parallel line shifted by a permanent (plastic) strain, so every cycle keeps
a residual strain and a positive signed hysteresis loop area.  Cycle 2
unloads with a softened modulus (0.9 * E_TRUE) to make the modulus
retention ratio differ from 1.
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
SCALE_MM_PER_PX = 0.05
AREA_MM2 = 25.0
E_TRUE = 70000.0        # MPa, loading and cycle-1 unloading
E_SOFT = 0.9            # cycle-2 unloading modulus retention
POISSON = 0.3
DT_S = 0.5
P1 = (56.0, 128.0)      # gauge endpoints, reference pixels
P2 = (200.0, 128.0)

# (load strains, unload strains, permanent strain after the cycle,
#  unload modulus)
CYCLES = [
    {"load": [0.0, 0.001, 0.002, 0.003],
     "unload": [0.0028, 0.0024, 0.0020, 0.0016, 0.0012, 0.0008, 0.0005],
     "eps_p": 0.0005, "E_unload": E_TRUE},
    {"load": [0.0005, 0.0015, 0.0025, 0.0035],
     "unload": [0.0033, 0.0029, 0.0025, 0.0021, 0.0017, 0.0013, 0.001],
     "eps_p": 0.001, "E_unload": E_SOFT * E_TRUE},
]


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
    out = Path(__file__).resolve().parent / "cyclic"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ref = make_reference(rng)
    Image.fromarray(ref, mode="L").save(out / "reference.png")

    csv_lines = ["frame_id,time_s,force_N,cycle_id,phase"]
    k = 0
    with zipfile.ZipFile(out / "frames.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for c, cyc in enumerate(CYCLES, start=1):
            eps_p = cyc["eps_p"]
            prev_p = CYCLES[c - 2]["eps_p"] if c > 1 else 0.0
            for eps in cyc["load"]:
                stress = E_TRUE * (eps - prev_p)
                k = _emit(zf, csv_lines, ref, k, c, "load", eps, stress)
            for eps in cyc["unload"]:
                stress = cyc["E_unload"] * (eps - eps_p)
                k = _emit(zf, csv_lines, ref, k, c, "unload", eps, stress)
    (out / "curve.csv").write_text("\n".join(csv_lines) + "\n")

    truth = {
        "seed": SEED, "size_px": [SIZE, SIZE],
        "E_true_MPa": E_TRUE,
        "cycles": [
            {"cycle_id": str(c + 1),
             "peak_strain": max(cyc["load"]),
             "peak_stress_MPa": E_TRUE * (max(cyc["load"])
                                          - (CYCLES[c - 1]["eps_p"] if c else 0.0)),
             "permanent_strain": cyc["eps_p"],
             "unload_modulus_MPa": cyc["E_unload"]}
            for c, cyc in enumerate(CYCLES)
        ],
        "poisson_ratio": POISSON,
        "request": {
            "scale_mm_per_px": SCALE_MM_PER_PX,
            "roi": [32, 32, 192, 192],
            "subset_size": 31, "grid_step": 16,
            "search_radius": 8, "max_iterations": 50,
            "area_mm2": AREA_MM2,
            "p1": P1, "p2": P2,
        },
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {out}/reference.png, frames.zip, curve.csv, truth.json")


def _emit(zf, csv_lines, ref, k, cycle, phase, eps, stress):
    fid = f"frame_{k:02d}"
    img = warp(ref, eps)
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    zf.writestr(f"{fid}.png", buf.getvalue())
    csv_lines.append(
        f"{fid},{(k + 1) * DT_S:.2f},{stress * AREA_MM2:.3f},{cycle},{phase}")
    return k + 1


if __name__ == "__main__":
    main()
