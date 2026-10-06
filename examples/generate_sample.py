"""Generate a reproducible synthetic speckle pair for demonstration.

The deformed image applies a known affine in-plane map
    x' = x + u0 + exx0*x + shx0*y
    y' = y + v0 + shy0*x + eyy0*x... (see parameters below)
plus an affine brightness change, written to examples/sample/.
Run: .venv/bin/python examples/generate_sample.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates


SEED = 218
SIZE = 256                 # square edge in pixels
N_DOTS = 4200
DOT_SIGMA = 1.1
U0, V0 = 3.2, -2.4          # translation, pixels
DUX, DUY = 0.012, 0.004     # du/dx, du/dy
DVX, DVY = 0.003, 0.018     # dv/dx, dv/dy
GAIN, OFFSET = 1.12, -14.0  # brightness change of deformed image


def make_reference(rng: np.random.Generator) -> np.ndarray:
    H = W = SIZE
    impulses = np.zeros((H, W), dtype=np.float64)
    ys = rng.integers(0, H, N_DOTS)
    xs = rng.integers(0, W, N_DOTS)
    intensities = rng.uniform(0.6, 1.0, N_DOTS)
    np.add.at(impulses, (ys, xs), intensities)
    field = gaussian_filter(impulses, DOT_SIGMA)
    field = field / field.max()
    img = 30.0 + 175.0 * field
    img += rng.normal(0, 1.2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def warp(ref: np.ndarray) -> np.ndarray:
    H, W = ref.shape
    x, y = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    cx = cy = SIZE / 2.0
    # Invert the forward map to sample reference from deformed grid coordinates.
    A = np.array([[1.0 + DUX, DUY], [DVX, 1.0 + DVY]])
    t = np.array([U0, V0])
    rel = np.vstack([(x - cx).ravel(), (y - cy).ravel()])
    src = np.linalg.solve(A, rel - t[:, None]) + np.array([[cx], [cy]])
    warped = map_coordinates(ref.astype(np.float64),
                             [src[1], src[0]], order=3, mode="reflect")
    warped = GAIN * warped + OFFSET
    return np.clip(warped.reshape(H, W), 0, 255).astype(np.uint8)


def main() -> None:
    out = Path(__file__).resolve().parent / "sample"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ref = make_reference(rng)
    defo = warp(ref)
    Image.fromarray(ref, mode="L").save(out / "reference.png")
    Image.fromarray(defo, mode="L").save(out / "deformed.png")
    meta = {
        "seed": SEED, "size_px": [SIZE, SIZE],
        "translation_px": [U0, V0],
        "F": [[1.0 + DUX, DUY], [DVX, 1.0 + DVY]],
        "brightness_gain": GAIN, "brightness_offset": OFFSET,
        "suggested_request": {
            "scale_mm_per_px": 0.05,
            "roi": [32, 32, 192, 192],
            "subset_size": 31, "grid_step": 16,
            "search_radius": 8, "max_iterations": 50,
        },
    }
    (out / "truth.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out/'reference.png'} and {out/'deformed.png'}")


if __name__ == "__main__":
    main()
