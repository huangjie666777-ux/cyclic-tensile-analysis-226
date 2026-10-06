import io
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cyclic_dic226.pipeline import run_measurement
from cyclic_dic226.validation import (RequestError, AnalysisParams, ROI,
                                          build_params, decode_gray_png,
                                          grid_points, validate_pair)


def _speckle_pair(seed=7, size=192, du=2.6, dv=-1.7,
                  F=None, gain=1.08, offset=-9.0):
    rng = np.random.default_rng(seed)
    H = W = size
    imp = np.zeros((H, W))
    ys = rng.integers(0, H, 2600)
    xs = rng.integers(0, W, 2600)
    np.add.at(imp, (ys, xs), rng.uniform(0.5, 1.0, 2600))
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(imp, 1.0)
    ref = np.clip(40 + 180 * field / field.max()
                  + rng.normal(0, 1.0, (H, W)), 0, 255).astype(np.uint8)
    F = F if F is not None else np.eye(2)
    x, y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    cx = cy = size / 2.0
    rel = np.vstack([(x - cx).ravel(), (y - cy).ravel()])
    t = np.array([du, dv])
    src = np.linalg.solve(F, rel - t[:, None]) + np.array([[cx], [cy]])
    warped = map_coordinates(ref.astype(float), [src[1], src[0]],
                             order=3, mode="reflect")
    defo = np.clip(gain * warped.reshape(H, W) + offset, 0, 255).astype(np.uint8)
    return ref, defo


def _params(**kw):
    base = dict(roi=ROI(40, 40, 112, 112), scale_mm_per_px=0.05,
                subset_size=29, grid_step=16, search_radius=8,
                max_iterations=50)
    base.update(kw)
    return AnalysisParams(**base)


def test_translation_recovered():
    ref, defo = _speckle_pair(du=2.6, dv=-1.7)
    p = _params()
    validate_pair(ref, defo, p)
    gx, gy = grid_points(p)
    m = run_measurement(ref, defo, gx, gy, p)
    assert m.n_valid >= int(0.8 * m.n_points)
    assert np.all(np.abs(np.nanmean(m.u_mm) / 0.05 - 2.6) < 0.15)
    assert np.all(np.abs(np.nanmean(m.v_mm) / 0.05 + 1.7) < 0.15)
    assert np.nanmin(m.correlation) > 0.9


def test_affine_strain_recovered():
    F = np.array([[1.012, 0.004], [0.003, 1.018]])
    ref, defo = _speckle_pair(du=1.5, dv=-1.0, F=F)
    p = _params(grid_step=12)
    gx, gy = grid_points(p)
    m = run_measurement(ref, defo, gx, gy, p)
    E = 0.5 * (F.T @ F - np.eye(2))
    assert np.nanmedian(m.exx) == pytest.approx(E[0, 0], abs=0.006)
    assert np.nanmedian(m.eyy) == pytest.approx(E[1, 1], abs=0.006)
    assert np.nanmedian(m.exy) == pytest.approx(E[0, 1], abs=0.006)


def test_flat_texture_point_fails_without_zero_fill():
    ref, defo = _speckle_pair()
    ref[40:90, 40:90] = 128  # flat block covering first ROI corner subset
    p = _params()
    gx, gy = grid_points(p)
    m = run_measurement(ref, defo, gx, gy, p)
    assert not m.valid[0]
    assert m.reasons[0] == "flat_texture"
    assert np.isnan(m.u_mm[0]) and np.isnan(m.exx[0])


def test_validation_rejects_bad_params():
    with pytest.raises(RequestError):
        build_params({"scale_mm_per_px": "-1", "roi_x": "0", "roi_y": "0",
                      "roi_w": "10", "roi_h": "10", "subset_size": "28",
                      "grid_step": "4", "search_radius": "3",
                      "max_iterations": "10"})
    with pytest.raises(RequestError):
        build_params({"scale_mm_per_px": "0.1", "roi_x": "0", "roi_y": "0",
                      "roi_w": "10", "roi_h": "10", "subset_size": "31",
                      "grid_step": "4", "search_radius": "20",
                      "max_iterations": "10"})


def test_decode_rejects_color_and_oversize():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    with pytest.raises(RequestError):
        decode_gray_png(buf.getvalue(), "reference")
    big = np.zeros((600, 8), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(big, mode="L").save(buf, format="PNG")
    with pytest.raises(RequestError):
        decode_gray_png(buf.getvalue(), "reference")


def test_grid_point_limit_and_inclusive_ends():
    p = _params(roi=ROI(10, 10, 400, 400), grid_step=4)
    with pytest.raises(RequestError):
        grid_points(p)
    p = _params()
    gx, gy = grid_points(p)
    # interior subset centers must span both inclusive ends
    half = p.subset_size // 2
    assert gx.min() == p.roi.x + half
    assert gx.max() == p.roi.x + p.roi.w - 1 - half
    assert gy.min() == p.roi.y + half
    assert gy.max() == p.roi.y + p.roi.h - 1 - half
