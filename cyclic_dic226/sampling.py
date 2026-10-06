"""Bicubic sub-pixel sampling helpers."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def sample_bicubic(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample 8-bit image (converted to float64) at continuous (x=col, y=row) points."""
    return map_coordinates(
        image.astype(np.float64),
        np.vstack([ys.ravel(), xs.ravel()]),
        order=3,
        mode="constant",
        cval=np.nan,
    )


def image_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradients: d/dx (columns) and d/dy (rows)."""
    f = image.astype(np.float64)
    gy, gx = np.gradient(f)
    return gx, gy


def affine_map(params: np.ndarray, lx: np.ndarray, ly: np.ndarray):
    """Map local subset coords with 6 affine parameters.

    params = [u, v, ux, uy, vx, vy] where
    x_def = x0 + lx + u + ux*lx + uy*ly,
    y_def = y0 + ly + v + vx*lx + vy*ly.
    Returns deformed local coordinates (dx, dy) relative to the centre.
    """
    u, v, ux, uy, vx, vy = params
    dx = lx + u + ux * lx + uy * ly
    dy = ly + v + vx * lx + vy * ly
    return dx, dy


def subset_coords(size: int) -> tuple[np.ndarray, np.ndarray]:
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    lx, ly = np.meshgrid(ax, ax)
    return lx.ravel(), ly.ravel()
