"""Strain recovery from fitted displacement gradients.

For each valid grid point, a planar (affine) field is least-squares fit to the
valid points in its 3x3 grid neighbourhood separately for u and v.  At least
three non-collinear valid neighbours are required.  The fitted gradients form
the deformation gradient F, and the Green-Lagrange strain
E = (F^T F - I)/2 is reported (Exy is the tensor shear component).
"""
from __future__ import annotations

import numpy as np

MIN_POINTS = 3
RANK_TOL = 1e-10


def fit_gradients(coords_mm: np.ndarray, values: np.ndarray) -> np.ndarray | None:
    """Fit value = a + bx*x + by*y; return [a, bx, by] or None if degenerate.

    coords_mm: (m,2) x/y in mm; values: (m,) in mm.
    """
    m = coords_mm.shape[0]
    if m < MIN_POINTS:
        return None
    A = np.column_stack([np.ones(m), coords_mm])
    try:
        beta, *_ = np.linalg.lstsq(A, values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    # Rank check on the centred [x, y] design block: non-collinear points.
    B = A[:, 1:] - A[:, 1:].mean(axis=0)
    sv = np.linalg.svd(B, compute_uv=False)
    if sv.size < 2 or sv[1] <= RANK_TOL * max(1.0, sv[0]):
        return None
    return beta


def green_lagrange(du_dx: float, du_dy: float, dv_dx: float, dv_dy: float
                   ) -> tuple[float, float, float]:
    F = np.array([[1.0 + du_dx, du_dy],
                  [dv_dx, 1.0 + dv_dy]])
    E = 0.5 * (F.T @ F - np.eye(2))
    return float(E[0, 0]), float(E[1, 1]), float(E[0, 1])


def compute_strains(grid_x: np.ndarray, grid_y: np.ndarray,
                    valid: np.ndarray, u_mm: np.ndarray, v_mm: np.ndarray,
                    scale_mm_per_px: float):
    """Per-point strains using the 3x3 grid neighbourhood.

    Grid arrays are pixel coordinates.  Returns Exx/Eyy/Exy arrays with NaN
    wherever the point itself is invalid or the neighbourhood is degenerate.
    """
    n = grid_x.size
    exx = np.full(n, np.nan)
    eyy = np.full(n, np.nan)
    exy = np.full(n, np.nan)
    x_mm = grid_x * scale_mm_per_px
    y_mm = grid_y * scale_mm_per_px

    idx = {}
    for i in range(n):
        idx[(round(float(grid_x[i]), 6), round(float(grid_y[i]), 6))] = i

    xs = np.unique(grid_x)
    ys = np.unique(grid_y)
    for i in range(n):
        if not valid[i]:
            continue
        xi = np.where(xs == grid_x[i])[0][0]
        yi = np.where(ys == grid_y[i])[0][0]
        nb = []
        for dxi in (-1, 0, 1):
            for dyi in (-1, 0, 1):
                xj, yj = xi + dxi, yi + dyi
                if 0 <= xj < xs.size and 0 <= yj < ys.size:
                    j = idx.get((round(float(xs[xj]), 6),
                                 round(float(ys[yj]), 6)))
                    if j is not None and valid[j]:
                        nb.append(j)
        if len(nb) < MIN_POINTS:
            continue
        nb = np.asarray(nb, dtype=int)
        coords = np.column_stack([x_mm[nb], y_mm[nb]])
        bu = fit_gradients(coords, u_mm[nb])
        bv = fit_gradients(coords, v_mm[nb])
        if bu is None or bv is None:
            continue
        exx[i], eyy[i], exy[i] = green_lagrange(bu[1], bu[2], bv[1], bv[2])
    return exx, eyy, exy
