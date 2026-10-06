"""Per-point DIC: integer search + 6-parameter affine sub-pixel optimisation.

The objective is the zero-mean normalised sum of squared differences (ZNSSD)
with an affine intensity model, which tolerates offset and proportional
brightness change.  ZNCC = 1 - ZNSSD/2 for the optimal intensity fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cyclic_dic226.sampling import (affine_map, image_gradients,
                                        sample_bicubic, subset_coords)

FLAT_STD_THRESHOLD = 2.0          # grey levels in the reference subset
PARAM_TOL = 1e-4                  # pixels per pixel-step for translation terms
GN_DAMPING = 1e-3
MIN_SIGMA_J = 1e-8


@dataclass
class PointResult:
    x: float
    y: float
    u: float | None          # mm, x right
    v: float | None          # mm, y down
    correlation: float | None
    converged: bool
    reason: str               # "" on success
    iterations: int
    params: np.ndarray | None = None  # [u,v,ux,uy,vx,vy] in pixels


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def integer_search(ref_patch: np.ndarray, defo: np.ndarray,
                   cx: int, cy: int, half: int, radius: int) -> tuple[int, int, float]:
    """Exhaustive integer shift search by ZNCC. Patches must stay fully in-frame."""
    H, W = defo.shape
    rs = _normalize(ref_patch.ravel())
    best_dx = best_dy = 0
    best_c = -2.0
    for dy in range(-radius, radius + 1):
        y0, y1 = cy + dy - half, cy + dy + half + 1
        if y0 < 0 or y1 > H:
            continue
        for dx in range(-radius, radius + 1):
            x0, x1 = cx + dx - half, cx + dx + half + 1
            if x0 < 0 or x1 > W:
                continue
            patch = defo[y0:y1, x0:x1].ravel()
            if patch.std() < 1e-9:
                continue
            c = float(np.dot(rs, _normalize(patch)))
            if c > best_c:
                best_c, best_dx, best_dy = c, dx, dy
    return best_dx, best_dy, best_c


def _affine_intensity_residual(f_ref_n: np.ndarray, g: np.ndarray, dg_dx: np.ndarray,
                               dg_dy: np.ndarray, lx: np.ndarray, ly: np.ndarray,
                               p: np.ndarray):
    """Normalised ZNSSD residual and its Jacobian for params [u,v,ux,uy,vx,vy].

    The intensity affine correction (offset + proportional change) is applied
    analytically: the minimised vector is the component of (g - g_mean) along
    the orthonormal normalised reference direction, expressed so that the
    returned per-pixel vector r has sum(r**2) = ZNSSD in [0, 4].
    """
    n = f_ref_n.size
    g_mean = g.mean()
    gc = g - g_mean
    g_norm = np.linalg.norm(gc)
    if g_norm < 1e-9:
        return None
    # Residual: r = (I - f f^T) gc / ||gc||  -> zero when gc || f_ref_n
    dot = float(np.dot(f_ref_n, gc))
    r = (gc - dot * f_ref_n) / g_norm

    # dr/dg = (I - f f^T - r r^T)/||gc|| evaluated on centred space; account
    # for centring via projection Q = I - 11^T/n.  The dense n-by-n matrix is
    # never formed: its action on a vector is rank-3 and applied directly.
    def project(v: np.ndarray) -> np.ndarray:
        q = v - v.mean()
        return q - f_ref_n * float(np.dot(f_ref_n, q)) - r * float(np.dot(r, q))

    dx, dy = affine_map(p, lx, ly)
    # partial deformed coords wrt parameters
    d_xy = np.empty((6, 2, n))
    d_xy[0] = [np.ones(n), np.zeros(n)]              # u
    d_xy[1] = [np.zeros(n), np.ones(n)]              # v
    d_xy[2] = [lx, np.zeros(n)]                      # ux
    d_xy[3] = [ly, np.zeros(n)]                      # uy
    d_xy[4] = [np.zeros(n), lx]                      # vx
    d_xy[5] = [np.zeros(n), ly]                      # vy

    J = np.empty((n, 6))
    for k in range(6):
        dg = dg_dx * d_xy[k, 0] + dg_dy * d_xy[k, 1]
        J[:, k] = project(dg) / g_norm
    return r, J, dx, dy


def solve_point(ref: np.ndarray, defo: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                cx: float, cy: float, size: int, radius: int,
                max_iter: int) -> PointResult:
    half = size // 2
    lx, ly = subset_coords(size)
    H, W = ref.shape
    icx, icy = int(round(cx)), int(round(cy))

    if (icx - half < 0 or icx + half >= W or icy - half < 0 or icy + half >= H):
        return PointResult(cx, cy, None, None, None, False, "out_of_bounds", 0)

    ref_patch = ref[icy - half:icy + half + 1, icx - half:icx + half + 1]
    if float(ref_patch.std()) < FLAT_STD_THRESHOLD:
        return PointResult(cx, cy, None, None, None, False, "flat_texture", 0)
    f_ref = ref_patch.astype(np.float64).ravel()
    f_ref_n = _normalize(f_ref - f_ref.mean())

    # Integer offsets must be able to move the whole subset inside the frame.
    reach = half + radius
    if icx - reach < 0 or icx + reach >= W or icy - reach < 0 or icy + reach >= H:
        # Search radius itself is bounded by the image edge here.
        pass
    dx0, dy0, c0 = integer_search(ref_patch, defo, icx, icy, half, radius)
    if c0 <= -1.5:
        return PointResult(cx, cy, None, None, None, False, "integer_search_out_of_bounds", 0)

    p = np.array([float(dx0), float(dy0), 0.0, 0.0, 0.0, 0.0])
    converged = False
    iters_used = 0
    for it in range(1, max_iter + 1):
        iters_used = it
        dx, dy = affine_map(p, lx, ly)
        sx = cx + dx
        sy = cy + dy
        if (sx.min() < 0 or sx.max() > W - 1 or sy.min() < 0 or sy.max() > H - 1):
            return PointResult(cx, cy, None, None, None, False, "subpixel_out_of_bounds", it)
        g = sample_bicubic(defo, sx, sy)
        if not np.all(np.isfinite(g)):
            return PointResult(cx, cy, None, None, None, False, "subpixel_out_of_bounds", it)
        dgx = sample_bicubic(gx, sx, sy)
        dgy = sample_bicubic(gy, sx, sy)
        out = _affine_intensity_residual(f_ref_n, g, dgx, dgy, lx, ly, p)
        if out is None:
            return PointResult(cx, cy, None, None, None, False, "flat_deformed_texture", it)
        r, J, _, _ = out
        JtJ = J.T @ J
        sv = np.linalg.svd(JtJ, compute_uv=False)
        if sv[0] <= 0 or sv[-1] / sv[0] < MIN_SIGMA_J:
            return PointResult(cx, cy, None, None, None, False, "degenerate_jacobian", it)
        try:
            delta = np.linalg.solve(JtJ + GN_DAMPING * sv[0] * np.eye(6), -J.T @ r)
        except np.linalg.LinAlgError:
            return PointResult(cx, cy, None, None, None, False, "degenerate_jacobian", it)
        if not np.all(np.isfinite(delta)):
            return PointResult(cx, cy, None, None, None, False, "solver_failure", it)
        p = p + delta
        if abs(delta[0]) < PARAM_TOL and abs(delta[1]) < PARAM_TOL and \
                np.max(np.abs(delta[2:])) < PARAM_TOL:
            converged = True
            break

    if not converged:
        return PointResult(cx, cy, None, None, None, False, "not_converged", iters_used)

    dx, dy = affine_map(p, lx, ly)
    g = sample_bicubic(defo, cx + dx, cy + dy)
    if g.std() < 1e-9:
        return PointResult(cx, cy, None, None, None, False, "flat_deformed_texture", iters_used)
    zncc = float(np.dot(f_ref_n, _normalize(g - g.mean())))
    return PointResult(cx, cy, p[0], p[1], zncc, True, "", iters_used, p)
