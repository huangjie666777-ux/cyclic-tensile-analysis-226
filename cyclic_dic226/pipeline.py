"""End-to-end measurement pipeline combining correlation and strain stages."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cyclic_dic226.correlation import solve_point
from cyclic_dic226.sampling import image_gradients
from cyclic_dic226.strain import compute_strains
from cyclic_dic226.validation import AnalysisParams


@dataclass
class Measurement:
    grid_x_px: np.ndarray
    grid_y_px: np.ndarray
    valid: np.ndarray
    u_mm: np.ndarray
    v_mm: np.ndarray
    correlation: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    reasons: list[str]
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray

    @property
    def n_points(self) -> int:
        return self.grid_x_px.size

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


def run_measurement(ref: np.ndarray, defo: np.ndarray, gx_px: np.ndarray,
                    gy_px: np.ndarray, p: AnalysisParams) -> Measurement:
    n = gx_px.size
    valid = np.zeros(n, dtype=bool)
    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    corr = np.full(n, np.nan)
    conv = np.zeros(n, dtype=bool)
    iters = np.zeros(n, dtype=int)
    reasons: list[str] = [""] * n

    grad_x, grad_y = image_gradients(defo)

    for i in range(n):
        res = solve_point(ref, defo, grad_x, grad_y,
                          float(gx_px[i]), float(gy_px[i]),
                          p.subset_size, p.search_radius, p.max_iterations)
        iters[i] = res.iterations
        reasons[i] = res.reason
        if res.converged and res.u is not None and res.v is not None:
            valid[i] = True
            conv[i] = True
            u[i] = res.u * p.scale_mm_per_px
            v[i] = res.v * p.scale_mm_per_px
            corr[i] = res.correlation

    exx, eyy, exy = compute_strains(gx_px, gy_px, valid, u, v,
                                   p.scale_mm_per_px)
    return Measurement(gx_px, gy_px, valid, u, v, corr, conv, iters,
                       reasons, exx, eyy, exy)
