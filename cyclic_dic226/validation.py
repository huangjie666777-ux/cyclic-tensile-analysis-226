"""Request validation and image decoding."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


MAX_EDGE = 512
MAX_GRID_POINTS = 400
MAX_SEARCH_RADIUS = 16
MAX_ITERATIONS = 100


class RequestError(ValueError):
    """Raised when an entire request must be rejected (HTTP 422)."""


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class AnalysisParams:
    roi: ROI
    scale_mm_per_px: float
    subset_size: int
    grid_step: int
    search_radius: int
    max_iterations: int


def parse_int(name: str, raw: str | int) -> int:
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{name} must be an integer") from exc
    return value


def parse_float(name: str, raw: str | float) -> float:
    try:
        value = float(str(raw))
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{name} must be a number") from exc
    if not np.isfinite(value):
        raise RequestError(f"{name} must be finite")
    return value


def build_params(form: dict[str, str]) -> AnalysisParams:
    required = ("roi_x", "roi_y", "roi_w", "roi_h", "subset_size",
                "grid_step", "search_radius", "max_iterations")
    missing = [k for k in required if form.get(k) in (None, "")]
    if missing:
        raise RequestError("missing fields: " + ", ".join(missing))

    roi = ROI(
        x=parse_int("roi_x", form["roi_x"]),
        y=parse_int("roi_y", form["roi_y"]),
        w=parse_int("roi_w", form["roi_w"]),
        h=parse_int("roi_h", form["roi_h"]),
    )
    scale = parse_float("scale_mm_per_px", form["scale_mm_per_px"])
    subset = parse_int("subset_size", form["subset_size"])
    step = parse_int("grid_step", form["grid_step"])
    radius = parse_int("search_radius", form["search_radius"])
    iters = parse_int("max_iterations", form["max_iterations"])

    if scale <= 0:
        raise RequestError("scale_mm_per_px must be positive")
    if subset <= 0 or subset % 2 == 0:
        raise RequestError("subset_size must be a positive odd integer")
    if step <= 0:
        raise RequestError("grid_step must be a positive integer")
    if radius < 0 or radius > MAX_SEARCH_RADIUS:
        raise RequestError(f"search_radius must be in [0, {MAX_SEARCH_RADIUS}]")
    if iters <= 0 or iters > MAX_ITERATIONS:
        raise RequestError(f"max_iterations must be in [1, {MAX_ITERATIONS}]")
    if roi.w <= 0 or roi.h <= 0:
        raise RequestError("roi_w and roi_h must be positive")

    return AnalysisParams(roi, scale, subset, step, radius, iters)


def decode_gray_png(data: bytes, label: str) -> np.ndarray:
    """Decode an 8-bit grayscale PNG; reject anything else."""
    if not data:
        raise RequestError(f"{label} image is empty")
    try:
        with Image.open(__import__("io").BytesIO(data)) as im:
            im.load()
            fmt = (im.format or "").upper()
            if fmt != "PNG":
                raise RequestError(f"{label} must be a PNG image (got {fmt or 'unknown'})")
            if im.mode in ("I", "I;16", "I;16B", "I;16L"):
                raise RequestError(
                    f"{label} must be 8-bit; 16-bit PNG is not supported (mode={im.mode})")
            if im.mode != "L":
                raise RequestError(f"{label} must be an 8-bit grayscale PNG (mode={im.mode})")
            arr = np.asarray(im, dtype=np.uint8)
    except RequestError:
        raise
    except Exception as exc:
        raise RequestError(f"{label} is not a decodable PNG: {exc}") from exc
    if arr.ndim != 2:
        raise RequestError(f"{label} must be a single-channel grayscale image")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise RequestError(f"{label} has empty dimensions")
    if arr.shape[0] > MAX_EDGE or arr.shape[1] > MAX_EDGE:
        raise RequestError(
            f"{label} edge exceeds {MAX_EDGE} px (got {arr.shape[1]}x{arr.shape[0]})")
    return arr


def validate_pair(ref: np.ndarray, defo: np.ndarray, p: AnalysisParams) -> None:
    if ref.shape != defo.shape:
        raise RequestError(
            f"reference and deformed images must share size "
            f"({ref.shape[1]}x{ref.shape[0]} vs {defo.shape[1]}x{defo.shape[0]})")
    H, W = ref.shape
    r = p.roi
    if r.x < 0 or r.y < 0 or r.x + r.w > W or r.y + r.h > H:
        raise RequestError("ROI lies outside the image")
    half = p.subset_size // 2
    if r.x + r.w - half <= r.x + half or r.y + r.h - half <= r.y + half:
        raise RequestError("ROI is too small for the requested subset size")


def grid_points(p: AnalysisParams) -> tuple[np.ndarray, np.ndarray]:
    """Reference grid (column, row), inclusive of both ends; every requested point kept."""
    half = p.subset_size // 2
    x0, y0 = p.roi.x + half, p.roi.y + half
    x1, y1 = p.roi.x + p.roi.w - 1 - half, p.roi.y + p.roi.h - 1 - half
    xs = np.arange(x0, x1 + 1, p.grid_step, dtype=np.float64)
    ys = np.arange(y0, y1 + 1, p.grid_step, dtype=np.float64)
    if xs[-1] < x1 - 1e-9:
        xs = np.append(xs, x1)
    if ys[-1] < y1 - 1e-9:
        ys = np.append(ys, y1)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    if len(pts) > MAX_GRID_POINTS:
        raise RequestError(
            f"grid has {len(pts)} points, exceeds maximum {MAX_GRID_POINTS}; "
            "increase grid_step or shrink ROI")
    return pts[:, 0], pts[:, 1]
