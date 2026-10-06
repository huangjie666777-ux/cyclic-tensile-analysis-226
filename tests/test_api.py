import io
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cyclic_dic226.app import app
from tests.test_pipeline import _speckle_pair

client = TestClient(app)


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


FORM = dict(scale_mm_per_px="0.05", roi_x="40", roi_y="40",
            roi_w="112", roi_h="112", subset_size="29",
            grid_step="16", search_radius="8", max_iterations="50")


def test_analyze_and_download():
    ref, defo = _speckle_pair(du=2.6, dv=-1.7)
    files = {"reference": ("r.png", _png(ref), "image/png"),
             "deformed": ("d.png", _png(defo), "image/png")}
    r = client.post("/analyze", data=FORM, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["n_points"] > 10
    assert body["n_valid_displacement"] == body["n_points"]
    assert all("grid_x_mm" in pt for pt in body["points"])

    files = {"reference": ("r.png", _png(ref), "image/png"),
             "deformed": ("d.png", _png(defo), "image/png")}
    r = client.post("/download", data=FORM, files=files)
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert set(zf.namelist()) == {"points.csv", "valid_mask.png", "result.json"}
    csv = zf.read("points.csv").decode()
    assert csv.splitlines()[0].startswith("index,grid_x_mm")
    # mask is a valid grayscale PNG
    mask = np.asarray(Image.open(io.BytesIO(zf.read("valid_mask.png"))))
    assert mask.ndim == 2 and mask.max() == 255


def test_mismatched_sizes_rejected():
    ref, defo = _speckle_pair()
    defo = np.pad(defo, ((0, 4), (0, 0)))  # 196 rows vs 192
    files = {"reference": ("r.png", _png(ref), "image/png"),
             "deformed": ("d.png", _png(defo), "image/png")}
    r = client.post("/analyze", data=FORM, files=files)
    assert r.status_code == 422


def test_even_subset_rejected():
    ref, defo = _speckle_pair()
    form = dict(FORM, subset_size="30")
    files = {"reference": ("r.png", _png(ref), "image/png"),
             "deformed": ("d.png", _png(defo), "image/png")}
    r = client.post("/analyze", data=form, files=files)
    assert r.status_code == 422
