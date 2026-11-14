"""Upload preprocessing (app.floorprep) without OCR/CLIP: synthetic plan images, model-free steps only."""

import numpy as np
from PIL import Image

from app.floorprep import preprocess


def _plan(bg=(200, 205, 190)) -> np.ndarray:
    """Grey-tinted 'scan' with a rectangular plan, a few specks and wide empty margins (BGR)."""
    img = np.full((800, 1200, 3), bg, np.uint8)
    img[200:204, 300:900] = 20
    img[600:604, 300:900] = 20
    img[200:604, 300:304] = 20
    img[200:604, 896:900] = 20
    img[400, 600] = 0  # single-pixel speck inside the plan
    return img


def test_preprocess_whitens_crops_and_pads():
    res = preprocess(_plan(), orient=False, split=False)
    assert len(res.images) == 1
    out = res.images[0]
    h, w = out.shape[:2]
    assert h % 32 == 0 and w % 32 == 0
    assert w < 1200 * res.info["scale"]  # margins cropped away
    assert np.median(out) >= 245  # tinted paper -> white
    assert res.info["orientation"] == {"flip": False, "rot90": 0}
    assert len(res.info["crops"]) == 1 and set(res.info["timing"]) >= {"resize", "whiten", "crop"}


def test_despeckle_removes_isolated_pixel_keeps_lines():
    from app.floorprep.preprocess import despeckle

    img = np.full((1000, 1000, 3), 255, np.uint8)
    img[100:900, 100:104] = 0  # wall line
    img[500, 500] = 0  # speck
    out = despeckle(img)
    assert out[500, 500].min() == 255
    assert (out[100:900, 100:104] == 0).all()


def test_upload_step_returns_flag_and_info(monkeypatch):
    import app.floorprep as fp
    from app import main

    monkeypatch.setattr(fp, "detect_penmarks", lambda img: True)
    monkeypatch.setattr(fp, "preprocess", lambda img, split: preprocess(img, orient=False, split=split))
    img = Image.fromarray(_plan()[:, :, ::-1])
    out, has_penmarks, info = main._preprocess_upload(img)
    assert has_penmarks is True
    assert out.mode == "RGB" and out.width % 32 == 0
    assert info["scale"] > 0 and "crops" in info


def test_pen_check_failure_does_not_block(monkeypatch):
    import app.floorprep as fp
    from app import main

    def boom(img):
        raise RuntimeError("no CLIP weights")

    monkeypatch.setattr(fp, "detect_penmarks", boom)
    monkeypatch.setattr(fp, "preprocess", lambda img, split: preprocess(img, orient=False, split=split))
    _, has_penmarks, _ = main._preprocess_upload(Image.fromarray(_plan()[:, :, ::-1]))
    assert has_penmarks is None
