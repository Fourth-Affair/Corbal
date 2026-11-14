"""Rotation-aware OCR of plan labels.

Text detection runs once (on an upscaled copy for small images). Each detected box is then recognised in
the orientations that make sense for its shape (vertical boxes at 90° and 270°, horizontal at 0° and 180°)
and the best-scoring reading is kept.
"""

from functools import lru_cache

import numpy as np
from PIL import Image

from app.profiling import stage

MIN_LONG_SIDE = 1600  # upscale small plans so 1-2 letter labels (MH, KH, K) are detectable


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def read_text(img: Image.Image, min_score: float = 0.5) -> list[tuple[str, tuple[float, float, float, float], float]]:
    """Return (text, bbox in original pixels, score)."""
    img = img.convert("RGB")
    W, H = img.size
    s = max(1.0, MIN_LONG_SIDE / max(W, H))
    work = img.resize((round(W * s), round(H * s)), Image.BICUBIC) if s > 1 else img
    arr = np.asarray(work)
    with stage("engine_load"):
        eng = _engine()
    with stage("detect"):
        boxes, _ = eng.text_det(arr) if hasattr(eng, "text_det") else (None, None)
    if boxes is None or len(boxes) == 0:
        return []

    crops, metas = [], []
    for box in boxes:
        xs, ys = box[:, 0], box[:, 1]
        x0, y0, x1, y1 = max(0, xs.min() - 2), max(0, ys.min() - 2), min(work.width, xs.max() + 2), min(work.height, ys.max() + 2)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        crop = work.crop((int(x0), int(y0), int(x1), int(y1)))
        angles = (90, 270) if crop.height > 1.3 * crop.width else (0, 180)
        for a in angles:
            crops.append(np.asarray(crop.rotate(a, expand=True) if a else crop))
            metas.append((x0, y0, x1, y1))
    if not crops:
        return []
    with stage("recognise"):
        rec, _ = eng.text_rec(crops)  # one batched call for all crops/orientations
    best: dict = {}
    for meta, (text, score) in zip(metas, rec):
        if float(score) > best.get(meta, ("", -1.0))[1]:
            best[meta] = (text, float(score))
    out = []
    for (x0, y0, x1, y1), (text, score) in best.items():
        if score >= min_score and text.strip():
            out.append((text.strip(), (x0 / s, y0 / s, x1 / s, y1 / s), score))
    return out


def recognise(img: Image.Image, box: tuple[float, float, float, float], vertical: bool,
              text_height: int = 40) -> tuple[str, float]:
    """Recognise one known text box (no detection): (text, score) of the better of the two orientations. Used to
    re-read text the detector split into pieces, e.g. a vertical dimension "3 100" read as "3" and "100"."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = img.convert("RGB").crop((max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)))
    if crop.width < 4 or crop.height < 4:
        return "", 0.0
    short = crop.width if vertical else crop.height
    s = max(1.0, text_height / short)
    if s > 1:
        crop = crop.resize((round(crop.width * s), round(crop.height * s)), Image.BICUBIC)
    crops = [np.asarray(crop.rotate(a, expand=True)) for a in ((90, 270) if vertical else (0, 180))]
    rec, _ = _engine().text_rec(crops)
    text, score = max(((t, float(sc)) for t, sc in rec), key=lambda r: r[1], default=("", 0.0))
    return text.strip(), score
