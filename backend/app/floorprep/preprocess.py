"""Simple floor-plan preprocessing for the CubiCasa model.

Steps (in order): orientation fix (optional) -> resize -> background whitening -> despeckle (optional)
-> crop / split into drawings -> pad.

The cropped/split images are meant as *model input*. Keep the original image for question answering:
the crop removes margins, logos and disclaimers, but can also drop title blocks, scale notes and
legends. `info` records every transform so model outputs can be mapped back to the original.
"""
import time
from dataclasses import dataclass, field
import cv2
import numpy as np
from ._ocr import get_ocr

WHITE = (255, 255, 255)
LATIN = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÅäöå0123456789")

DEFAULTS = dict(
    # orientation (OCR on 8 rotation/flip variants)
    orient_ocr_side=1200, orient_min_conf=0.5, orient_margin=1.3, orient_min_gain=10.0, orient_rot90_margin=1.3,
    # resize
    max_long_side=1500, min_long_side=1000, max_upscale=4.0,
    # background whitening ("paper" estimate)
    bg_work_side=500, bg_dilate=7, bg_median=41, paper_frac=0.65, paper_s_max=45, bg_blur=21, white_point=0.95,
    # despeckle
    speck_area=14, dark_thresh=180,
    # crop / split
    cluster_dilate=0.015, min_cluster_frac=0.25, crop_pad=0.02,
    # pad
    pad_margin=16, pad_multiple=32,
)


def load_image(path):
    """Read any PNG/JPG as 3-channel BGR uint8 (alpha composited onto white)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return img


def _scaled(img, side):
    s = min(1.0, side / max(img.shape[:2]))
    if s == 1.0:
        return img, 1.0
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def _odd(k):
    k = int(k)
    return k if k % 2 else k + 1


# ---------------- orientation ----------------
def apply_orientation(img, flip, k):
    """Horizontal flip (optional) then rotate k*90 degrees counter-clockwise."""
    out = cv2.flip(img, 1) if flip else img
    return np.ascontiguousarray(np.rot90(out, k))


def orientation_scores(img, c=DEFAULTS):
    """OCR readability score for each of the 8 (flip, k) variants: confidence x Latin characters."""
    ocr = get_ocr()
    small, _ = _scaled(img, c["orient_ocr_side"])
    scores = {}
    for flip in (False, True):
        base = cv2.flip(small, 1) if flip else small
        for k in range(4):
            res, _ = ocr(np.ascontiguousarray(np.rot90(base, k)))
            scores[(flip, k)] = sum(float(sc) * sum(ch in LATIN for ch in t)
                                    for _, t, sc in (res or []) if float(sc) >= c["orient_min_conf"])
    return scores


def choose_orientation(scores, c=DEFAULTS):
    best = max(scores, key=scores.get)
    if best == (False, 0):
        return best
    need = c["orient_margin"] * scores[(False, 0)] + c["orient_min_gain"]
    if best[1] in (1, 3):   # 90-degree turns must also clearly beat the best 0/180 candidate
        need = max(need, c["orient_rot90_margin"] * max(scores[(f, k)] for f in (False, True) for k in (0, 2)))
    return best if scores[best] >= need else (False, 0)


def fix_orientation(img, c=DEFAULTS):
    """Returns (image, (flip, k)). Images without readable text are left unchanged."""
    best = choose_orientation(orientation_scores(img, c), c)
    return (apply_orientation(img, *best) if best != (False, 0) else img), best


# ---------------- size / photometric ----------------
def resize(img, c=DEFAULTS):
    """Long side into [min_long_side, max_long_side]. Returns (image, scale)."""
    L = max(img.shape[:2])
    if L > c["max_long_side"]:
        s, it = c["max_long_side"] / L, cv2.INTER_AREA
    elif L < c["min_long_side"]:
        s, it = min(c["min_long_side"] / L, c["max_upscale"]), cv2.INTER_LANCZOS4
    else:
        return img, 1.0
    return cv2.resize(img, None, fx=s, fy=s, interpolation=it), s


def whiten_background(img, c=DEFAULTS):
    """Divide by an estimate of the paper colour: removes grey wash, uneven light and colour casts,
    while large fills (grey balconies) are excluded from the estimate and survive."""
    h, w = img.shape[:2]
    small, _ = _scaled(img, c["bg_work_side"])
    bg = cv2.dilate(small, np.ones((int(c["bg_dilate"]),) * 2, np.uint8))      # erase dark lines
    ref = cv2.medianBlur(bg, _odd(c["bg_median"]))
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    v_ref = np.percentile(cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)[..., 2], 90)
    paper = (hsv[..., 2] >= c["paper_frac"] * v_ref) & (hsv[..., 1] <= c["paper_s_max"])
    if paper.mean() > 0.05:
        bg = cv2.inpaint(bg, (~paper).astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        bg = cv2.GaussianBlur(bg, (0, 0), c["bg_blur"])
    else:
        bg = ref
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = img.astype(np.float32) / np.maximum(bg, 1) * 255 / c["white_point"]
    return np.clip(out, 0, 255).astype(np.uint8)


def despeckle(img, c=DEFAULTS):
    """White out tiny isolated dark specks (scan/photocopy noise)."""
    out = img.copy()
    dark = (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) < c["dark_thresh"]).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    lut = stats[:, cv2.CC_STAT_AREA] < c["speck_area"]
    lut[0] = False
    out[lut[lab]] = 255
    return out


# ---------------- layout ----------------
def crop_drawings(img, c=DEFAULTS, split=True):
    """Find the drawing(s) on the sheet. Returns (crops, boxes) with boxes as [x, y, w, h] in `img`.
    Small clusters (logos, disclaimers) are dropped; with split=True, separate plans become separate crops."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    content = ((hsv[..., 2] < 200) | (hsv[..., 1] > 60)).astype(np.uint8)
    k = max(3, int(c["cluster_dilate"] * max(h, w)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(content, np.ones((k, k), np.uint8)))
    if n <= 1:
        return [img], [[0, 0, w, h]]
    areas = stats[1:, cv2.CC_STAT_AREA]
    boxes = [stats[i + 1, :4].tolist() for i, a in enumerate(areas) if a >= c["min_cluster_frac"] * areas.max()]
    if not split or len(boxes) == 1:
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        boxes = [[x0, y0, x1 - x0, y1 - y0]]
    else:
        boxes.sort(key=lambda b: (b[0], b[1]))
    p = int(c["crop_pad"] * max(h, w))
    out_boxes = []
    for x, y, bw, bh in boxes:
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(w, x + bw + p), min(h, y + bh + p)
        out_boxes.append([x0, y0, x1 - x0, y1 - y0])
    return [img[y:y + bh, x:x + bw] for x, y, bw, bh in out_boxes], out_boxes


def pad(img, c=DEFAULTS):
    """White margin, size rounded up to a multiple of pad_multiple (network stride)."""
    h, w = img.shape[:2]
    m, q = int(c["pad_margin"]), int(c["pad_multiple"])
    H, W = -(-(h + 2 * m) // q) * q, -(-(w + 2 * m) // q) * q
    t, l = (H - h) // 2, (W - w) // 2
    return cv2.copyMakeBorder(img, t, H - h - t, l, W - w - l, cv2.BORDER_CONSTANT, value=WHITE)


# ---------------- pipeline ----------------
@dataclass
class PreprocessResult:
    images: list                       # model-ready images (one per drawing)
    info: dict = field(default_factory=dict)


def preprocess(img, orient=True, despeckle_=True, split=True, params=None):
    """Run the pipeline on a BGR image.
    info = {orientation: {flip, rot90}, scale, crops: [[x,y,w,h] in oriented+resized coords],
            timing: {stage: seconds}}"""
    c = dict(DEFAULTS, **(params or {}))
    timing, info = {}, {"input_size": [img.shape[1], img.shape[0]]}

    def timed(name, fn, *a, **kw):
        t = time.perf_counter()
        r = fn(*a, **kw)
        timing[name] = round(time.perf_counter() - t, 4)
        return r

    flip, k = False, 0
    if orient:
        img, (flip, k) = timed("orientation", fix_orientation, img, c)
    info["orientation"] = {"flip": bool(flip), "rot90": int(k)}
    img, s = timed("resize", resize, img, c)
    info["scale"] = round(float(s), 4)
    img = timed("whiten", whiten_background, img, c)
    if despeckle_:
        img = timed("despeckle", despeckle, img, c)
    crops, boxes = timed("crop", crop_drawings, img, c, split)
    info["crops"] = boxes
    images = timed("pad", lambda cs: [pad(x, c) for x in cs], crops)
    info["output_sizes"] = [[x.shape[1], x.shape[0]] for x in images]
    info["timing"] = timing
    return PreprocessResult(images, info)
