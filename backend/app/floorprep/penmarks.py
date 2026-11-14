"""Binary pen-mark flag: does this drawing carry hand-written annotations? (yes / no)

How it decides (all local, CPU, open source):
1. Candidate regions: thin irregular strokes in any non-print colour, plus every text box found by
   RapidOCR's detector (which also finds black pen and pencil).
2. Each candidate is scored by CLIP (ViT-B-32, OpenAI weights): "handwritten" vs printed text, logos,
   drawing lines, fills.
3. The drawing is flagged when any candidate is confidently handwritten (P >= 0.85).

Flagged drawings are meant to go to a stronger model that interprets what the marks mean.
"""
import cv2
import numpy as np
from ._ocr import get_ocr
from .preprocess import whiten_background, DEFAULTS as _PRE

THRESHOLD = 0.85
WORK_SIDE = 1500
WHITE = (255, 255, 255)

PEN = dict(  # colour-stroke proposal parameters (tuned on the CubiCasa sample)
    ink_v_min=40, pen_v_max=215, pen_s_min=70, pen_strict_s=100, ink_max_frac=0.03,
    pen_line_len=31, pen_print_px=1500, pen_print_frac=0.35,
    ink_max_width=7, ink_line_fill=0.5, ink_line_aspect=6.0, pen_min_stroke_px=30, max_colour_regions=40,
)
HUE_BANDS = {"red": 0, "orange": 15, "yellow": 30, "green": 60, "cyan": 90, "blue": 120, "purple": 150}
HUE_HALF = {"red": 7, "orange": 8, "yellow": 15, "green": 15, "cyan": 15, "blue": 15, "purple": 15}
HAND = ["handwritten text written with a ballpoint pen",
        "a handwritten note scribbled on paper",
        "hand-drawn pen scribbles and crosses"]
OTHER = ["printed text in a typed font", "a printed label on a technical drawing",
         "printed numbers and dimensions", "a company logo", "lines of an architectural floor plan",
         "a door or window symbol on a floor plan", "a coloured area of a printed floor plan", "blank paper"]


# ---------------- colour stroke proposals ----------------
def _colour_masks(img, c=PEN):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    vok = (v >= c["ink_v_min"]) & (v <= c["pen_v_max"])
    near_black = cv2.dilate(((v < 90) & (s < 80)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    sat = (s >= c["pen_s_min"]) & vok & ~near_black
    strict = (s >= c["pen_strict_s"]) & vok & ~near_black
    L = int(c["pen_line_len"])
    k_h, k_v = np.ones((1, L), np.uint8), np.ones((L, 1), np.uint8)

    def straight(m8):
        return cv2.morphologyEx(m8, cv2.MORPH_OPEN, k_h) | cv2.morphologyEx(m8, cv2.MORPH_OPEN, k_v)

    def is_print(m8, st):   # print colour: large coverage or many long axis-aligned runs
        n, ns = int(m8.sum() // 255), int(st.sum() // 255)
        return m8.mean() / 255 > c["ink_max_frac"] or (ns >= c["pen_print_px"] and ns / max(n, 1) >= c["pen_print_frac"])

    out = {}
    for name, h0 in HUE_BANDS.items():
        d = np.abs(h.astype(np.int16) - h0)
        band = np.minimum(d, 180 - d) <= HUE_HALF[name]
        m = (band & sat).astype(np.uint8) * 255
        st = straight(m)
        if is_print(m, st):
            m = (band & strict).astype(np.uint8) * 255
            st = straight(m)
            if is_print(m, st):
                continue
        m = cv2.bitwise_and(m, cv2.bitwise_not(cv2.dilate(st, np.ones((3, 3), np.uint8))))
        if m.any():
            out[name] = m
    return out


def _stroke_mask(m, c=PEN):
    """Thin, non-straight strokes (straight thin pieces = printed windows)."""
    if m.mean() / 255 > c["ink_max_frac"]:
        return None
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(c["ink_max_width"]),) * 2)
    thick = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, se), np.ones((5, 5), np.uint8))
    thin = cv2.dilate(cv2.bitwise_and(m, cv2.bitwise_not(thick)), np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < c["pen_min_stroke_px"]:
            continue
        ys, xs = np.nonzero(lab[y:y + bh, x:x + bw] == i)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        is_line = area / max(rw * rh, 1) >= c["ink_line_fill"] and max(rw, rh) >= c["ink_line_aspect"] * max(min(rw, rh), 1)
        keep[i] = not is_line
    return (keep[lab] * 255).astype(np.uint8) if keep.any() else None


def _colour_regions(work, c=PEN):
    union = np.zeros(work.shape[:2], np.uint8)
    for m in _colour_masks(work, c).values():
        sm = _stroke_mask(m, c)
        if sm is not None:
            union |= sm
    if not union.any():
        return []
    k = max(3, int(0.02 * WORK_SIDE))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(union, np.ones((k, k), np.uint8)))
    H, W = work.shape[:2]
    regs = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        px = int(((lab[y:y + h, x:x + w] == i) & (union[y:y + h, x:x + w] > 0)).sum())
        if px == 0:
            continue
        p = max(0.05 * max(w, h), 0.01 * WORK_SIDE)
        x0, y0 = max(0, int(x - p)), max(0, int(y - p))
        x1, y1 = min(W, int(x + w + p)), min(H, int(y + h + p))
        regs.append((px, [x0, y0, x1 - x0, y1 - y0]))
    regs.sort(key=lambda r: -r[0])
    return [b for _, b in regs[:int(c["max_colour_regions"])]]


# ---------------- text proposals ----------------
def _text_boxes(work):
    boxes, _ = get_ocr().text_detector(work)
    out = []
    for b in (boxes if boxes is not None else []):
        b = np.asarray(b)
        x0, y0 = b.min(0); x1, y1 = b.max(0)
        if y1 - y0 >= 8 and x1 - x0 >= 8:
            out.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    return out


def _iou(a, b):
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1, a[2] * a[3] + b[2] * b[3] - inter)


def _square_crop(img, box, ctx=0.15):
    x, y, w, h = box
    p = int(ctx * max(w, h)) + 4
    H, W = img.shape[:2]
    c = img[max(0, y - p):min(H, y + h + p), max(0, x - p):min(W, x + w + p)]
    s = max(c.shape[:2])
    t, l = (s - c.shape[0]) // 2, (s - c.shape[1]) // 2
    return cv2.copyMakeBorder(c, t, s - c.shape[0] - t, l, s - c.shape[1] - l, cv2.BORDER_CONSTANT, value=WHITE)


# ---------------- CLIP ----------------
_CLIP = None
def _clip():
    global _CLIP
    if _CLIP is None:
        import torch, open_clip
        model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model.eval()
        with torch.no_grad():
            t = model.encode_text(open_clip.get_tokenizer("ViT-B-32")(HAND + OTHER))
        _CLIP = (model, prep, t / t.norm(dim=-1, keepdim=True))
    return _CLIP


def _p_hand(crops, batch=64):
    import torch
    from PIL import Image
    model, prep, text = _clip()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = torch.stack([prep(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops[i:i + batch]])
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            out.extend((100.0 * f @ text.T).softmax(dim=-1)[:, :len(HAND)].sum(1).tolist())
    return out


def penmark_score(img):
    """Highest P(handwritten) over all candidate regions (0..1). Mainly for tuning the threshold."""
    s = WORK_SIDE / max(img.shape[:2])
    work = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    work = whiten_background(work, dict(_PRE, white_point=0.90))
    props = _colour_regions(work)
    for b in _text_boxes(work):
        if all(_iou(b, p) < 0.5 for p in props):
            props.append(b)
    if not props:
        return 0.0
    return float(max(_p_hand([_square_crop(work, b) for b in props])))


def detect_penmarks(img, threshold=THRESHOLD):
    """True if the drawing (BGR image, original orientation is fine) has hand-written pen/pencil marks."""
    return penmark_score(img) >= threshold
