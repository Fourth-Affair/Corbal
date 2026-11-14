"""Perception layers of a plan, each with citable items: walls, symbols (doors, windows, fixtures) and text (OCR).

These back the /api/walls, /api/symbols and /api/text endpoints and the `layer` option of /api/render.
Each layer is cached per image (and per model weights where they can change).

walls    wallseg.py (dedicated U-Net: solid wall + openings) when its weights exist, else the CubiCasa net's
         vectorised walls (cubinet.py post-processing). Items: connected wall regions and openings.
symbols  symdet.py (YOLO11 detector). Items: one box per door / window / fixture, with confidence.
text     ocr.py (RapidOCR, rotation-aware). Tokens: text + box; room labels mapped to a room type.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from app.config import CACHE_DIR
from app.models import Layer, PlanItem, TextToken

LAYER_CACHE = CACHE_DIR / "layers"
MIN_REGION_PX = 30  # ignore specks


def _image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _box(sl) -> tuple[float, float, float, float]:
    ys, xs = sl
    return float(xs.start), float(ys.start), float(xs.stop), float(ys.stop)


# ---------------------------------------------------------------- walls

def wall_classes(rel: str, path: Path) -> tuple[np.ndarray, str]:
    """(HxW uint8 map: 0 background, 1 solid wall, 2 opening; model name). Cached per weights file."""
    from app.cubicasa import wallseg

    if wallseg.available():
        st = wallseg.WEIGHTS.stat()
        model, key = "wallseg", f"wallseg-{st.st_size:x}-{int(st.st_mtime):x}"
    else:
        model, key = "cubicasa-net", "cubicasa-net"
    cache = LAYER_CACHE / "walls" / f"{Path(rel).stem}.{key}.png"
    if cache.is_file():
        return np.asarray(Image.open(cache)), model
    img = _image(path)
    if model == "wallseg":
        p = wallseg.predict(img)
        cls = p.argmax(0).astype(np.uint8)
    else:  # 2019 net: wall pixels include openings, so there is no opening class
        from app.cubicasa.walls import predict_walls

        cls = predict_walls(img, vector=False)["mask"].astype(np.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cls).save(cache, optimize=True)
    return cls, model


def walls(rel: str, path: Path) -> Layer:
    cls, model = wall_classes(rel, path)
    H, W = cls.shape
    items: list[PlanItem] = []
    for k, category in ((1, "wall"), (2, "opening")):
        lab, n = ndimage.label(cls == k, structure=np.ones((3, 3)))
        if not n:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
        for i, (sl, sz) in enumerate(zip(ndimage.find_objects(lab), sizes)):
            if sz >= MIN_REGION_PX:
                items.append(PlanItem(id=f"{category}-{i}", category=category, bbox=_box(sl), raw=model,
                                      text=f"{int(sz)} px"))
    stats = {"wall_px": float((cls == 1).sum()), "opening_px": float((cls == 2).sum()),
             "wall_fraction": float((cls > 0).mean())}
    return Layer(image=rel, width=W, height=H, layer="walls", model=model, items=items, stats=stats)


# ---------------------------------------------------------------- symbols

def symbols(rel: str, path: Path) -> Layer:
    """Raises FileNotFoundError if no detector weights are installed."""
    from app.cubicasa import symdet
    from app.cubicasa.detect import _det_items

    if not symdet.available():
        raise FileNotFoundError("symbol detector weights not installed (models/symbol-det.onnx)")
    img = _image(path)
    items = _det_items(symdet.detect(rel, img))
    counts: dict[str, float] = {}
    for it in items:
        k = it.cls or it.category
        counts[k] = counts.get(k, 0) + 1
    return Layer(image=rel, width=img.width, height=img.height, layer="symbols", model=symdet.weights_path().name,
                 items=items, stats=counts)


# ---------------------------------------------------------------- text

def text(rel: str, path: Path) -> Layer:
    from app.cubicasa.features import ocr_tokens
    from app.cubicasa.rooms_dict import room_type
    from app.cubicasa.token_table import lookup

    d = ocr_tokens(rel, _image(path))
    tokens = []
    for t in d["ocr"]:
        rt = room_type(t["text"]) or lookup(t["text"])
        tokens.append(TextToken(text=t["text"], bbox=tuple(t["bbox"]), score=t["score"],
                                room_type=rt if rt and rt != "other" else None))
    return Layer(image=rel, width=d["width"], height=d["height"], layer="text", model="rapidocr", tokens=tokens,
                 stats={"tokens": float(len(tokens)), "room_labels": float(sum(t.room_type is not None for t in tokens))})


LAYERS = {"walls": walls, "symbols": symbols, "text": text}


# ---------------------------------------------------------------- rendering

def render(name: str, rel: str, path: Path) -> Image.Image:
    """The plan with one layer drawn on top (for quick human checks)."""
    from PIL import ImageDraw

    img = _image(path)
    if name == "walls":
        cls, _ = wall_classes(rel, path)
        a = np.asarray(img).astype(np.float32) * 0.4 + 255 * 0.6
        a[cls == 1] = (38, 44, 58)
        a[cls == 2] = (30, 136, 229)
        return Image.fromarray(a.astype(np.uint8))
    layer = LAYERS[name](rel, path)
    d = ImageDraw.Draw(img)
    lw = max(2, round(max(img.size) / 600))
    if name == "symbols":
        colours = {"door": (220, 30, 30), "window": (30, 136, 229)}
        for it in layer.items:
            d.rectangle(it.bbox, outline=colours.get(it.category, (46, 125, 79)), width=lw)
    else:
        for t in layer.tokens:
            d.rectangle(t.bbox, outline=(0, 137, 123) if t.room_type else (140, 140, 150), width=lw)
            if t.room_type:
                d.text((t.bbox[0], max(0, t.bbox[1] - 12)), t.room_type, fill=(0, 137, 123))
    return img
