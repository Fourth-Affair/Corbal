"""Symbol detector (doors, windows, fixtures): a small YOLO/RT-DETR model trained on CubiCasa SVG boxes.

  python -m app.cubicasa.symdet --splits val test train     # cache detections for the fusion fit

Unlike the CubiCasa icon segmentation, a detector separates touching symbols (e.g. a row of windows), so its
counts are the evidence for the door/window/fixture templates. Detections are cached per image as JSON,
keyed by a fingerprint of the weights file, and feed `app.cubicasa.fit` as extra `det_*` features.
Train with `scripts/verda_train.py` (Verda GPU VM) or `app.cubicasa.export_yolo` + `app.cubicasa.train`.
"""

import argparse
import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.config import CACHE_DIR, DATASET_DIR, MODELS_DIR
from app.cubicasa.features import FEAT_DIR

DET_NAMES = ["door", "window", "toilet", "sink", "double_sink", "shower", "bathtub", "sauna_stove"]
# PlanItem (category, cls) per detector class
DET_ITEMS = {"door": ("door", ""), "window": ("window", ""), "toilet": ("fixture", "toilet"),
             "sink": ("fixture", "sink"), "double_sink": ("fixture", "double_sink"),
             "shower": ("fixture", "shower"), "bathtub": ("fixture", "bathtub"),
             "sauna_stove": ("fixture", "sauna_stove")}
# Count features are computed for each class and for these unions (match the question templates)
DET_GROUPS = {"sinks": ("sink", "double_sink"), "showers_baths": ("shower", "bathtub")}
CONF_THRESHOLDS = (0.25, 0.4, 0.55, 0.7)
MIN_CONF = 0.1  # what is cached; features threshold higher
ITEM_CONF = 0.4  # detections shown as evidence / citations
TILE_OVERLAP = 256  # px of native resolution shared by neighbouring tiles (> the largest symbol side)
# Ultralytics keeps predictor state on the model object, so one model must not predict from two threads at once
# (API requests run in a thread pool: an image analysis and a PDF question can both reach the detector).
_predict_lock = threading.Lock()


def weights_meta() -> dict:
    """How the weights were trained: imgsz and tile (native-resolution tile side, 0 = whole image).

    Read from the `<weights>.json` sidecar written by app.cubicasa.train; SYMDET_IMGSZ / SYMDET_TILE override it.
    Without a sidecar: the original detector (whole image at 1280)."""
    meta = {"imgsz": 1280, "tile": 0}
    p = weights_path()
    side = p.with_suffix(".json") if p else None
    if side and side.is_file():
        meta |= {k: v for k, v in json.loads(side.read_text(encoding="utf-8")).items() if k in meta}
    for k in meta:
        env = os.environ.get(f"SYMDET_{k.upper()}")
        if env:
            meta[k] = int(env)
    return meta


def weights_path() -> Path | None:
    """ONNX (CPU runtime) preferred, else the .pt checkpoint. None if no detector has been trained."""
    env = os.environ.get("SYMDET_WEIGHTS")
    cands = [Path(env)] if env else [MODELS_DIR / "symbol-det.onnx", MODELS_DIR / "symbol-det.pt"]
    return next((p for p in cands if p.is_file()), None)


def available() -> bool:
    return weights_path() is not None


def fingerprint() -> str | None:
    """Identifies the detector weights by content (not mtime: a git checkout must not invalidate the fusion models)."""
    p = weights_path()
    if p is None:
        return None
    st = p.stat()
    return _content_hash(p, st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=4)
def _content_hash(p: Path, size: int, mtime_ns: int) -> str:  # size/mtime only key the cache
    import hashlib

    return f"{p.name}-{hashlib.sha256(p.read_bytes()).hexdigest()[:12]}"


@lru_cache(maxsize=1)
def _model():
    from ultralytics import YOLO

    p = weights_path()
    if p is None:
        raise FileNotFoundError("symbol detector weights not found (models/symbol-det.onnx)")
    return YOLO(str(p), task="detect")


def det_path(rel: str) -> Path:
    # SYMDET_CACHE_DIR keeps the caches of several candidate detectors apart (one file per image otherwise)
    return Path(os.environ.get("SYMDET_CACHE_DIR") or FEAT_DIR) / f"{Path(rel).stem}.det.json"


def image_path(rel: str) -> Path:
    """Challenge PNG if present, else the byte-identical original `<variant>/<id>/F1_original.png`."""
    from app.config import CUBICASA_ORIG_DIR

    p = DATASET_DIR / rel
    if p.is_file():
        return p
    variant, _, num = Path(rel).stem.rpartition("-")
    q = CUBICASA_ORIG_DIR / variant / num / "F1_original.png"
    return q if q.is_file() else p


def cached(rel: str) -> list[dict] | None:
    """Cached detections for the current weights and inference settings, or None."""
    p = det_path(rel)
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    meta = weights_meta()
    ok = d.get("weights") == fingerprint() and d.get("imgsz", 1280) == meta["imgsz"] and d.get("tile", 0) == meta["tile"]
    return d["dets"] if ok else None


def tile_starts(length: int, tile: int, overlap: int = TILE_OVERLAP) -> list[int]:
    """Tile origins covering [0, length): stride tile - overlap, the last tile flush with the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile - overlap))
    return starts + [length - tile]


def nms(dets: list[dict], iou: float = 0.5, ios: float = 0.7) -> list[dict]:
    """Class-wise greedy NMS over merged tile outputs. Besides IoU, suppresses a box mostly inside a stronger one
    (intersection over the smaller box >= ios): a symbol cut by a tile border is a fragment of the full box."""
    keep: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        a = d["bbox"]
        aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        dup = False
        for k in keep:
            if k["cls"] != d["cls"]:
                continue
            b = k["bbox"]
            iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            bb = (b[2] - b[0]) * (b[3] - b[1])
            if inter / (aa + bb - inter) >= iou or inter / max(1e-6, min(aa, bb)) >= ios:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def _predict(img: Image.Image, imgsz: int, ox: float = 0.0, oy: float = 0.0) -> list[dict]:
    with _predict_lock:
        res = _model().predict(img, imgsz=imgsz, conf=MIN_CONF, iou=0.5, max_det=1000, verbose=False)[0]
    names = res.names
    return [{"cls": names[int(k)], "conf": round(float(c), 4),
             "bbox": [round(x0 + ox, 1), round(y0 + oy, 1), round(x1 + ox, 1), round(y1 + oy, 1)]}
            for (x0, y0, x1, y1), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]


def predict_image(img: Image.Image, conf: float = MIN_CONF) -> list[dict]:
    """Uncached detections on a PIL image, whole or tiled as the weights were trained."""
    img = img.convert("RGB")
    meta = weights_meta()
    tile = meta["tile"]
    if not tile:
        return [d for d in _predict(img, meta["imgsz"]) if d["conf"] >= conf]
    dets = []
    for y in tile_starts(img.height, tile):
        for x in tile_starts(img.width, tile):
            crop = img.crop((x, y, min(x + tile, img.width), min(y + tile, img.height)))
            dets += _predict(crop, meta["imgsz"], x, y)
    return [d for d in nms(dets) if d["conf"] >= conf]


def detect(rel: str, img: Image.Image | None = None) -> list[dict]:
    """Detections as [{"cls", "conf", "bbox": [x0, y0, x1, y1]}] in pixels of the challenge PNG (cached).

    Tiled weights (sidecar tile > 0) run on native-resolution tiles; boxes are merged with `nms`."""
    hit = cached(rel)
    if hit is not None:
        return hit
    img = (img or Image.open(image_path(rel))).convert("RGB")
    dets = predict_image(img)
    meta = weights_meta()
    p = det_path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps({"weights": fingerprint(), **meta, "dets": dets}))
        tmp.replace(p)
    finally:
        tmp.unlink(missing_ok=True)
    return dets


def det_features(dets: list[dict]) -> dict[str, float]:
    """Counts above several confidence thresholds, per class and per template group."""
    groups = {n: (n,) for n in DET_NAMES} | DET_GROUPS
    f = {}
    for g, members in groups.items():
        confs = [d["conf"] for d in dets if d["cls"] in members]
        for t in CONF_THRESHOLDS:
            f[f"det_{g}_c{int(t * 100)}"] = float(sum(c >= t for c in confs))
        f[f"det_{g}_sum"] = float(sum(confs))  # expected count if confidences were calibrated
    return f


def main() -> None:
    from app.eval import load_questions, split_of

    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--all", dest="only_featured", action="store_false",
                    help="every image, not only those in the fit table (default: only images with perception features)")
    args = ap.parse_args()
    if not available():
        raise SystemExit("no detector weights; train one first (scripts/verda_train.py)")
    from app.cubicasa.fit import TABLE

    featured = None
    if args.only_featured and TABLE.exists():
        featured = {json.loads(line)["path"] for line in TABLE.read_text(encoding="utf-8").splitlines()}
    from app.config import DATASET_DIR as DS

    if (DS / "questions.jsonl").is_file():
        paths = sorted({q["path"] for q in load_questions()})
    else:  # e.g. on the training VM: every image with ground-truth labels
        paths = sorted(f"data/cubicasa5k/{f.stem}.png" for f in (CACHE_DIR / "svg").glob("*.json"))
    todo = [p for p in paths if split_of(p) in args.splits and (featured is None or p in featured)]
    t0 = time.time()
    for i, rel in enumerate(todo, 1):
        if cached(rel) is None:
            detect(rel)
        if i % 50 == 0:
            print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("done", len(todo), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
