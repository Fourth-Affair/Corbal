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
import time
from functools import lru_cache
from pathlib import Path

from PIL import Image

from app.config import DATASET_DIR, MODELS_DIR
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
IMGSZ = int(os.environ.get("SYMDET_IMGSZ", 1280))  # must match training


def weights_path() -> Path | None:
    """ONNX (CPU runtime) preferred, else the .pt checkpoint. None if no detector has been trained."""
    env = os.environ.get("SYMDET_WEIGHTS")
    cands = [Path(env)] if env else [MODELS_DIR / "symbol-det.onnx", MODELS_DIR / "symbol-det.pt"]
    return next((p for p in cands if p.is_file()), None)


def available() -> bool:
    return weights_path() is not None


def fingerprint() -> str | None:
    p = weights_path()
    if p is None:
        return None
    st = p.stat()
    return f"{p.name}-{st.st_size:x}-{int(st.st_mtime):x}"


@lru_cache(maxsize=1)
def _model():
    from ultralytics import YOLO

    p = weights_path()
    if p is None:
        raise FileNotFoundError("symbol detector weights not found (dataset/models/symbol-det.onnx)")
    return YOLO(str(p), task="detect")


def det_path(rel: str) -> Path:
    return FEAT_DIR / f"{Path(rel).stem}.det.json"


def cached(rel: str) -> list[dict] | None:
    """Cached detections for the current weights, or None."""
    p = det_path(rel)
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    return d["dets"] if d.get("weights") == fingerprint() else None


def detect(rel: str, img: Image.Image | None = None) -> list[dict]:
    """Detections as [{"cls", "conf", "bbox": [x0, y0, x1, y1]}] in pixels of the challenge PNG (cached)."""
    hit = cached(rel)
    if hit is not None:
        return hit
    img = (img or Image.open(DATASET_DIR / rel)).convert("RGB")
    res = _model().predict(img, imgsz=IMGSZ, conf=MIN_CONF, iou=0.5, max_det=1000, verbose=False)[0]
    dets = []
    names = res.names
    for (x0, y0, x1, y1), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
        dets.append({"cls": names[int(k)], "conf": round(float(c), 4),
                     "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]})
    p = det_path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"weights": fingerprint(), "imgsz": IMGSZ, "dets": dets}))
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
    ap.add_argument("--only-featured", action="store_true", default=True,
                    help="only images that already have perception features (the fit table)")
    args = ap.parse_args()
    if not available():
        raise SystemExit("no detector weights; train one first (scripts/verda_train.py)")
    from app.cubicasa.fit import TABLE

    featured = None
    if args.only_featured and TABLE.exists():
        featured = {json.loads(line)["path"] for line in TABLE.read_text().splitlines()}
    paths = sorted({q["path"] for q in load_questions()})
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
