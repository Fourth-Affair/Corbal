"""Image-only runtime parser.

perception:  CubiCasa net (walls/rooms/icons segmentation + junction heatmaps), rotation-aware OCR and,
             when trained, the symbol detector (app.cubicasa.symdet: doors, windows, fixtures as instances)
evidence:    detector boxes, connected components of icon/room classes and dictionary-matched OCR room
             labels -> PlanItems
counting:    per-template fusion models (app.cubicasa.fit) over the evidence features -> PlanGraph.counts
"""

import json
from functools import lru_cache

import joblib
import numpy as np
from PIL import Image
from scipy import ndimage

from app.cubicasa import symdet
from app.cubicasa.features import compute, feature_paths
from app.cubicasa.fit import FEATURE_VERSION, FUSION_PATH, _predict
from app.cubicasa.fuse import ICON, ROOM, components, image_features
from app.cubicasa.rooms_dict import room_type
from app.models import PlanGraph, PlanItem

ICON_ITEMS = {"window": ("window", ""), "door": ("door", ""), "toilet": ("fixture", "toilet"),
              "sink": ("fixture", "sink"), "bathtub": ("fixture", "bathtub"), "sauna_bench": ("fixture", "sauna_bench")}
ROOM_ITEMS = {"bedroom": "bedroom", "bath": "bathroom", "kitchen": "kitchen", "outdoor": "balcony"}


@lru_cache(maxsize=1)
def _bundle() -> dict:
    if not FUSION_PATH.is_file():
        raise FileNotFoundError(f"fusion models not found at {FUSION_PATH} (run: python -m app.cubicasa.fit --final)")
    bundle = joblib.load(FUSION_PATH)
    if bundle["version"] != FEATURE_VERSION:
        raise FileNotFoundError("fusion models are stale; re-run app.cubicasa.fit --final")
    det = bundle.get("symdet")
    if det and det != symdet.fingerprint():
        # det_* features from different weights would silently shift every count
        raise FileNotFoundError(f"fusion models were fitted with symbol detector {det}, found {symdet.fingerprint()}; "
                                "restore those weights or re-run app.cubicasa.fit --final")
    return bundle


def _fusion() -> dict:
    return _bundle()["templates"]


def _det_items(dets: list[dict]) -> list[PlanItem]:
    items = []
    for k, d in enumerate(dets):
        if d["conf"] >= symdet.ITEM_CONF and d["cls"] in symdet.DET_ITEMS:
            cat, cls = symdet.DET_ITEMS[d["cls"]]
            items.append(PlanItem(id=f"det-{d['cls']}-{k}", category=cat, cls=cls, score=d["conf"],
                                  bbox=tuple(d["bbox"]), raw="detector"))
    return items


def _evidence(rel: str, dets: list[dict] | None = None) -> list[PlanItem]:
    npz, js = feature_paths(rel)
    z = np.load(npz)
    meta = json.loads(js.read_text())
    icons, iconf, rooms, rconf = z["icons"], z["icons_conf"], z["rooms"], z["rooms_conf"]
    H, W = icons.shape
    items: list[PlanItem] = []
    if dets is not None:  # detector instances replace segmentation blobs for the classes it knows
        items += _det_items(dets)
    for name, (cat, cls) in ICON_ITEMS.items():
        if dets is not None and name in symdet.DET_ITEMS:
            continue
        _, sizes, mconf, boxes = components(icons == ICON[name], iconf)
        for k, (sz, c, sl) in enumerate(zip(sizes, mconf, boxes)):
            if sz >= 20 and c >= 0.5:
                ys, xs = sl
                items.append(PlanItem(id=f"{name}-{k}", category=cat, cls=cls, score=round(float(c), 3),
                                      bbox=(float(xs.start), float(ys.start), float(xs.stop), float(ys.stop))))
    for name, cls in ROOM_ITEMS.items():
        m = ndimage.binary_opening(rooms == ROOM[name], iterations=2)
        _, sizes, mconf, boxes = components(m, rconf)
        for k, (sz, c, sl) in enumerate(zip(sizes, mconf, boxes)):
            if sz >= 2e-3 * H * W:
                ys, xs = sl
                items.append(PlanItem(id=f"room-{name}-{k}", category="room", cls=cls, score=round(float(c), 3),
                                      bbox=(float(xs.start), float(ys.start), float(xs.stop), float(ys.stop))))
    for k, tok in enumerate(meta["ocr"]):
        t = room_type(tok["text"])
        if t and t != "other":
            items.append(PlanItem(id=f"label-{k}", category="room", cls=t, text=tok["text"],
                                  score=tok["score"], bbox=tuple(tok["bbox"]), raw="ocr"))
    # storeys: large building blobs
    fg = ndimage.binary_closing(rooms != ROOM["background"], iterations=max(2, int(0.01 * max(H, W))))
    _, sizes, _, boxes = components(fg)
    for k, (sz, sl) in enumerate(zip(sizes, boxes)):
        if sz >= 0.03 * H * W:
            ys, xs = sl
            items.append(PlanItem(id=f"floor-{k}", category="floor",
                                  bbox=(float(xs.start), float(ys.start), float(xs.stop), float(ys.stop))))
    return items


def graph_from_image(rel: str, path) -> PlanGraph:
    img = Image.open(path).convert("RGB")
    compute(rel, img)  # cached perception outputs
    f = image_features(rel)
    dets = symdet.detect(rel, img) if symdet.available() else None
    if dets is not None:
        f.update(symdet.det_features(dets))
    elif _bundle().get("symdet"):
        raise FileNotFoundError("fusion models need the symbol detector (models/symbol-det.onnx)")
    counts = {}
    for t, spec in _fusion().items():
        X = np.array([[f.get(k, 0.0) for k in spec["keys"]]], dtype=np.float64)
        counts[t] = int(_predict(spec["model"], spec["kind"], X)[0])
    return PlanGraph(image=rel, width=img.width, height=img.height, source="model", items=_evidence(rel, dets),
                     counts=counts)
