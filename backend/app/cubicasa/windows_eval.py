"""Window counts against the SVG ground truth: detector boxes as they are vs split at mullions (app.cubicasa.windows).

  DATASET_DIR=../dataset/junction-dataset-preview uv run python -m app.cubicasa.windows_eval [--all]

Plans with cached detections only (unless --all). Split parameters are chosen on the even-indexed plans and
reported on the odd half; the counting model's windows answer (graph cache) is shown where available. 197 of the
241 preview plans are in the counting model's training split, so its numbers here are optimistic; the split rule
is not fitted on these plans.
"""

import argparse
import itertools
import json
from pathlib import Path

from PIL import Image

from app.config import CACHE_DIR, DATASET_DIR
from app.cubicasa import symdet, windows
from app.cubicasa.answer import count
from app.models import PlanGraph

CONFS = (0.35, 0.4, 0.45, 0.5)


def _hit(b, g) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return g[0] - 2 <= cx <= g[2] + 2 and g[1] - 2 <= cy <= g[3] + 2


def collect(all_plans: bool) -> list[dict]:
    from app.cubicasa.graph import _cache_path

    recs = []
    for n, p in enumerate(sorted((DATASET_DIR / "data" / "cubicasa5k").glob("*.png"))):
        rel = f"data/cubicasa5k/{p.name}"
        svg = CACHE_DIR / "svg" / f"{p.stem}.json"
        if not svg.is_file() or (not all_plans and symdet.cached(rel) is None):
            continue
        gt = PlanGraph.model_validate_json(svg.read_text(encoding="utf-8"))
        img = Image.open(p)
        dets = [d for d in symdet.detect(rel, img) if symdet.DET_ITEMS.get(d["cls"], ("",))[0] == "window"]
        gc = _cache_path(rel, "model")
        fusion = json.loads(gc.read_text(encoding="utf-8"))["counts"].get("windows") if gc.is_file() else None
        recs.append({"plan": p.stem, "idx": n, "gt": count(gt, "windows").answer,
                     "gt_boxes": [it.bbox for it in gt.items if it.category == "window"],
                     "dets": [(d["conf"], tuple(d["bbox"])) for d in dets], "fusion": fusion, "ink": windows._ink(img)})
    return recs


def score(recs, conf, params=None) -> dict:
    if params:
        windows.COVER, windows.MIN_PIECE, windows.GAP_ROW_MAX = params
    exact = err = merged = split_ok = split_bad = 0
    for r in recs:
        boxes = [b for c, b in r["dets"] if c >= conf]
        if params:
            out = []
            for b in boxes:
                pieces = windows.split_box(r["ink"], b)
                if len(pieces) > 1:  # each piece should hold its own ground-truth window
                    owners = [next((k for k, g in enumerate(r["gt_boxes"]) if _hit(pc, g)), None) for pc in pieces]
                    good = len({o for o in owners if o is not None}) == len(pieces)
                    split_ok += good
                    split_bad += not good
                out += pieces
            boxes = out
        n = len(boxes)
        exact += n == r["gt"]
        err += abs(n - r["gt"])
        merged += sum(sum(_hit_center(g, b) for g in r["gt_boxes"]) >= 2 for b in boxes)
    k = max(1, len(recs))
    return {"exact": exact / k, "mae": err / k, "merged_boxes": merged, "splits_ok": split_ok, "splits_bad": split_bad}


def _hit_center(g, b) -> bool:
    return _hit(((g[0] + g[2]) / 2 - 0.5, (g[1] + g[3]) / 2 - 0.5, (g[0] + g[2]) / 2 + 0.5, (g[1] + g[3]) / 2 + 0.5), b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also run the detector on plans without cached detections")
    a = ap.parse_args()
    recs = collect(a.all)
    dev = [r for r in recs if r["idx"] % 2 == 0]
    held = [r for r in recs if r["idx"] % 2 == 1]
    print(f"{len(recs)} plans ({len(dev)} dev / {len(held)} held-out)")
    grid = list(itertools.product((0.35, 0.45, 0.55, 0.65, 0.8), (0.6, 0.8, 1.0, 1.3), (0.25, 0.35)))
    best_conf_raw = max(CONFS, key=lambda c: score(dev, c)["exact"])
    best = max(((c, p) for c in CONFS for p in grid), key=lambda cp: (score(dev, *cp)["exact"], -score(dev, *cp)["mae"]))
    print(f"dev: raw boxes best conf {best_conf_raw}: {score(dev, best_conf_raw)}")
    print(f"dev: split best conf {best[0]} cover/min_piece/gap_row_max {best[1]}: {score(dev, *best)}")
    print("held-out half:")
    print(f"  raw boxes  (conf {best_conf_raw}): {score(held, best_conf_raw)}")
    print(f"  split      (conf {best[0]}, {best[1]}): {score(held, *best)}")
    fus = [r for r in held if r["fusion"] is not None]
    if fus:
        print(f"  counting model on the {len(fus)} held-out plans with a cached graph: exact "
              f"{sum(r['fusion'] == r['gt'] for r in fus) / len(fus):.1%}, MAE "
              f"{sum(abs(r['fusion'] - r['gt']) for r in fus) / len(fus):.2f}  | split on the same plans: "
              f"{score(fus, *best)}")
    out = CACHE_DIR / "windows_eval.json"
    out.write_text(json.dumps({"best": best, "best_raw_conf": best_conf_raw}), encoding="utf-8")


if __name__ == "__main__":
    main()
