"""Walls: ground truth from the original CubiCasa SVGs, predictions from the CubiCasa net, and accuracy metrics.

  python -m app.cubicasa.walls --split test --limit 60                     # CubiCasa net, held-out plans
  python -m app.cubicasa.walls --split test --limit 60 --model wallseg     # dedicated wall segmenter (wallseg.py)

Predictions come in two forms:
  pixel  - the net's room segmentation, class "Wall" (argmax per pixel)
  vector - the original CubiCasa post-processing (junction heatmaps -> wall lines -> wall polygons)
Ground truth: every `<g class="Wall ...">` polygon, rescaled from F1_scaled to the challenge PNG. Door and window
openings sit inside wall polygons in the SVG; `solid` removes them (CubiCasa's segmentation labels openings as wall).

`*_annotated` scores only inside the floors the SVG annotates: some plans show extra floors without labels.
Metrics are on the image pixel grid: IoU, and precision/recall/F1 with a small distance tolerance (`tol` px),
because a 1-2 px shift of a wall edge is annotation noise, not a wrong wall.
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from app.config import CACHE_DIR, DATASET_DIR
from app.cubicasa.labels_from_svg import Matrix, _first_polygon, _mul, _parse_transform, svg_path_for

WALL_CLASS = 2  # cubinet.ROOM_CLASSES index of "Wall"
OUT_DIR = CACHE_DIR / "walls"


def gt_walls(rel: str) -> tuple[list, list, tuple[int, int]]:
    """(wall polygons, opening polygons, (W, H)) in challenge-PNG pixels."""
    svg = svg_path_for(rel)
    with Image.open(DATASET_DIR / rel) as im:
        W, H = im.size
    with Image.open(svg.parent / "F1_scaled.png") as im:
        sw, sh = im.size
    walls, openings = [], []

    def walk(el: ET.Element, m: Matrix) -> None:
        if "display: none" in (el.get("style") or ""):
            return
        head = (el.get("class") or "").split()[:1]
        m2 = _mul(m, _parse_transform(el.get("transform")))
        if head == ["Wall"]:
            poly = _first_polygon(el, m)
            if poly and len(poly) >= 3:
                walls.append(poly)
        elif head in (["Door"], ["Window"]):
            poly = _first_polygon(el, m)
            if poly and len(poly) >= 3:
                openings.append(poly)
            return
        for ch in el:
            walk(ch, m2)

    walk(ET.parse(svg).getroot(), (W / sw, 0, 0, H / sh, 0, 0))
    return walls, openings, (W, H)


def rasterize(polys: list, size: tuple[int, int]) -> np.ndarray:
    img = Image.new("1", size, 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        d.polygon([tuple(map(float, xy)) for xy in p], fill=1, outline=1)
    return np.asarray(img, dtype=bool)


def predict_walls(img: Image.Image, max_side: int = 1536, vector: bool = True) -> dict:
    """{"mask": HxW bool (pixel walls), "prob": HxW float wall probability, "polygons": vector wall polygons}."""
    import torch

    from app.cubicasa import cubinet

    pred = cubinet.predict(img, max_side=max_side)
    rooms = torch.softmax(pred[0, 21:33], 0)
    out = {"mask": (rooms.argmax(0) == WALL_CLASS).numpy(), "prob": rooms[WALL_CLASS].numpy(), "polygons": []}
    if vector:
        polygons, types, *_ = cubinet.parse(pred)
        out["polygons"] = [np.asarray(p).tolist() for p, t in zip(polygons, types) if t["type"] == "wall"]
    return out


def metrics(pred: np.ndarray, gt: np.ndarray, tol: int = 3) -> dict:
    inter, union = np.logical_and(pred, gt).sum(), np.logical_or(pred, gt).sum()
    # distance-tolerant matching: a predicted wall pixel counts as correct if a true wall pixel is within tol px
    gt_near = ndimage.binary_dilation(gt, iterations=tol) if tol else gt
    pred_near = ndimage.binary_dilation(pred, iterations=tol) if tol else pred
    p = float(np.logical_and(pred, gt_near).sum() / max(pred.sum(), 1))
    r = float(np.logical_and(gt, pred_near).sum() / max(gt.sum(), 1))
    return {"iou": float(inter / max(union, 1)), "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def annotated_region(rel: str, size: tuple[int, int], walls: list, margin: float = 0.03) -> np.ndarray:
    """Bounding boxes (+ margin) of the floors whose walls the SVG annotates. Some plans have floors with rooms and
    doors labelled but no wall polygons; walls found there are real but have no ground truth, so the `*_annotated`
    metrics ignore predictions outside this region."""
    from app.cubicasa.labels_from_svg import graph_from_svg

    W, H = size
    m = margin * max(W, H)
    centres = [(np.mean([x for x, _ in w]), np.mean([y for _, y in w])) for w in walls]
    boxes = [b for b in (it.bbox for it in graph_from_svg(rel).items if it.category == "floor")
             if any(b[0] - m <= cx <= b[2] + m and b[1] - m <= cy <= b[3] + m for cx, cy in centres)]
    region = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1 in boxes or [(0, 0, W, H)]:
        region[max(0, int(y0 - m)):int(y1 + m) + 1, max(0, int(x0 - m)):int(x1 + m) + 1] = True
    return region


def evaluate(rel: str, tol: int = 3, vector: bool = True, model: str = "cubicasa", device: str = "cpu") -> dict:
    walls, openings, size = gt_walls(rel)
    gt = rasterize(walls, size)
    solid = gt & ~rasterize(openings, size)
    img = Image.open(DATASET_DIR / rel).convert("RGB")
    t0 = time.time()
    if model == "wallseg":
        from app.cubicasa import wallseg

        pr = wallseg.predict_walls(img, device=device)
        vector = False
    else:
        pr = predict_walls(img, vector=vector)
    pred_solid = pr.get("solid", pr["mask"])  # the CubiCasa net has no separate solid-wall output
    region = annotated_region(rel, size, walls)
    res = {"path": rel, "model": model, "size": size, "n_gt_walls": len(walls), "seconds": round(time.time() - t0, 2),
           "pixel": metrics(pr["mask"], gt, tol), "pixel_solid": metrics(pred_solid, solid, tol),
           "pixel_annotated": metrics(pr["mask"] & region, gt, tol),
           "solid_annotated": metrics(pred_solid & region, solid, tol)}
    if vector:
        vm = rasterize(pr["polygons"], size)
        res["n_pred_walls"] = len(pr["polygons"])
        res["vector"] = metrics(vm, gt, tol)
        res["vector_solid"] = metrics(vm, solid, tol)
        res["vector_annotated"] = metrics(vm & region, gt, tol)
    return res, pr, gt, solid


def main() -> None:
    from app.eval import load_questions, split_of

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--no-vector", action="store_true")
    ap.add_argument("--model", choices=["cubicasa", "wallseg"], default="cubicasa")
    ap.add_argument("--device", default="cpu", help="wallseg only: cpu / cuda (ROCm also uses cuda)")
    args = ap.parse_args()
    paths = sorted({q["path"] for q in load_questions() if split_of(q["path"]) == args.split})
    paths = paths[:: max(1, len(paths) // args.limit)][: args.limit] if args.limit else paths  # spread over variants
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, rel in enumerate(paths, 1):
        try:
            res, *_ = evaluate(rel, args.tol, not args.no_vector, args.model, args.device)
        except Exception as e:  # noqa: BLE001
            print("ERR", rel, type(e).__name__, e, flush=True)
            continue
        rows.append(res)
        print(f"{i}/{len(paths)} {Path(rel).stem:<40} pixel F1 {res['pixel_annotated']['f1']:.3f} IoU {res['pixel']['iou']:.3f}"
              + (f" | vector F1 {res['vector']['f1']:.3f}" if "vector" in res else "") + f"  {res['seconds']}s", flush=True)
    (OUT_DIR / f"walls_{args.split}{'' if args.model == 'cubicasa' else '_' + args.model}.json").write_text(json.dumps(rows))
    for key in ("pixel", "pixel_annotated", "pixel_solid", "solid_annotated", "vector", "vector_annotated", "vector_solid"):
        if rows and key in rows[0]:
            m = {k: np.mean([r[key][k] for r in rows]) for k in ("iou", "precision", "recall", "f1")}
            print(f"{key:<13} " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()) + f"   (n={len(rows)}, tol={args.tol}px)")


if __name__ == "__main__":
    main()
