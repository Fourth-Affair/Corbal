"""Symbol detector evaluation against the SVG ground truth: boxes and, above all, counts.

  python -m app.cubicasa.det_eval --splits val test --out det_eval.json [--viz 20 --viz-dir viz/]

Per class and per template group (symdet.DET_GROUPS): AP50 and recall at IoU 0.5, and count exact-match / mean
absolute error at confidence thresholds. The threshold is chosen on the first split (val) and applied unchanged to
the others, so the test number is not tuned on test. Error buckets for missed boxes: `small` (short side < 12 px)
and `merged` (a prediction of the same class covers it and another ground-truth box). Detections come from
symdet.detect (cached; SYMDET_WEIGHTS / SYMDET_CACHE_DIR select the candidate).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.config import CACHE_DIR
from app.cubicasa import symdet
from app.cubicasa.export_yolo import _boxes
from app.cubicasa.graph import get_graph

THRESHOLDS = [round(float(t), 2) for t in np.arange(0.1, 0.91, 0.05)]
GROUPS = {n: (n,) for n in symdet.DET_NAMES} | symdet.DET_GROUPS
SMALL = 12


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _inside(gt, pred) -> float:
    """Fraction of the ground-truth box covered by the prediction."""
    iw, ih = min(gt[2], pred[2]) - max(gt[0], pred[0]), min(gt[3], pred[3]) - max(gt[1], pred[1])
    return 0.0 if iw <= 0 or ih <= 0 else iw * ih / max(1e-6, (gt[2] - gt[0]) * (gt[3] - gt[1]))


def _match(gts: list, preds: list) -> tuple[list[tuple[float, bool]], list[bool]]:
    """Greedy matching by confidence at IoU 0.5: [(conf, is_tp)] per prediction, matched flag per ground truth."""
    used = [False] * len(gts)
    scored = []
    for p in sorted(preds, key=lambda d: -d["conf"]):
        best, bi = 0.5, -1
        for i, g in enumerate(gts):
            if not used[i]:
                v = _iou(g, p["bbox"])
                if v >= best:
                    best, bi = v, i
        if bi >= 0:
            used[bi] = True
        scored.append((p["conf"], bi >= 0))
    return scored, used


def _ap(scored: list[tuple[float, bool]], n_gt: int) -> float:
    """VOC-style all-point interpolated AP."""
    if n_gt == 0:
        return float("nan")
    if not scored:
        return 0.0
    scored = sorted(scored, key=lambda x: -x[0])
    tp = np.cumsum([s[1] for s in scored])
    fp = np.cumsum([not s[1] for s in scored])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def split_paths(split: str) -> list[str]:
    from app.eval import split_of

    return sorted(p for p in (f"data/cubicasa5k/{f.stem}.png" for f in (CACHE_DIR / "svg").glob("*.json"))
                  if split_of(p) == split)


def evaluate(paths: list[str], viz: int = 0, viz_dir: Path | None = None, thr: dict | None = None) -> dict:
    """Metrics for one split. `thr` (group -> threshold) fixes the count thresholds; None = report all."""
    scored, n_gt = defaultdict(list), defaultdict(int)
    missed = defaultdict(lambda: {"total": 0, "small": 0, "merged": 0})
    counts = defaultdict(list)  # group -> [(truth, {thr: predicted})]
    per_image_err = defaultdict(list)
    for rel in paths:
        if not symdet.image_path(rel).is_file():
            continue
        g = get_graph(rel, "svg")
        gts_all = _boxes(g)
        dets = symdet.detect(rel)
        for ci, name in enumerate(symdet.DET_NAMES):
            gts = [b[1:] for b in gts_all if b[0] == ci]
            preds = [d for d in dets if d["cls"] == name]
            s, used = _match(gts, preds)
            scored[name] += s
            n_gt[name] += len(gts)
            for gi, ok in enumerate(used):
                if ok:
                    continue
                gb = gts[gi]
                missed[name]["total"] += 1
                if min(gb[2] - gb[0], gb[3] - gb[1]) < SMALL:
                    missed[name]["small"] += 1
                if any(_inside(gb, p["bbox"]) >= 0.5 and any(_inside(o, p["bbox"]) >= 0.5 for j, o in enumerate(gts) if j != gi)
                       for p in preds if p["conf"] >= symdet.ITEM_CONF):
                    missed[name]["merged"] += 1
        for grp, members in GROUPS.items():
            ids = [symdet.DET_NAMES.index(m) for m in members]
            truth = sum(b[0] in ids for b in gts_all)
            confs = [d["conf"] for d in dets if d["cls"] in members]
            pred = {t: sum(c >= t for c in confs) for t in THRESHOLDS}
            counts[grp].append((truth, pred))
            if thr:
                per_image_err[grp].append((abs(pred[thr[grp]] - truth), rel, truth, pred[thr[grp]]))
    out = {"images": len(counts["door"]), "classes": {}, "groups": {}}
    for name in symdet.DET_NAMES:
        rec = sum(s[1] for s in scored[name]) / max(1, n_gt[name])
        out["classes"][name] = {"n_gt": n_gt[name], "ap50": round(_ap(scored[name], n_gt[name]), 4),
                                "recall_any_conf": round(rec, 4), "missed": dict(missed[name])}
    for grp, rows in counts.items():
        exact = {t: float(np.mean([p[t] == y for y, p in rows])) for t in THRESHOLDS}
        mae = {t: float(np.mean([abs(p[t] - y) for y, p in rows])) for t in THRESHOLDS}
        t_best = thr[grp] if thr else max(THRESHOLDS, key=lambda t: (exact[t], -mae[t]))
        out["groups"][grp] = {"thr": t_best, "exact": round(exact[t_best], 4), "mae": round(mae[t_best], 3),
                              "mean_truth": round(float(np.mean([y for y, _ in rows])), 2)}
    if viz and viz_dir and thr:
        _viz(per_image_err, viz, viz_dir, thr)
    return out


def _viz(per_image_err: dict, n: int, viz_dir: Path, thr: dict) -> None:
    """The n worst images per group: ground truth green, detections above the threshold red."""
    from PIL import Image, ImageDraw

    for grp, rows in per_image_err.items():
        if grp in symdet.DET_GROUPS:
            continue
        d = viz_dir / grp
        d.mkdir(parents=True, exist_ok=True)
        ci = symdet.DET_NAMES.index(grp)
        for err, rel, truth, pred in sorted(rows, reverse=True)[:n]:
            if err == 0:
                break
            im = Image.open(symdet.image_path(rel)).convert("RGB")
            dr = ImageDraw.Draw(im)
            for b in _boxes(get_graph(rel, "svg")):
                if b[0] == ci:
                    dr.rectangle(b[1:], outline=(0, 170, 0), width=3)
            for det in symdet.detect(rel):
                if det["cls"] == grp and det["conf"] >= thr[grp]:
                    dr.rectangle(det["bbox"], outline=(220, 0, 0), width=2)
            im.save(d / f"{Path(rel).stem}_gt{truth}_pred{pred}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"], help="the first split picks the thresholds")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--viz", type=int, default=0, help="draw the N worst images per class (last split)")
    ap.add_argument("--viz-dir", type=Path, default=Path("det_viz"))
    args = ap.parse_args()
    if not symdet.available():
        raise SystemExit("no detector weights")
    res, thr = {"weights": symdet.fingerprint(), "meta": symdet.weights_meta()}, None
    for i, split in enumerate(args.splits):
        paths = split_paths(split)[: args.limit or None]
        last = i == len(args.splits) - 1
        r = evaluate(paths, args.viz if last else 0, args.viz_dir, thr)
        if thr is None:
            thr = {g: v["thr"] for g, v in r["groups"].items()}
        res[split] = r
        print(f"== {split}: {r['images']} images ({res['weights']}, {res['meta']})")
        print(f"{'class':<14}{'n_gt':>6}{'AP50':>7}{'recall':>8}{'missed':>8}{'small':>7}{'merged':>8}")
        for n, c in r["classes"].items():
            m = c["missed"]
            print(f"{n:<14}{c['n_gt']:>6}{c['ap50']:>7.3f}{c['recall_any_conf']:>8.3f}{m.get('total', 0):>8}"
                  f"{m.get('small', 0):>7}{m.get('merged', 0):>8}")
        print(f"{'count group':<14}{'thr':>6}{'exact':>7}{'MAE':>8}{'mean':>8}")
        for g, c in r["groups"].items():
            print(f"{g:<14}{c['thr']:>6.2f}{c['exact']:>7.1%}{c['mae']:>8.3f}{c['mean_truth']:>8.2f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
