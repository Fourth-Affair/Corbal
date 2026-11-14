"""Deterministic perception benchmark: what the image pipeline found vs the CubiCasa SVG labels. No questions, no
router, no LLM, so a change to a detector, a segmentation net or the fusion models can be checked in seconds.

  uv run python -m app.bench perception --set fast              # 16 test images (cached graphs: a few seconds)
  uv run python -m app.bench perception --set full --walls      # all test images, plus wall segmentation
  uv run python -m app.bench diff <runA>.jsonl <runB>.jsonl     # per-class F1 and count deltas, changed images

Three things are scored per image (official test split only; needs dataset/cubicasa5k-original for the labels):

- symbols: PlanGraph evidence items (the boxes answers cite) against label boxes, one-to-one, greedy by IoU, a match
  needs IoU >= --iou. Precision / recall / F1 per class and the mean IoU of the matches. This is the localisation
  check the question benchmark lacks: a right count with boxes on the wrong objects scores badly here.
- rooms: evidence items are region boxes and OCR labels, several per room, so matching is by centre: a predicted
  item is right if its centre lies in a labelled room of that type; a labelled room is found if any does.
  Low balcony precision is expected: every "outdoor" region is balcony evidence, the labels only count PARV* rooms.
- counts: the fusion-model count per template against the label count (exact match). Same numbers the questions
  ask for, without the question.
- walls (--walls): wall-pixel F1 (3 px tolerance, annotated floors only) of the wall segmenter, or of the CubiCasa
  net when the segmenter's weights are missing.
"""

import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.models import PlanGraph, PlanItem

# symbol classes: name -> (category, classes); empty classes = any
SYMBOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "door": ("door", ()),
    "window": ("window", ()),
    "toilet": ("fixture", ("toilet",)),
    "sink": ("fixture", ("sink", "double_sink")),
    "shower_bath": ("fixture", ("shower", "bathtub")),
    "sauna_stove": ("fixture", ("sauna_stove",)),
}
# no "sauna": the SVGs rarely label sauna rooms as such (Undefined / Bath), so there is nothing to match against
ROOMS = ("bedroom", "bathroom", "wc", "kitchen", "balcony")
COUNT_TEMPLATES = ("doors", "windows", "floors", "wc_seats", "sinks", "showers_baths", "saunas", "bedrooms",
                   "bathrooms", "wc_rooms", "kitchens", "balconies")

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_boxes(pred: list[Box], truth: list[Box], thr: float) -> dict:
    """One-to-one greedy matching by descending IoU. {"tp", "fp", "fn", "iou": sum of matched IoUs}."""
    pairs = sorted(((iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)), reverse=True)
    used_p, used_t, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            total += v
    tp = len(used_p)
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(truth) - tp, "iou": round(total, 4)}


def _inside(x: float, y: float, room: PlanItem) -> bool:
    x0, y0, x1, y1 = room.bbox
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return False
    if not room.polygon or len(room.polygon) < 3:
        return True
    inside, pts = False, room.polygon  # ray casting
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def match_rooms(pred: list[PlanItem], truth: list[PlanItem]) -> dict:
    """Centre-in-room matching, many predictions to one room. tp/fn count labelled rooms, fp predicted items."""
    found, wrong = set(), 0
    for p in pred:
        cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
        hits = [j for j, t in enumerate(truth) if _inside(cx, cy, t)]
        found.update(hits)
        wrong += not hits
    return {"tp": len(found), "fp": wrong, "fn": len(truth) - len(found), "pred": len(pred)}


def _visible(it: PlanItem, w: int, h: int) -> bool:
    return it.bbox[2] > 0 and it.bbox[3] > 0 and it.bbox[0] < w and it.bbox[1] < h


def compare(model: PlanGraph, labels: PlanGraph, iou_thr: float = 0.3) -> dict:
    """One image: {"symbols": {cls: tp/fp/fn/iou}, "rooms": {type: tp/fp/fn/pred}, "counts": {template: [pred, truth]}}."""
    from app.cubicasa.answer import count

    w, h = labels.width, labels.height
    truth = [it for it in labels.items if _visible(it, w, h)]  # labels outside the image cannot be found
    symbols = {}
    for name, (cat, classes) in SYMBOLS.items():
        sel = lambda items: [it.bbox for it in items if it.category == cat and (not classes or it.cls in classes)]  # noqa: E731
        symbols[name] = match_boxes(sel(model.items), sel(truth), iou_thr)
    rooms = {}
    for room in ROOMS:
        sel = lambda items: [it for it in items if it.category == "room" and it.cls == room]  # noqa: E731
        rooms[room] = match_rooms(sel(model.items), sel(truth))
    counts = {t: [model.counts[t], count(labels, t).answer] for t in COUNT_TEMPLATES if t in model.counts}
    return {"symbols": symbols, "rooms": rooms, "counts": counts}


# ---------------------------------------------------------------------------------------------- run


def _wall_f1(rel: str) -> dict:
    from app.cubicasa import walls, wallseg

    model = "wallseg" if wallseg.available() else "cubicasa"
    res, *_ = walls.evaluate(rel, vector=False, model=model)
    return {"model": model, "f1": round(res["pixel_annotated"]["f1"], 4),
            "solid_f1": round(res["solid_annotated"]["f1"], 4)}


def run(set_name: str, limit: int, iou_thr: float, with_walls: bool, label: str) -> Path:
    from app import bench
    from app.cubicasa import graph as graph_mod
    from app.cubicasa.graph import get_graph

    set_path = bench.BENCH_DIR / f"{set_name}.jsonl"
    if not set_path.is_file():
        raise SystemExit(f"{set_path} missing: run `python -m app.bench build` first")
    paths = sorted({it["path"] for it in bench._read_jsonl(set_path)
                    if it["kind"] == "dataset" and it["dataset"] == "cubicasa5k"})
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no cubicasa images in {set_path}")

    meta = {"kind": "perception", "set": set_name, "set_hash": hashlib.sha1("\n".join(paths).encode()).hexdigest()[:12],
            "iou": iou_thr, "walls": with_walls, "label": label, "git": bench._git(),
            "graph_cache": graph_mod._cache_path(paths[0], "model").parent.name,
            "started": datetime.now(UTC).isoformat(timespec="seconds")}
    recs, t0 = [], time.perf_counter()
    for i, rel in enumerate(paths, 1):
        rec = {"path": rel, "error": None}
        try:
            rec.update(compare(get_graph(rel, "model"), get_graph(rel, "svg"), iou_thr))
            if with_walls:
                rec["walls"] = _wall_f1(rel)
        except FileNotFoundError as e:  # weights or labels missing
            rec["error"] = str(e)
        recs.append(rec)
        if i % 25 == 0 or i == len(paths):
            print(f"  images {i}/{len(paths)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    meta["duration_s"] = round(time.perf_counter() - t0, 1)

    bench.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    name = "-".join(x for x in [stamp, meta["git"]["sha"] + ("+" if meta["git"]["dirty"] else ""), set_name,
                                "perception", label] if x)
    out = bench.RUNS_DIR / f"{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    report(out)
    print(f"\nrun saved: {out}")
    return out


# ---------------------------------------------------------------------------------------------- report


def _load(path: Path) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[0]["meta"], [r for r in rows[1:] if not r.get("error")]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r, 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def summary(recs: list[dict]) -> dict:
    """{"symbols"/"rooms": {cls: tp, fp, fn, iou, f1, ...}, "counts": {template: (exact, n)}, "walls": {...}}."""
    out: dict = {"symbols": {}, "rooms": {}, "counts": {}}
    for group in ("symbols", "rooms"):
        tot = defaultdict(lambda: defaultdict(float))
        for r in recs:
            for cls, m in r[group].items():
                for k, v in m.items():
                    tot[cls][k] += v
        for cls, m in tot.items():
            p, rc, f1 = _prf(int(m["tp"]), int(m["fp"]), int(m["fn"]))
            out[group][cls] = {"n": int(m["tp"] + m["fn"]), "precision": p, "recall": rc, "f1": f1,
                               "iou": m["iou"] / m["tp"] if group == "symbols" and m["tp"] else float("nan")}
    hits = defaultdict(lambda: [0, 0])
    for r in recs:
        for t, (pred, truth) in r["counts"].items():
            hits[t][0] += pred == truth
            hits[t][1] += 1
    out["counts"] = {t: tuple(v) for t, v in hits.items()}
    walls = [r["walls"] for r in recs if r.get("walls")]
    if walls:
        out["walls"] = {"model": walls[0]["model"], "n": len(walls),
                        "f1": sum(w["f1"] for w in walls) / len(walls),
                        "solid_f1": sum(w["solid_f1"] for w in walls) / len(walls)}
    return out


def _f(x: float) -> str:
    return f"{x:8.3f}" if x == x else f"{'-':>8}"


def report(path: Path) -> None:
    from app.bench import _cluster_ci

    meta, recs = _load(Path(path))
    s = summary(recs)
    g = meta["git"]
    print(f"\nperception set={meta['set']} ({meta['set_hash']}) images={len(recs)} iou>={meta['iou']} "
          f"git={g['sha']}{'+dirty' if g['dirty'] else ''} graphs={meta['graph_cache']}")

    pairs = [(r["path"], 0.0, float(p == t)) for r in recs for p, t in r["counts"].values()]
    if pairs:
        lo, hi = _cluster_ci(pairs)
        print(f"\nHEADLINE counts exact: {sum(b for _, _, b in pairs) / len(pairs):.1%} "
              f"[95% CI {lo:.1%}-{hi:.1%}]  (n={len(pairs)}, {len(recs)} images)")

    print(f"\n{'symbols (boxes)':<16}{'labels':>7}{'prec':>8}{'recall':>8}{'F1':>8}{'IoU':>8}")
    for cls in SYMBOLS:
        if cls in s["symbols"]:
            m = s["symbols"][cls]
            print(f"{cls:<16}{m['n']:>7}{_f(m['precision'])}{_f(m['recall'])}{_f(m['f1'])}{_f(m['iou'])}")
    print(f"\n{'rooms (centre)':<16}{'labels':>7}{'prec':>8}{'recall':>8}{'F1':>8}")
    for cls in ROOMS:
        if cls in s["rooms"]:
            m = s["rooms"][cls]
            print(f"{cls:<16}{m['n']:>7}{_f(m['precision'])}{_f(m['recall'])}{_f(m['f1'])}")
    print(f"\n{'counts':<16}{'n':>7}{'exact':>8}")
    for t in COUNT_TEMPLATES:
        if t in s["counts"]:
            ok, n = s["counts"][t]
            print(f"{t:<16}{n:>7}{ok / n:>8.1%}")
    if "walls" in s:
        w = s["walls"]
        print(f"\nwalls ({w['model']}, {w['n']} images): F1 {w['f1']:.3f}  solid-wall F1 {w['solid_f1']:.3f}")
    print(f"\n{meta.get('duration_s')}s")


def diff(path_a: Path, path_b: Path, show: int = 10) -> None:
    from app.bench import _cluster_ci

    ma, ra = _load(Path(path_a))
    mb, rb = _load(Path(path_b))
    if ma.get("kind") != "perception" or mb.get("kind") != "perception":
        raise SystemExit("both runs must be perception runs")
    a, b = {r["path"]: r for r in ra}, {r["path"]: r for r in rb}
    shared = sorted(set(a) & set(b))
    if ma["set_hash"] != mb["set_hash"]:
        print(f"WARNING: different image sets: comparing the {len(shared)} shared images only")
    if ma["iou"] != mb["iou"]:
        print(f"WARNING: different IoU thresholds ({ma['iou']} vs {mb['iou']}): the symbol rows are not comparable")
    label = lambda m: f"{m['git']['sha']}{'+' if m['git']['dirty'] else ''} {m.get('label') or ''} graphs={m['graph_cache']}"  # noqa: E731
    print(f"A = {label(ma)}  ({Path(path_a).name})\nB = {label(mb)}  ({Path(path_b).name})")
    sa, sb = summary([a[p] for p in shared]), summary([b[p] for p in shared])

    for group, classes in (("symbols", SYMBOLS), ("rooms", ROOMS)):
        print(f"\n{group + ' F1':<16}{'labels':>7}{'A':>8}{'B':>8}{'delta':>8}")
        for cls in classes:
            if cls in sa[group] and cls in sb[group]:
                fa, fb = sa[group][cls]["f1"], sb[group][cls]["f1"]
                print(f"{cls:<16}{sa[group][cls]['n']:>7}{_f(fa)}{_f(fb)}{fb - fa:>+8.3f}")

    print(f"\n{'counts exact':<16}{'n':>7}{'A':>8}{'B':>8}{'delta':>8} 95% CI (bootstrap over images; * = excludes 0)")

    def line(name: str, templates) -> None:
        pairs = [(p, float(a[p]["counts"][t][0] == a[p]["counts"][t][1]),
                  float(b[p]["counts"][t][0] == b[p]["counts"][t][1]))
                 for p in shared for t in templates if t in a[p]["counts"] and t in b[p]["counts"]]
        if not pairs:
            return
        pa, pb = sum(x for _, x, _ in pairs) / len(pairs), sum(y for _, _, y in pairs) / len(pairs)
        lo, hi = _cluster_ci(pairs)
        print(f"{name:<16}{len(pairs):>7}{pa:>8.1%}{pb:>8.1%}{pb - pa:>+8.1%} [{lo:+.1%}, {hi:+.1%}] "
              f"{'*' if lo > 0 or hi < 0 else ' '}")

    line("ALL", COUNT_TEMPLATES)
    for t in COUNT_TEMPLATES:
        line(t, [t])
    if "walls" in sa and "walls" in sb:
        print(f"\nwalls F1 {sa['walls']['f1']:.3f} -> {sb['walls']['f1']:.3f}   "
              f"solid {sa['walls']['solid_f1']:.3f} -> {sb['walls']['solid_f1']:.3f}")

    def errors(r: dict) -> int:  # unmatched boxes + wrong counts on one image
        return (sum(m["fp"] + m["fn"] for m in r["symbols"].values())
                + sum(p != t for p, t in r["counts"].values()))

    moved = sorted(((errors(b[p]) - errors(a[p]), p) for p in shared if errors(b[p]) != errors(a[p])))
    print(f"\nimages better {sum(d < 0 for d, _ in moved)}   worse {sum(d > 0 for d, _ in moved)}   "
          f"(errors = unmatched symbol boxes + wrong counts)")
    for d, p in moved[:show] + moved[max(show, len(moved) - show):]:
        print(f"  {'WORSE ' if d > 0 else 'BETTER'} {d:+4d}  {p.rsplit('/', 1)[-1]}")
