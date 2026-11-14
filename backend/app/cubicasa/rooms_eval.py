"""Room geometry (app.cubicasa.rooms) against the CubiCasa SVG ground truth.

  DATASET_DIR=../dataset/junction-dataset-preview uv run python -m app.cubicasa.rooms_eval [--limit 120] [--label x]

Plans: preview plans whose model graph is cached (no perception runs here). Per plan:
  merged      our spaces holding >= 2 ground-truth rooms (open plan kitchen/living/dining/hall counted apart: CubiCasa
              splits open plans with invisible lines, so those merges are acceptable)
  bad merged  merged spaces holding a bedroom, bath, WC or sauna together with another room
  over-split  ground-truth rooms whose area is spread over >= 2 of our spaces (each >= 25% of it)
  counts      bedrooms / bathrooms / wc_rooms / kitchens / saunas from the strong rooms, exact match vs the ground
              truth, next to the counting model's answer (graph.counts; optimistic on preview, 197/241 in its train)
Even / odd plan halves are reported separately (tune on even, read odd).
"""

import argparse
import json
import time

from shapely.geometry import Polygon

from app.config import CACHE_DIR, DATASET_DIR
from app.models import PlanGraph

OPEN_PLAN = {"Kitchen", "LivingRoom", "Dining", "Entry", "Hall", "DraughtLobby", "Room", "Undefined"}
PRIVATE = {"Bedroom", "Bath", "Sauna"}
TEMPLATES = ("bedrooms", "bathrooms", "wc_rooms", "kitchens", "saunas")


def _main_class(it) -> str:
    toks = (it.raw or "").split()
    return toks[1] if len(toks) > 1 else ""


def geometry_counts(geo) -> dict[str, int]:
    strong = [r for r in geo.rooms if r.tier == "strong"]

    def n(t):
        return sum(t in (r.purposes or [r.type]) for r in strong)
    return {"bedrooms": n("bedroom"), "bathrooms": n("bathroom"), "wc_rooms": n("wc"), "kitchens": n("kitchen"),
            "saunas": n("sauna")}


def evaluate_plan(rel: str) -> dict:
    from app.cubicasa.answer import count
    from app.cubicasa.graph import get_graph
    from app.cubicasa.rooms import geometry_from_image

    stem = rel.rsplit("/", 1)[-1][:-4]
    gt = PlanGraph.model_validate_json((CACHE_DIR / "svg" / f"{stem}.json").read_text(encoding="utf-8"))
    graph = get_graph(rel, "model")
    geo = geometry_from_image(rel, DATASET_DIR / rel, graph)
    rooms = [it for it in gt.items if it.category == "room" and it.polygon and len(it.polygon) >= 3
             and _main_class(it) not in ("Outdoor",)]
    gpolys = [Polygon(g.polygon).buffer(0) for g in rooms]
    ours = [Polygon(r.polygon).buffer(0) for r in geo.rooms if len(r.polygon) >= 3]
    merged = bad = 0
    for P in ours:
        inside = [g for g, gp in zip(rooms, gpolys) if gp.area > 0 and P.contains(gp.representative_point())]
        if len(inside) >= 2:
            classes = {_main_class(g) for g in inside}
            if not classes <= OPEN_PLAN:
                merged += 1
                bad += bool(classes & PRIVATE)
    over = 0
    for gp in gpolys:
        if gp.area <= 0:
            continue
        parts = [P.intersection(gp).area / gp.area for P in ours if P.intersects(gp)]
        over += sum(p >= 0.25 for p in parts) >= 2
    gc = geometry_counts(geo)
    return {"plan": stem, "spaces": len(geo.rooms), "gt_rooms": len(rooms), "merged": merged, "bad_merged": bad,
            "over_split": over,
            "counts": {t: [gc[t], graph.counts.get(t), count(gt, t).answer] for t in TEMPLATES}}


def report(recs: list[dict], name: str) -> None:
    n = max(1, len(recs))
    tot = {k: sum(r[k] for r in recs) for k in ("spaces", "gt_rooms", "merged", "bad_merged", "over_split")}
    line = ", ".join(f"{k} {v}" for k, v in tot.items())
    geo = {t: sum(r["counts"][t][0] == r["counts"][t][2] for r in recs) / n for t in TEMPLATES}
    fus = {t: sum(r["counts"][t][1] == r["counts"][t][2] for r in recs) / n for t in TEMPLATES}
    print(f"{name} ({len(recs)} plans): {line}")
    print("   exact  " + "  ".join(f"{t} geo {geo[t]:.0%} / model {fus[t]:.0%}" for t in TEMPLATES))


def main() -> None:
    from app.cubicasa.graph import _cache_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="current")
    a = ap.parse_args()
    imgs = sorted((DATASET_DIR / "data" / "cubicasa5k").glob("*.png"))
    plans = [(n, f"data/cubicasa5k/{p.name}") for n, p in enumerate(imgs)
             if _cache_path(f"data/cubicasa5k/{p.name}", "model").is_file() and (CACHE_DIR / "svg" / f"{p.stem}.json").is_file()]
    if a.limit:
        plans = plans[: a.limit]
    recs, t0 = [], time.time()
    for k, (n, rel) in enumerate(plans):
        try:
            r = evaluate_plan(rel)
            r["idx"] = k  # halves alternate over the evaluated plans
            recs.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"  {rel}: {e!r}")
    out = CACHE_DIR / "rooms_eval" / f"{a.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    print(f"{a.label}: {time.time() - t0:.0f} s -> {out}")
    report([r for r in recs if r["idx"] % 2 == 0], "even")
    report([r for r in recs if r["idx"] % 2 == 1], "odd ")


if __name__ == "__main__":
    main()
