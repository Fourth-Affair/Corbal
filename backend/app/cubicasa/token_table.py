"""Learn an OCR-token -> room-type table from train images (OCR tokens matched to ground-truth room polygons).

Complements the hand-written dictionary in rooms_dict.py with drawn variants and typical OCR misreads.

  python -m app.cubicasa.token_table
"""

import json
from collections import Counter, defaultdict

from shapely.geometry import Point, Polygon

from app.config import MODELS_DIR
from app.cubicasa.features import feature_paths
from app.cubicasa.rooms_dict import normalise

TABLE_PATH = MODELS_DIR / "ocr_token_table.json"
TYPES = ("bedroom", "bathroom", "wc", "sauna", "kitchen", "balcony")


def learn(min_count: int = 3, min_purity: float = 0.7) -> dict[str, str]:
    from app.cubicasa.graph import get_graph
    from app.eval import load_questions, split_of

    stats: dict[str, Counter] = defaultdict(Counter)
    paths = sorted({q["path"] for q in load_questions() if split_of(q["path"]) == "train"})
    n_img = 0
    for rel in paths:
        js = feature_paths(rel)[1]
        if not js.exists():
            continue
        n_img += 1
        ocr = json.loads(js.read_text(encoding="utf-8"))["ocr"]
        rooms = [(Polygon(it.polygon).buffer(0), it.cls) for it in get_graph(rel, "svg").items
                 if it.category == "room" and it.polygon and len(it.polygon) >= 3]
        for tok in ocr:
            key = normalise(tok["text"])
            if not key or len(key) > 20:
                continue
            x0, y0, x1, y1 = tok["bbox"]
            c = Point((x0 + x1) / 2, (y0 + y1) / 2)
            cls = next((cl for poly, cl in rooms if poly.contains(c)), "outside")
            stats[key][cls] += 1
    table = {}
    for key, cnt in stats.items():
        total = sum(cnt.values())
        cls, k = cnt.most_common(1)[0]
        if total >= min_count and k / total >= min_purity and cls in TYPES:
            table[key] = cls
    print(f"learned {len(table)} tokens from {n_img} train images")
    return table


_TABLE: dict[str, str] | None = None


def lookup(token: str) -> str | None:
    global _TABLE
    if _TABLE is None:
        _TABLE = json.loads(TABLE_PATH.read_text(encoding="utf-8")) if TABLE_PATH.exists() else {}
    return _TABLE.get(normalise(token))


if __name__ == "__main__":
    t = learn()
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(json.dumps(t, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    by = defaultdict(list)
    for k, v in t.items():
        by[v].append(k)
    for v, ks in sorted(by.items()):
        print(f"{v:<9} {len(ks):>3}  {sorted(ks)[:25]}")
