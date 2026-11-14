"""Exact-match eval on cubicasa5k questions, per template.

  python -m app.eval --method mode            # majority-answer baseline
  python -m app.eval --method svg             # oracle: original CubiCasa labels (upper bound)
  python -m app.eval --method model --limit 200
"""

import argparse
import json
import random
import zlib
from collections import Counter, defaultdict

from app.config import CUBICASA_ORIG_DIR, DATASET_DIR
from app.cubicasa.answer import count
from app.cubicasa.graph import get_graph
from app.router import route


def load_questions() -> list[dict]:
    out = []
    with open(DATASET_DIR / "questions.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["path"].startswith("data/cubicasa5k/"):
                out.append(q)
    return out


def _orig_split() -> dict[str, str]:
    """Map 'variant-id' -> train/val/test using the original CubiCasa5K split files, if present."""
    m = {}
    for split in ("train", "val", "test"):
        f = CUBICASA_ORIG_DIR / f"{split}.txt"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").split():
                parts = line.strip("/").split("/")
                if len(parts) >= 2:
                    m[f"{parts[-2]}-{parts[-1]}"] = split
    return m


_SPLIT = None


def split_of(path: str) -> str:
    global _SPLIT
    if _SPLIT is None:
        _SPLIT = _orig_split()
    stem = path.rsplit("/", 1)[-1].removesuffix(".png")
    if stem in _SPLIT:
        return _SPLIT[stem]
    h = zlib.crc32(stem.encode()) % 10  # fallback: deterministic 80/10/10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["mode", "svg", "model"], default="mode")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    ap.add_argument("--all-languages", action="store_true", help="keep EN+FI duplicates (same answers)")
    ap.add_argument("--show-errors", type=int, default=0)
    args = ap.parse_args()

    qs, seen = [], set()
    for q in load_questions():  # EN and FI variants share answers; keep one per (image, template)
        key = (q["path"], route(q["question"]))
        if args.all_languages or key not in seen:
            seen.add(key)
            qs.append(q)
    train = [q for q in qs if split_of(q["path"]) == "train"]
    test = [q for q in qs if split_of(q["path"]) == args.split]
    if args.limit:
        paths = sorted({q["path"] for q in test})
        random.Random(0).shuffle(paths)
        keep = set(paths[: args.limit])
        test = [q for q in test if q["path"] in keep]

    mode = {}
    if args.method == "mode":
        by_t = defaultdict(Counter)
        for q in train:
            by_t[route(q["question"])][q["answer"]] += 1
        mode = {t: c.most_common(1)[0][0] for t, c in by_t.items()}

    hit, tot, errors, failed = Counter(), Counter(), [], Counter()
    for q in test:
        t = route(q["question"])
        if args.method == "mode":
            pred = mode.get(t)
        else:
            try:
                pred = count(get_graph(q["path"], args.method), t).answer
            except FileNotFoundError:
                failed[t] += 1
                pred = None
        tot[t] += 1
        if pred == q["answer"]:
            hit[t] += 1
        else:
            errors.append((q["path"], t, pred, q["answer"]))

    print(f"method={args.method} split={args.split} images={len({q['path'] for q in test})} questions={len(test)}")
    print(f"{'template':<15}{'n':>6}{'exact':>8}")
    for t in sorted(tot):
        extra = f"  ({failed[t]} unparsed)" if failed[t] else ""
        print(f"{t:<15}{tot[t]:>6}{hit[t] / tot[t]:>8.1%}{extra}")
    print(f"{'ALL':<15}{sum(tot.values()):>6}{sum(hit.values()) / max(1, sum(tot.values())):>8.1%}")
    for e in errors[: args.show_errors]:
        print("ERR", *e)


if __name__ == "__main__":
    main()
