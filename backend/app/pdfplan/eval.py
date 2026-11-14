"""Score the PDF parser on questions.jsonl: counts exact, areas within 2% (the challenge's scoring rules).

  uv run python -m app.pdfplan.eval                                         # every question about a .pdf
  uv run python -m app.pdfplan.eval --prefix data/swiss_dwellings/ --split eval
  uv run python -m app.pdfplan.eval --prefix data/swiss_dwellings/ --no-dxf # PDF only: text + image detector

--split dev/eval halves the selected files (sorted by path, even/odd). The vector rules were written while looking
at the even-indexed swiss_dwellings plans of the preview set, so the odd half is the held-out number there.
"""

import argparse
import json
import time
from collections import defaultdict

from app.config import DATASET_DIR
from app.pdfplan.answer import answer_question, parse_question


def correct(pred, truth, unit: str) -> bool:
    if pred is None or isinstance(pred, str):
        return False
    if unit == "count":
        return pred == truth
    try:
        return abs(float(pred) - float(truth)) <= 0.02 * abs(float(truth))
    except (TypeError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="", help="only questions whose path starts with this")
    ap.add_argument("--split", choices=["all", "dev", "eval"], default="all")
    ap.add_argument("--no-dxf", action="store_true", help="ignore same-name DXF files")
    ap.add_argument("--show", type=int, default=10, help="print this many misses")
    args = ap.parse_args()

    qs = [json.loads(line) for line in open(DATASET_DIR / "questions.jsonl", encoding="utf-8")]
    qs = [q for q in qs if q["path"].lower().endswith(".pdf") and q["path"].startswith(args.prefix)]
    files = sorted({q["path"] for q in qs})
    keep = {"all": set(files), "dev": set(files[0::2]), "eval": set(files[1::2])}[args.split]
    qs = [q for q in qs if q["path"] in keep]

    score = defaultdict(lambda: [0, 0])
    misses = []
    t0 = time.time()
    for q in qs:
        parsed = parse_question(q["question"])
        template = parsed[0] if parsed else "unparsed"
        try:
            a = answer_question(q["path"], q["question"], use_dxf=not args.no_dxf)
            pred = a.answer
        except Exception as e:  # noqa: BLE001  one broken file must not stop the eval
            pred = None
            template = f"error:{type(e).__name__}"
        ok = correct(pred, q["answer"], q["unit"])
        for key in (template, "ALL"):
            score[key][0] += ok
            score[key][1] += 1
        if not ok:
            misses.append((q["path"], q["question"][:90], pred, q["answer"]))

    print(f"{len(keep)} PDFs, {len(qs)} questions ({args.split}{', no DXF' if args.no_dxf else ''}) "
          f"in {time.time() - t0:.0f}s")
    for k in sorted(score, key=lambda k: (k == "ALL", k)):
        ok, n = score[k]
        print(f"  {k:24s} {ok:5d}/{n:<5d} {100 * ok / n:5.1f}%")
    for m in misses[: args.show]:
        print("  miss:", *m)


if __name__ == "__main__":
    main()
