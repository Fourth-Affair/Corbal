"""Score the PDF parser on questions.jsonl: counts exact, areas within 2% (the challenge's scoring rules).

  uv run python -m app.pdfplan.eval                                         # every question about a .pdf
  uv run python -m app.pdfplan.eval --prefix data/swiss_dwellings/ --split eval
  uv run python -m app.pdfplan.eval --prefix data/swiss_dwellings/ --no-dxf # PDF only: text + image detector
  uv run python -m app.pdfplan.eval --agent --every 10                      # through the Gemini agent (PdfTools), as
                                                                            # /api/ask; every 10th question

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
    ap.add_argument("--agent", action="store_true", help="answer through the Gemini agent with PdfTools (fallback: "
                    "the deterministic answerer when Gemini is unavailable), as /api/ask does")
    ap.add_argument("--every", type=int, default=1, help="only every Nth question (a sample across plans and types)")
    ap.add_argument("--workers", type=int, default=8, help="parallel questions with --agent")
    args = ap.parse_args()

    qs = [json.loads(line) for line in open(DATASET_DIR / "questions.jsonl", encoding="utf-8")]
    qs = [q for q in qs if q["path"].lower().endswith(".pdf") and q["path"].startswith(args.prefix)]
    files = sorted({q["path"] for q in qs})
    keep = {"all": set(files), "dev": set(files[0::2]), "eval": set(files[1::2])}[args.split]
    qs = [q for q in qs if q["path"] in keep][:: max(1, args.every)]

    def ask(q):
        if not args.agent:
            return answer_question(q["path"], q["question"], use_dxf=not args.no_dxf)
        from app.answering.agent import answer_with
        from app.pdfplan.tools import PdfTools

        tools = PdfTools(q["path"], use_dxf=not args.no_dxf)
        return answer_with(q["question"], tools) or tools.fallback(q["question"])

    def one(q):
        try:
            return ask(q), None
        except Exception as e:  # noqa: BLE001  one broken file must not stop the eval
            return None, f"error:{type(e).__name__}"

    t0 = time.time()
    if args.agent:  # parse each PDF once, sequentially (pdfium is not thread-safe), then ask in parallel
        from concurrent.futures import ThreadPoolExecutor

        from app.bench import _load_dotenv, _warm_documents

        _load_dotenv()
        _warm_documents(sorted({q["path"] for q in qs}))
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(one, qs))
    else:
        results = [one(q) for q in qs]

    score = defaultdict(lambda: [0, 0])
    misses = []
    for q, (a, error) in zip(qs, results, strict=True):
        parsed = parse_question(q["question"])
        template = error or (parsed[0] if parsed else "unparsed")
        pred = a.answer if a is not None else None
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
