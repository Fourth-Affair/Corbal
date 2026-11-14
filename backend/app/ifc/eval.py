"""Score the deterministic IFC answerer on questions.jsonl.

  uv run python -m app.ifc.eval                          # every .ifc question of DATASET_DIR (full dataset)
  uv run python -m app.ifc.eval --preview                # the preview release (= our test set; never tune on it)
  uv run python -m app.ifc.eval --preview --agent --type count   # through the Gemini agent (IfcTools), as the API
  uv run python -m app.ifc.eval --type count --show 30   # one answer type, print misses

Scoring:
  count                  exact integer
  area / volume / length within 2% relative error (the challenge rule)
  text                   the gold answers are free text written by people / an LLM, so this is approximate:
                         - gold says the information is not in the model -> correct if we answer not_found or "No ..."
                         - short gold (<= 8 words, no numbers) -> normalised containment either way
                         - gold with numbers -> correct if >= 80% of its numbers (not taken from the question) appear in
                           our answer + explanation (2% tolerance)
                         - other long gold -> content-word overlap >= 50%
Reported separately: accuracy (correct / all), coverage (answered = not not_found / all), precision (correct
answered / answered). A correct not_found (gold says "not available") counts for accuracy, not for coverage.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from app.config import DATASET_DIR

_NA = re.compile(r"not (directly )?(available|contained|included|present|stored|specified|possible|defined|modell?ed|provided|found)|"
                 r"cannot be|no information|no (explicit |dedicated )?[\w /-]+ (were |was )?found|does not contain|isn't available|"
                 r"is not available|are not included|not in the (bim )?model|no data", re.I)
_NUM = re.compile(r"(?<![\w.])-?\d+(?:[ ,]\d{3})*(?:[.,]\d+)?(?![\w])")
_STOP = set("the a an of in on and or is are was were with for to by this that these those there their its it as at be "
            "from has have total all each per which what model building".split())


def _numbers(s: str) -> list[float]:
    out = []
    for t in _NUM.findall(s or ""):
        t = t.replace(" ", "")
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", t):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-zäöüåß0-9]+", (s or "").casefold()) if w not in _STOP and len(w) > 2}


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 0.02 * abs(b) + 1e-9


def score_text(pred_answer, explanation: str, confidence: str, gold: str, question: str) -> bool:
    ours = f"{pred_answer if pred_answer is not None else ''}\n{explanation or ''}"
    gold = str(gold)
    if _NA.search(gold) and not _numbers(gold.split("However")[0]) or re.match(r"^\s*no\b", gold, re.I):
        return confidence == "not_found" or bool(re.search(r"(^|\n|\. )no\b|\bnone of\b|not (found|available)|\b0 ", ours, re.I))
    qnums = _numbers(question)
    gnums = [x for x in _numbers(gold) if not any(_close(x, q) for q in qnums)]
    if confidence == "not_found":
        return False
    if gnums:
        onums = _numbers(ours)
        hit = sum(1 for g in gnums if any(_close(o, g) for o in onums))
        return hit >= 0.8 * len(gnums)
    gw, ow = _words(gold), _words(ours)
    if len(gold.split()) <= 8:
        g, o = " ".join(sorted(gw)), ours.casefold()
        return bool(gw) and (all(w in o for w in gw) or (bool(ow) and ow <= gw))
    return bool(gw) and len(gw & ow) >= 0.5 * len(gw)


def correct(a, q) -> bool:
    gold, typ = q["answer"], q["type"]
    if typ == "text":
        return score_text(a.answer, a.explanation, a.confidence, gold, q["question"])
    pred = a.answer
    if pred is None or isinstance(pred, str) or isinstance(pred, bool):
        return False
    try:
        if typ == "count" or q.get("unit") == "count":
            return float(pred) == float(gold)
        return _close(float(pred), float(gold))
    except (TypeError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="use dataset/junction-dataset-preview (test set)")
    ap.add_argument("--dataset", type=Path, default=None, help="dataset root with questions.jsonl and data/")
    ap.add_argument("--prefix", default="", help="only questions whose path starts with this")
    ap.add_argument("--type", default=None, help="only this answer type (count, area, volume, length, text)")
    ap.add_argument("--show", type=int, default=0, help="print this many misses")
    ap.add_argument("--out", type=Path, default=None, help="write per-question results (jsonl)")
    ap.add_argument("--agent", action="store_true", help="answer through the Gemini agent with IfcTools (fallback: "
                    "the deterministic answerer when Gemini is unavailable), as /api/ask does")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (after filters)")
    ap.add_argument("--workers", type=int, default=8, help="parallel questions with --agent")
    args = ap.parse_args()

    root = args.dataset or (DATASET_DIR.parent / "junction-dataset-preview" if args.preview else DATASET_DIR)
    import app.ifc.model as model_mod

    model_mod.DATASET_DIR = root  # resolve paths against the chosen release
    from app.ifc.answer import answer_question

    if args.agent:
        from app.answering.agent import answer_with
        from app.bench import _load_dotenv
        from app.ifc.agent_tools import IfcTools
        from app.ifc.model import load_model

        _load_dotenv()

        def answer_question(path, question):  # noqa: F811  same routing as /api/ask
            tools = IfcTools(load_model(path), path)
            return answer_with(question, tools) or tools.fallback(question)

    qs = [json.loads(line) for line in open(root / "questions.jsonl", encoding="utf-8")]
    qs = [q for q in qs if q["path"].lower().endswith(".ifc") and q["path"].startswith(args.prefix)
          and (args.type is None or q["type"] == args.type)]
    if args.limit:
        qs = qs[: args.limit]
    stats = defaultdict(lambda: [0, 0, 0, 0])  # n, correct, answered, correct&answered
    misses, out = [], []
    t0 = time.time()
    def one(q):
        from app.models import Answer

        try:
            return answer_question(q["path"], q["question"])
        except FileNotFoundError:
            return Answer(answer=None, confidence="not_found", explanation="file missing")
        except Exception as e:  # noqa: BLE001  one broken question must not stop the eval
            return Answer(answer=None, confidence="not_found", explanation=f"error: {type(e).__name__}: {e}")

    if args.agent:  # models parsed once, sequentially, then questions in parallel (network-bound)
        from concurrent.futures import ThreadPoolExecutor

        from app.ifc.model import load_model

        for path in sorted({q["path"] for q in qs}):
            try:
                load_model(path)
            except Exception:  # noqa: BLE001
                pass
        with ThreadPoolExecutor(args.workers) as pool:
            answers = list(pool.map(one, qs))
    else:
        answers = [one(q) for q in qs]
    for q, a in zip(qs, answers, strict=True):
        ok = correct(a, q)
        answered = a.confidence != "not_found" and a.answer is not None
        for key in (q["type"], f"{q['type']}:{a.template or '-'}", "ALL"):
            s = stats[key]
            s[0] += 1
            s[1] += ok
            s[2] += answered
            s[3] += ok and answered
        if not ok:
            misses.append((q["type"], q["path"].rsplit("/", 1)[-1], q["question"][:110], str(a.answer)[:80].replace("\n", " | "),
                           str(q["answer"])[:80].replace("\n", " | ")))
        if args.out:
            out.append({**q, "pred": a.answer, "confidence": a.confidence, "template": a.template, "correct": ok,
                        "n_citations": len(a.citations), "explanation": a.explanation[:2000]})
    print(f"{root.name}: {len({q['path'] for q in qs})} IFC files, {len(qs)} questions in {time.time() - t0:.0f}s")
    print(f"  {'group':40s} {'n':>5s} {'accuracy':>9s} {'coverage':>9s} {'precision':>9s}")
    for k in sorted(stats, key=lambda k: (k == "ALL", ":" in k, k)):
        n, c, ans, ca = stats[k]
        prec = f"{100 * ca / ans:8.1f}%" if ans else "       -"
        print(f"  {k:40s} {n:5d} {100 * c / n:8.1f}% {100 * ans / n:8.1f}% {prec}")
    for mm in misses[: args.show]:
        print("  miss:", " || ".join(mm))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
