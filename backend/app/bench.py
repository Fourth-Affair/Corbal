"""Benchmark: fixed question sets -> the /api/ask answer path -> per-question records -> summary and diff.

  uv run python -m app.bench build                                   # fixed sets -> dataset/bench/{fast,full}.jsonl
  uv run python -m app.bench run --set fast                          # as the API runs (Gemini agent + fallback)
  uv run python -m app.bench run --set fast --answerer template      # offline: router + template counter
  uv run python -m app.bench run --set fast --source svg --answerer template  # oracle (labels; upper bound)
  uv run python -m app.bench run --set fast --no-cache --label rerun  # fresh Gemini answers (run-to-run noise)
  uv run python -m app.bench run --set fast --kind open              # breadth only: the open questions
  uv run python -m app.bench run --set fast --kind open --source svg --answerer agent  # same, perfect perception
  uv run python -m app.bench report dataset/runs/bench/<run>.jsonl
  uv run python -m app.bench diff <runA>.jsonl <runB>.jsonl --show 20  # paired delta, 95% CI, flipped questions
  uv run python -m app.bench perception --set fast                   # no questions, no LLM: evidence vs SVG labels
                                                                     # (app.bench_perception; diff works on its runs too)

Sets (deterministic from --seed; stored under dataset/, never committed):
- dataset questions: cubicasa5k images of the official *test* split only (both languages, so EN/FI consistency can
  be checked), plus a sample of every other dataset. Plans are routed by file type like the API does: images ->
  image pipeline (agent or template counter), PDFs -> app.pdfplan (deterministic, whatever --answerer says). File
  types without a parser (.ifc) show up as 0% coverage, so the blind spot stays visible.
- paraphrases (app.answering.paraphrases) on test images: unseen wordings and languages. Ground truth = the SVG
  label count for the expected template; trap / out-of-scope questions expect not_found.
- open questions (app.answering.openq) on test images: questions no count template covers (items in one room, rooms
  that have something, yes/no, comparisons, sums, adjacency, relative size) in EN / FI / SV, with the answer computed
  from the SVG labels. This is the breadth measure: the report has a table per question family, and
  `--source svg --answerer agent` shows how far the agent gets with perfect perception.

Scoring: counts exact, other numbers within --tol relative error (areas, lengths, volumes; default 2%, the
challenge rule), short text answers
normalised string match, long free-text answers are not scored (coverage only). When yes or no is expected, the
answer's first word decides ("Kyllä, keittiössä on ikkuna" = yes).
A not_found is never correct unless the expected answer is "not answerable". Accuracy = correct / scored;
precision = correct / answered, so a tweak that trades honest misses for wrong answers shows up.
The first headline weights each dataset's accuracy by its share of questions.jsonl (the samples here are not
proportional: cubicasa is over-represented); datasets with only unscorable free-text answers are left out of it.
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from app.config import CACHE_DIR, DATASET_DIR
from app.eval import split_of
from app.router import route

_REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(os.environ.get("BENCH_DIR", CACHE_DIR.parent / "bench"))
RUNS_DIR = Path(os.environ.get("BENCH_RUNS_DIR", CACHE_DIR.parent / "runs" / "bench"))

SETS = {  # cubicasa test images, paraphrase rounds (one image per paraphrase per round), per other dataset
    # open_per_family: open questions (app.answering.openq) per question family
    "fast": {"cubicasa_images": 16, "paraphrase_rounds": 1, "per_dataset": 10, "open_per_family": 9},
    "full": {"cubicasa_images": 0, "paraphrase_rounds": 5, "per_dataset": 100, "open_per_family": 60},  # 0 = all test images
}
CONFIDENCES = ["high", "medium", "low", "not_found"]

# ---------------------------------------------------------------------------------------------- helpers

_LANG_PATTERNS = [  # first match wins; checked on lowercased text
    ("sv", r"\b(hur|många|finns|ritningen|lägenheten|våningar)\b|å"),
    ("fi", r"\b(kuinka|montako|onko|mikä|monta|pohjapiirustuksessa|asunnossa|piirustuksessa)\b"),
    ("de", r"\b(wie|viele|gibt|diesem|geschoss|welche|räume)\b"),
    ("it", r"\b(quanti|quante|questo|piano|locali)\b"),
    ("es", r"¿|\b(cuál|cuántos|cuántas|número)\b"),
    ("fr", r"\b(combien|quelle|niveau|pièces)\b"),
    ("et", r"\b(mitu|kui|korrusel|ruumi)\b"),
]


def lang_of(question: str) -> str:
    q = question.lower()
    for lang, pat in _LANG_PATTERNS:
        if re.search(pat, q):
            return lang
    return "en"


_YES = {"yes", "kyllä", "ja", "joo", "oui", "sí", "si", "true"}
_NO = {"no", "ei", "nej", "nein", "non", "false"}
_YES_LEAD = _YES | {"on", "jo", "finns"}  # "On." / "Jo, det finns ..." open a yes; only checked on a first word


def _polarity(v) -> str | None:
    """yes / no from the first word of an answer sentence, None if it does not start with either."""
    words = re.findall(r"\w+", str(v).lower())
    if not words:
        return None
    return "yes" if words[0] in _YES_LEAD else "no" if words[0] in _NO else None


def _norm_text(v) -> str:
    s = re.sub(r"[\s.!]+$", "", str(v).strip().lower())
    if s in _YES:
        return "yes"
    if s in _NO:
        return "no"
    return re.sub(r"\s+", " ", s)


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return None


def score(item: dict, pred, tol: float) -> bool | None:
    """True / False, or None when the expected answer cannot be scored automatically (long free text)."""
    exp = item["answer"]
    if exp is None:  # expected: not answerable
        return pred is None
    if pred is None:
        return False
    e = _num(exp)
    if e is not None:
        p = _num(pred)
        if p is None:
            return False
        if item.get("unit") in (None, "count"):
            return p == e
        return abs(p - e) <= tol * max(abs(e), 1e-9)
    if len(str(exp)) > 40 or "\n" in str(exp):
        return None
    if _norm_text(exp) in ("yes", "no"):
        return _polarity(pred) == _norm_text(exp)
    return _norm_text(pred) == _norm_text(exp)


def _qid(path: str, question: str) -> str:
    return hashlib.sha1(f"{path}\n{question}".encode()).hexdigest()[:12]


def _git() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"sha": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def _load_dotenv() -> None:
    """The API gets .env through docker compose; the CLI reads the repo-root .env itself (without overriding)."""
    f = _REPO / ".env"
    if not f.is_file():
        return
    for line in f.read_text().splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip("'\"")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _pct(x: float) -> str:
    return f"{x:6.1%}"


# ---------------------------------------------------------------------------------------------- build


def build(seed: int) -> None:
    from app.answering.paraphrases import PARAPHRASES
    from app.cubicasa.answer import count
    from app.cubicasa.graph import get_graph

    questions = _read_jsonl(DATASET_DIR / "questions.jsonl")
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_ds[q["path"].split("/")[1]].append(q)
    cubi_test = sorted({q["path"] for q in by_ds.pop("cubicasa5k", []) if split_of(q["path"]) == "test"})
    cubi_qs = defaultdict(list)
    for q in questions:
        if q["path"] in cubi_test:
            cubi_qs[q["path"]].append(q)
    if not cubi_test:
        raise SystemExit("no cubicasa test images found (is the dataset unpacked?)")

    def item(q, kind, dataset, template=None):
        return {"id": _qid(q["path"], q["question"]), "kind": kind, "dataset": dataset, "path": q["path"],
                "question": q["question"], "answer": q["answer"], "unit": q.get("unit"), "type": q.get("type"),
                "lang": q.get("lang") or lang_of(q["question"]), "template": template,
                **({"family": q["family"], "asks": q["asks"]} if "family" in q else {})}

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in SETS.items():
        rng = random.Random(f"{seed}-{name}")
        imgs = list(cubi_test)
        rng.shuffle(imgs)
        if cfg["cubicasa_images"]:
            imgs = sorted(imgs[: cfg["cubicasa_images"]])
        items = [item(q, "dataset", "cubicasa5k", route(q["question"])) for p in imgs for q in cubi_qs[p]]

        # paraphrases: round r puts paraphrase i on a different test image; truth from the SVG labels
        para_imgs = list(cubi_test)
        rng.shuffle(para_imgs)
        svg_ok = 0
        for r in range(cfg["paraphrase_rounds"]):
            for i, (question, labels) in enumerate(PARAPHRASES):
                path = para_imgs[(r * len(PARAPHRASES) + i) % len(para_imgs)]
                template = next(iter(labels))
                if template == "none":
                    answer = None
                else:
                    try:
                        answer = count(get_graph(path, "svg"), template).answer
                    except FileNotFoundError:
                        continue  # no SVG labels -> no truth
                    svg_ok += 1
                items.append(item({"path": path, "question": question, "answer": answer, "unit": "count",
                                   "type": "count"}, "paraphrase", "paraphrase",
                                  None if template == "none" else template))

        for ds in sorted(by_ds):
            qs = sorted(by_ds[ds], key=lambda q: _qid(q["path"], q["question"]))
            rng.shuffle(qs)
            items += [item(q, "dataset", ds) for q in qs[: cfg["per_dataset"]]]

        if cfg.get("open_per_family"):  # open questions: truth from the SVG labels of the test images
            from app.answering import openq

            labelled = {}
            for p in cubi_test:
                try:
                    labelled[p] = get_graph(p, "svg")
                except FileNotFoundError:
                    pass
            # own generator: adding or changing open questions leaves the rest of the set as it was
            open_rng = random.Random(f"{seed}-{name}-open")
            items += [item(q, "open", "open") for q in openq.sample(labelled, cfg["open_per_family"], open_rng)]

        out = BENCH_DIR / f"{name}.jsonl"
        out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
        c = Counter(it["dataset"] for it in items)
        print(f"{name}: {len(items)} questions, {len(imgs)} cubicasa test images -> {out}")
        print("   ", ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


# ---------------------------------------------------------------------------------------------- run


def _dataset_weights() -> dict[str, int]:
    """Questions per dataset in questions.jsonl: the weights of the overall headline."""
    c: Counter = Counter()
    with open(DATASET_DIR / "questions.jsonl", encoding="utf-8") as f:
        for line in f:
            m = re.search(r'"path":\s*"data/([^/"]+)/', line)
            if m:
                c[m.group(1)] += 1
    return dict(c)


def _answer_one(item: dict, graphs: dict, answerer: str, use_cache: bool) -> dict:
    """One question through the same routing as /api/ask: by file type, never by dataset folder."""
    from app.main import IMAGE_SUFFIXES, _is_pdf, _template_answer
    from app.models import Answer

    rec = {"id": item["id"], "fell_back": False, "error": None}
    t0 = time.perf_counter()
    path = item["path"]
    graph = graphs.get(path)
    ids = None  # ids a citation may use; None = not checked (PDF answers build their items per question)
    if _is_pdf(path):
        from app.pdfplan.answer import answer_question

        try:
            res = answer_question(path, item["question"])
        except FileNotFoundError as e:
            res = Answer(answer=None, confidence="not_found", explanation=f"Parser unavailable: {e}")
            rec["error"] = "parser"
        except Exception as e:  # noqa: BLE001  the API would return a 500: count it, keep the run going
            res = Answer(answer=None, confidence="not_found", explanation=f"{type(e).__name__}: {e}")
            rec["error"] = "crash"
    elif Path(path).suffix.lower() not in IMAGE_SUFFIXES:
        res = Answer(answer=None, confidence="not_found", explanation="unsupported file type")
        rec["error"] = "unsupported"
    elif isinstance(graph, str):
        res = Answer(answer=None, template=route(item["question"]), confidence="not_found", explanation=graph)
        rec["error"] = "parser"
    else:
        ids = {it.id for it in graph.items}
        res = None
        if answerer in ("llm", "agent"):
            from app.answering import agent

            res = agent.answer(item["question"], graph, use_cache=use_cache)
            if res is None:
                rec["error"] = "agent_failed"
        if res is None and answerer != "agent":
            rec["fell_back"] = answerer == "llm"
            res = _template_answer(item["question"], graph)
        if res is None:
            res = Answer(answer=None, confidence="not_found", explanation="agent failed")
    rec["latency_s"] = round(time.perf_counter() - t0, 3)

    num = _num(res.answer)
    rec.update(pred=res.answer, unit=res.unit, template=res.template, confidence=res.confidence,
               answered_by=res.answered_by, n_citations=len(res.citations), clarification=res.clarification_needed,
               # cited evidence exists in the plan, and a non-zero count has at least one cited item
               citations_ok=(ids is None or all(c.id in ids for c in res.citations))
               and not (num and not res.citations))
    return rec


def run(set_name: str, answerer: str, source: str, use_cache: bool, workers: int, tol: float, label: str,
        limit: int, kind: str = "") -> Path:
    from app.cubicasa import graph as graph_mod
    from app.cubicasa.graph import get_graph

    _load_dotenv()
    set_path = BENCH_DIR / f"{set_name}.jsonl"
    if not set_path.is_file():
        raise SystemExit(f"{set_path} missing: run `python -m app.bench build` first")
    items = _read_jsonl(set_path)
    if kind:
        items = [it for it in items if it["kind"] == kind]
    if limit:
        items = items[:limit]
    set_hash = hashlib.sha1(set_path.read_bytes()).hexdigest()[:12]
    if answerer in ("llm", "agent") and not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY not set: the agent is unavailable"
              + (", every answer falls back to the template path" if answerer == "llm" else ""))

    # graphs first, sequentially (perception may run the nets once per image; cached afterwards)
    from app.main import IMAGE_SUFFIXES, _is_pdf

    paths = sorted({it["path"] for it in items if Path(it["path"]).suffix.lower() in IMAGE_SUFFIXES})
    graphs: dict = {}
    t0 = time.perf_counter()
    for i, p in enumerate(paths, 1):
        try:
            graphs[p] = get_graph(p, source)
        except FileNotFoundError as e:
            graphs[p] = f"Parser unavailable: {e}"
        if i % 25 == 0 or i == len(paths):
            print(f"  graphs {i}/{len(paths)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    graph_cache = graph_mod._cache_path(paths[0], source).parent.name if paths else None

    meta = {"set": set_name, "set_hash": set_hash, "answerer": answerer, "source": source, "use_cache": use_cache,
            "tol": tol, "label": label, "git": _git(), "graph_cache": graph_cache,
            "started": datetime.now(UTC).isoformat(timespec="seconds"), "dataset_weights": _dataset_weights()}
    if answerer in ("llm", "agent"):
        from app.answering import agent

        meta.update(gemini_model=agent.GEMINI_MODEL, gemini_thinking=agent.GEMINI_THINKING,
                    prompt_version=agent.PROMPT_VERSION)

    t0 = time.perf_counter()
    # PDFs first, one at a time: pdfium is not thread-safe, and that path makes no network calls anyway
    pdf_recs = {it["id"]: _answer_one(it, graphs, answerer, use_cache) for it in items if _is_pdf(it["path"])}
    if pdf_recs:
        print(f"  answered {len(pdf_recs)} PDF questions ({time.perf_counter() - t0:.0f}s)", flush=True)
    recs, done = [], 0
    with ThreadPoolExecutor(max(1, workers if answerer != "template" else 1)) as pool:
        for rec in pool.map(lambda it: pdf_recs.get(it["id"]) or _answer_one(it, graphs, answerer, use_cache), items):
            recs.append(rec)
            done += 1
            if done % 100 == 0 or done == len(items):
                print(f"  answered {done}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    meta["duration_s"] = round(time.perf_counter() - t0, 1)
    for it, rec in zip(items, recs, strict=True):
        rec["correct"] = score(it, rec["pred"], tol)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    name = "-".join(x for x in [stamp, meta["git"]["sha"] + ("+" if meta["git"]["dirty"] else ""), set_name,
                                answerer, source, label] if x)
    out = RUNS_DIR / f"{name}.jsonl"
    with open(out, "w") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    report(out)
    print(f"\nrun saved: {out}")
    return out


# ---------------------------------------------------------------------------------------------- report


def _load_run(path: Path) -> tuple[dict, list[dict], dict[str, dict]]:
    rows = _read_jsonl(path)
    meta, recs = rows[0]["meta"], rows[1:]
    items = {it["id"]: it for it in _read_jsonl(BENCH_DIR / f"{meta['set']}.jsonl")}
    return meta, recs, items


def _cluster_ci(pairs: list[tuple[str, float, float]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of mean(b - a) with a bootstrap over images (questions on one image are correlated)."""
    by = defaultdict(lambda: [0.0, 0])
    for key, a, b in pairs:
        by[key][0] += b - a
        by[key][1] += 1
    groups = list(by.values())
    if not groups:
        return 0.0, 0.0
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = n = 0
        for _ in groups:
            d, k = groups[rng.randrange(len(groups))]
            s += d
            n += k
        stats.append(s / n)
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def _row_stats(rs: list[tuple[dict, dict]]) -> dict:
    scored = [(it, r) for it, r in rs if r["correct"] is not None]
    answered = [(it, r) for it, r in scored if r["pred"] is not None]
    return {
        "n": len(rs),
        "acc": sum(r["correct"] for _, r in scored) / len(scored) if scored else float("nan"),
        "cov": sum(r["pred"] is not None for _, r in rs) / len(rs) if rs else float("nan"),
        "prec": sum(r["correct"] for _, r in answered) / len(answered) if answered else float("nan"),
    }


def _table(title: str, groups: dict[str, list]) -> None:
    print(f"\n{title:<29}{'n':>6}{'acc':>8}{'cover':>8}{'prec':>8}")
    for k in sorted(groups, key=str):
        s = _row_stats(groups[k])
        print(f"{str(k):<29}{s['n']:>6}{_pct(s['acc']):>8}{_pct(s['cov']):>8}{_pct(s['prec']):>8}")


def report(path: Path) -> None:
    meta, recs, items = _load_run(Path(path))
    rs = [(items[r["id"]], r) for r in recs if r["id"] in items]
    m = meta
    print(f"\nset={m['set']} ({m['set_hash']}) answerer={m['answerer']} source={m['source']} "
          f"git={m['git']['sha']}{'+dirty' if m['git']['dirty'] else ''} cache={m['use_cache']} "
          f"graphs={m['graph_cache']}" + (f" model={m.get('gemini_model')} prompt=v{m.get('prompt_version')}"
                                         if m.get("gemini_model") else ""))

    by_ds = defaultdict(list)
    for it, r in rs:
        if it["kind"] == "dataset":
            by_ds[it["dataset"]].append((it, r))
    weights = m.get("dataset_weights") or {}
    acc = {ds: _row_stats(g)["acc"] for ds, g in by_ds.items()}
    used = [ds for ds in acc if acc[ds] == acc[ds] and weights.get(ds)]  # scored here, and weighted
    if used:
        total = sum(weights[ds] for ds in used)
        left_out = sorted(set(by_ds) - set(used))
        print(f"\nHEADLINE all datasets, weighted by their share of questions.jsonl: acc "
              f"{sum(weights[ds] * acc[ds] for ds in used) / total:.1%}  (tol {m['tol']:.0%}; "
              + ", ".join(f"{ds} {weights[ds] / total:.0%}" for ds in sorted(used, key=lambda d: -weights[d]))
              + (f"; not scorable, left out: {', '.join(left_out)}" if left_out else "") + ")")

    cubi = [(it, r) for it, r in rs if it["dataset"] == "cubicasa5k"]
    s = _row_stats(cubi)
    lo, hi = _cluster_ci([(it["path"], 0.0, float(r["correct"])) for it, r in cubi])
    print(f"HEADLINE cubicasa test: acc {s['acc']:.1%} [95% CI {lo:.1%}-{hi:.1%}]  coverage {s['cov']:.1%}  "
          f"precision {s['prec']:.1%}  (n={s['n']})")

    _table("dataset", {k: [x for x in rs if x[0]["dataset"] == k] for k in {it["dataset"] for it, _ in rs}})
    tmpl = defaultdict(list)
    for it, r in rs:
        if it["dataset"] in ("cubicasa5k", "paraphrase"):
            tmpl[it["template"] or "(unsupported)"].append((it, r))
        else:  # other datasets have no expected template: group by the one the parser chose
            tmpl[f"{it['dataset'][:5]}:{r.get('template') or '(none)'}"[:28]].append((it, r))
    _table("template", tmpl)
    fam = defaultdict(list)
    for it, r in rs:
        if it["kind"] == "open":
            fam[it["family"]].append((it, r))
    if fam:
        _table("open question family", fam)
    _table("language", {k: [x for x in rs if x[0]["lang"] == k] for k in {it["lang"] for it, _ in rs}})

    print(f"\n{'confidence':<16}{'n':>6}{'acc':>8}   (calibration: high should be ~always right)")
    for c in CONFIDENCES:
        g = [r for _, r in rs if r["confidence"] == c and r["correct"] is not None]
        if g:
            print(f"{c:<16}{len(g):>6}{_pct(sum(r['correct'] for r in g) / len(g)):>8}")

    answered = [r for _, r in rs if r["pred"] is not None]
    lat = sorted(r["latency_s"] for _, r in rs if r["error"] != "unsupported")
    groups = defaultdict(set)  # EN/FI variants of one dataset question share an answer: so should we
    for it, r in cubi:
        if it["template"]:
            groups[(it["path"], it["template"])].add(json.dumps(r["pred"]))
    multi = [g for g in groups.values() if g]
    print("\nother")
    if answered:
        print(f"  citations valid          {sum(r['citations_ok'] for r in answered) / len(answered):.1%} of answered")
    if multi:
        print(f"  EN/FI consistent         {sum(len(g) == 1 for g in multi) / len(multi):.1%} of {len(multi)} groups")
    print(f"  agent fallbacks          {sum(r['fell_back'] for _, r in rs)}   errors "
          f"{dict(Counter(r['error'] for _, r in rs if r['error']))}")
    print(f"  answered by              {dict(Counter(r['answered_by'] for r in answered))}")
    if lat:
        print(f"  latency p50/p95          {lat[len(lat) // 2]:.2f}s / {lat[int(len(lat) * 0.95) - 1]:.2f}s   "
              f"wall {m.get('duration_s')}s")
    asked_back = sum(bool(r.get("clarification")) for _, r in rs)
    if asked_back:
        print(f"  asked a follow-up instead  {asked_back}  (counted as not answered)")
    unscored = sum(r["correct"] is None for _, r in rs)
    if unscored:
        print(f"  unscored (free text)     {unscored}")


def diff(path_a: Path, path_b: Path, show: int) -> None:
    if "kind" in _read_jsonl(Path(path_a))[0]["meta"]:  # perception runs have their own records
        from app import bench_perception

        return bench_perception.diff(Path(path_a), Path(path_b), show)
    ma, ra, items = _load_run(Path(path_a))
    mb, rb, items_b = _load_run(Path(path_b))
    if ma["set_hash"] != mb["set_hash"]:
        print(f"WARNING: different sets ({ma['set']} {ma['set_hash']} vs {mb['set']} {mb['set_hash']}): "
              "comparing shared questions only")
    items = {**items_b, **items}
    a = {r["id"]: r for r in ra}
    b = {r["id"]: r for r in rb}
    ids = [i for i in a if i in b and i in items and a[i]["correct"] is not None and b[i]["correct"] is not None]
    label = lambda m: f"{m['answerer']}/{m['source']}@{m['git']['sha']}{'+' if m['git']['dirty'] else ''} {m.get('label') or ''}"  # noqa: E731
    print(f"A = {label(ma)}  ({Path(path_a).name})\nB = {label(mb)}  ({Path(path_b).name})")

    def line(name, sel):
        if not sel:
            return
        pa = sum(a[i]["correct"] for i in sel) / len(sel)
        pb = sum(b[i]["correct"] for i in sel) / len(sel)
        lo, hi = _cluster_ci([(items[i]["path"], float(a[i]["correct"]), float(b[i]["correct"])) for i in sel])
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"{name:<16}{len(sel):>6}{_pct(pa):>8}{_pct(pb):>8}{pb - pa:>+8.1%} [{lo:+.1%}, {hi:+.1%}] {sig}")

    print(f"\n{'slice':<16}{'n':>6}{'A':>8}{'B':>8}{'delta':>8} 95% CI (bootstrap over images; * = excludes 0)")
    line("ALL", ids)
    for ds in sorted({items[i]["dataset"] for i in ids}):
        line(ds, [i for i in ids if items[i]["dataset"] == ds])
    print()
    for t in sorted({items[i]["template"] for i in ids if items[i]["template"]}):
        line(t, [i for i in ids if items[i]["template"] == t])
    for f in sorted({items[i]["family"] for i in ids if items[i].get("family")}):
        line(f, [i for i in ids if items[i].get("family") == f])

    cov = lambda rs, sel: sum(rs[i]["pred"] is not None for i in sel) / max(1, len(sel))  # noqa: E731
    print(f"\ncoverage {cov(a, ids):.1%} -> {cov(b, ids):.1%}")
    fixed = [i for i in ids if not a[i]["correct"] and b[i]["correct"]]
    broke = [i for i in ids if a[i]["correct"] and not b[i]["correct"]]
    print(f"fixed (wrong -> right) {len(fixed)}   broke (right -> wrong) {len(broke)}")
    for title, sel in (("BROKE", broke), ("FIXED", fixed)):
        for i in sel[:show]:
            it = items[i]
            print(f"  {title} {it['path'].rsplit('/', 1)[-1]:<24} expected={it['answer']!s:<6} "
                  f"A={a[i]['pred']!s:<6}({a[i]['confidence']}) B={b[i]['pred']!s:<6}({b[i]['confidence']})  "
                  f"{it['question'][:70]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("run")
    p.add_argument("--set", default="fast")
    p.add_argument("--answerer", choices=["llm", "agent", "template"], default=os.environ.get("ANSWERER", "llm"),
                   help="llm = agent with template fallback (the API); agent = no fallback; template = offline path")
    p.add_argument("--source", choices=["model", "svg"], default=os.environ.get("PLAN_SOURCE", "model"))
    p.add_argument("--no-cache", action="store_true", help="ignore cached agent answers")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tol", type=float, default=0.02, help="relative tolerance for non-count numbers")
    p.add_argument("--label", default="")
    p.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    p.add_argument("--kind", choices=["dataset", "paraphrase", "open"], default="",
                   help="only this kind of question, e.g. open = the breadth questions")
    p = sub.add_parser("perception", help="deterministic: model evidence vs SVG labels (no questions, no LLM)")
    p.add_argument("--set", default="fast", help="images of this set (cubicasa test split)")
    p.add_argument("--limit", type=int, default=0, help="first N images only")
    p.add_argument("--iou", type=float, default=0.3, help="a symbol box matches a label box at this IoU")
    p.add_argument("--walls", action="store_true", help="also score wall segmentation (runs the net: ~3 s/image)")
    p.add_argument("--label", default="")
    p = sub.add_parser("report")
    p.add_argument("run")
    p = sub.add_parser("diff")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--show", type=int, default=10)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.seed)
    elif args.cmd == "run":
        run(args.set, args.answerer, args.source, not args.no_cache, args.workers, args.tol, args.label, args.limit,
            args.kind)
    elif args.cmd == "perception":
        from app import bench_perception

        bench_perception.run(args.set, args.limit, args.iou, args.walls, args.label)
    elif args.cmd == "report":
        if "kind" in _read_jsonl(Path(args.run))[0]["meta"]:
            from app import bench_perception

            bench_perception.report(Path(args.run))
        else:
            report(Path(args.run))
    else:
        diff(Path(args.a), Path(args.b), args.show)


if __name__ == "__main__":
    main()
