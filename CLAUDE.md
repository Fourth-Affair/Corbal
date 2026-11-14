# CLAUDE.md

Junction 2026 hackathon entry for the **Stoa** challenge: answer natural-language questions about
construction drawings and BIM models **accurately and verifiably**.

## The challenge in one paragraph

Given a dataset of real technical drawings (many file formats, several languages — expect Finnish,
Swedish, English) plus example questions, build a program that answers questions like a site
engineer would ask them. A great answer is **correct** and **points to where in the drawing it came
from** (file, sheet/page, region, or BIM element) so a human can verify it in seconds. Judges favour
approaches that won't be obsoleted by next year's frontier models: hybrids of LLMs + deterministic
parsing, computer vision, geometry/3D, or small fine-tuned models.

## Design principles (apply these when making decisions)

1. **Evidence first, LLM second.** Extract everything we can deterministically (vector text with
   coordinates, title blocks, IFC properties, DXF entities) into structured, addressable evidence.
   The LLM plans, calls tools, and composes — it does not "eyeball" whole sheets.
2. **Every claim cites evidence.** Answers carry citations (`doc_id`, `page`, `bbox` in PDF points,
   or IFC `GlobalId`). The backend validates that cited evidence exists before returning; uncited
   answers are flagged as low confidence.
3. **Compute, don't guess.** Counts, areas, sums and unit conversions are done in code (tools), not
   by the model.
4. **Vision on crops, not pages.** When raster/VLM analysis is needed, send zoomed regions with
   known coordinates, so the result maps back to a bbox.
5. **Say "not found."** A confident wrong answer is worse than an honest miss.
6. **Measure.** The example questions are the eval set; every pipeline change gets run against it.

## Repo layout

- `backend/` — FastAPI (Python 3.12, **uv-managed**: `pyproject.toml` + `uv.lock`). Entry: `backend/app/main.py`.
  - `app/router.py` — question (EN/FI) → template id.
  - `app/cubicasa/` — cubicasa5k pipeline:
    - perception: `cubinet.py` (CubiCasa5K pretrained multi-task net, vendored in `vendor/floortrans`, CC BY-NC)
      + `ocr.py` (RapidOCR, detect once, recognise crops at 0/180 or 90/270°); cached by `features.py`.
    - evidence → counts: `fuse.py` (component counts, probability-threshold counts, heatmap peaks, OCR labels via
      `rooms_dict.py`), `fit.py` (per-template rule / gradient-boosting models, fitted on val with K-fold CV,
      reported on test), `detect.py` (runtime: PlanGraph with evidence items + fusion counts).
    - ground truth: `labels_from_svg.py`; YOLO tooling (`export_yolo.py`, `train.py`, `classes.py`) is unused for now.
  - `app/eval.py` — exact-match eval per template.
- `frontend/` — Next.js 15 / React 19 / TypeScript. Calls backend via `BACKEND_URL`.
- `compose.yaml` — runs both with hot reload; mounts `./dataset` at `/dataset`.
- `dataset/` (gitignored, never commit) — `junction-dataset/` (challenge data + `questions.jsonl`),
  `cubicasa5k-original/` (Zenodo 2613548, labels for training only), `cache/`, `yolo-cubicasa/`, `runs/`, `models/`.

## Commands

```sh
docker compose up --build                 # full stack: frontend :3000, backend :8000 (docs at /docs)
PLAN_SOURCE=svg docker compose up backend # demo with ground-truth labels instead of the model

cd backend
uv sync                                   # install deps (CPU torch); `uv add <pkg>` to add one
uv run pytest -q
uv run python -m app.cubicasa.build_cache --source svg      # parse all original SVGs (~10 s)
uv run python -m app.eval --method svg --split val          # oracle (should be 100%)
uv run python -m app.eval --method mode --split val         # majority baseline (~57%)
uv run python -m app.cubicasa.features --splits val test     # cache net maps + OCR (~3 s/image on CPU)
uv run python -m app.cubicasa.fit --test --fit-on train+val          # CV on val, score on test
uv run python -m app.cubicasa.fit --test --final --fit-on train+val  # + save fusion models -> dataset/models/
uv run python -m app.eval --method model --split test       # end-to-end score through the runtime path
```

Rebuild the Docker image after changing `backend/pyproject.toml` or `frontend/package.json`.

## cubicasa5k facts (verified)

- Challenge PNG `<variant>-<id>.png` is byte-identical to original `<variant>/<id>/F1_original.png`.
  SVG coords are in `F1_scaled.png` space → rescale by original/scaled size.
- The questions were generated from the original `model.svg`: the SVG oracle scores 100% on train/val/test.
  Counting rules (in `labels_from_svg.py` / `answer.py`): showers = `Shower` + `Bathtub*` (not ShowerCab/Screen);
  sinks = any `*Sink*`; saunas = sauna stoves + sauna rooms without a stove; balcony = Outdoor space named PARV*;
  WC room = Bath space named WC*.
- The pretrained CubiCasa net was trained on the original train split, so its outputs on train images could be
  optimistic. Measured: they are not (single-feature accuracy on train ≈ val), and adding train images to the
  fusion fit improves test (74.7% → 75.7% with 331 train images). Select models by CV on val, report on test,
  never fit on test.
- SVG labels are for training/validation only. The runtime path (`PLAN_SOURCE=model`) must be image-only.
  Split = original CubiCasa `train/val/test.txt`; never train on test.

## Current cubicasa results (exact match, held-out test split, 387 images / 3398 questions)

Majority baseline 52.8% → pipeline **77.2%** (same number via the runtime path, `app.eval --method model`). Per template: kitchens 95, bathrooms 94, floors 93, saunas 93,
WC seats 89, WC rooms 85, balconies 84, bedrooms 77, showers+baths 75, sinks 74, doors 51, windows 38.
Weakest = doors/windows: adjacent symbols merge in the segmentation; a CPU-trained YOLO was too slow to help
(~100 min/epoch). Next steps: GPU-train an instance detector (door/window/shower) with `export_yolo.py --det`.

## Conventions

- Pydantic models for every API request/response; keep the `Answer`/`Citation` schema stable — the
  frontend renders highlights from it.
- Coordinates: image pixels (PNG) or PDF user-space points, origin top-left, per page.
- Secrets (e.g. `ANTHROPIC_API_KEY`) via `.env`, never committed.
- Default LLM: latest Claude models (`claude-opus-5-5` for reasoning, `claude-haiku-5-5` for cheap
  bulk steps like page classification).
