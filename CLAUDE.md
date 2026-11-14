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

## What we are optimising for

**The goal is to handle a wide variety of natural-language questions, not to get the best possible
score on the dataset.** The dataset questions come from a handful of templates; real users will
phrase things differently and ask things no template covers. When the two pull apart, prefer the
change that makes the system answer more kinds of questions (general tools, robust question
understanding, honest "not found") over one that squeezes out dataset accuracy with template- or
dataset-specific rules. The benchmark is a regression check, not the objective.

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
6. **Measure, but don't overfit.** The example questions are the eval set and every pipeline change
   gets run against it to catch regressions. A higher score is not the aim in itself: also test
   paraphrases and question types outside the dataset templates, and don't add rules that only
   serve the dataset's phrasings.

## Repo layout

- `backend/` — FastAPI (Python 3.12, **uv-managed**: `pyproject.toml` + `uv.lock`). Entry: `backend/app/main.py`.
  - `app/router.py` — question (EN/FI) → template id.
  - `app/router.py` is no longer the gate when Gemini is on: it is the offline fallback and a cross-check.
  - `app/answering/agent.py` — question → Gemini (`GEMINI_MODEL`, default `gemini-3.8-flash`) tool loop →
    `final_answer` → `validate()`: value must equal a number a tool returned (else not_found), cited ids must come
    from tool output, confidence from the producing tool (count_template → counting model; raw items → low; router
    disagreement → low), explanation must state the value and use existing `[k]` refs. Answers cached in
    `CACHE_DIR/agent` (key includes `PROMPT_VERSION`: bump it when the prompt or tools change). `GEMINI_THINKING`
    (default `MEDIUM` in code; `LOW` was measured 2-3x faster with the same answers on the dataset questions;
    `MINIMAL` is rejected by gemini-3.8-flash). `GEMINI_MAX_PER_HOUR` caps uncached questions per process (0 = off). Each Gemini call retries `GEMINI_RETRIES` (default 1) times on timeout / network error / 429 / 5xx
    (`GEMINI_TIMEOUT_S` per call, default 30). Yes/no answers are linked to, and checked against, a count_template
    result. No `GEMINI_API_KEY` / API error → router + template counter. `ANSWERER=template` disables it.
  - `app/answering/tools.py` — deterministic tools over a PlanGraph: `plan_summary`, `count_template` (fusion
    counts), `list_items`, `items_in` (containment by the room's outline when it has one: fixtures by centre, doors
    and windows when they run along it; bounding box otherwise; a room label uses the room region around it),
    `calculate` (add / subtract; every operand must be a number an earlier tool call returned, the answer takes the
    confidence of its least reliable operand and cites the operands' items). Room regions of the image pipeline
    carry their segmentation outline (`GRAPH_VERSION` in `graph.py` is part of the graph cache key: bump it when
    `graph_from_image` changes what it stores).
  - `scripts/routing_experiment.py` — Gemini tool choice vs router on dataset phrasings + paraphrases.
  - `app/cubicasa/` — cubicasa5k pipeline:
    - perception: `cubinet.py` (CubiCasa5K pretrained multi-task net, vendored in `vendor/floortrans`, CC BY-NC)
      + `ocr.py` (RapidOCR, detect once, recognise crops at 0/180 or 90/270°); cached by `features.py`.
    - evidence → counts: `fuse.py` (component counts, probability-threshold counts, heatmap peaks, OCR labels via
      `rooms_dict.py`), `fit.py` (per-template rule / gradient-boosting models, fitted on val with K-fold CV,
      reported on test), `detect.py` (runtime: PlanGraph with evidence items + fusion counts).
    - symbol detector: `symdet.py` (YOLO11s boxes for door, window, toilet, sink, double sink, shower, bathtub,
      sauna stove; `symbol-det.pt` + `symbol-det.json` sidecar (imgsz 1600) on CPU (.onnx also accepted); cached per image, added as `det_*` fusion features and used as citations). Trained
      via `export_yolo.py` + `train.py`, or `backend/scripts/verda_train.py` (+ `verda_remote.sh`, runs on the VM).
      Val mAP50: door 0.95, window 0.90, all 0.87. Optional: everything runs without its weights.
    - walls: `walls.py` (SVG wall polygons as ground truth, wall metrics with 3 px tolerance, scored only on floors
      whose walls the SVG annotates) and `wallseg.py` (dedicated U-Net wall segmenter: background / solid wall /
      opening at native resolution, sliding-window inference). CubiCasa net baseline on 60 test plans: wall F1 0.91
      (median 0.96), weakest on scanned architectural drawings; it paints openings as wall.
    - API layers: `layers.py` backs `/api/walls`, `/api/symbols`, `/api/text` and `/api/render?layer=` (one
      `Layer` response shape; cached per image and weights).
    - rooms (web app geometry): `rooms.py` (spaces from the wall map, `geometry_from_image` -> `PlanGeometry`),
      `room_split.py` (cut a space holding two strong room labels), `room_types.py` (type from OCR label + fixtures +
      net pixels, hand-set weights in `EVIDENCE_WEIGHTS`, strong / weak / unknown tier), `wallvec.py` (wall
      centreline segments with length / thickness / rooms on each side), `windows.py` (split one detector box over a
      mullioned window run). Evals: `rooms_eval.py`, `windows_eval.py`. `token_table.py` learns OCR token -> room type.
    - ground truth: `labels_from_svg.py`.
    - scale: `dimensions.py` (printed dimension figures + tick-to-tick span of their lines; >= 2 figures agreeing
      within 4% and inside the door/fixture size band of `size_priors.py`), else written room areas (`rooms.py`,
      bare decimals under a label count). Dimension lines win; a > 10% disagreement with written areas is a warning.
  - Routing is by file type, never by dataset folder: `.png/.jpg` -> image pipeline (`app/cubicasa/`), `.pdf` ->
    `app/pdfplan/`, `.ifc` -> `app/ifc/`. Uploaded PDFs (frontend) are stored in `CACHE_DIR/uploads/`, shown as page 1 rendered to PNG,
    saved as a `PlanGraph` with `source="pdf"` (PDF points) and answered by `app/pdfplan/`.
  - `app/pdfplan/` — vector PDF plans, no vision. `text.py`: text runs with boxes (pdfium). `parse.py`: rooms from
    centred "name / number / area" stacks and apartment tags ("Whg. A") in the PDF text; if a same-name `.dxf` sits
    next to the PDF, room/apartment outlines and door / window / fixture symbols from it (layers matched by keyword,
    FIXTURE outline + inner marks -> toilet/washbasin/bathtub/shower/kitchen/stairs), joined to the PDF through shared
    texts with an affine fit. Without a DXF, symbol counts use the image detector on the rendered page (low
    confidence) and apartment areas say not found. `answer.py`: EN/DE/FR/IT questions -> 12 templates.
    `python -m app.pdfplan.eval [--prefix data/swiss_dwellings/] [--split eval] [--no-dxf]`.
  - `app/ifc/` — IFC (BIM) models, deterministic, no API key. `model.py`: ifcopenshell -> IfcModel evidence (header /
    IfcApplication / schema, project / sites / buildings / storeys, every element with class, effective kind (IFC2X3
    generic class + type object -> IfcSanitaryTerminal etc.), storey, type, materials + layer thicknesses, all
    properties and quantities converted to m / m² / m³ in code, spaces with floor area / volume / height measured
    from their 3D shape, systems / zones); cached in `CACHE_DIR/ifc/`. Space net area = NetFloorArea quantity > body
    geometry > GrossFloorArea > area property (the dataset's "net floor area" equals the geometry, not Revit's Area
    property). `tools.py`: general tools (count / select / group by class + storey + filters, find spaces, sum /
    extreme of a quantity, types, materials, storeys, metadata, systems), every result carries GlobalIds.
    `answer.py`: EN/DE/FI/SV question -> operation -> tools -> Answer; citations are PlanItems with id = IFC
    GlobalId, raw = IFC class, bbox (0,0,0,0). Unmapped -> not_found. Not yet offered to the Gemini agent.
    `python -m app.ifc.eval [--preview]` (counts exact, numbers within 2%, text scored approximately).
  - `app/eval.py` — exact-match eval per template (counting models only, bypasses the agent).
  - `app/bench.py` — benchmark of the `/api/ask` answer path, routed by file type like the API (images -> agent /
    template counter, PDFs -> `app/pdfplan/`, IFC -> `app/ifc/`, anything else = unsupported): fixed sets (`dataset/bench/{fast,full}.jsonl`: cubicasa
    test split EN+FI, paraphrases/traps from `app/answering/paraphrases.py`, samples of every other dataset), runs in
    `dataset/runs/bench/` (meta: git sha, set hash, graph cache, Gemini model/prompt version), metrics: accuracy with
    image-bootstrap CI, coverage, precision, calibration per confidence, citation validity, EN/FI consistency,
    latency. Numbers within 2% (`--tol`, the challenge rule). First headline = dataset accuracies weighted by their
    share of `questions.jsonl`. `diff` gives paired deltas with CIs and the flipped questions.
  - `app/answering/openq.py` — the breadth measure: "open" questions no count template covers, written for a
    labelled plan in EN / FI / SV with the answer computed from the SVG labels. Nine families: `in_room_count`
    (sinks in the kitchen), `in_room_yesno`, `rooms_with` (bathrooms that have a window), `plan_yesno`,
    `compare_yesno` (more windows than doors?), `arithmetic` (sum / difference of two counts), `subtype_count`
    (bathtubs only), `adjacent_yesno`, `size_yesno`. A question is only generated where the labels give one
    defensible answer (one room of that type, no corner-clipping windows, clear wall / clear gap, 1.5x area).
    In the bench sets as kind `open` (fast 81, full 540), reported per family. Baseline on fast (2026-11-14):
    offline template path 5%, agent 58% (coverage 72%, precision 81%), agent on SVG labels 79% (precision 97%).
    After adding `calculate` and outline containment: agent on SVG labels 93% (`arithmetic` 22% -> 100%,
    `in_room_count` 56% -> 89%); agent on the real pipeline 59% (coverage 78%), within noise of before: what is
    left there is perception (door / window counts, rough room regions) and Gemini runs that end without a final
    answer and fall back to the template counter. With 9 questions per family the fast set only shows large
    effects; use `--set full --kind open` to compare small ones. `paraphrases.py` keeps only traps that a plan
    without a scale cannot answer.
  - `app/bench_perception.py` — `bench perception`: deterministic, no questions / router / LLM. Model PlanGraph vs
    SVG labels on the set's cubicasa test images: symbol boxes (one-to-one IoU matching -> precision/recall/F1 per
    class), rooms (centre-in-room), fusion counts exact per template, `--walls` wall F1. Seconds on cached graphs.
  - `app/main.py` web-app endpoints `/api/plans/{id}/...` (analyze, image, ask, geometry, scale, outline, corrections,
    feedback): source file from Supabase Storage, analysis saved in Supabase `plan_knowledge` (+ a local copy in
    `CACHE_DIR/plan-graphs/`), geometry in `plan_geometry`. Room-area questions are answered in `ask_plan` from the
    user's scale + accepted outlines before the agent is called. Analysis is a background job (`app/jobs.py`):
    `POST /analyze` -> 200 graph if saved, else 202; the page polls `GET /analysis` (state, stage, queue position).
  - `app/profiling.py` — `Profile` + `stage("name")` (nested, dotted names, contextvar: a no-op without an active
    profile, so library code marks its own stages) + `steps()` for long sequential functions; peak RSS per stage via
    psutil. Jobs log one `analysis_profile` JSON line each. `app/warmup.py` loads every model at startup (`WARMUP`).
  - `scripts/profile_analysis.py` — the upload path (pipeline + geometry) on preview plans with an empty cache: stage
    table, `--save-outputs` / `--diff` (prove a speed-up changes nothing), `--cpus N` (emulate a smaller server),
    `--parallel-walls`, `--cprofile`.
  - `scripts/inspect_plans.py` — inspection sheets (rooms, walls, detections, ground truth side by side; `--uploads N`
    for uploaded plans). `scripts/fetch_questions.py` — `questions.jsonl` out of the dataset zip (HTTP range reads).
- `frontend/` — Next.js 16 / React 19 / TypeScript. The browser calls only Next.js API routes
  (`app/api/plans/[planId]/*`), which proxy to the backend via `BACKEND_URL` and parse every backend response as JSON.
  Uploads go through `upload/route.ts` (anon key) to Supabase Storage. Plan viewer: `app/plans/[planId]/page.tsx`;
  several floors at once: `app/building/page.tsx` (`/api/buildings/ask`). Both wait for analyses through
  `app/analysis-client.ts` (`runAnalysis`: POST /analyze, poll GET /analysis); never treat a 2xx from /analyze as
  the graph, it is 202 while the job runs.
- `supabase/migrations/` — tables, upload RPCs and the Storage policy; apply in timestamp order.
- `compose.yaml` — runs both with hot reload; mounts `./dataset` at `/dataset`.
- `compose.prod.yaml` + `deploy/` — production on one server (Caddy -> frontend -> backend), see "Production" below.
- `dataset/` (gitignored, never commit) — `junction-dataset/` (full challenge data + `questions.jsonl`),
  `junction-dataset-preview/` (preview release = our test set, never trained on), `cubicasa5k-original/`
  (Zenodo 2613548, labels + official split for training/eval only), `cache/`, `yolo-cubicasa-det/`, `runs/`,
  `models/` is now at the repo root (see below).
- `models/` (`MODELS_DIR`, tracked in git, all files < 50 MB): `cubicasa-fusion.joblib`, `ocr_token_table.json`, `symbol-det.pt` + `.json` sidecar, `wallseg.pt` (fp16); checksums in `models/MANIFEST.json`. Not committed: `cubicasa-net.pkl` (208 MB CubiCasa net; the share bundle's `dataset/cubicasa-pretrained/` path also works). The detector fingerprint is a content hash, so retraining the detector requires refitting the fusion models.
- `docs/pipeline.md` — how the whole system works, stage by stage (for the team; keep it current when a stage changes).
- `docs/literature-review.md` — floor-plan recognition since CubiCasa5K and the roadmap.

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
uv run python -m app.cubicasa.fit --test --final --fit-on train+val  # + save fusion models -> models/
uv run python -m app.eval --method model --split test       # end-to-end score through the runtime path

# benchmark (every pipeline change): fast ≈ 460 q (~6 min with the agent, CI ±6 pts), full ≈ 7.9k q
uv run python -m app.bench build                            # once (deterministic)
uv run python -m app.bench run --set fast                   # API path: agent + fallback (--answerer template = offline)
uv run python -m app.bench run --set fast --no-cache --label rerun   # fresh Gemini answers: measures noise
uv run python -m app.bench diff <runA.jsonl> <runB.jsonl>   # paired delta, 95% CI, broke/fixed questions
uv run python -m app.bench run --set fast --answerer template   # no Gemini: router + counters + PDF parser (~10 s)
uv run python -m app.bench run --set fast --kind open           # breadth only: the open questions (~4 min)
uv run python -m app.bench run --set fast --kind open --source svg --answerer agent   # same with perfect perception:
                                                            # what is left is question understanding and tools
uv run python scripts/profile_analysis.py                  # where an upload's analysis spends time (6 preview plans)
uv run python scripts/profile_analysis.py --save-outputs ../dataset/runs/profile/before   # then after a change:
uv run python scripts/profile_analysis.py --diff ../dataset/runs/profile/before ../dataset/runs/profile/after
uv run python -m app.bench perception --set fast [--walls]  # no questions at all: evidence boxes/rooms/counts vs SVG
                                                            # labels (diff works on these runs too)

# symbol detector on a Verda GPU VM (VERDA_CLIENT_ID/SECRET in .env). The VM downloads CubiCasa5K itself;
# the challenge preview images (= test) are held out. VM is deleted at the end unless --keep.
uv run python scripts/verda_train.py list                   # GPU types, prices, availability
uv run python scripts/verda_train.py run                    # -> dataset/models/sweep/ (staging; copy to models/ + refit to adopt)
# sweep on an existing VM (never deleted by the script): trains presets A-D from verda_remote.sh (A = shipped recipe,
# B/C = imgsz 1600 + rot90, D = 1024 native-res tiles), evaluates each + "current" with det_eval (val picks thresholds,
# test reported), caches detections and refits the fusion models on the VM (fit-on val and train+val; questions from
# the dataset zip). Works from Windows (tar over ssh). --skip-upload reuses an earlier upload; attach --host re-follows.
uv run python scripts/verda_train.py run --host <ip> --sweep "current A B C D" --questions-zip ~/junction-dataset.zip
uv run python -m app.cubicasa.det_eval --splits val test    # per-class AP50/recall + count exact-match, error buckets
# local alternative (needs dataset/cubicasa5k-original):
uv run python -m app.cubicasa.export_yolo --holdout ../dataset/junction-dataset-preview/questions.jsonl
uv run python -m app.cubicasa.train --device 0 --onnx      # -> dataset/models/ (staging; copy to models/ + refit to adopt)
uv run python -m app.cubicasa.symdet --splits train val test # cache detections, then refit (fit --final)

# walls (needs dataset/cubicasa5k-original). Local AMD GPU: ~/.venvs/stoa-rocm (torch rocm7.1), PYTHONPATH=. 
uv run python -m app.cubicasa.walls --split test --limit 60                  # CubiCasa net wall accuracy
uv run python -m app.cubicasa.wallseg prepare --holdout ../dataset/junction-dataset-preview/questions.jsonl
~/.venvs/stoa-rocm/bin/python -m app.cubicasa.wallseg train --device cuda    # -> dataset/models/wallseg.pt
uv run python -m app.cubicasa.walls --split test --limit 60 --model wallseg  # compare
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

**Now (detector sweep preset B: YOLO11s at imgsz 1600 + rot90, counting models refit on train+val on the VM):
81.8%**. Per template: floors 93, kitchens 95, saunas 94, bathrooms 93, WC seats 91, WC rooms 89, showers+baths 85,
balconies 85, sinks 84, bedrooms 78, doors 66, windows 44. Previous detector 81.3% (same VM run). Detector test recall
door 0.957, window 0.917; 93 test windows are still "merged" (one box over a mullioned run; the ground truth itself
labels such runs inconsistently, see `windows_eval.py`).

The 81.8% was measured with three `var_*` features parsed from the dataset **file name** (the CubiCasa variant), which
an upload never has. Removed 2026-11-15 (`fuse.image_features`; `fit.py` excludes `var_*`): on the 241 preview plans,
zeroing them changes 2 of 2892 counts and leaves accuracy at 80.5%. Re-report the test score at the next refit (VM).
No feature may come from a file name or path.

History: majority baseline 52.8% → pipeline **77.2%** without the detector (same number via the runtime path, `app.eval --method model`). Per template: kitchens 95, bathrooms 94, floors 93, saunas 93,
WC seats 89, WC rooms 85, balconies 84, bedrooms 77, showers+baths 75, sinks 74, doors 51, windows 38.
Reproduced 77.1% with `fit --test --fit-on train+val`. Weakest = doors/windows: adjacent symbols merge in the
segmentation. The symbol detector and the refit with its `det_*` features gave the 81.8% above. 197 of the 241 preview plans are in the official *train* split, so preview scores of
the counting models are optimistic: report on the official test split.

## Current PDF results (preview set; counts exact, areas within 2%)

swiss_dwellings (65 plans / 2358 questions), rules designed on the even-indexed plans (sorted by path): dev 99.5%,
held-out odd half **99.0%**. All counts, room labels and per-room areas are 100% on both halves; the misses are
apartment / floor area totals 3–10% too high (unnamed numbered rooms counted as indoor). PDF only (`--no-dxf`):
52.8% — room labels 100%, but many plans print no areas and symbol counts fall back to the CubiCasa-trained
detector. All 71 preview PDFs (2644 questions, incl. IFC renders and drawing sets): 88.5%.

## Current IFC results (deterministic, `app.ifc.eval`)

Full dataset (1188 questions): counts 100% (76), room / storey areas 100% (292), text ≈49% (820; approximate scoring
of free-text reference answers); all 64.9%, coverage 91.2%, precision 67.6%. Preview (1076 questions, test set): areas
99.0% (692), counts 48% / lengths 50% / volumes 50% (the misses are Danish / Estonian phrasings the word lists do not
cover, answered not_found), text ≈54%; all 82.2%, coverage 90.8%, precision 88.5%.

## Analysis speed (`scripts/profile_analysis.py`, 6 preview plans 0.03–12 MP, laptop, models loaded)

Upload path = image pipeline + room / wall geometry. Before (2026-11-14): 12 MP plan **70.6 s**, 6.5 GB peak; plans
up to 1 MP 3–9 s. Biggest stages at 12 MP: wall segmenter 27 s (native-resolution 512 px tiles), `fuse` features
16 s, CubiCasa net 5 s + full-resolution maps 4 s, OCR 5 s (detection 4 s), evidence 3 s. On 4 pinned cores the 12 MP
plan takes 92 s and small plans 10–16 s.

- Done: `fuse.components` counts only labelled pixels (ndimage.sum/mean bincounted every pixel 4x per call):
  features 16.2 -> 7.3 s, evidence 3.4 -> 1.4 s at 12 MP; 12 MP total 61 s. Outputs byte-identical (6 plans
  graph + geometry, `image_features` on 253 cached plans).
- Done: model warm-up at startup (`app/warmup.py`, `WARMUP`): 7–18 s per process no longer lands on the first upload.
- Measured, off by default: `WALLSEG_MAX_PIXELS` (wall segmenter on a downscaled copy). On the 13 preview plans
  over 4 MP: wall-seg time 251 s -> 180 s (6 MP cap) / 120 s (4 MP); agreement with the native map wall F1 mean
  0.984 / 0.974, worst plan 0.936 / 0.902, openings worst 0.83. Agreement is not accuracy: turn it on only after
  `app.cubicasa.walls --model wallseg` (ground-truth wall F1, needs the original CubiCasa5K) shows no drop.
- Measured, off by default: `PARALLEL_WALLS` (wall segmenter in a thread during the pipeline). On 4 cores: 160.8 ->
  157.5 s over the 6 plans, the 12 MP plan slower (91.5 -> 94.8 s) and +0.5 GB peak: both networks compete for
  the same cores. Worth re-measuring only on a bigger server.
- Not done (needs a counting-model refit): full-resolution softmax / argmax / quantisation in `features.compute`
  (`maps`, 4–5 s at 12 MP) and full-resolution morphology in `image_features` / `_evidence`. Other candidates:
  ONNX Runtime for the wall U-Net, OCR detection on a downscaled copy.

## Production (`deploy/README.md` has the operator's version)

Every push to `main` runs the backend tests and then deploys to the single server (`.github/workflows/deploy.yml` ->
`deploy/deploy.sh ci`): code only, the server keeps its `.env`, `models/cubicasa-net.pkl` and `dataset/cache/`. So:

- A change that needs a new env var, weights file or Supabase migration breaks production until someone adds it by
  hand. Give new env vars a working default in code; pass them through `compose.prod.yaml` (with the same default:
  an unset `${VAR:-}` becomes an empty string, not "unset"), list them in `.env.example`, and mention migrations in
  the commit.
- `models/` is mounted read-only and `dataset/` too (only `dataset/cache` is writable): runtime code must not write
  next to the weights or the dataset.
- One uvicorn process (`--workers 1`). Uploads are analysed by the in-process job queue (`app/jobs.py`, one worker
  per slot; a restart forgets it and the page resubmits). Heavy work (`graph_from_image`, `_compute_geometry`) goes
  through `_bounded_analysis` (`MAX_CONCURRENT_ANALYSES` slots; jobs wait, request handlers get a 503 "busy" after
  `ANALYSIS_QUEUE_TIMEOUT`); keep new heavy paths inside it, and mark their stages with `app.profiling.stage`. Memory peaks at ~1.7 GB + 0.27 GB per megapixel, so uploads are downscaled to
  `MAX_ANALYSIS_PIXELS` (`config.py`, default 16 MP) and PDF pages render at most that large (`pdf_render_scale`).
  Downscaling is safe because the viewer draws the graph in its own `width`/`height` over the original image.
- Unhandled backend errors return JSON `{"detail": ...}` (exception handler in `main.py`). The Next.js proxies show a
  non-JSON body as "Could not reach the backend". Node's `fetch` in those proxies gives up after 300 s.
- Gemini failures (bad key, retired `GEMINI_MODEL`, quota) are silent for users: answers fall back to the template
  counter. Check backend logs for `Gemini agent failed`.
- `deploy/*.sh` must keep LF line endings (`.gitattributes`). Writing them from Python on Windows produces CRLF.

## Conventions

- Pydantic models for every API request/response; keep the `Answer`/`Citation` schema stable — the
  frontend renders highlights from it.
- Coordinates: image pixels (PNG) or PDF user-space points, origin top-left, per page.
- Secrets (`GEMINI_API_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `VERDA_*`) via `.env`, never committed. Only `NEXT_PUBLIC_*`
  values reach the browser, and they are baked in at `next build`.
- LLM: the answering agent uses Gemini through `google-genai` (`GEMINI_MODEL`). For new LLM components, prefer the
  latest Claude models (`claude-opus-5-5` for reasoning, `claude-haiku-5-5` for cheap bulk steps like page
  classification).
