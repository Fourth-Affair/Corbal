# How the pipeline works

What happens between "user drops a floor plan" and "answer with highlighted evidence": every stage, which model
or rule decides what, where results are stored, and how we know how good each part is. Written for the team. For
commands and current scores see [CLAUDE.md](../CLAUDE.md); for running the server see
[deploy/README.md](../deploy/README.md).

Contents: [1 Overview](#1-overview) · [2 Upload and job queue](#2-upload-and-the-job-queue) ·
[3 Image pipeline](#3-the-image-pipeline-png--jpg--webp) · [4 Room and wall geometry](#4-room-and-wall-geometry) ·
[5 Vector PDFs](#5-vector-pdfs) · [6 IFC models](#6-ifc-models) · [7 Answering](#7-answering-a-question) ·
[8 Human corrections, scale, outlines](#8-human-corrections-scale-and-outlines) ·
[9 Storage and caches](#9-storage-and-caches) · [10 Evaluation and data hygiene](#10-evaluation-and-data-hygiene) ·
[11 Speed and memory](#11-speed-and-memory) · [12 Known weaknesses](#12-known-weaknesses)

---

## 1. Overview

```
 browser ──▶ Next.js (frontend/app/api/plans/*) ──▶ FastAPI (backend/app/main.py)
                                                        │
             upload ──▶ Supabase Storage + plans row    │
                                                        ▼
                         POST /analyze ──▶ job queue (app/jobs.py) ──▶ by file type:
                                                 .png/.jpg/.webp ─▶ image pipeline (app/cubicasa/)  ─┐
                                                 .pdf            ─▶ vector parser  (app/pdfplan/)    ├▶ PlanGraph
                                                 .ifc (API only) ─▶ IFC model     (app/ifc/)         ┘   + PlanGeometry
                                                        │
                         POST /ask ──▶ Gemini tool loop over the PlanGraph / PlanGeometry (app/answering/)
                                       └▶ fallback: regex router + counting model (no LLM)
                                                        │
                                                        ▼
                                  Answer {value, confidence, explanation, citations[bbox / polygon / GlobalId]}
```

The rule everything follows: **evidence first, LLM second.** Models and parsers turn the drawing into addressable
evidence: items with an id, a category, a box in drawing coordinates and a score. The LLM never looks at the
drawing. It calls tools that count, select and compute in code, and its final answer is rejected unless its number
came out of a tool and its citations are real item ids.

Two data structures carry everything (`backend/app/models.py`):

| Structure | What it holds | Coordinates |
|---|---|---|
| `PlanGraph` | `items` (rooms, doors, windows, fixtures, floors, text labels; each with `bbox`, optional `polygon`, `score`, `raw` = which source produced it), `counts` per question template (from the counting models), `floor_outlines`, `width`/`height`, `source` (`model` / `pdf` / `svg`) | image pixels, or PDF points, origin top-left |
| `PlanGeometry` | rooms as polygons with type, tier (strong / weak / unknown), evidence list, conflicts, neighbours, doors/windows connecting them; walls, openings, straight wall segments with thickness; the plan scale and where it came from | same as the graph |

`source="svg"` is the CubiCasa ground truth (the original labels). It exists only for training, evaluation and
demos (`PLAN_SOURCE=svg`). Production always uses `model`.

---

## 2. Upload and the job queue

1. **Upload** (`frontend/app/api/plans/[planId]/upload/route.ts`): the Next.js server calls the Supabase RPC
   `create_plan_with_image` (a `plans` row plus a `plan_images` row with status `pending`), uploads the file to the
   private `plans` bucket with the anon key, then marks the image `uploaded` (or `failed`). The plan id is a UUID
   generated in the browser.
2. **Analyse** (`POST /api/plans/{id}/analyze`, `main.py`): if the plan already has a saved analysis
   (`plan_knowledge`, or the local copy in `CACHE_DIR/plan-graphs/`), it is returned at once (200). Otherwise a
   **background job** is submitted and the call returns 202 `{state, stage, position}`.
3. **Job queue** (`app/jobs.py`): a thread pool with `MAX_CONCURRENT_ANALYSES` workers (default 1). One job per
   plan: submitting again returns the running job, a failed job is retried. The job runs `_analyze_plan` inside
   a `Profile`, so the current stage name is visible to the poller. Finished jobs stay in memory for 15 minutes; the
   result itself is persisted anyway.
4. **Polling** (`GET /api/plans/{id}/analysis`): the plan page polls every 1.5 s and shows the queue position or the
   current step ("Reading text", "Finding walls", ...). A 404 `unknown` (backend restarted) makes the page submit
   again.
5. **The job** (`_analyze_plan`): download the file from Storage → by content (`%PDF-` → PDF parser, else image) →
   decode, and downscale if above `MAX_ANALYSIS_PIXELS` (16 MP) → pipeline → save the graph (Supabase
   `plan_knowledge` + local file) → compute and save the geometry (`plan_geometry`). The geometry step never
   fails the analysis: if it errors, it is computed again on first use.

Heavy work (pipeline, geometry) also takes a slot from `_analysis_slots`, shared with request handlers that need
geometry for an older plan, so the server never runs more heavy jobs than it has memory for. A model **warm-up**
(`app/warmup.py`) runs in a background thread at startup and loads every network, so the first upload after a
deploy doesn't pay ~10–20 s for it. Jobs wait for it.

---

## 3. The image pipeline (PNG / JPG / WebP)

`app/cubicasa/detect.py: graph_from_image`. Every stage is cached per image (`CACHE_DIR/features/<plan>.*`), so
a second run only does the cheap parts.

### 3.1 Perception (`features.compute`)

| Step | Model | Input → output | Notes |
|---|---|---|---|
| `cubinet` | CubiCasa5K multi-task network (2019, pretrained, `models/cubicasa-net.pkl`, CC BY-NC) | image scaled to ≤ 1536 px → 44 channels: 21 junction heatmaps, 12 room classes, 11 icon classes | outputs are upsampled back to full resolution |
| `maps` | – | softmax / argmax / confidence, quantised to uint8, saved as `.npz` | full resolution, the costliest non-network step |
| `ocr` | RapidOCR (ONNX) | text detection once (small plans upscaled so the long side ≥ 1600 px), then each box recognised in 2 orientations (0/180° or 90/270° for vertical boxes) | tokens with box and score, `.json` |

### 3.2 Symbol detector (`symdet.py`)

YOLO11s trained by us on CubiCasa5K (`models/symbol-det.pt`; the sidecar `symbol-det.json` says imgsz 1600, whole
image, trained with 90° rotations). Eight classes: door, window, toilet, sink, double sink, shower, bathtub, sauna
stove. Detections ≥ 0.1 are cached; ≥ 0.4 become evidence items (`det-<class>-<k>`). Test recall: door 0.957, window
0.917. Its weaknesses: a run of windows separated only by mullions often comes out as one box (`windows.py` tries
to split them; the ground truth itself labels such runs inconsistently).

### 3.3 Features → counts (`fuse.py`, `fit.py`)

The question templates ("how many windows / bedrooms / saunas ...", 12 of them) are answered by small **per-template
counting models**, not by counting boxes directly. Reason: every evidence source over- or under-counts in its own
way, and a model fitted on labelled plans learns how to combine them.

- `image_features` turns the maps into ~600 numbers per plan: connected components of each icon / room class at
  several size and confidence thresholds, components after erosion (to split touching symbols), probability-map
  peaks, junction-heatmap peaks, room regions, "storeys" (big building blobs), OCR room labels (all, inside the
  building, unique), dimension texts like `12x21` (window / door sizes), plus the detector's counts at four
  confidence thresholds (`det_*`).
- `fit.py` fits three candidates per template: `rule` (the single best feature + an integer offset: fully
  transparent), `hgb_c` (gradient-boosted classifier over the count) and `hgb_p` (gradient-boosted Poisson regressor,
  rounded). Selection is by K-fold CV on the official **val** split. The chosen models are refit on train+val and
  reported once on **test**. Saved in `models/cubicasa-fusion.joblib`, with the feature version and the
  detector's content hash: a different detector refuses to load with old models, because `det_*` would silently
  shift.
- No feature may come from the file name (see [§10](#10-evaluation-and-data-hygiene)).

Result on the held-out test split (387 plans, 3398 questions): **81.8 %** exact. Kitchens 95, floors 93, bathrooms
93, ... bedrooms 78, doors 66, windows 44.

### 3.4 Evidence items (`_evidence`)

So that a count can be **shown**, the graph also carries the items behind it: detector boxes for the eight
detector classes; for the other icon classes (e.g. appliances, sauna benches), CubiCasa net blobs; room regions
from the net's room classes (with an outline polygon); OCR tokens that name a room (`rooms_dict.py` +
`token_table.py`, Finnish / Swedish / English abbreviations such as MH, KPH, WC, K, OH); floor outlines.
`answer.count` sorts the evidence by score and cites the top *n* for a count of *n*. When the number of evidence
items differs from the count, the answer says so and the confidence drops to `low`.

---

## 4. Room and wall geometry

`app/cubicasa/rooms.py: geometry_from_image`, run once per plan after the pipeline, stored in `plan_geometry`. This
is what the room tools, the room overlay and area questions use.

1. **Walls** (`layers.wall_classes`): a dedicated U-Net (ResNet-34 encoder, `models/wallseg.pt`) segments every
   pixel into background / solid wall / opening (door or window gap), at native resolution with 512 px tiles,
   overlap blending. Wall F1 0.925 on 60 test plans (CubiCasa net baseline 0.91, which also paints openings as wall).
2. **Thickness**: twice the median distance-to-background along the wall skeleton.
3. **Spaces**: walls + openings are closed by about a quarter of the wall thickness, then every enclosed region
   becomes a space (too small or touching the image border = not a room).
4. **Split** (`room_split.py`): a space holding two strong labels that shouldn't share a space (bedroom open to the
   living room) is cut along the straight line between the labels that crosses the least free space. Open-plan
   kitchen / living / dining / hall stay together.
5. **Labels and fixtures into rooms**: OCR tokens by their centre, fixtures by the detector boxes and net icons
   inside the outline.
6. **Type** (`room_types.py`): every piece of evidence adds points to room types. Weights are hand-set in
   `EVIDENCE_WEIGHTS`: a readable label +6, a decisive fixture +4..5 (toilet → WC, shower / bathtub → bathroom,
   sauna stove → sauna), net pixels ≤ +1.5. The tier is **strong** with a label or decisive fixture and a clear
   margin, **weak** on weak evidence or a conflict (label says kitchen, a toilet sits inside), and **unknown**
   with no evidence. The UI shows weak rooms dashed and explains each type ("why").
7. **Openings and wall segments** (`wallvec.py`): openings link rooms (doors / windows per room). The wall map is
   skeletonised and traced into straight centreline segments with length, thickness and the room on each side.
8. **Scale** (metres per pixel), first that applies:
   - the user's two-point calibration;
   - printed dimension lines (`dimensions.py`): ≥ 2 figures agreeing within 4 % with their tick-to-tick span, and
     inside the door / fixture size band of `size_priors.py`;
   - areas written inside rooms (`12.5` under a label): m/px = √(m² / room pixels). ≥ 2 rooms agreeing within 15 %;
     else a single room when it is the only one that passes every check (`rooms.single_room_scale`): one clean
     reading (a number with a unit, or one bare decimal at OCR score ≥ 0.9, not a sum), one room purpose, a
     convex-ish outline (area / hull ≥ 0.85), a written area plausible for the room type, and an m/px inside the
     door / fixture band. Two passing rooms that disagree give no scale;
   - the room-area questions and the outline review use this scale when the user has not calibrated one;
   - otherwise only a rough range from door and fixture sizes (shown as `~8–12 m²`, never as a measured area).

---

## 5. Vector PDFs

`app/pdfplan/`, deterministic, no vision. `text.py` reads every text run with its box (pdfium). `parse.py` finds
rooms as centred "name / number / area" stacks and apartment tags. If a same-name `.dxf` exists (dataset only), it
adds room and apartment outlines and door / window / fixture symbols (layers matched by keyword, joined to the PDF
with an affine fit through shared texts). Without a DXF, which is always the case for uploads, symbol counts come
from the image detector on the page rendered at 2× (low confidence), and apartment areas are "not found". Questions
are matched to 12 templates in EN / DE / FR / IT (`answer.py`). Swiss dwellings held-out half: 99.0 %. Uploaded
PDFs are answered by this parser, not the Gemini agent.

## 6. IFC models

`app/ifc/`, deterministic. ifcopenshell reads every element with class, storey, type, materials and all properties
and quantities (converted to m / m² / m³ in code), and measures each space's floor area, volume and height from its
3D shape. General tools (count / select / group by class + storey + filters, sums, extremes, types, materials,
systems) return IFC GlobalIds as citations. Available through `POST /api/ask` with a dataset path. Not yet in the
web app or the agent.

---

## 7. Answering a question

`POST /api/plans/{id}/ask` (`main.ask_plan`). In order:

1. **Load** the saved graph and apply the user's corrections ([§8](#8-human-corrections-scale-and-outlines)).
2. **Area questions** ("how big is the kitchen", "pinta-ala") are answered in code: they need a saved scale, and
   a room outline the user has accepted. Area = polygon area × (m per px)². Without these, the answer says what is
   missing.
3. **PDF plans** go to the PDF parser.
4. **Gemini agent** (`app/answering/agent.py`, `GEMINI_MODEL`):
   - The system prompt limits it to the plan and to building topics. It cannot see the drawing.
   - Tools (`tools.py`): `plan_summary`, `count_template` (the counting models), `list_items`, `items_in`
     (containment by the room's outline), `calculate` (add / subtract numbers earlier tools returned),
     `covering_quantity`, `list_rooms`, `room_details`, `list_walls`. At most 12 steps.
   - **Validation** of its `final_answer`:
     - the number must equal a number a tool returned (else not found);
     - cited ids must come from tool output;
     - confidence comes from the tool that produced the value (counting model → its confidence; a raw item list
       → low; a calculation → its least reliable operand; disagreement with the regex router → low);
     - a yes/no answer is checked against the matching count;
     - the explanation must state the value.
   - Answers without chat history are cached (`CACHE_DIR/agent`, key includes the prompt version).
5. **Fallback** (no API key, API error, timeout, hourly cap `GEMINI_MAX_PER_HOUR` reached, or the agent ended
   without an answer): the regex router (`router.py`, EN + FI patterns → template) plus the counting model.
   Anything else is "not supported yet".

Breadth, measured on questions outside the dataset templates (`openq.py`, EN / FI / SV): the agent answers 59 %
correctly on the real pipeline and 93 % with perfect perception. The gap is mostly perception (door / window
counts, rough room regions).

---

## 8. Human corrections, scale and outlines

- **Corrections** (`plan_corrections`, append-only events): users add, retype, move or delete rooms, doors,
  windows and fixtures. The effective graph = the original graph with the events replayed. Template counts are
  adjusted by the *difference* the corrections make to the evidence counts, so the counting model's calibration
  is kept. Answers note "includes user-reviewed corrections".
- **Scale** (`plan_scales`): two points + a known distance. It overrides the plan's own scale for every metric value.
- **Outlines** (`plan_outlines`): suggested floor / room boundaries the user accepts, edits or rejects; area
  answers use only accepted ones.
- **Feedback** (`plan_answer_feedback`): helpful / incorrect / unclear per answer, with the evidence ids, for later
  evaluation.

---

## 9. Storage and caches

| Where | What | Can it be deleted? |
|---|---|---|
| Supabase Storage `plans/` | uploaded files | no: the source of truth |
| Supabase tables | `plans`, `plan_images`, `plan_knowledge` (graph), `plan_geometry`, `plan_scales`, `plan_outlines`, `plan_corrections`, `plan_answer_feedback` | no |
| `CACHE_DIR/uploads/` | decoded upload PNG / PDF | yes (downloaded again) |
| `CACHE_DIR/features/` | net maps (`.npz`), OCR (`.json`), detections (`.det.json`) | yes (recomputed: a full analysis) |
| `CACHE_DIR/layers/walls/` | wall maps per weights file | yes |
| `CACHE_DIR/plan-graphs/` | local copy of each plan's graph (used when Supabase is down) | yes |
| `CACHE_DIR/agent/` | Gemini answers | yes (costs Gemini calls) |
| `CACHE_DIR/model-…-g3/` | graphs of *dataset* images (bench, `/api/ask`); the folder name holds the counting models' size / mtime, the detector hash and `GRAPH_VERSION`, so any of them changing starts a fresh cache | yes |
| `models/` | weights, tracked in git, except `cubicasa-net.pkl` (208 MB) | no |

---

## 10. Evaluation and data hygiene

**Splits.** CubiCasa5K's official train / val / test split. Counting models: selected by CV on val, refit on
train+val, reported once on test. The symbol detector and the wall segmenter were trained with the challenge
preview plans held out. **Never train on test.**

**The preview set is not clean for the counting models.** 197 of the 241 preview plans are in CubiCasa's *train*
split, which both the pretrained CubiCasa net and our counting models saw. Preview counting scores are therefore
optimistic. Report on the official test split.

**What the production path can and cannot see.** Uploads go through `graph_from_image`, which reads only the image
and the trained weights. Ground-truth labels are only read by `PLAN_SOURCE=svg`, training and the evaluators
(`labels_from_svg`, `eval`, `bench*`, `*_eval`, `openq`). Production pins `PLAN_SOURCE=model`.

**File-name leak, fixed 2026-11-15.** Until then the counting features included the CubiCasa *variant* parsed from
the dataset file name (`high_quality_architectural-123.png` → `var_high_quality_architectural = 1`). An upload's
name carries no variant, so in production those features were always 0, an input the models never saw. Measured
on the 241 preview plans, zeroing them changes 2 of 2892 counts, with the same accuracy (80.5 %). `image_features`
no longer computes them, and `fit.py` refuses `var_*` columns, so the next refit is clean. The shipped models still
list the keys, which now always read 0.

**Tuning that has seen held-out data** (known, accepted, keep in mind when reading numbers):
- room geometry rules (`room_split`, `room_types` weights) were tuned with `rooms_eval` on the *even* half of the
  preview plans; read the *odd* half;
- detector preset B was adopted after comparing test-split scores (81.8 vs 81.3 %, same run);
- the agent prompt has been iterated against the benchmark, which contains test-split plans.

**Benchmarks** (`app/bench.py`): fixed question sets (`fast` ≈ 460, `full` ≈ 7.9k) through the real `/api/ask`
path, with accuracy and bootstrap CIs, coverage, precision, calibration per confidence, citation validity, EN/FI
consistency and latency. `--kind open` = the breadth questions. `bench perception` scores boxes / rooms / counts
against the labels without any question. The goal is breadth of questions answered, not the dataset score: the
benchmark is a regression check.

---

## 11. Speed and memory

Measured with `scripts/profile_analysis.py` (laptop, models loaded; details in CLAUDE.md "Analysis speed"):

| Plan | Analysis + geometry | Peak memory |
|---|---|---|
| ≤ 1 MP (most CubiCasa plans) | 3–9 s | 1–2 GB |
| 12 MP | ~61 s (70.6 s before the 2026-11 speed-up) | ~6 GB |

At 12 MP, the wall segmenter (~27 s, native resolution) is the largest cost, then the CubiCasa net + its
full-resolution maps (~9 s), OCR (~5 s) and features (~7 s). Memory grows by ~0.27 GB per megapixel because walls
and OCR work at native resolution. That is why uploads are capped at 16 MP. Every analysis logs an
`analysis_profile` line with all stage timings.

---

## 12. Known weaknesses

- **Doors and windows** are the weakest counts (66 / 44 % exact): runs of windows behind mullions, doors drawn
  without an arc, and the labels themselves are inconsistent.
- **Room regions** come from the wall map: a missing wall merges rooms, a wall stub splits them. Open-plan areas
  are one space by design.
- **Scale**: plans without dimension lines or written areas only get a rough size range until the user calibrates.
- **Non-CubiCasa drawings** (CAD exports, scans with hatching, other drawing styles) are out of the training
  distribution of every network here. Expect lower accuracy and more "not found".
- **Uploaded PDFs** have no DXF: no room outlines, and symbols come from the image detector at low confidence.
- **Licences**: the CubiCasa5K data and pretrained net are CC BY-NC; Ultralytics is AGPL-3.0.
