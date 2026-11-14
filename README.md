# Stoa: answering questions about construction drawings, with evidence

Junction 2026 entry for the Stoa challenge. Upload a floor plan (image or PDF) and ask about it in plain language,
in English, Finnish or Swedish ("how many windows are in the bedroom?", "montako WC-istuinta?"). Every answer
points to the evidence it came from (a box or outline on the plan, or an IFC `GlobalId`), so you can check it in seconds.

The design rule is **evidence first, LLM second**: deterministic parsers and small vision models turn the drawing into
addressable evidence (rooms, walls, doors, windows, fixtures, text). The LLM (Gemini) only plans and calls tools that
count, measure and do the arithmetic in code. Answers it cannot back with tool output are flagged or become "not found".

## What it handles

| Input | Pipeline | How it reads the drawing |
|---|---|---|
| Plan **image** (`.png`, `.jpg`, `.webp` upload) | `backend/app/cubicasa/` | Wall segmenter (U-Net), YOLO11s symbol detector (door, window, toilet, sink, shower, bathtub, sauna stove), rotation-aware OCR for room labels, CubiCasa5K network for room classes; rooms from the wall map, typed from labels + fixtures |
| Vector **PDF** | `backend/app/pdfplan/` | Text with coordinates (rooms, numbers, areas); outlines and symbols from a same-name `.dxf` when present, else the image detector on the rendered page |
| **IFC** model (API only, not in the web app yet) | `backend/app/ifc/` | ifcopenshell: classes, storeys, types, materials, quantities, spaces measured from their geometry |

Questions go through a Gemini tool loop (`app/answering/agent.py`) over these tools. Without a `GEMINI_API_KEY`, or
when the API fails or the hourly cap is reached, a deterministic router and per-question-type counters answer instead.

## Results

Held-out data only; full numbers and history in [CLAUDE.md](CLAUDE.md).

- **Plan images, counting** (official CubiCasa5K test split, 387 plans / 3398 questions): **81.8 %** exact
  (majority baseline 52.8 %). Strongest: kitchens 95 %, floors 93 %, bathrooms 93 %. Weakest: doors 66 %, windows 44 %.
  Most window misses are runs of windows behind mullions, which the ground truth itself labels inconsistently.
- **Vector PDFs** (swiss_dwellings, held-out half): **99.0 %**. All 71 preview PDFs: 88.5 %.
- **IFC** (preview set): areas 99 %, overall 82 % with 91 % coverage.
- **Breadth** (questions outside the dataset templates, `app/answering/openq.py`): 59 % on the real pipeline,
  93 % with perfect perception. The remaining gap is mostly perception, not question understanding.

## The web app

1. The browser uploads the file through the frontend to a private Supabase Storage bucket and creates a `plans` row.
2. `POST /api/plans/{id}/analyze`: the backend downloads the file and runs the image pipeline or the PDF parser.
   It stores the resulting `PlanGraph` in `plan_knowledge` and the room / wall geometry in `plan_geometry`, so each
   plan is analysed only once.
3. The plan page shows the drawing with detected rooms and objects. You can ask questions (cited items are
   highlighted), calibrate the scale with two points, measure distances, review room outlines, and correct
   detections. Corrections are stored as append-only events, and answers use the corrected plan.

The browser only talks to Next.js. Next.js API routes proxy to FastAPI, and only the backend holds the Supabase
service-role key.

## Setup

Needs [uv](https://docs.astral.sh/uv/), Docker (for the full stack), and Node 22 (only to run the frontend outside Docker).

1. **Model weights.** Everything except one file is in git under [`models/`](models/) (see `models/MANIFEST.json`).
   The 208 MB CubiCasa5K network is not: get `stoa-share.zip` from the team drive, unzip it in the repo root, and
   copy it into place:

   ```sh
   unzip stoa-share.zip
   cp dataset/cubicasa-pretrained/model_best_val_loss_var.pkl models/cubicasa-net.pkl   # optional: the old path also works
   ```

   The bundle also holds cached features and parsed CubiCasa SVG labels (`dataset/cache/`) for refitting models
   without rerunning the networks. Its `dataset/models/` copies are older than `models/` and are not used.
2. **Supabase** (for the web app): create the project, apply `supabase/migrations/` and create the `plans` bucket.
   See [deploy/README.md](deploy/README.md#2-supabase).
3. **Environment.** `cp .env.example .env` and fill in `GEMINI_API_KEY` and the Supabase values. `.env` is gitignored.
4. **Datasets** (only for evaluation and training, not needed to run the app). Everything under `dataset/` is gitignored:
   - Challenge data: unzip `junction-dataset.zip` into `dataset/junction-dataset/`. The preview release goes in
     `dataset/junction-dataset-preview/`; it is our test set and is never trained on.
   - Original CubiCasa5K (training, wall evaluation): unzip `cubicasa5k.zip` from
     [Zenodo 2613548](https://zenodo.org/records/2613548) into `dataset/cubicasa5k-original/`.
5. **Smoke test:**

   ```sh
   cd backend
   uv sync
   uv run pytest -q
   ```

## Run it

```sh
docker compose up --build
```

Frontend at <http://localhost:3000>. API at <http://localhost:8000>, with interactive docs at
<http://localhost:8000/docs>. Both containers hot-reload on source changes. Rebuild after changing
`backend/pyproject.toml` or `frontend/package.json`. The first analysis of a plan takes 10–90 s on CPU, depending
on its size, because the models are loaded and walls and OCR run at native resolution.

Without Docker: `cd backend && uv run uvicorn app.main:app --reload`, and
`cd frontend && npm ci && npm run dev` with `BACKEND_URL=http://localhost:8000` and the Supabase values in
`frontend/.env.local`.

The dataset endpoints work without Supabase: `POST /api/ask` with `{"path": "data/...", "question": "..."}` for a
dataset image, PDF or IFC. `GET /api/render?path=...&question=...` draws the cited evidence on the plan, and
`/api/walls`, `/api/symbols` and `/api/text` return one perception layer.

## Production

A single CPU server behind Caddy (HTTPS, per-IP rate limits). Every push to `main` is tested and deployed by GitHub
Actions. See [deploy/README.md](deploy/README.md) for setup, cost limits and operating notes.

## Evaluation and training

Every pipeline change is checked with the benchmark (`uv run python -m app.bench run --set fast`), and with
`--kind open` for questions outside the dataset templates. The benchmark is a regression check, not the goal.
Training the detector on a rented GPU, the wall segmenter, and refitting the counting models: see
[CLAUDE.md](CLAUDE.md#commands). Background: [docs/literature-review.md](docs/literature-review.md).

## Licences

CubiCasa5K data and the pretrained CubiCasa network are CC BY-NC(-SA): non-commercial use only. Ultralytics (used to
train and run the detector) is AGPL-3.0. Fine for the hackathon; revisit before any commercial use.
