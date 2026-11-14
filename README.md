# Stoa: answering questions about floor plans, with evidence

Junction 2026 entry for the Stoa challenge. The backend reads a floor plan image, extracts addressable
evidence (walls, symbols, text), and answers count questions ("how many windows?", in English or Finnish)
with citations that point to where on the plan each counted item is.

## What the backend can see

| Layer | Model | Endpoint | What you get |
|---|---|---|---|
| **Walls** | `wallseg.py`: U-Net (ResNet-34) trained on CubiCasa5K walls, native resolution, sliding window. Falls back to the CubiCasa5K network if its weights are missing | `GET /api/walls?path=` | Wall regions and door/window **openings** in walls, as boxes, plus wall pixel stats |
| **Symbols** | `symdet.py`: YOLO11s detector (ONNX, CPU) for door, window, toilet, sink, double sink, shower, bathtub, sauna stove | `GET /api/symbols?path=` | One box per object, with confidence and per-class counts |
| **Text** | `ocr.py`: RapidOCR, rotation-aware | `GET /api/text?path=` | Every text token with its box; room labels such as `MH` or `KPH` mapped to a room type |

Questions go to `POST /api/ask` (`{"path", "question"}`); the answer carries its citations.
`GET /api/render?path=&layer=walls|symbols|text` (or `&question=...`) returns the plan with that layer drawn on
top. All coordinates are pixels of the plan image, origin top-left. Interactive docs: http://localhost:8000/docs.

Measured on held-out plans (official CubiCasa5K test split):

- **Counting:** 77.1% exact answers before the symbol detector (doors 51%, windows 38%). The detector
  reaches mAP50 0.95 on doors and 0.90 on windows on the validation split; refitting the counting models
  with it is the next step.
- **Walls:** the CubiCasa network scores wall F1 0.91 (median plan 0.96, 3 px tolerance). The dedicated
  segmenter also separates openings from solid wall, which the older network cannot; on sample plans,
  solid-wall F1 is 0.98–1.00 against 0.82–0.89.

Background and next steps: [docs/literature-review.md](docs/literature-review.md).

## Teammate setup

The `dataset/` folder is gitignored (tens of GB, plus licence terms), so it is set up by hand:

1. **Challenge data.** Download the full Junction dataset (`junction-dataset.zip`) and unzip it into
   `dataset/` so that `dataset/junction-dataset/questions.jsonl` and `dataset/junction-dataset/data/` exist.
   If you also have the preview release, unzip it as `dataset/junction-dataset-preview/`: we treat its plans
   as test data, and the training commands exclude them (`--holdout`).
2. **Original CubiCasa5K** (only for training and wall evaluation): download `cubicasa5k.zip` from
   [Zenodo 2613548](https://zenodo.org/records/2613548) (5.5 GB) and unzip it into `dataset/cubicasa5k-original/`.
   It holds the official train/val/test split and the SVG labels.
3. **Share bundle.** Get `stoa-share.zip` from the team drive and unzip it in the repo root:

   ```sh
   unzip stoa-share.zip
   ```

   It adds these files under `dataset/`:

   | Path | What it is |
   |---|---|
   | `models/cubicasa-fusion.joblib`, `models/ocr_token_table.json` | Counting models and OCR word table; required by `/api/ask` |
   | `models/symbol-det.onnx` | Symbol detector; required by `/api/symbols` |
   | `models/wallseg.pt` | Wall segmenter; `/api/walls` falls back to the CubiCasa network without it |
   | `cubicasa-pretrained/model_best_val_loss_var.pkl` | CubiCasa5K network weights (CC BY-NC, non-commercial) |
   | `cache/features_table.jsonl`, `cache/features/*.det.json` | Cached evidence features and detections, so the counting models can be refitted without re-running the networks |
   | `cache/svg/` | Ground-truth labels parsed from the original CubiCasa SVGs |

4. **Python environment and smoke test** (needs [uv](https://docs.astral.sh/uv/)):

   ```sh
   cd backend
   uv sync
   uv run pytest -q
   uv run python -m app.eval --method model --split test --limit 20
   ```

   The first run on each image takes 3–5 s on CPU (networks + OCR); results are then cached in `dataset/cache/`.

The bundle is a snapshot. Whoever retrains a model rebuilds it from the repo root and re-uploads it:

```sh
zip -r stoa-share.zip dataset/models/cubicasa-fusion.joblib dataset/models/ocr_token_table.json \
  dataset/models/symbol-det.onnx dataset/models/wallseg.pt \
  dataset/cubicasa-pretrained/model_best_val_loss_var.pkl dataset/cache/features_table.jsonl \
  dataset/cache/features/*.det.json dataset/cache/svg
```

Training and evaluation commands (symbol detector on a rented Verda GPU, wall segmenter on a local GPU,
counting-model fits) are in [CLAUDE.md](CLAUDE.md).

## Start the app

```sh
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000) for the frontend. The API runs at
[http://localhost:8000](http://localhost:8000), with interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

The frontend and backend source folders are mounted into their containers, and both reload when their source
changes. Rebuild after changing dependencies (`backend/pyproject.toml`, `frontend/package.json`):

```sh
docker compose up --build
```

Stop the stack with `Ctrl+C`, or run `docker compose down` in another terminal.

## Licences

CubiCasa5K data and the pretrained CubiCasa network are CC BY-NC(-SA): non-commercial use only. Ultralytics
(used to train and run the detector) is AGPL-3.0. Fine for the hackathon; revisit before any commercial use.
## Store uploaded images in Supabase Storage

Image uploads are sent directly from the browser to a Supabase Storage bucket. PDFs continue to be previewed locally and are not uploaded. The frontend needs the Supabase project URL and the public anon/publishable key; never put a `service_role` or secret key in frontend environment variables.

1. In the Supabase dashboard, open **Storage** and create a bucket named `plans` (or choose another name). Keep it private unless you specifically want uploaded images to be publicly readable.
2. Open **Project Settings → API** and copy the project URL and anon/publishable key.
3. Create `frontend/.env.local` using `frontend/.env.example`, and fill in the values. For Docker Compose, also copy those same three variables into a root `.env` file (next to `compose.yaml`), since Compose passes them to the frontend container.
4. Allow browser uploads by adding this policy in **SQL Editor**. It allows anonymous users to add objects to this bucket; access to read or delete objects remains restricted.

   ```sql
   create policy "Allow browser image uploads to plans"
   on storage.objects
   for insert
   to anon
   with check (
     bucket_id = 'plans'
     and (storage.extension(name) = 'png'
       or storage.extension(name) = 'jpg'
       or storage.extension(name) = 'jpeg'
       or storage.extension(name) = 'webp'
       or storage.extension(name) = 'gif')
   );
   ```

   If you use a different bucket name, change `bucket_id` in the policy and set `NEXT_PUBLIC_SUPABASE_BUCKET` to that name in both env files.
5. Restart the frontend (`docker compose up --build` for Docker, or restart `npm run dev` if running locally). Select or drop an image; the UI reports whether the upload succeeded. Objects appear under **Storage → plans**.

This setup uses the anon key, which is intended to be exposed in browser apps. The Storage policy controls what that key can do. For a production app with user accounts, restrict inserts to authenticated users and their own folder, and set appropriate bucket file-size and MIME-type limits in Supabase.
