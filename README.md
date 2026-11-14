# Next.js + FastAPI starter

This repository runs a Next.js frontend and a FastAPI backend with Docker Compose.

## Teammate setup

The `dataset/` folder is gitignored (about 35 GB, plus licence terms), so it is set up by hand:

1. **Challenge data.** Download the Junction dataset from the challenge and unzip it so that
   `dataset/junction-dataset/questions.jsonl` and `dataset/junction-dataset/data/` exist.
2. **Share bundle.** Get `stoa-share.zip` from the team drive and unzip it in the repo root:

   ```sh
   unzip stoa-share.zip
   ```

   It adds the files below under `dataset/`:

   | Path | What it is |
   |---|---|
   | `models/cubicasa-fusion.joblib`, `models/ocr_token_table.json` | Trained counting models and OCR word table; required by the API |
   | `cubicasa-pretrained/model_best_val_loss_var.pkl` | CubiCasa5K segmentation network weights (CC BY-NC, non-commercial) |
   | `cache/features_table.jsonl` | Per-image evidence features; lets you refit (`app.cubicasa.fit`) without re-running extraction |
   | `cache/svg/` | Ground-truth labels parsed from the original CubiCasa SVGs (for `eval --method svg` and `token_table`) |

3. **Python environment and smoke test** (needs [uv](https://docs.astral.sh/uv/)):

   ```sh
   cd backend
   uv sync
   uv run pytest -q
   uv run python -m app.eval --method model --split test --limit 20
   ```

   The first run on each image takes 3–5 s on CPU (segmentation network + OCR); results are then cached in
   `dataset/cache/`.

The bundle is a snapshot: whoever refits the models rebuilds it from the repo root and re-uploads it:

```sh
zip -r stoa-share.zip dataset/models/cubicasa-fusion.joblib dataset/models/ocr_token_table.json \
  dataset/cubicasa-pretrained/model_best_val_loss_var.pkl dataset/cache/features_table.jsonl dataset/cache/svg
```

More commands (extraction, fitting, evaluation) are in `CLAUDE.md`.

## Start the app

```sh
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000) for the frontend. The API is available at [http://localhost:8000](http://localhost:8000), with interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

The frontend and backend source folders are mounted into their containers for development. Both services reload when their source changes. Rebuild the services after changing their dependencies:

```sh
docker compose up --build
```

Stop the stack with `Ctrl+C`, or run `docker compose down` in another terminal.
