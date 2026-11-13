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

- `backend/` — FastAPI (Python 3.12). Entry: `backend/app/main.py`.
- `frontend/` — Next.js 15 / React 19 / TypeScript. Calls backend via `BACKEND_URL`.
- `compose.yaml` — runs both with hot reload.
- `dataset/` (gitignored, create locally) — challenge dataset; never commit it.

## Commands

```sh
docker compose up --build          # full stack: frontend :3000, backend :8000 (docs at /docs)
docker compose up --build backend  # backend only
```

Rebuild after changing `backend/requirements.txt` or `frontend/package.json`.

## Conventions

- Pydantic models for every API request/response; keep the `Answer`/`Citation` schema stable — the
  frontend renders highlights from it.
- Coordinates: PDF user-space points, origin top-left (PyMuPDF convention), per page.
- Secrets (e.g. `ANTHROPIC_API_KEY`) via `.env`, never committed.
- Default LLM: latest Claude models (`claude-opus-5-5` for reasoning, `claude-haiku-5-5` for cheap
  bulk steps like page classification).
