import io
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.config import DATASET_DIR
from app.cubicasa.answer import count
from app.cubicasa.graph import get_graph, resolve_dataset_path
from app.models import Answer, AskRequest
from app.router import route

app = FastAPI(title="Junc API", version="0.1.0")

# "model" = image-only pipeline (default). "svg" = original CubiCasa labels, for demos/debugging only.
PLAN_SOURCE = os.environ.get("PLAN_SOURCE", "model")


def _check_path(path: str):
    try:
        p = resolve_dataset_path(path)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {path}")
    if p.parent != (DATASET_DIR / "data" / "cubicasa5k").resolve() or p.suffix != ".png":
        raise HTTPException(400, "Only data/cubicasa5k/*.png files are supported so far.")
    return p


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
def ask(req: AskRequest) -> Answer:
    _check_path(req.path)
    template = route(req.question)
    if template is None:
        return Answer(answer=None, confidence="not_found",
                      explanation="This question type is not supported yet.")
    try:
        graph = get_graph(req.path, PLAN_SOURCE)
    except FileNotFoundError as e:  # e.g. model weights or original labels missing
        return Answer(answer=None, template=template, confidence="not_found",
                      explanation=f"Parser unavailable: {e}")
    return count(graph, template)


@app.get("/api/image")
def image(path: str = Query(...)):
    return FileResponse(_check_path(path), media_type="image/png")


@app.get("/api/render")
def render(path: str = Query(...), question: str = Query(...)):
    """The plan with the counted items outlined. Handy for quick human verification."""
    from PIL import Image, ImageDraw

    answer = ask(AskRequest(path=path, question=question))
    img = Image.open(_check_path(path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, it in enumerate(answer.citations, 1):
        if it.polygon:
            draw.polygon(it.polygon, outline=(220, 30, 30), width=4)
        else:
            draw.rectangle(it.bbox, outline=(220, 30, 30), width=4)
        draw.text((it.bbox[0] + 4, it.bbox[1] + 2), str(i), fill=(220, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
