import io
import json
import os
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.config import DATASET_DIR
from app.cubicasa.answer import count
from app.cubicasa.graph import get_graph, resolve_dataset_path
from app.models import Answer, AskRequest, Layer, PlanGraph, PlanItem, PlanOutlinesRequest, PlanQuestionRequest
from app.router import route

app = FastAPI(title="Junc API", version="0.1.0")

# "model" = image-only pipeline (default). "svg" = original CubiCasa labels, for demos/debugging only.
PLAN_SOURCE = os.environ.get("PLAN_SOURCE", "model")
# "llm" = Gemini agent with tools over the plan (falls back to "template" without GEMINI_API_KEY or on API errors).
ANSWERER = os.environ.get("ANSWERER", "llm")


def _check_path(path: str):
    try:
        p = resolve_dataset_path(path)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {path}") from None
    if p.parent != (DATASET_DIR / "data" / "cubicasa5k").resolve() or p.suffix != ".png":
        raise HTTPException(400, "Only data/cubicasa5k/*.png files are supported so far.")
    return p


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/hello")
async def hello() -> dict[str, str]:
    return {"message": "Hello from FastAPI"}


@app.get("/api/plans/{plan_id}")
def get_plan(plan_id: UUID) -> dict:
    """Return a plan and its linked Storage objects by plan UUID."""
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured")

    query = urlencode({"id": f"eq.{plan_id}", "select": "id,created_at,metadata,plan_images(*)"})
    request = Request(
        f"{supabase_url.rstrip('/')}/rest/v1/plans?{query}",
        headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            plans = json.loads(response.read())
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="Supabase plan lookup failed") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase") from error

    if not plans:
        raise HTTPException(status_code=404, detail="Plan not found")
    return plans[0]


def _knowledge_request(plan_id: UUID, method: str = "GET", body: dict | None = None):
    """Read or persist the analyzed graph using backend-only Supabase credentials."""
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured")
    query = urlencode({"on_conflict": "plan_id"}) if method == "POST" else urlencode({"plan_id": f"eq.{plan_id}", "select": "graph"})
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}", "Content-Type": "application/json"}
    if method == "POST":
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    data = json.dumps(body).encode() if body is not None else None
    request = Request(f"{supabase_url.rstrip('/')}/rest/v1/plan_knowledge?{query}", data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read()) if response.status != 204 and response.headers.get("Content-Length") != "0" else None
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="Supabase plan knowledge request failed") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase") from error


def _saved_graph(plan_id: UUID) -> PlanGraph | None:
    rows = _knowledge_request(plan_id)
    if not rows:
        return None
    try:
        graph = PlanGraph.model_validate(rows[0]["graph"])
        if not graph.floor_outlines and graph.source == "model":
            from app.cubicasa.detect import floor_outline_candidates

            graph.floor_outlines = floor_outline_candidates(graph.image)
            if graph.floor_outlines:
                graph.items = [item for item in graph.items if item.category != "floor"] + [
                    PlanItem(id=outline.id, category="floor", bbox=(min(x for x, _ in outline.polygon),
                        min(y for _, y in outline.polygon), max(x for x, _ in outline.polygon),
                        max(y for _, y in outline.polygon)), polygon=outline.polygon, score=outline.score)
                    for outline in graph.floor_outlines
                ]
                _knowledge_request(plan_id, "POST", {"plan_id": str(plan_id), "graph": graph.model_dump(mode="json")})
        return graph
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Saved plan analysis is invalid") from error


@app.get("/api/plans/{plan_id}/scale")
def get_plan_scale(plan_id: UUID) -> dict | None:
    return _scale_request(plan_id)


@app.put("/api/plans/{plan_id}/scale")
def save_plan_scale(plan_id: UUID, calibration: dict) -> dict:
    try:
        points = calibration["points"]
        x1, y1 = (float(value) for value in points[0])
        x2, y2 = (float(value) for value in points[1])
        distance = float(calibration["distance"])
        unit = calibration["unit"]
    except (KeyError, TypeError, ValueError, IndexError):
        raise HTTPException(status_code=422, detail="Provide two points, a distance, and a unit") from None
    if unit not in ("m", "cm", "ft", "in") or distance <= 0 or (x1 == x2 and y1 == y2):
        raise HTTPException(status_code=422, detail="Distance and points must be valid")
    plan = get_plan(plan_id)
    image = next((item for item in plan.get("plan_images", []) if item.get("role") == "source"), None)
    if not image:
        raise HTTPException(status_code=409, detail="Plan has no source image")
    return _scale_request(plan_id, "PUT", {"plan_id": str(plan_id), "points": [[x1, y1], [x2, y2]], "distance": distance, "unit": unit})


def _scale_request(plan_id: UUID, method: str = "GET", body: dict | None = None):
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured")
    query = urlencode({"on_conflict": "plan_id"}) if method == "PUT" else urlencode({"plan_id": f"eq.{plan_id}", "select": "points,distance,unit"})
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}", "Content-Type": "application/json"}
    if method == "PUT":
        headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    # PostgREST upserts use POST + on_conflict, even though our app endpoint is PUT.
    request = Request(f"{supabase_url.rstrip('/')}/rest/v1/plan_scales?{query}", data=json.dumps(body).encode() if body else None, headers=headers, method="POST" if method == "PUT" else "GET")
    try:
        with urlopen(request, timeout=20) as response:
            rows = json.loads(response.read())
        return rows[0] if rows else None
    except HTTPError as error:
        reason = error.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Plan scale request failed: {reason or error.reason}") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase") from error


def _outline_request(plan_id: UUID, method: str = "GET", body: dict | None = None):
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured")
    query = urlencode({"on_conflict": "plan_id"}) if method == "PUT" else urlencode({"plan_id": f"eq.{plan_id}", "select": "outlines"})
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}", "Content-Type": "application/json"}
    if method == "PUT":
        headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    request = Request(f"{supabase_url.rstrip('/')}/rest/v1/plan_outlines?{query}",
                      data=json.dumps(body).encode() if body else None, headers=headers,
                      method="POST" if method == "PUT" else "GET")
    try:
        with urlopen(request, timeout=20) as response:
            rows = json.loads(response.read())
        return rows[0]["outlines"] if rows else []
    except HTTPError as error:
        reason = error.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Floor outline request failed: {reason or error.reason}") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase") from error


@app.get("/api/plans/{plan_id}/outline")
def get_plan_outline(plan_id: UUID) -> dict:
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    return {"candidates": [outline.model_dump(mode="json") for outline in graph.floor_outlines],
            "saved": _outline_request(plan_id)}


@app.put("/api/plans/{plan_id}/outline")
def save_plan_outline(plan_id: UUID, req: PlanOutlinesRequest) -> dict:
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    for outline in req.outlines:
        if any(x < 0 or x > graph.width or y < 0 or y > graph.height for x, y in outline.polygon):
            raise HTTPException(status_code=422, detail="Outline points must be inside the plan image")
    saved = _outline_request(plan_id, "PUT", {"plan_id": str(plan_id),
        "outlines": [outline.model_dump(mode="json") for outline in req.outlines]})
    return {"saved": saved}


@app.get("/api/plans/{plan_id}/image")
def get_plan_image(plan_id: UUID) -> Response:
    """Stream the private source image for a plan to its frontend viewer."""
    plan = get_plan(plan_id)
    images = [image for image in plan.get("plan_images", []) if image.get("role") == "source"]
    if not images:
        raise HTTPException(status_code=404, detail="Plan has no source image")
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    image_ref = images[0]
    object_path = "/".join(quote(part, safe="") for part in image_ref["object_path"].split("/"))
    bucket = quote(image_ref["bucket_id"], safe="")
    request = Request(
        f"{supabase_url}/storage/v1/object/{bucket}/{object_path}",
        headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
    )
    try:
        with urlopen(request, timeout=30) as upstream:
            content = upstream.read()
            media_type = upstream.headers.get("Content-Type", "application/octet-stream")
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="Could not download the uploaded plan image") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase Storage") from error
    return Response(content, media_type=media_type, headers={"Cache-Control": "private, max-age=300"})


@app.post("/api/plans/{plan_id}/analyze", response_model=PlanGraph)
def analyze_plan(plan_id: UUID) -> PlanGraph:
    """Download the uploaded source image from Supabase and analyze it with the plan models."""
    if graph := _saved_graph(plan_id):
        return graph
    plan = get_plan(plan_id)
    images = [image for image in plan.get("plan_images", []) if image.get("role") == "source"]
    if not images:
        raise HTTPException(status_code=409, detail="Plan has no source image")

    image_ref = images[0]
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    object_path = "/".join(quote(part, safe="") for part in image_ref["object_path"].split("/"))
    bucket = quote(image_ref["bucket_id"], safe="")
    image_url = f"{supabase_url}/storage/v1/object/{bucket}/{object_path}"
    request = Request(image_url, headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"})
    try:
        with urlopen(request, timeout=60) as response:
            image_bytes = response.read()
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="Could not download the uploaded plan image") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase Storage") from error

    # Use the real uploaded image with the existing image-only detector pipeline. A plan-specific
    # cache key prevents unrelated uploads from sharing feature files.
    from PIL import Image, UnidentifiedImageError
    from app.cubicasa.detect import graph_from_image

    from app.config import CACHE_DIR

    cache_key = f"uploads/{plan_id}.png"
    cache_path = CACHE_DIR / cache_key
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(cache_path, format="PNG")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=422, detail="Uploaded source is not a readable image") from error
    try:
        graph = graph_from_image(cache_key, cache_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=f"Plan analysis models are unavailable: {error}") from error
    _knowledge_request(plan_id, "POST", {"plan_id": str(plan_id), "graph": graph.model_dump(mode="json")})
    return graph


@app.post("/api/plans/{plan_id}/ask", response_model=Answer)
def ask_plan(plan_id: UUID, req: PlanQuestionRequest) -> Answer:
    """Answer a question using the analysis saved for this plan in Supabase."""
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    if ANSWERER == "llm":
        from app.answering.agent import answer

        if (result := answer(req.question, graph, req.history, req.scale)) is not None:
            return result
    return _template_answer(req.question, graph)


def _template_answer(question: str, graph) -> Answer:
    """Regex router + template counter: the offline path, and the fallback when Gemini is unavailable."""
    template = route(question)
    if template is None:
        return Answer(answer=None, confidence="not_found", explanation="This question type is not supported yet.")
    return count(graph, template)


@app.post("/api/ask")
def ask(req: AskRequest) -> Answer:
    _check_path(req.path)
    try:
        graph = get_graph(req.path, PLAN_SOURCE)
    except FileNotFoundError as error:  # e.g. model weights or original labels missing
        return Answer(answer=None, template=route(req.question), confidence="not_found",
                      explanation=f"Parser unavailable: {error}")
    if ANSWERER == "llm":
        from app.answering.agent import answer

        if (result := answer(req.question, graph)) is not None:
            return result
    return _template_answer(req.question, graph)


@app.get("/api/image")
def image(path: str = Query(...)):
    return FileResponse(_check_path(path), media_type="image/png")


def _layer(name: str, path: str) -> Layer:
    from app.cubicasa import layers

    p = _check_path(path)
    try:
        return layers.LAYERS[name](path, p)
    except FileNotFoundError as error:  # model weights not installed
        raise HTTPException(503, str(error)) from error


@app.get("/api/walls")
def walls(path: str = Query(...)) -> Layer:
    """Wall regions and door/window openings in the wall (wall segmenter, or the CubiCasa net as fallback)."""
    return _layer("walls", path)


@app.get("/api/symbols")
def symbols(path: str = Query(...)) -> Layer:
    """One box per door, window and fixture from the symbol detector."""
    return _layer("symbols", path)


@app.get("/api/text")
def text(path: str = Query(...)) -> Layer:
    """OCR tokens with boxes; room labels (e.g. "MH", "KPH") are mapped to a room type."""
    return _layer("text", path)


@app.get("/api/render")
def render(path: str = Query(...), question: str | None = Query(None),
           layer: Literal["answer", "walls", "symbols", "text"] = Query("answer")):
    """Render the answer citations or one perception layer over a plan image."""
    from PIL import Image, ImageDraw

    from app.cubicasa import layers

    p = _check_path(path)
    if layer == "answer":
        if not question:
            raise HTTPException(400, "layer=answer needs question")
        answer = ask(AskRequest(path=path, question=question))
        img = Image.open(p).convert("RGB")
        draw = ImageDraw.Draw(img)
        for i, item in enumerate(answer.citations, 1):
            if item.polygon:
                draw.polygon(item.polygon, outline=(220, 30, 30), width=4)
            else:
                draw.rectangle(item.bbox, outline=(220, 30, 30), width=4)
            draw.text((item.bbox[0] + 4, item.bbox[1] + 2), str(i), fill=(220, 30, 30))
    else:
        try:
            img = layers.render(layer, path, p)
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
