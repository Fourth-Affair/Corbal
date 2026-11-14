import io
import json
import logging
import math
import os
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.cubicasa.answer import count
from app.cubicasa.graph import get_graph, resolve_dataset_path
from app.models import (Answer, AskRequest, Layer, PlanGeometry, PlanGraph, PlanItem, PlanOutlinesRequest,
                        PlanQuestionRequest, PlanScaleCalibration)
from app.router import route

app = FastAPI(title="Junc API", version="0.1.0")
log = logging.getLogger(__name__)

# "model" = image-only pipeline (default). "svg" = original CubiCasa labels, for demos/debugging only.
PLAN_SOURCE = os.environ.get("PLAN_SOURCE", "model")
# "llm" = Gemini agent with tools over the plan (falls back to "template" without GEMINI_API_KEY or on API errors).
ANSWERER = os.environ.get("ANSWERER", "llm")

# Plans are routed by file type, never by folder: images -> image pipeline, PDFs -> vector PDF parser.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".pdf": "application/pdf"}


def _is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def _check_path(path: str, pdf_ok: bool = False):
    """Resolve a dataset-relative path to a plan file: an image, or (where supported) a PDF."""
    try:
        p = resolve_dataset_path(path)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {path}") from None
    suffix = p.suffix.lower()
    if suffix in IMAGE_SUFFIXES or (pdf_ok and suffix == ".pdf"):
        return p
    kinds = "an image (.png, .jpg)" + (" or a .pdf" if pdf_ok else "")
    raise HTTPException(400, f"Expected {kinds}, got {p.name}.")


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


def _download_source(plan_id: UUID) -> bytes:
    """The plan's uploaded source file (image or PDF) from Supabase Storage."""
    plan = get_plan(plan_id)
    images = [image for image in plan.get("plan_images", []) if image.get("role") == "source"]
    if not images:
        raise HTTPException(status_code=409, detail="Plan has no source file")
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    image_ref = images[0]
    object_path = "/".join(quote(part, safe="") for part in image_ref["object_path"].split("/"))
    bucket = quote(image_ref["bucket_id"], safe="")
    request = Request(f"{supabase_url}/storage/v1/object/{bucket}/{object_path}",
                      headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"})
    try:
        with urlopen(request, timeout=60) as upstream:
            return upstream.read()
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="Could not download the uploaded plan file") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase Storage") from error


def _is_pdf_bytes(content: bytes) -> bool:
    return content[:5] == b"%PDF-"


def _upload_pdf(plan_id: UUID, content: bytes | None = None):
    """Local copy of an uploaded PDF (the PDF parser reads files); downloaded again if the cache was cleared."""
    from app.config import CACHE_DIR

    path = CACHE_DIR / "uploads" / f"{plan_id}.pdf"
    if not path.is_file():
        content = content if content is not None else _download_source(plan_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pdf.tmp")
        tmp.write_bytes(content)
        tmp.replace(path)
    return path


def _render_pdf_page(path_or_bytes, scale: float = 2.0):
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path_or_bytes)
    try:
        return pdf[0].render(scale=scale).to_pil().convert("RGB")
    finally:
        pdf.close()


@app.get("/api/plans/{plan_id}/image")
def get_plan_image(plan_id: UUID) -> Response:
    """The plan's source image for the frontend viewer; a PDF is served as its first page rendered to PNG."""
    content = _download_source(plan_id)
    if _is_pdf_bytes(content):
        buf = io.BytesIO()
        _render_pdf_page(content).save(buf, format="PNG")
        content, media_type = buf.getvalue(), "image/png"
    else:
        from PIL import Image, UnidentifiedImageError

        try:
            media_type = Image.MIME.get(Image.open(io.BytesIO(content)).format, "application/octet-stream")
        except (UnidentifiedImageError, OSError):
            media_type = "application/octet-stream"
    return Response(content, media_type=media_type, headers={"Cache-Control": "private, max-age=300"})


@app.post("/api/plans/{plan_id}/analyze", response_model=PlanGraph)
def analyze_plan(plan_id: UUID) -> PlanGraph:
    """Download the uploaded source file from Supabase and analyze it (image pipeline, or the PDF parser)."""
    if graph := _saved_graph(plan_id):
        return graph
    image_bytes = _download_source(plan_id)
    if _is_pdf_bytes(image_bytes):  # vector PDF: deterministic parser, coordinates in PDF points
        from app.pdfplan.answer import plan_graph

        try:
            graph = plan_graph(str(_upload_pdf(plan_id, image_bytes)), image=f"uploads/{plan_id}.pdf")
        except Exception as error:  # noqa: BLE001  unreadable or encrypted PDF
            raise HTTPException(status_code=422, detail=f"Uploaded PDF could not be read: {error}") from error
        _knowledge_request(plan_id, "POST", {"plan_id": str(plan_id), "graph": graph.model_dump(mode="json")})
        _store_geometry_safely(plan_id, graph)
        return graph

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
    _store_geometry_safely(plan_id, graph)
    return graph


# ---------------------------------------------------------------- plan geometry (rooms, walls, openings)

def _geometry_request(plan_id: UUID, method: str = "GET", body: dict | None = None) -> dict | None:
    """Read or upsert the plan_geometry row (backend-only credentials, like plan_knowledge)."""
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured")
    query = urlencode({"on_conflict": "plan_id"}) if method == "POST" else urlencode(
        {"plan_id": f"eq.{plan_id}", "select": "geometry"})
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}", "Content-Type": "application/json"}
    if method == "POST":
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    request = Request(f"{supabase_url.rstrip('/')}/rest/v1/plan_geometry?{query}",
                      data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            if method == "POST":
                return None
            rows = json.loads(response.read())
            return rows[0]["geometry"] if rows else None
    except HTTPError as error:
        reason = error.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Plan geometry request failed: {reason or error.reason}") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase") from error


def _compute_geometry(plan_id: UUID, graph: PlanGraph) -> PlanGeometry:
    from app.cubicasa.rooms import geometry_from_graph, geometry_from_image

    if graph.source == "pdf":
        return geometry_from_graph(graph)
    from app.config import CACHE_DIR

    rel = f"uploads/{plan_id}.png"
    path = CACHE_DIR / rel
    if not path.is_file():  # cache cleared since the analysis: fetch the source image again
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(_download_source(plan_id))).convert("RGB").save(path, format="PNG")
    return geometry_from_image(rel, path, graph)


def _store_geometry(plan_id: UUID, geometry: PlanGeometry) -> None:
    _geometry_request(plan_id, "POST", {
        "plan_id": str(plan_id), "geometry": geometry.model_dump(mode="json", exclude={"rooms": {"__all__": {"area_m2"}}}),
        "model": geometry.model, "labeled_rooms": sum(r.confidence == "labeled" for r in geometry.rooms),
        "rough_rooms": sum(r.confidence == "rough" for r in geometry.rooms)})


def _store_geometry_safely(plan_id: UUID, graph: PlanGraph) -> None:
    """Geometry is an extra layer: a failure here must not fail the analysis (it is retried on first read)."""
    try:
        _store_geometry(plan_id, _compute_geometry(plan_id, graph))
    except Exception as error:  # noqa: BLE001
        log.warning("plan %s: geometry not stored (%s)", plan_id, error)


def _metres_per_unit(scale: PlanScaleCalibration) -> float | None:
    """Metres per image pixel (PDF point) from the user's two-point calibration."""
    (x1, y1), (x2, y2) = scale.points
    span = math.hypot(x2 - x1, y2 - y1)
    if span <= 0:
        return None
    unit_m = {"m": 1.0, "cm": 0.01, "ft": 0.3048, "in": 0.0254}[scale.unit]
    return scale.distance * unit_m / span


def with_areas(geometry: PlanGeometry, scale: PlanScaleCalibration | None) -> PlanGeometry:
    if scale and (m := _metres_per_unit(scale)):
        for room in geometry.rooms:
            room.area_m2 = round(room.area_px * m * m, 2)
    return geometry


def _plan_geometry(plan_id: UUID, graph: PlanGraph) -> PlanGeometry:
    """Stored geometry, computed and stored on first use for plans analysed before it existed."""
    stored = _geometry_request(plan_id)
    if stored is not None:
        try:
            return PlanGeometry.model_validate(stored)
        except ValueError:
            log.warning("plan %s: stored geometry is invalid, recomputing", plan_id)
    geometry = _compute_geometry(plan_id, graph)
    _store_geometry(plan_id, geometry)
    return geometry


def _saved_scale(plan_id: UUID) -> PlanScaleCalibration | None:
    try:
        row = _scale_request(plan_id)
        return PlanScaleCalibration.model_validate(row) if row else None
    except (HTTPException, ValueError):
        return None


@app.get("/api/plans/{plan_id}/geometry", response_model=PlanGeometry)
def get_plan_geometry(plan_id: UUID, rough: bool = Query(True, description="include rough (unlabelled) rooms")
                      ) -> PlanGeometry:
    """Rooms (polygons, purpose read from the drawing, labeled vs rough), walls and openings of an analysed plan.
    Room areas in m² are filled in when the plan has a scale calibration."""
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    try:
        geometry = _plan_geometry(plan_id, graph)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=f"Wall model unavailable: {error}") from error
    if not rough:
        geometry.rooms = [r for r in geometry.rooms if r.confidence == "labeled"]
    return with_areas(geometry, _saved_scale(plan_id))


@app.post("/api/plans/{plan_id}/ask", response_model=Answer)
def ask_plan(plan_id: UUID, req: PlanQuestionRequest) -> Answer:
    """Answer a question using the analysis saved for this plan in Supabase."""
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    if graph.source == "pdf":
        from app.pdfplan.answer import answer_question

        return answer_question(str(_upload_pdf(plan_id)), req.question)
    if ANSWERER == "llm":
        from app.answering.agent import answer

        geometry = None
        try:  # room tools for the agent; without geometry it answers with the item tools only
            geometry = with_areas(_plan_geometry(plan_id, graph), req.scale)
        except Exception as error:  # noqa: BLE001
            log.warning("plan %s: no geometry for the agent (%s)", plan_id, error)
        if (result := answer(req.question, graph, req.history, req.scale, geometry=geometry)) is not None:
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
    _check_path(req.path, pdf_ok=True)
    if _is_pdf(req.path):  # vector plan: deterministic parser, citations in PDF points
        from app.pdfplan.answer import answer_question

        try:
            return answer_question(req.path, req.question)
        except FileNotFoundError as error:
            return Answer(answer=None, confidence="not_found", explanation=f"Parser unavailable: {error}")
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
    p = _check_path(path, pdf_ok=True)
    return FileResponse(p, media_type=MEDIA[p.suffix.lower()])


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

    p = _check_path(path, pdf_ok=layer == "answer")
    if layer == "answer":
        if not question:
            raise HTTPException(400, "layer=answer needs question")
        answer = ask(AskRequest(path=path, question=question))
        k = 1.0
        if _is_pdf(path):  # citations are PDF points: render page 1 at 2x
            import pypdfium2 as pdfium

            k = 2.0
            pdf = pdfium.PdfDocument(str(p))
            img = pdf[0].render(scale=k).to_pil().convert("RGB")
            pdf.close()
        else:
            img = Image.open(p).convert("RGB")
        draw = ImageDraw.Draw(img)
        for i, item in enumerate(answer.citations, 1):
            if item.polygon:
                draw.polygon([(x * k, y * k) for x, y in item.polygon], outline=(220, 30, 30), width=4)
            else:
                draw.rectangle([v * k for v in item.bbox], outline=(220, 30, 30), width=4)
            draw.text((item.bbox[0] * k + 4, item.bbox[1] * k + 2), str(i), fill=(220, 30, 30))
    else:
        try:
            img = layers.render(layer, path, p)
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
