import io
import json
import logging
import math
import os
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.cubicasa.answer import count
from app.cubicasa.graph import get_graph, resolve_dataset_path
from app.models import (Answer, AskRequest, Layer, PlanCorrectionRequest, PlanFeedbackRequest, PlanGeometry,
                        PlanGraph, PlanItem, PlanOutlinesRequest, PlanQuestionRequest, PlanScale,
                        PlanScaleCalibration)
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
        if graph.source == "model":
            from app.cubicasa.detect import floor_outline_candidates, room_outline_candidates

            current_candidates = floor_outline_candidates(graph.image)
            current_rooms = room_outline_candidates(graph.image)
            changed = (
                (current_candidates and [item.model_dump(mode="json") for item in current_candidates] !=
                 [item.model_dump(mode="json") for item in graph.floor_outlines])
                or (current_rooms and [item.model_dump(mode="json") for item in current_rooms] !=
                    [item.model_dump(mode="json") for item in graph.room_outlines])
            )
            if changed:
                graph.floor_outlines = current_candidates
                graph.room_outlines = current_rooms
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


def _corrections_request(plan_id: UUID, body: dict | None = None):
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured")
    method = "POST" if body else "GET"
    query = urlencode({"plan_id": f"eq.{plan_id}", "select": "action,item_id,original_item,corrected_item,created_at",
                       "order": "created_at.asc,id.asc"})
    headers = {"apikey": service_key, "Authorization": f"Bearer {service_key}", "Content-Type": "application/json"}
    if body:
        headers["Prefer"] = "return=minimal"
    request = Request(f"{supabase_url.rstrip('/')}/rest/v1/plan_corrections?{query}",
                      data=json.dumps(body).encode() if body else None, headers=headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return None if body or response.status == 204 else json.loads(response.read())
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="Supabase correction request failed") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase") from error


def _apply_corrections(graph: PlanGraph, events: list[dict]) -> PlanGraph:
    """Apply append-only human edits to a copy of the model graph, retaining its original record."""
    if not events:
        return graph
    original_items = list(graph.items)
    items = list(graph.items)
    for event in events:
        item_id = event["item_id"]
        items = [item for item in items if item.id != item_id]
        corrected = event.get("corrected_item")
        if corrected:
            replacement = PlanItem.model_validate(corrected)
            items.append(replacement)
    if graph.counts:
        baseline = graph.model_copy(deep=True)
        baseline.items = original_items
        baseline.counts = {}
        edited = graph.model_copy(deep=True)
        edited.items = items
        edited.counts = {}
        adjusted = dict(graph.counts)
        for template, base_count in graph.counts.items():
            before = count(baseline, template).answer or 0
            after = count(edited, template).answer or 0
            adjusted[template] = max(0, base_count + after - before)
        graph = graph.model_copy(deep=True)
        graph.counts = adjusted
    graph = graph.model_copy(deep=True)
    graph.items = items
    graph.user_correction_count = len(events)
    return graph


@app.get("/api/plans/{plan_id}/corrections")
def get_plan_corrections(plan_id: UUID) -> dict:
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    events = _corrections_request(plan_id) or []
    corrected = _apply_corrections(graph, events)
    return {"items": [item.model_dump(mode="json") for item in corrected.items], "events": events}


@app.post("/api/plans/{plan_id}/corrections")
def add_plan_correction(plan_id: UUID, req: PlanCorrectionRequest) -> dict:
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    existing_events = _corrections_request(plan_id) or []
    effective = _apply_corrections(graph, existing_events)
    if req.action == "add":
        if req.item is None or req.item_id is not None:
            raise HTTPException(status_code=422, detail="Adding an item requires item and no item_id")
        item = req.item.model_copy(update={"id": f"user-{uuid4()}", "score": None, "raw": None, "user_verified": True})
        if item.category in ("floor", "wall", "opening"):
            raise HTTPException(status_code=422, detail="Only rooms, doors, windows, and fixtures can be added")
        if (any(value < 0 for value in item.bbox) or item.bbox[2] > graph.width or item.bbox[3] > graph.height
                or item.bbox[0] >= item.bbox[2] or item.bbox[1] >= item.bbox[3]):
            raise HTTPException(status_code=422, detail="Item bounds must be inside the plan image")
        item_id = item.id
        before = None
        after = item.model_dump(mode="json")
    else:
        if not req.item_id or (req.action == "delete" and req.item is not None) or (req.action == "update" and req.item is None):
            raise HTTPException(status_code=422, detail="Updating requires item_id and item; deleting requires item_id only")
        current = next((item for item in effective.items if item.id == req.item_id), None)
        if current is None:
            raise HTTPException(status_code=404, detail="Plan item not found")
        item_id = current.id
        before = current.model_dump(mode="json")
        after = None
        if req.action == "update":
            if req.item.category not in ("room", "door", "window", "fixture"):
                raise HTTPException(status_code=422, detail="Only rooms, doors, windows, and fixtures can be corrected")
            if (any(value < 0 for value in req.item.bbox) or req.item.bbox[2] > graph.width or req.item.bbox[3] > graph.height
                    or req.item.bbox[0] >= req.item.bbox[2] or req.item.bbox[1] >= req.item.bbox[3]):
                raise HTTPException(status_code=422, detail="Item bounds must be inside the plan image")
            if req.item.polygon and any(x < 0 or y < 0 or x > graph.width or y > graph.height for x, y in req.item.polygon):
                raise HTTPException(status_code=422, detail="Item outline must be inside the plan image")
            # Keep identity and source fields while saving the reviewer's corrected type and geometry.
            after = current.model_copy(update={"category": req.item.category, "cls": req.item.cls,
                                               "bbox": req.item.bbox, "polygon": req.item.polygon,
                                               "score": None, "user_verified": True}).model_dump(mode="json")
    event = {"plan_id": str(plan_id), "action": req.action, "item_id": item_id,
             "original_item": before, "corrected_item": after}
    _corrections_request(plan_id, event)
    events = _corrections_request(plan_id) or []
    corrected = _apply_corrections(graph, events)
    return {"items": [item.model_dump(mode="json") for item in corrected.items], "events": events}


@app.post("/api/plans/{plan_id}/feedback")
def add_plan_answer_feedback(plan_id: UUID, req: PlanFeedbackRequest) -> dict:
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured")
    body = {"plan_id": str(plan_id), **req.model_dump()}
    request = Request(f"{supabase_url.rstrip('/')}/rest/v1/plan_answer_feedback",
                      data=json.dumps(body).encode(), headers={
                          "apikey": service_key, "Authorization": f"Bearer {service_key}",
                          "Content-Type": "application/json", "Prefer": "return=minimal",
                      }, method="POST")
    try:
        with urlopen(request, timeout=20):
            pass
    except HTTPError as error:
        raise HTTPException(status_code=502, detail="Supabase feedback request failed") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=502, detail="Could not reach Supabase") from error
    return {"saved": True}


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
    elif method == "DELETE":
        headers["Prefer"] = "return=minimal"
    request = Request(f"{supabase_url.rstrip('/')}/rest/v1/plan_outlines?{query}",
                      data=json.dumps(body).encode() if body else None, headers=headers,
                      method="POST" if method == "PUT" else method)
    try:
        with urlopen(request, timeout=20) as response:
            if method == "DELETE":
                return []
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
    candidates = graph.floor_outlines + graph.room_outlines
    return {"candidates": [outline.model_dump(mode="json") for outline in candidates],
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


@app.delete("/api/plans/{plan_id}/outline")
def delete_plan_outline(plan_id: UUID) -> dict:
    _outline_request(plan_id, "DELETE")
    return {"saved": []}


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
        "plan_id": str(plan_id),
        # metric values and size checks depend on the user's calibration at read time: stored in pixels only
        "geometry": geometry.model_dump(mode="json", exclude={"rooms": {"__all__": {"area_m2", "area_m2_range",
                                                                                  "size_check"}},
                                                              "wall_segments": {"__all__": {"length_m", "thickness_m"}}}),
        "model": geometry.model, "strong_rooms": sum(r.tier == "strong" for r in geometry.rooms),
        "weak_rooms": sum(r.tier == "weak" for r in geometry.rooms)})


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


def with_metrics(geometry: PlanGeometry, calibration: PlanScaleCalibration | None) -> PlanGeometry:
    """Fill room areas (m²) and wall lengths / thicknesses (m). The user's calibration wins over the scale derived from
    written room areas; geometry.scale says which one was used."""
    if calibration and (m := _metres_per_unit(calibration)):
        geometry.scale = PlanScale(m_per_px=m, source="calibration")
    if geometry.scale:
        m = geometry.scale.m_per_px
        for room in geometry.rooms:
            room.area_m2 = round(room.area_px * m * m, 2)
        for seg in geometry.wall_segments:
            seg.length_m = round(seg.length_px * m, 2)
            seg.thickness_m = round(seg.thickness_px * m, 3)
    from app.cubicasa.size_priors import check_sizes

    check_sizes(geometry)  # rough ranges from symbol sizes + plausibility (may lower a room's tier)
    return geometry


def _plan_geometry(plan_id: UUID, graph: PlanGraph) -> PlanGeometry:
    """Stored geometry, computed and stored on first use for plans analysed before it existed."""
    stored = _geometry_request(plan_id)
    if stored is not None and stored.get("version", 1) >= PlanGeometry.model_fields["version"].default:
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
def get_plan_geometry(plan_id: UUID, uncertain: bool = Query(True, description="include weak / unknown rooms")
                      ) -> PlanGeometry:
    """Rooms (polygons, type from combined evidence with a strong / weak / unknown tier), wall segments, walls and
    openings of an analysed plan. Metric areas and lengths use the user's calibration, else written room areas."""
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    try:
        geometry = _plan_geometry(plan_id, graph)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=f"Wall model unavailable: {error}") from error
    if not uncertain:
        geometry.rooms = [r for r in geometry.rooms if r.tier == "strong"]
    return with_metrics(geometry, _saved_scale(plan_id))


@app.post("/api/plans/{plan_id}/ask", response_model=Answer)
def ask_plan(plan_id: UUID, req: PlanQuestionRequest) -> Answer:
    """Answer a question using the analysis saved for this plan in Supabase."""
    graph = _saved_graph(plan_id)
    if graph is None:
        raise HTTPException(status_code=409, detail="Plan analysis is not ready yet")
    graph = _apply_corrections(graph, _corrections_request(plan_id) or [])
    question = req.question.casefold()
    area_question = any(term in question for term in ("area", "size", "how large", "how big", "square metre", "square meter", "m²", "pinta-ala", "pinta ala", "kuinka suuri", "yta", "kvadratmeter"))
    if area_question:
        if req.scale is None:
            return Answer(answer=None, confidence="not_found", explanation="I need a known distance to establish the drawing scale before estimating floor area.")
        room_terms = (("living room", "living"), ("livingroom", "living"), ("kitchen", "kitchen"), ("bedroom", "bedroom"),
                      ("bathroom", "bathroom"), ("washroom", "bathroom"), ("bath", "bathroom"), ("balcony", "balcony"),
                      ("sauna", "sauna"), ("garage", "garage"), ("storage", "storage"), ("entry", "entry"),
                      ("hallway", "entry"), ("wc", "wc"), ("toilet room", "wc"))
        requested_room = next((room_type for term, room_type in room_terms if term in question), None)
        if requested_room:
            outlines = [outline for outline in req.floor_outlines if outline.scope == "room" and outline.room_type == requested_room and outline.decision != "rejected"]
            suggested_ids = {outline.id for outline in graph.room_outlines
                             if outline.scope == "room" and outline.room_type == requested_room}
            reviewed_ids = {outline.id for outline in req.floor_outlines
                            if outline.scope == "room" and outline.room_type == requested_room
                            and outline.decision in ("accepted", "rejected")}
            if suggested_ids - reviewed_ids:
                return Answer(answer=None, confidence="not_found",
                              explanation=f"Some suggested {requested_room} boundaries still need to be reviewed before I can calculate their areas.")
            if not outlines:
                return Answer(answer=None, confidence="not_found", explanation=f"The {requested_room} boundary still needs to be reviewed or drawn before I can estimate its area.")
        else:
            outlines = [outline for outline in req.floor_outlines if outline.scope == "floor" and outline.decision != "rejected"]
            if not outlines:
                return Answer(answer=None, confidence="not_found", explanation="The drawing scale is available, but a whole-floor boundary still needs to be reviewed or drawn before I can estimate floor area.")
        calibration_pixels = ((req.scale.points[1][0] - req.scale.points[0][0]) ** 2 +
                              (req.scale.points[1][1] - req.scale.points[0][1]) ** 2) ** 0.5
        if calibration_pixels <= 0:
            return Answer(answer=None, confidence="not_found", explanation="The scale reference points are too close together. Please recalibrate using a longer known distance.")
        meters_per_unit = {"m": 1.0, "cm": 0.01, "ft": 0.3048, "in": 0.0254}[req.scale.unit]
        meters_per_pixel = req.scale.distance * meters_per_unit / calibration_pixels
        room_areas = []
        for outline in outlines:
            polygon = outline.polygon
            pixel_area = abs(sum(x * polygon[(index + 1) % len(polygon)][1] - polygon[(index + 1) % len(polygon)][0] * y
                                 for index, (x, y) in enumerate(polygon))) / 2
            room_areas.append((outline, pixel_area * meters_per_pixel * meters_per_pixel))
        area = sum(value for _, value in room_areas)
        if requested_room and len(room_areas) > 1:
            details = "; ".join(f"{outline.label or requested_room} {index + 1}: {value:.1f} m²"
                                 for index, (outline, value) in enumerate(room_areas))
            return Answer(answer=details + f". Total: {area:.1f} m².", unit=None, confidence="low",
                          explanation="Areas are calculated separately from the accepted room outlines and saved drawing scale. Check each highlighted boundary.",
                          answered_by="template")
        return Answer(answer=round(area, 1), unit="m²", confidence="low",
                      explanation=f"Estimated {outlines[0].label.lower()} area from the saved scale and user-accepted outline. Check the highlighted boundary before relying on this area.",
                      answered_by="template")
    if graph.source == "pdf":
        from app.pdfplan.answer import answer_question

        return answer_question(str(_upload_pdf(plan_id)), req.question)
    if ANSWERER == "llm":
        from app.answering.agent import answer

        geometry = None
        try:  # room tools for the agent; without geometry it answers with the item tools only
            geometry = with_metrics(_plan_geometry(plan_id, graph), req.scale)
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
