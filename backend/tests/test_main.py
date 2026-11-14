"""API hardening: JSON error bodies (the Next.js proxies parse every response as JSON) and the render-size cap."""

from fastapi.testclient import TestClient

from app.config import MAX_ANALYSIS_PIXELS, pdf_render_scale
from app.main import app


def test_unhandled_error_is_json():
    path = "/__test_unhandled_error"

    def boom():
        raise RuntimeError("boom")

    app.add_api_route(path, boom)
    try:
        response = TestClient(app, raise_server_exceptions=False).get(path)
    finally:
        app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != path]
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal error (RuntimeError)"}


def test_pdf_render_scale_caps_large_pages():
    assert pdf_render_scale(595, 842) == 2.0  # A4 renders at 2x
    w, h = 14400, 14400  # the largest page PDF allows
    k = pdf_render_scale(w, h)
    assert k < 2.0 and (w * k) * (h * k) <= MAX_ANALYSIS_PIXELS * 1.0001


def test_large_upload_is_downscaled_before_analysis(monkeypatch, tmp_path):
    import io
    from uuid import uuid4

    from PIL import Image

    import app.config
    import app.cubicasa.detect as detect
    import app.main as main
    from app.models import PlanGraph

    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, format="PNG")
    seen = {}

    def fake_graph(rel, path):
        seen["size"] = Image.open(path).size
        return PlanGraph(image=rel, width=seen["size"][0], height=seen["size"][1], source="model")

    monkeypatch.setattr(main, "MAX_ANALYSIS_PIXELS", 30_000)
    monkeypatch.setattr(app.config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(main, "_saved_graph", lambda plan_id: None)
    monkeypatch.setattr(main, "_download_source", lambda plan_id: buf.getvalue())
    monkeypatch.setattr(main, "_persist_graph", lambda plan_id, graph: None)
    monkeypatch.setattr(main, "_store_geometry_safely", lambda plan_id, graph: None)
    monkeypatch.setattr(detect, "graph_from_image", fake_graph)
    graph = main.analyze_plan(uuid4())
    w, h = seen["size"]
    assert w * h <= 30_000 and abs(w / h - 4 / 3) < 0.02
    assert (graph.width, graph.height) == (w, h)
