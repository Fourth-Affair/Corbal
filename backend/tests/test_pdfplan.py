import pytest

from app.config import DATASET_DIR
from app.pdfplan.answer import answer_question, parse_question

# a vector PDF plan that has a same-name DXF next to it
PLAN = "data/swiss_dwellings/site1000_bldg2082_floor9385.pdf"


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How many apartments (residential units) are shown on this floor plan?", ("count:apartments", None)),
        ("Combien d'appartements (logements) figurent sur ce plan d'étage ?", ("count:apartments", None)),
        ("Wie viele Türen (Innentüren und Eingangstüren zusammen) sind auf diesem Grundriss eingezeichnet?", ("count:doors", None)),
        ("Quante porte (porte interne e porte d'ingresso insieme) sono disegnate in questa pianta?", ("count:doors", None)),
        ("Combien de fenêtres sont dessinées sur ce plan ?", ("count:windows", None)),
        ("Wie viele WCs sind auf diesem Grundriss eingezeichnet?", ("count:toilets", None)),
        ("Quanti lavabi sono disegnati in questa pianta?", ("count:washbasins", None)),
        ("Wie viele Badewannen sind auf diesem Grundriss eingezeichnet?", ("count:bathtubs", None)),
        ("Quante docce sono disegnate in questa pianta?", ("count:showers", None)),
        ("How many rooms labelled 'Bad' are there on this floor?", ("label_count", "Bad")),
        ("How many rooms labelled 'Cage d'escalier' are there on this floor?", ("label_count", "Cage d'escalier")),
        ("Combien de pièces portant la désignation « Cage d'escalier » y a-t-il à cet étage ?", ("label_count", "Cage d'escalier")),
        ("Quanti locali con la denominazione «Vano scala» ci sono in questo piano?", ("label_count", "Vano scala")),
        ("Wie gross ist die gesamte Nettofläche in m² aller Räume mit der Bezeichnung „Zimmer“ in diesem Geschoss?", ("label_area", "Zimmer")),
        ("What is the net floor area of room U1.15 ('Korridor') in m²?", ("room_area", "U1.15")),
        ("Qual è la superficie netta dell'appartamento C in m² (somma di tutti i locali, esclusi balconi)?", ("apartment_area", "C")),
        ("Wie gross ist die Nettowohnfläche von Wohnung B in m² (Summe aller Räume ohne Balkone)?", ("apartment_area", "B")),
        ("What is the total net floor area in m² of all numbered indoor rooms on this floor?", ("floor_area", None)),
        ("What is the weather like?", None),
    ],
)
def test_parse_question(question, expected):
    assert parse_question(question) == expected


needs_data = pytest.mark.skipif(not (DATASET_DIR / PLAN).is_file(), reason="sample PDF plan not available")


@needs_data
@pytest.mark.parametrize(
    "question,value",
    [
        ("How many apartments (residential units) are shown on this floor plan?", 4),
        ("How many doors (interior doors and entrance doors together) are drawn on this floor plan?", 36),
        ("How many windows are drawn on this floor plan?", 28),
        ("How many toilets (WC) are drawn on this floor plan?", 4),
        ("How many washbasins are drawn on this floor plan?", 4),
        ("How many bathtubs are drawn on this floor plan?", 4),
        ("How many rooms labelled 'Zimmer' are there on this floor?", 12),
        ("What is the net floor area of room 2.10 ('Zimmer') in m²?", 16.78),
        ("What is the total net area in m² of all rooms labelled 'Zimmer' on this floor?", 184.93),
    ],
)
def test_answers_with_citations(question, value):
    a = answer_question(PLAN, question)
    assert a.answer == pytest.approx(value, rel=0.02)
    assert a.confidence in ("high", "medium")
    assert a.citations, "every answer must point at evidence"
    for c in a.citations:  # citations are PDF points on an A3 landscape page
        x0, y0, x1, y1 = c.bbox
        assert 0 <= x0 < x1 <= 1191 and 0 <= y0 < y1 <= 842
    if isinstance(value, int) and "labelled" not in question:
        assert len(a.citations) == value


@needs_data
def test_pdf_text_alone():
    """Without the DXF, labels and printed areas still come from the PDF text; apartment areas say not found."""
    a = answer_question(PLAN, "What is the net floor area of room 2.10 ('Zimmer') in m²?", use_dxf=False)
    assert a.answer == pytest.approx(16.78) and a.confidence == "high"
    assert answer_question(PLAN, "How many rooms labelled 'Zimmer' are there on this floor?", use_dxf=False).answer == 12
    a = answer_question(PLAN, "What is the net floor area of apartment B in m², summing all its rooms?", use_dxf=False)
    assert a.answer is None and a.confidence == "not_found"


@needs_data
def test_api_routes_by_file_type():
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    r = c.post("/api/ask", json={"path": PLAN, "question": "Wie viele Fenster sind auf diesem Grundriss eingezeichnet?"})
    assert r.status_code == 200 and r.json()["answer"] == 28 and r.json()["template"] == "pdf_count_windows"
    dxf = PLAN.removesuffix(".pdf") + ".dxf"
    assert c.post("/api/ask", json={"path": dxf, "question": "How many windows?"}).status_code == 400
    assert c.get("/api/image", params={"path": PLAN}).headers["content-type"] == "application/pdf"


@needs_data
def test_uploaded_pdf_flow(tmp_path, monkeypatch):
    """Upload flow with Supabase mocked: analyze -> rendered page for the viewer -> PDF parser answers."""
    import uuid

    from fastapi.testclient import TestClient

    import app.main as m
    from app import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    store: dict = {}

    def knowledge(pid, method="GET", body=None):
        if method == "POST":
            store[str(pid)] = body
            return None
        return [store[str(pid)]] if str(pid) in store else []

    import time

    pdf = (DATASET_DIR / PLAN).read_bytes()
    monkeypatch.setattr(m, "_download_source", lambda pid: pdf)
    monkeypatch.setattr(m, "_knowledge_request", knowledge)
    monkeypatch.setattr(m, "_corrections_request", lambda pid, body=None: [])  # no user corrections
    c, pid = TestClient(m.app), uuid.uuid4()
    r = c.post(f"/api/plans/{pid}/analyze")  # starts the background job (202) or returns the saved graph (200)
    deadline = time.monotonic() + 30
    while r.status_code == 202 and time.monotonic() < deadline:
        time.sleep(0.05)
        r = c.get(f"/api/plans/{pid}/analysis")
    body = r.json()
    assert r.status_code == 200 and body.get("state", "done") == "done", body
    g = body.get("graph", body)
    assert g["source"] == "pdf" and (g["width"], g["height"]) == (1191, 842)
    assert c.get(f"/api/plans/{pid}/image").headers["content-type"] == "image/png"
    a = c.post(f"/api/plans/{pid}/ask", json={"question": "How many rooms labelled 'Zimmer' are there on this floor?"}).json()
    assert a["answer"] == 12 and len(a["citations"]) == 12


@needs_data
def test_agent_tools():
    from app.pdfplan.tools import PdfTools

    t = PdfTools(PLAN)
    out, trace = t.call("pdf_lookup", {"template": "label_count", "argument": "Zimmer"})
    assert out["value"] == answer_question(PLAN, "How many rooms labelled 'Zimmer' are there?").answer == 12
    assert trace.template_answer is not None and all(t.item(i) for i in trace.item_ids)
    assert "error" in t.call("pdf_lookup", {"template": "room_area"})[0]  # needs a room number
    out, trace = t.call("list_rooms", {"apartment": "A"})
    areas = [r["area_m2"] for r in out["rooms"] if "area_m2" in r]
    assert out["count"] == 9 and areas and all(a in trace.numbers for a in areas)
    out, trace = t.call("list_symbols", {"kind": "door"})
    assert out["count"] == len(trace.item_ids) > 0 and t.item(trace.item_ids[0]).category == "door"
    _, total = t.call("calculate", {"operation": "add", "operands": areas[:2]})
    assert total.numbers == [round(areas[0] + areas[1], 6)]
