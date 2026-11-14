import pytest

from app.config import CUBICASA_ORIG_DIR, DATASET_DIR
from app.router import route

SAMPLE = "data/cubicasa5k/high_quality_architectural-1.png"


@pytest.mark.parametrize(
    "question,template",
    [
        ("How many sinks and wash basins (kitchen and bathroom combined; a double-bowl sink counts as one) are drawn on this floor plan?", "sinks"),
        ("Kuinka monta suihkua ja kylpyammetta pohjapiirustukseen on piirretty yhteensä?", "showers_baths"),
        ("How many bathrooms or washrooms (e.g. KH, KPH, PH, PESUH, SH) are on this floor plan, not counting separate WC rooms?", "bathrooms"),
        ("Kuinka monta erillistä WC-tilaa pohjapiirustuksessa on?", "wc_rooms"),
        ("How many toilet bowls (WC seats) are drawn on this floor plan?", "wc_seats"),
        ("How many floors (storeys) does this floor plan show?", "floors"),
        ("How many windows are shown on this floor plan?", "windows"),
        ("Kuinka monta ovea pohjapiirustuksessa on?", "doors"),
        ("What is the weather like?", None),
    ],
)
def test_route(question, template):
    assert route(question) == template


needs_data = pytest.mark.skipif(
    not (DATASET_DIR / SAMPLE).is_file() or not CUBICASA_ORIG_DIR.is_dir(), reason="dataset not available"
)


@needs_data
def test_svg_oracle_sample():
    from app.cubicasa.answer import count
    from app.cubicasa.labels_from_svg import graph_from_svg

    g = graph_from_svg(SAMPLE)
    expected = {"doors": 11, "windows": 5, "bedrooms": 1, "kitchens": 1, "bathrooms": 2, "wc_rooms": 1,
                "saunas": 2, "wc_seats": 2, "sinks": 4, "showers_baths": 3, "floors": 2}
    assert {t: count(g, t).answer for t in expected} == expected
    # every citation lies inside the image (true for this sample)
    for it in g.items:
        x0, y0, x1, y1 = it.bbox
        assert 0 <= x0 < x1 <= g.width and 0 <= y0 < y1 <= g.height


@needs_data
def test_api_ask_svg(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    monkeypatch.setattr(main, "PLAN_SOURCE", "svg")
    client = TestClient(main.app)
    r = client.post("/api/ask", json={"path": SAMPLE, "question": "Kuinka monta saunaa pohjapiirustuksessa on?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == 2 and len(body["citations"]) == 2
    assert client.post("/api/ask", json={"path": "data/cubicasa5k/../../questions.jsonl", "question": "doors"}).status_code == 400


@pytest.mark.parametrize(
    "token,expected",
    [("MH", "bedroom"), ("MH 12,5 m²", "bedroom"), ("KH", "bathroom"), ("PESUH.", "bathroom"), ("WC", "wc"),
     ("SAUNA", "sauna"), ("KT", "kitchen"), ("TUPAK.+HUONE", "kitchen"), ("Lasitettu parveke", "balcony"),
     ("PARVEKE", "balcony"), ("OH", "other"), ("TK", "other"), ("+108.382", None), ("70,0 m²", None)],
)
def test_room_dictionary(token, expected):
    from app.cubicasa.rooms_dict import room_type

    assert room_type(token) == expected


def test_apartment_header():
    from app.cubicasa.rooms_dict import apartment_type

    assert apartment_type("3H+K+S") == {"rooms": 3, "sauna": True, "kitchen": True}
    assert apartment_type("2h+kt") == {"rooms": 2, "sauna": False, "kitchen": True}
    assert apartment_type("KH") is None
