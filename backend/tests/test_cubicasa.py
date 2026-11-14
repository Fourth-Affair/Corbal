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


def test_det_features_thresholds_and_groups():
    from app.cubicasa.symdet import det_features

    dets = [{"cls": "window", "conf": 0.9, "bbox": [0, 0, 1, 1]},
            {"cls": "window", "conf": 0.3, "bbox": [2, 0, 3, 1]},
            {"cls": "sink", "conf": 0.6, "bbox": [0, 0, 1, 1]},
            {"cls": "double_sink", "conf": 0.45, "bbox": [0, 0, 1, 1]}]
    f = det_features(dets)
    assert f["det_window_c25"] == 2 and f["det_window_c40"] == 1 and f["det_window_c70"] == 1
    assert f["det_sinks_c40"] == 2 and f["det_sinks_c55"] == 1
    assert f["det_door_c25"] == 0 and f["det_door_sum"] == 0
    assert abs(f["det_window_sum"] - 1.2) < 1e-9


def test_det_items_replace_segmentation_classes():
    from app.cubicasa.detect import _det_items

    items = _det_items([{"cls": "door", "conf": 0.8, "bbox": [1, 2, 3, 4]},
                        {"cls": "shower", "conf": 0.2, "bbox": [1, 2, 3, 4]}])  # below ITEM_CONF
    assert [(it.category, it.cls, it.raw) for it in items] == [("door", "", "detector")]


def test_export_classes_match_detector():
    from app.cubicasa.export_yolo import DET_CLASSES
    from app.cubicasa.symdet import DET_ITEMS, DET_NAMES

    assert DET_CLASSES == [DET_ITEMS[n] for n in DET_NAMES]


@pytest.mark.parametrize("size", [(300, 200), (700, 530), (1200, 1100)])
def test_wallseg_sliding_window_shapes(size):
    import numpy as np
    import torch
    from PIL import Image

    from app.cubicasa import wallseg

    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, len(wallseg.CLASSES), 3, padding=1).eval()  # stand-in for the U-Net
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    p = wallseg.predict(img, model=model, batch=4)
    assert p.shape == (len(wallseg.CLASSES), size[1], size[0])
    assert np.allclose(p.sum(0), 1, atol=1e-4) and p.min() >= 0


def test_walls_layer_items(monkeypatch, tmp_path):
    import numpy as np

    from app.cubicasa import layers

    cls = np.zeros((100, 120), dtype=np.uint8)
    cls[10:14, 10:110] = 1  # one wall
    cls[50:90, 60:64] = 1  # another, not touching
    cls[10:14, 40:60] = 2  # an opening inside the first wall
    cls[0:2, 0:2] = 1  # speck: ignored
    monkeypatch.setattr(layers, "wall_classes", lambda rel, path: (cls, "test"))
    layer = layers.walls("data/cubicasa5k/x.png", tmp_path / "x.png")
    cats = sorted(it.category for it in layer.items)
    assert cats == ["opening", "wall", "wall", "wall"]  # the opening splits the first wall in two; speck dropped
    assert layer.layer == "walls" and layer.model == "test" and layer.stats["opening_px"] == 80


@needs_data
def test_api_layers():
    from fastapi.testclient import TestClient

    import app.main as main
    from app.cubicasa import symdet

    client = TestClient(main.app)
    r = client.get("/api/text", params={"path": SAMPLE})
    assert r.status_code == 200 and r.json()["layer"] == "text" and r.json()["tokens"]
    r = client.get("/api/symbols", params={"path": SAMPLE})
    assert r.status_code == (200 if symdet.available() else 503)
    assert client.get("/api/render", params={"path": SAMPLE}).status_code == 400  # layer=answer needs a question
    assert client.get("/api/walls", params={"path": "data/cubicasa5k/../../questions.jsonl"}).status_code == 400
