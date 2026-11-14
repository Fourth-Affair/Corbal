"""Plan geometry: rooms from a wall map, tiers, openings, wall segments, purpose labels, room tools."""

import numpy as np
import pytest

from app.cubicasa import layers, rooms
from app.cubicasa.rooms_dict import room_purpose
from app.models import Layer, PlanGraph, PlanScaleCalibration, TextToken


@pytest.mark.parametrize("token,purpose", [
    ("MH", "bedroom"), ("MH2", "bedroom"), ("OH", "living"), ("ET", "entry"), ("VH", "closet"), ("KPH", "bathroom"),
    ("K", "kitchen"), ("Kitchen", "kitchen"), ("Living room", "living"), ("OH+K", "kitchen"),
    ("H2388", None), ("h187o", None), ("10.747", None), ("5K", None), ("NA", None),
    # OCR damage: label fused with a dimension, separator lost, umlaut lost, slash read as a letter
    ("VAR4956", "storage"), ("MHTYOH", "bedroom"), ("MH/TYÖH", "bedroom"), ("KEITTIO", "kitchen"),
    ("MHL", "bedroom"), ("AULAI", "entry"), ("KH", "bathroom"), ("KHH", "utility"), ("S", "sauna"),
    # German / French / Italian names printed on vector PDFs
    ("Zimmer", "bedroom"), ("Pièce", "bedroom"), ("Camera da letto", "bedroom"), ("Küche", "kitchen"),
    ("Cuisine", "kitchen"), ("Bagno", "bathroom"), ("Bain", "bathroom"), ("Wohnen/Essen", "living"),
    ("Séjour/repas", "living"), ("Korridor", "entry"), ("Réduit", "storage"), ("Cage d'escalier", "stairs"),
    ("Balkon", "balcony"),
])
def test_room_purpose(token, purpose):
    assert room_purpose(token) == purpose


def _plan(monkeypatch, tokens):
    """200x100 plan: outer wall, a middle wall at x=100 with a doorway (opening class) at y 40-60."""
    cls = np.zeros((100, 200), np.uint8)
    cls[:6, :] = cls[-6:, :] = cls[:, :6] = cls[:, -6:] = 1
    cls[:, 97:103] = 1
    cls[40:60, 97:103] = 2
    monkeypatch.setattr(layers, "wall_classes", lambda rel, path: (cls, "wallseg"))
    monkeypatch.setattr(layers, "text", lambda rel, path: Layer(image=rel, width=200, height=100, layer="text",
                                                                 model="test", tokens=tokens))
    monkeypatch.setattr(rooms, "_net_maps", lambda rel, shape: (None, None))
    return rooms.geometry_from_image("uploads/x.png", None)


def test_rooms_strong_and_unknown(monkeypatch):
    geo = _plan(monkeypatch, [TextToken(text="MH", bbox=(40, 40, 60, 50), score=0.95),
                              TextToken(text="H2388", bbox=(140, 20, 170, 30), score=0.95)])
    assert len(geo.rooms) == 2
    left, right = sorted(geo.rooms, key=lambda r: r.bbox[0])
    assert (left.tier, left.type, left.name) == ("strong", "bedroom", "MH")
    assert (right.tier, right.type) == ("unknown", "unknown")  # a ceiling height is not a room label
    assert left.neighbours == [right.id] and right.neighbours == [left.id]
    assert len(geo.openings) == 1 and geo.openings[0].rooms == sorted([left.id, right.id])
    assert geo.openings[0].id in left.connections


def test_wall_segments_only_on_the_building(monkeypatch):
    """A sheet frame and a scale bar around the plan are line work, not walls: no segments there."""
    cls = np.zeros((300, 400), np.uint8)
    cls[2:5, 2:398] = cls[295:298, 2:398] = cls[2:298, 2:5] = cls[2:298, 395:398] = 1  # sheet frame
    cls[270:276, 30:150] = 1  # scale bar
    cls[100:106, 100:300] = cls[194:200, 100:300] = cls[100:200, 100:106] = cls[100:200, 294:300] = 1  # building
    cls[100:200, 197:203] = 1
    monkeypatch.setattr(layers, "wall_classes", lambda rel, path: (cls, "wallseg"))
    monkeypatch.setattr(layers, "text", lambda rel, path: Layer(image=rel, width=400, height=300, layer="text",
                                                                 model="test", tokens=[]))
    monkeypatch.setattr(rooms, "_net_maps", lambda rel, shape: (None, None))
    geo = rooms.geometry_from_image("uploads/x.png", None)
    assert len(geo.rooms) == 2 and geo.wall_segments
    for sg in geo.wall_segments:
        assert all(90 <= x <= 310 and 90 <= y <= 210 for x, y in (sg.p0, sg.p1)), sg
    assert [sg.id for sg in geo.wall_segments] == [f"seg-{j + 1}" for j in range(len(geo.wall_segments))]


def test_pdf_geometry_uses_the_drawing_labels(monkeypatch, tmp_path):
    """Uploaded PDF without a DXF: rooms and walls from the rendered page (k = 2 px per point), typed by the PDF's
    own room labels, which replace the OCR reading of the same text; everything returned in PDF points."""
    from PIL import Image

    from app.models import PlanItem

    png = tmp_path / "page.png"
    Image.new("RGB", (200, 100), "white").save(png)
    graph = PlanGraph(image="uploads/p.pdf", width=100, height=50, source="pdf", items=[
        PlanItem(id="room-1.01", category="room", bbox=(20, 20, 30, 25), text="Küche 1.01 12.5 m²"),
        PlanItem(id="room-1.02", category="room", bbox=(70, 20, 80, 25), text="Zimmer 1.02")])
    cls = np.zeros((100, 200), np.uint8)
    cls[:6, :] = cls[-6:, :] = cls[:, :6] = cls[:, -6:] = 1
    cls[:, 97:103] = 1
    cls[40:60, 97:103] = 2
    monkeypatch.setattr(layers, "wall_classes", lambda rel, path: (cls, "wallseg"))
    monkeypatch.setattr(layers, "text", lambda rel, path: Layer(image=rel, width=200, height=100, layer="text",
                                                                 model="test", tokens=[
        TextToken(text="Kuche", bbox=(42, 41, 58, 49), score=0.9),  # OCR of the drawing's own label: dropped
        TextToken(text="WC", bbox=(150, 70, 160, 80), score=0.9)]))  # text the PDF parser has no room for: kept
    monkeypatch.setattr(rooms, "_net_maps", lambda rel, shape: (None, None))
    geo = rooms.geometry_from_pdf(graph, "uploads/p.pdf.png", png, 2.0)
    assert (geo.image, geo.width, geo.height, geo.model) == ("uploads/p.pdf", 100, 50, "wallseg")
    left, right = sorted(geo.rooms, key=lambda r: r.bbox[0])
    assert (left.type, left.tier, left.labeled_area_m2) == ("kitchen", "strong", 12.5)
    assert [lb.text for lb in left.labels] == ["Küche 1.01 12.5 m²"] and left.labels[0].bbox == (20, 20, 30, 25)
    assert left.evidence[0].source == "drawing"
    assert right.labeled_area_m2 is None and "Zimmer 1.02" in [lb.text for lb in right.labels]
    assert all(0 <= x <= 100 and 0 <= y <= 50 for rm in geo.rooms for x, y in rm.polygon)
    assert 1900 < left.area_px < 2100  # ~91 x 88 px inside the walls = ~8000 px² = ~2000 pt²
    assert all(0 <= v <= 100 for sg in geo.wall_segments for v in (*sg.p0, *sg.p1))


def test_pdf_geometry_keeps_dxf_outlines(tmp_path):
    from app.models import PlanItem

    graph = PlanGraph(image="p.pdf", width=100, height=50, source="pdf", items=[
        PlanItem(id="room-1.01", category="room", bbox=(0, 0, 10, 10), text="Bad 1.01 4.0 m²",
                 polygon=[(0, 0), (10, 0), (10, 10), (0, 10)])])
    geo = rooms.geometry_from_pdf(graph, "unused.png", tmp_path / "unused.png", 2.0)  # no render needed
    assert geo.model == "pdf" and geo.rooms[0].type == "bathroom" and geo.rooms[0].polygon[1] == (10, 0)


def test_low_confidence_label_gives_no_type(monkeypatch):
    geo = _plan(monkeypatch, [TextToken(text="MH", bbox=(40, 40, 60, 50), score=0.4)])
    assert all(r.tier == "unknown" for r in geo.rooms)


def test_room_tools(monkeypatch):
    from app.answering.tools import PlanTools
    from app.main import with_metrics

    geo = _plan(monkeypatch, [TextToken(text="OH", bbox=(40, 40, 60, 50), score=0.9)])
    geo = with_metrics(geo, PlanScaleCalibration(points=((0, 0), (100, 0)), distance=5, unit="m"))  # 0.05 m / px
    assert geo.scale.source == "calibration" and geo.wall_segments and geo.wall_segments[0].length_m
    graph = PlanGraph(image="uploads/x.png", width=200, height=100, source="model")
    tools = PlanTools(graph, geo)
    assert [d["name"] for d in tools.declarations()][-3:] == ["list_rooms", "room_details", "list_walls"]
    out, trace = tools.list_rooms()
    assert out["strong_rooms"] == 1 and out["unknown_spaces"] == 1
    (row,) = out["rooms"]
    assert row["type"] == "living" and row["tier"] == "strong" and 'label "OH"' in row["because"]
    assert row["area_m2"] in trace.numbers
    assert row["id"] in tools.by_id and tools.by_id[row["id"]].polygon  # citable with its outline
    details, dtrace = tools.room_details(row["id"])
    assert details["openings"][0]["leads_to"] != [] and details["outline_m"]
    assert details["walls"] and all(w["length_m"] in dtrace.numbers for w in details["walls"])
    assert any(w["exterior"] for w in details["walls"])
    walls, wtrace = tools.list_walls(exterior_only=True)
    assert walls["count"] > 0 and all(w["exterior"] for w in walls["walls"])
    assert all(i in tools.by_id and tools.by_id[i].category == "wall" for i in wtrace.item_ids)  # citable
    # plans without geometry keep the original tool set (benchmark path unchanged)
    assert "list_rooms" not in [d["name"] for d in PlanTools(graph).declarations()]


def _lab(text, purpose=None, score=0.95):
    from app.models import RoomLabel
    return RoomLabel(text=text, bbox=(0, 0, 1, 1), score=score, room_type=purpose)


def test_written_area_without_unit():
    assert rooms._labeled_area([_lab("MH", "bedroom"), _lab("11.5")]) == 11.5
    assert rooms._labeled_area([_lab("OH", "living"), _lab("26,5"), _lab("KEITTIÖ", "kitchen"), _lab("17.0")]) == 43.5
    assert rooms._labeled_area([_lab("12,5 m2")]) == 12.5
    assert rooms._labeled_area([_lab("11.5")]) is None  # no purpose label: a dimension, not an area
    assert rooms._labeled_area([_lab("MH", "bedroom"), _lab("20")]) is None  # integers are usually misread "2.0"
    assert rooms._labeled_area([_lab("MH", "bedroom"), _lab("11.5", score=0.6)]) is None  # unsure reading
