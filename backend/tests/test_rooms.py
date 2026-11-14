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
