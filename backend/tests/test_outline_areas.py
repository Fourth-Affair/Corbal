"""outline_areas: floor / room areas for the agent, computed from outlines x the drawing scale."""

from app.answering.tools import PlanTools
from app.models import PlanGeometry, PlanGraph, PlanOutline, PlanScale

SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def _tools(outlines=None, scale=True):
    graph = PlanGraph(image="x.png", width=200, height=200, source="model",
                      floor_outlines=[PlanOutline(id="floor-1", polygon=SQUARE, scope="floor")])
    geo = PlanGeometry(image="x.png", width=200, height=200, model="t", rooms=[],
                       scale=PlanScale(m_per_px=0.05, source="written_areas", rooms=["a", "b"]) if scale else None)
    return PlanTools(graph, geo, outlines)


def test_suggested_and_accepted_outlines():
    out, trace = _tools().call("outline_areas", {"scope": "floor"})
    (row,) = out["outlines"]
    assert row["area_m2"] == 25.0 and row["status"].startswith("suggested")  # 100 x 100 px at 5 cm
    assert 25.0 in trace.numbers and trace.item_ids == ["floor-1"]
    accepted = PlanOutline(id="floor-1", polygon=[(0.0, 0.0), (50.0, 0.0), (50.0, 100.0), (0.0, 100.0)],
                           scope="floor", decision="accepted")
    tools = _tools([accepted])
    out, _ = tools.call("outline_areas", {"scope": "floor"})
    (row,) = out["outlines"]
    assert row["area_m2"] == 12.5 and row["status"].startswith("accepted")  # the user's edit replaces the suggestion
    assert tools.area_scope == "floor"


def test_a_missing_room_is_not_the_floor():
    tools = _tools()
    out, trace = tools.call("outline_areas", {"scope": "room", "room_type": "study"})
    assert out["outlines"] == [] and not trace.numbers
    assert tools.area_scope == "study"


def test_no_scale_no_numbers():
    out, trace = _tools(scale=False).call("outline_areas", {})
    assert "scale" in out["note"] and not trace.numbers
