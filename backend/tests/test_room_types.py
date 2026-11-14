"""Room type from combined evidence (hand-set weights): OCR labels strong, decisive fixtures strong, 2019 net weak."""

import numpy as np

from app.cubicasa.room_types import Fixture, classify
from app.cubicasa.rooms import scale_from_written_areas
from app.cubicasa.wallvec import vectorise
from app.models import RoomLabel, RoomShape


def lb(text, purpose, score=0.95):
    return RoomLabel(text=text, bbox=(0, 0, 1, 1), score=score, room_type=purpose)


def test_toilet_alone_is_a_strong_wc():
    c = classify([], [Fixture("toilet", "det-toilet-1")], {}, 0.01)
    assert (c.type, c.tier) == ("wc", "strong")
    assert any(e.item_id == "det-toilet-1" for e in c.evidence)


def test_toilet_with_shower_is_a_strong_bathroom():
    c = classify([], [Fixture("toilet"), Fixture("shower")], {}, 0.02)
    assert (c.type, c.tier) == ("bathroom", "strong")


def test_label_is_strong():
    c = classify([lb("K", "kitchen")], [], {}, 0.05)
    assert (c.type, c.tier, c.purposes) == ("kitchen", "strong", ["kitchen"])


def test_net_pixels_alone_are_weak():
    c = classify([], [], {"bedroom": 0.8}, 0.05)
    assert (c.type, c.tier) == ("bedroom", "weak")
    assert classify([], [], {"bedroom": 0.3}, 0.05).tier == "unknown"  # too little to say anything


def test_label_beats_net_and_conflicting_fixture_drops_tier():
    assert classify([lb("MH", "bedroom")], [], {"kitchen": 1.0}, 0.05).type == "bedroom"
    c = classify([lb("K", "kitchen")], [Fixture("toilet")], {}, 0.05)
    assert (c.type, c.tier, c.conflicts) == ("kitchen", "weak", ["wc"])


def test_toilet_in_labelled_bathroom_is_no_conflict():
    c = classify([lb("KPH", "bathroom")], [Fixture("toilet"), Fixture("sink")], {"bath": 1.0}, 0.02)
    assert (c.type, c.tier, c.conflicts) == ("bathroom", "strong", [])
    c = classify([lb("MH", "bedroom")], [Fixture("shower")], {}, 0.05)  # a shower gives a bedroom nothing
    assert (c.tier, c.conflicts) == ("weak", ["bathroom"])


def test_open_plan_keeps_both_purposes():
    c = classify([lb("OH", "living"), lb("K", "kitchen")], [Fixture("appliance")], {}, 0.1)
    assert c.tier == "strong" and set(c.purposes) == {"living", "kitchen"}


def test_unclear_label_and_sink_are_weak():
    assert classify([lb("MH", "bedroom", score=0.6)], [], {}, 0.05).tier == "weak"
    assert classify([], [Fixture("sink")], {}, 0.05).tier == "weak"


def test_no_evidence_is_unknown():
    c = classify([lb("H2388", None)], [], {}, 0.05)
    assert (c.type, c.tier) == ("unknown", "unknown")


def test_scale_from_two_agreeing_written_areas():
    def room(i, area_px, m2):
        return RoomShape(id=f"space-{i}", type="bedroom", tier="strong", polygon=[(0, 0), (1, 0), (1, 1)],
                         bbox=(0, 0, 1, 1), area_px=area_px, labeled_area_m2=m2)
    # 0.05 m/px -> 1 px = 0.0025 m²
    s = scale_from_written_areas([room(1, 4000, 10.0), room(2, 6000, 15.1), room(3, 5000, 40.0)])
    assert s is not None and s.source == "written_areas" and abs(s.m_per_px - 0.05) / 0.05 < 0.01
    assert set(s.rooms) == {"space-1", "space-2"}  # the misread 40 m² is left out
    assert scale_from_written_areas([room(1, 4000, 10.0)]) is None  # one room alone is not trusted


def test_wall_segments_of_two_rooms():
    """200x100 plan, 6 px walls: outer rectangle + a middle wall at x=100."""
    cls = np.zeros((100, 200), np.uint8)
    cls[:6, :] = cls[-6:, :] = cls[:, :6] = cls[:, -6:] = 1
    cls[:, 97:103] = 1
    lab = np.zeros_like(cls, dtype=np.int32)
    lab[6:-6, 6:97] = 1
    lab[6:-6, 103:-6] = 2
    ids = {1: "space-1", 2: "space-2"}
    segs = vectorise(cls, lab, lambda v: ids.get(v), 6.0)
    middle = [s for s in segs if s.orientation == "vertical" and 90 < s.p0[0] < 110]
    assert middle and {*middle[0].rooms_left, *middle[0].rooms_right} == {"space-1", "space-2"}
    assert not middle[0].exterior and abs(middle[0].thickness_px - 6) <= 2
    outer = [s for s in segs if s.exterior]
    assert outer and all("outside" in s.rooms_left + s.rooms_right for s in outer)
    top = [s for s in segs if s.orientation == "horizontal" and s.p0[1] < 10]
    assert top and sum(s.length_px for s in top) > 150  # the top wall spans most of the 200 px width


def test_symbol_scale_band_and_size_checks():
    from app.cubicasa.size_priors import check_sizes, symbol_scale
    from app.models import PlanGeometry, PlanGraph, PlanItem

    def item(i, cat, cls, side):
        return PlanItem(id=f"{cat}-{i}", category=cat, cls=cls, bbox=(0, 0, side, side * 0.9), score=0.9)

    # doors 40 px (0.7-1.0 m) and toilets 28 px (0.5-0.8 m) -> about 2.1 cm per pixel
    graph = PlanGraph(image="x.png", width=500, height=500, source="model",
                      items=[item(i, "door", "", 40) for i in range(4)] + [item(9, "fixture", "toilet", 28),
                                                                          item(8, "window", "", 120)])
    band = symbol_scale(graph)
    assert band is not None and band.source == "symbols" and band.low < 0.021 < band.high
    assert len(band.items) == 5  # windows are not used
    assert symbol_scale(PlanGraph(image="x", width=1, height=1, source="model", items=graph.items[:2])) is None

    def room(i, rtype, tier, area_px):
        return RoomShape(id=f"space-{i}", type=rtype, tier=tier, polygon=[(0, 0), (1, 0), (1, 1)], bbox=(0, 0, 1, 1),
                         area_px=area_px)
    living_ok, living_tiny, speck = room(1, "living", "strong", 60000), room(2, "living", "strong", 5000), \
        room(3, "unknown", "unknown", 300)
    geo = PlanGeometry(image="x", width=500, height=500, model="t", rooms=[living_ok, living_tiny, speck],
                       symbol_scale=band)
    check_sizes(geo)
    lo, hi = living_ok.area_m2_range
    assert lo < 0.021 ** 2 * 60000 < hi and living_ok.size_check is None and living_ok.tier == "strong"
    assert living_tiny.tier == "weak" and "implausibly small" in living_tiny.size_check  # ~2 m² living room
    assert any(e.source == "geometry" and e.weight < 0 for e in living_tiny.evidence)
    assert "door swing" in speck.size_check and speck.tier == "unknown"


def test_written_scale_disagreeing_with_symbols_gets_a_warning():
    from app.cubicasa.size_priors import check_sizes
    from app.models import PlanGeometry, PlanScale

    geo = PlanGeometry(image="x", width=1, height=1, model="t",
                       scale=PlanScale(m_per_px=0.2, source="written_areas", rooms=["a", "b"]),
                       symbol_scale=PlanScale(m_per_px=0.02, source="symbols", low=0.017, high=0.025))
    check_sizes(geo)
    assert geo.scale.warning and "disagrees" in geo.scale.warning
