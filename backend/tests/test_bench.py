from app.bench import score
from app.bench_perception import compare, match_boxes, match_rooms, summary
from app.models import PlanGraph, PlanItem


def test_score_rules():
    assert score({"answer": 3, "unit": "count"}, 3, 0.02)
    assert not score({"answer": 3, "unit": "count"}, 4, 0.02)
    assert score({"answer": 100.0, "unit": "m2"}, 101.9, 0.02)  # areas: within 2%
    assert not score({"answer": 100.0, "unit": "m2"}, 103.0, 0.02)
    assert not score({"answer": 3, "unit": "count"}, None, 0.02)  # not_found is a miss...
    assert score({"answer": None, "unit": "count"}, None, 0.02)  # ...unless nothing can be answered
    assert score({"answer": "x" * 50}, "anything", 0.02) is None  # long free text is not scored


def test_match_boxes_is_one_to_one():
    truth = [(0, 0, 10, 10), (100, 100, 110, 110)]
    pred = [(1, 1, 11, 11), (0, 0, 10, 10), (300, 300, 310, 310)]  # two on the first label, one stray
    m = match_boxes(pred, truth, 0.3)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 2, 1)
    assert m["iou"] == 1.0  # the exact box wins the match
    assert match_boxes([], truth, 0.3) == {"tp": 0, "fp": 0, "fn": 2, "iou": 0.0}


def _item(i, category, cls, bbox, polygon=None):
    return PlanItem(id=f"{category}-{i}", category=category, cls=cls, bbox=bbox, polygon=polygon)


def test_match_rooms_by_centre():
    # L-shaped room: its bounding box covers (60, 60), the polygon does not
    room = _item(0, "room", "bedroom", (0, 0, 100, 100), [(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)])
    inside = [_item(1, "room", "bedroom", (10, 10, 30, 30)), _item(2, "room", "bedroom", (5, 50, 35, 90))]
    outside = _item(3, "room", "bedroom", (50, 50, 70, 70))
    assert match_rooms(inside + [outside], [room]) == {"tp": 1, "fp": 1, "fn": 0, "pred": 3}
    assert match_rooms([outside], [room])["fn"] == 1


def test_compare_and_summary():
    labels = PlanGraph(image="x.png", width=200, height=200, source="svg", items=[
        _item(0, "door", "", (0, 0, 10, 10)), _item(1, "door", "", (50, 0, 60, 10)),
        _item(2, "fixture", "toilet", (100, 100, 110, 110)),
        _item(3, "door", "", (-50, -50, -10, -10)),  # outside the image: counted by the questions, not findable
    ])
    model = PlanGraph(image="x.png", width=200, height=200, source="model", counts={"doors": 3, "wc_seats": 2},
                      items=[_item(0, "door", "", (0, 0, 10, 10)), _item(1, "fixture", "toilet", (100, 100, 110, 111))])
    rec = compare(model, labels)
    assert rec["symbols"]["door"] == {"tp": 1, "fp": 0, "fn": 1, "iou": 1.0}
    assert rec["symbols"]["toilet"]["tp"] == 1
    assert rec["counts"] == {"doors": [3, 3], "wc_seats": [2, 1]}
    s = summary([rec])
    assert s["symbols"]["door"]["recall"] == 0.5 and s["symbols"]["door"]["precision"] == 1.0
    assert s["counts"] == {"doors": (1, 1), "wc_seats": (0, 1)}
