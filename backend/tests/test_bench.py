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


def test_yes_no_scored_on_first_word():
    assert score({"answer": "yes"}, "Kyllä, keittiössä on ikkuna.", 0.02)
    assert score({"answer": "yes"}, "On.", 0.02)
    assert score({"answer": "no"}, "Nej, det finns inget badkar.", 0.02)
    assert not score({"answer": "no"}, "Yes", 0.02)
    assert not score({"answer": "yes"}, "There are 2 windows.", 0.02)  # no polarity stated
    assert not score({"answer": "yes"}, None, 0.02)


def _open_plan():
    """Kitchen | bathroom share a wall; a bedroom far away. Door size 10 -> walls of ~2 px."""
    sq = lambda x0, y0, x1, y1: [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # noqa: E731
    room = lambda i, cls, raw, b: PlanItem(id=f"room-{i}", category="room", cls=cls, raw=raw, bbox=b, polygon=sq(*b))  # noqa: E731
    return PlanGraph(image="x.png", width=400, height=200, source="svg", items=[
        room(0, "kitchen", "Space Kitchen", (0, 0, 100, 100)),
        room(1, "bathroom", "Space Bath", (102, 0, 142, 40)),
        room(2, "bedroom", "Space Bedroom", (300, 0, 390, 90)),
        room(3, "other", "Space LivingRoom", (0, 102, 100, 200)),
        _item(4, "fixture", "sink", (10, 10, 20, 20)), _item(5, "fixture", "double_sink", (30, 10, 50, 20)),
        _item(6, "fixture", "toilet", (110, 10, 120, 20)),
        _item(7, "window", "", (20, -3, 60, 0)),  # along the kitchen's top wall
        _item(8, "window", "", (98, -3, 106, 0)),  # across the kitchen / bathroom corner: unclear for both
        _item(9, "window", "", (40, 100, 70, 102)),  # glazing between kitchen and living room
        _item(10, "door", "", (300, 90, 310, 100)), _item(11, "door", "", (142, 10, 152, 20)),
    ])


def test_open_question_truth():
    from app.answering import openq

    plan = openq.Plan(_open_plan())
    kitchen, bath, bed = plan.only("kitchen"), plan.only("bathroom"), plan.only("bedroom")
    assert plan.only("living") is not None  # read from the SVG class
    assert plan.n_in(kitchen, "sink") == 2 and plan.n_in(bath, "sink") == 0 and plan.n_in(bath, "toilet") == 1
    assert plan.n_in(kitchen, "window") is None  # a corner window and shared glazing: not asked
    assert plan.n_in(bed, "window") == 0
    assert plan.adjacent(kitchen, bath) is True and plan.adjacent(kitchen, bed) is False
    assert plan.larger(kitchen, bath) is True and plan.larger(bath, kitchen) is False
    assert plan.larger(kitchen, bed) is None  # 10000 vs 8100 px²: too close to call

    by = {(c.family, c.key): c.answer for c in openq.candidates(_open_plan())}
    assert by[("in_room_count", "kitchen/sink")] == 2 and by[("in_room_yesno", "bathroom/toilet")] == "yes"
    assert ("in_room_count", "kitchen/window") not in by
    assert by[("arithmetic", "wc_seats+sinks")] == 3 and by[("compare_yesno", "sinks>wc_seats")] == "yes"
    assert by[("adjacent_yesno", "kitchen~bathroom")] == "yes" and by[("plan_yesno", "sauna")] == "no"


def test_open_question_wordings_and_sampling():
    import random

    from app.answering import openq

    rng = random.Random(0)
    for c in openq.candidates(_open_plan()):
        for lang in openq.LANGS:
            for _ in range(4):
                q = openq.question(c, lang, rng)
                assert "{" not in q and q[0].isupper() and q[-1] in "?.", q
    graphs = {f"p{i}.png": _open_plan() for i in range(openq.MIN_PLANS)}
    items = openq.sample(graphs, 6, random.Random(0))
    assert {it["family"] for it in items} == set(openq.FAMILIES)
    assert {it["lang"] for it in items} == set(openq.LANGS)
    assert all(it["kind"] == "open" and (it["unit"] == "count") == isinstance(it["answer"], int) for it in items)
    assert items == openq.sample(graphs, 6, random.Random(0))  # deterministic
