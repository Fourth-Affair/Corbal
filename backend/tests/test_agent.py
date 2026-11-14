from types import SimpleNamespace

import pytest

from app.answering import agent
from app.answering.tools import PlanTools
from app.models import PlanGraph, PlanItem


def _item(id, category, bbox, cls="", raw=None, text=None, score=0.9):
    return PlanItem(id=id, category=category, cls=cls, bbox=bbox, raw=raw, text=text, score=score)


GRAPH = PlanGraph(image="data/cubicasa5k/x.png", width=1000, height=1000, source="model", items=[
    _item("room-kitchen-0", "room", (0, 0, 400, 400), "kitchen"),
    _item("room-bath-0", "room", (500, 0, 800, 300), "bathroom"),
    _item("label-1", "room", (600, 100, 650, 120), "bathroom", raw="ocr", text="KPH"),
    _item("sink-0", "fixture", (100, 100, 120, 120), "sink"),
    _item("sink-1", "fixture", (550, 50, 570, 70), "double_sink"),
    _item("toilet-0", "fixture", (700, 200, 720, 220), "toilet"),
    _item("door-0", "door", (390, 100, 410, 140)),
    _item("door-1", "door", (480, 100, 500, 140)),
], counts={"sinks": 2, "doors": 2, "kitchens": 1})


@pytest.fixture
def tools():
    return PlanTools(GRAPH)


def test_count_template_uses_fusion_count(tools):
    out, trace = tools.count_template("sinks")
    assert out["value"] == 2 and trace.numbers == [2] and set(trace.item_ids) == {"sink-0", "sink-1"}


def test_items_in_room_and_label(tools):
    out, _ = tools.items_in("room-kitchen-0", "fixture", "sink")
    assert out["count"] == 1 and out["items"][0]["id"] == "sink-0"
    out, trace = tools.items_in("label-1", "fixture")  # label -> segmented bathroom region around it
    assert out["container"]["id"] == "room-bath-0" and out["count"] == 2 and trace.numbers == [2]


def test_bad_tool_calls_return_errors(tools):
    assert "error" in tools.call("items_in", {"container_id": "nope", "category": "door"})[0]
    assert "error" in tools.call("count_template", {"template": "radiators"})[0]
    assert "error" in tools.call("list_items", {"bogus": 1})[0]
    assert "error" in tools.call("rm_rf", {})[0]


def _final(**kw):
    return {"not_found": False, "explanation": "", **kw}


def test_validate_template_count(tools):
    _, trace = tools.count_template("sinks")
    a = agent.validate("How many sinks?", _final(value=2, cited_ids=["sink-0", "sink-1"],
                                                 explanation="There are 2 sinks [1], [2]."), [trace], tools)
    assert (a.answer, a.template, a.answered_by) == (2, "sinks", "llm")
    assert [c.id for c in a.citations] == ["sink-0", "sink-1"]


def test_validate_rejects_unbacked_number(tools):
    _, trace = tools.count_template("sinks")
    a = agent.validate("How many sinks?", _final(value=3, explanation="3 sinks."), [trace], tools)
    assert a.answer is None and a.confidence == "not_found"


def test_validate_drops_uncited_ids_and_fixes_explanation(tools):
    _, trace = tools.count_template("doors")
    a = agent.validate("How many doors?", _final(value=2, cited_ids=["door-0", "toilet-0", "ghost"],
                                                 explanation="Two doors [3]."), [trace], tools)
    assert [c.id for c in a.citations] == ["door-0"]  # toilet-0 exists but no tool returned it
    assert "2" in a.explanation and "[3]" not in a.explanation


def test_validate_router_disagreement_lowers_confidence(tools):
    _, trace = tools.count_template("kitchens")
    a = agent.validate("How many sinks are drawn?", _final(value=1, explanation="1 kitchen [1].",
                                                           cited_ids=["room-kitchen-0"]), [trace], tools)
    assert a.answer == 1 and a.confidence == "low"


def test_validate_raw_items_are_low_confidence(tools):
    _, trace = tools.items_in("room-kitchen-0", "fixture", "sink")
    a = agent.validate("How many sinks are in the kitchen?", _final(value=1, cited_ids=["sink-0"],
                                                                    explanation="1 sink [1]."), [trace], tools)
    assert a.answer == 1 and a.confidence == "low" and a.template is None


def test_validate_not_found(tools):
    a = agent.validate("Area of the kitchen?", {"not_found": True, "explanation": "No scale."}, [], tools)
    assert a.answer is None and a.confidence == "not_found" and a.explanation == "No scale."


class FakeModels:
    """Scripted Gemini: each turn returns the next list of (name, args) function calls."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def generate_content(self, model, contents, config):
        self.seen.append(len(contents))
        calls = [SimpleNamespace(id=f"c{i}", name=n, args=a) for i, (n, a) in enumerate(self.turns.pop(0))]
        return SimpleNamespace(function_calls=calls, text=None, candidates=[SimpleNamespace(content=None)])


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(agent, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(agent, "_client", None)

    def install(turns):
        models = FakeModels(turns)
        monkeypatch.setattr(agent, "_get_client", lambda: SimpleNamespace(models=models))
        monkeypatch.setattr(agent, "run_loop", _real_loop_with_plain_contents)
        return models

    return install


def _real_loop_with_plain_contents(question, tools, history=None):
    """run_loop without google.genai types (the fake client does not need them)."""
    traces, contents = [], [question]
    for _ in range(agent.MAX_STEPS):
        r = agent._get_client().models.generate_content(model="m", contents=contents, config=None)
        for call in r.function_calls:
            if call.name == "final_answer":
                return call.args, traces
        for call in r.function_calls:
            traces.append(tools.call(call.name, call.args)[1])
        contents.append("turn")
    raise ValueError("no final answer")


def test_answer_end_to_end_and_cache(fake):
    models = fake([
        [("count_template", {"template": "doors"})],
        [("final_answer", _final(value=2, cited_ids=["door-0", "door-1"], explanation="Ovia on 2 [1], [2]."))],
    ])
    a = agent.answer("Kuinka monta ovea?", GRAPH)
    assert (a.answer, a.template, len(a.citations)) == (2, "doors", 2)
    assert agent.answer("Kuinka monta ovea?", GRAPH) == a and len(models.seen) == 2  # second call from cache


def test_answer_returns_none_on_failure(fake, monkeypatch):
    fake([[("plan_summary", {})]] * agent.MAX_STEPS)  # never finishes
    assert agent.answer("How many doors?", GRAPH) is None
    monkeypatch.delenv("GEMINI_API_KEY")
    assert agent.answer("How many doors?", GRAPH) is None


@pytest.mark.parametrize("text,ok", [
    ("2 doors [1], [2].", True), ("2 doors [1-2].", True), ("2 doors [1, 2].", True),
    ("2 doors [1]–[3].", False), ("2 doors [1-3].", False), ("2 doors [2, 3].", False), ("Two doors.", False),
])
def test_explanation_refs(text, ok):
    assert (agent.explanation_problem(text, 2, 2) is None) == ok


def test_list_rooms_merges_label_into_region(tools):
    out, trace = tools.list_items("room", "bathroom")
    assert out["count"] == 1 and out["items"][0]["id"] == "room-bath-0" and out["items"][0]["labels"] == ["KPH"]
    assert set(trace.item_ids) == {"room-bath-0", "label-1"} and trace.numbers == [1]


def test_yes_no_linked_to_count(tools):
    _, trace = tools.count_template("kitchens")
    a = agent.validate("Is there a kitchen?", _final(text="Yes", explanation="Yes, there is 1 kitchen [1]."),
                       [trace], tools)
    assert (a.answer, a.template, [c.id for c in a.citations]) == ("Yes", "kitchens", ["room-kitchen-0"])
    _, trace = tools.count_template("saunas")
    a = agent.validate("Onko asunnossa saunaa?", _final(text="Kyllä", explanation="Kyllä."), [trace], tools)
    assert a.answer is None and a.confidence == "not_found"  # saunas = 0 contradicts "yes"
    a = agent.validate("Onko asunnossa saunaa?", _final(text="Ei", explanation="Saunaa ei löytynyt."), [trace], tools)
    assert (a.answer, a.template) == ("Ei", "saunas")


@pytest.mark.parametrize("text,pol", [("Yes", True), ("No.", False), ("Kyllä, yksi", True), ("Ei ole", False),
                                      ("Nej", False), ("Ja", True), ("WC-istuin ja suihku", None)])
def test_yes_no(text, pol):
    assert agent.yes_no(text) is pol
