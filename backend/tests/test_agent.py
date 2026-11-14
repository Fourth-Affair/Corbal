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


def _real_loop_with_plain_contents(question, tools, history=None, scale=None):
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


def test_answer_returns_none_when_gemini_unavailable(fake, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("503 overloaded")

    monkeypatch.setattr(agent, "run_loop", broken)  # API / network error: the caller may fall back
    assert agent.answer("How many doors?", GRAPH) is None
    monkeypatch.delenv("GEMINI_API_KEY")
    assert agent.answer("How many doors?", GRAPH) is None


class RecordingModels(FakeModels):
    """FakeModels for the real run_loop: records each call's allowed function names; a turn [] = no function call."""

    def __init__(self, turns):
        super().__init__(turns)
        self.allowed = []

    def generate_content(self, model, contents, config):
        self.allowed.append(config.tool_config.function_calling_config.allowed_function_names)
        return super().generate_content(model, contents, config)


@pytest.fixture
def real(monkeypatch, tmp_path):
    """The real run_loop (with google.genai types) against scripted responses."""
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(agent, "CACHE_DIR", tmp_path)

    def install(turns):
        models = RecordingModels(turns)
        monkeypatch.setattr(agent, "_get_client", lambda: SimpleNamespace(models=models))
        return models

    return install


def test_step_limit_forces_final_answer(real):
    turns = [[("count_template", {"template": "doors"})]] + [[("plan_summary", {})]] * (agent.MAX_STEPS - 2)
    models = real(turns + [[("final_answer", _final(value=2, explanation="2 doors."))]])
    a = agent.answer("How many doors?", GRAPH)
    assert (a.answer, a.template) == (2, "doors")
    assert models.allowed[:-1] == [None] * (agent.MAX_STEPS - 1) and models.allowed[-1] == ["final_answer"]


def test_agent_that_never_finishes_answers_not_found(real):
    real([[("plan_summary", {})]] * agent.MAX_STEPS)
    a = agent.answer("How many doors?", GRAPH)
    assert a is not None and a.answer is None and a.confidence == "not_found" and agent.gave_up(a)


def test_no_function_call_is_reprompted_for_a_final_answer(real):
    models = real([[], [("final_answer", _final(not_found=True, explanation="Not on this plan."))]])
    a = agent.answer("What colour is the roof?", GRAPH)
    assert a.confidence == "not_found" and not agent.gave_up(a)
    assert models.allowed == [None, ["final_answer"]]
    real([[], []])  # still no call after the reminder: give up, do not fall back
    assert agent.gave_up(agent.answer("What colour is the facade?", GRAPH))


def test_gave_up_is_not_replaced_by_the_fallback(real, monkeypatch):
    import app.main as m

    real([[("plan_summary", {})]] * agent.MAX_STEPS)
    monkeypatch.setattr(m, "ANSWERER", "llm")

    class Tools(PlanTools):
        def fallback(self, question):
            raise AssertionError("fallback must not run when the model gave up")

    assert agent.gave_up(m._agent_or_fallback("How many doors?", Tools(GRAPH)))


@pytest.mark.parametrize("value,number,ok", [
    (45.237, 45.237, True), (45.2, 45.237, True), (45.24, 45.237, True), (45.3, 45.237, False),
    (45, 45.237, False), (3, 2.9, False), (2, 2, True), (2.0, 2, True), (True, 1, False),
    (90.65, 90.645, True), (90.64, 90.645, True), (90.66, 90.645, False),  # half-up and half-even both accepted
])
def test_number_matches(value, number, ok):
    from app.answering.tools import number_matches

    assert number_matches(value, number) == ok


def test_numbers_written_with_trailing_zero():
    from app.answering.tools import backed, written_decimals

    assert written_decimals("87.0") == 1 and written_decimals("74,51") == 2 and written_decimals("12") == 0
    assert backed(87.0, [87.012], written_decimals("87.0")) and not backed(87.0, [87.012])


def test_explanation_states_value_in_local_formats():
    assert agent.explanation_problem("Största rummet är 74,51 m².", 74.509, 1) is None
    assert agent.explanation_problem("Total 74.509 m².", 74.509, 1) is None
    assert agent.explanation_problem("Total 75 m².", 74.509, 1) is not None
    assert agent.explanation_problem("Huoneita on 12.", 2, 1) is not None


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


class FlakyModels:
    """Raises the scripted errors first, then returns a response."""

    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def _flaky(monkeypatch, errors, retries=1):
    models = FlakyModels(errors)
    monkeypatch.setattr(agent, "_get_client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(agent, "GEMINI_RETRIES", retries)
    monkeypatch.setattr(agent, "GEMINI_RETRY_BACKOFF_S", 0)
    return models


def test_generate_retries_transient_errors(monkeypatch):
    import httpx
    from google.genai import errors

    models = _flaky(monkeypatch, [httpx.ReadTimeout("The read operation timed out")])
    assert agent._generate([], None) == "ok" and models.calls == 2
    models = _flaky(monkeypatch, [errors.ServerError(503, {"error": {"message": "overloaded"}})])
    assert agent._generate([], None) == "ok" and models.calls == 2
    models = _flaky(monkeypatch, [errors.ClientError(429, {"error": {"message": "quota"}})])
    assert agent._generate([], None) == "ok" and models.calls == 2


def test_generate_does_not_retry_bad_requests_and_gives_up(monkeypatch):
    import httpx
    from google.genai import errors

    models = _flaky(monkeypatch, [errors.ClientError(400, {"error": {"message": "bad"}})])
    with pytest.raises(errors.ClientError):
        agent._generate([], None)
    assert models.calls == 1
    models = _flaky(monkeypatch, [httpx.ReadTimeout("t"), httpx.ReadTimeout("t")])
    with pytest.raises(httpx.ReadTimeout):
        agent._generate([], None)
    assert models.calls == 2


def test_hourly_cap_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(agent, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(agent, "GEMINI_MAX_PER_HOUR", 1)
    monkeypatch.setattr(agent, "_budget", agent.deque())
    calls = []
    monkeypatch.setattr(agent, "run_loop", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(RuntimeError("x")))
    assert agent.answer("how many sinks?", GRAPH) is None and len(calls) == 1
    assert agent.answer("how many doors?", GRAPH) is None and len(calls) == 1  # cap reached: Gemini not called


def test_calculate_only_combines_tool_numbers(tools):
    assert "error" in tools.call("calculate", {"operation": "add", "operands": [2, 2]})[0]  # nothing returned yet
    _, sinks = tools.call("count_template", {"template": "sinks"})
    _, doors = tools.call("count_template", {"template": "doors"})
    out, trace = tools.call("calculate", {"operation": "add", "operands": [2, 2]})
    assert out["result"] == 4 and trace.numbers == [4] and trace.derived_from == [sinks]  # both 2s are the sinks count
    assert "error" in tools.call("calculate", {"operation": "add", "operands": [2, 7]})[0]  # 7 is the model's own
    assert "error" in tools.call("calculate", {"operation": "times", "operands": [2, 2]})[0]
    assert "error" in tools.call("calculate", {"operation": "add", "operands": [2]})[0]
    out, _ = tools.call("calculate", {"operation": "subtract", "operands": [4, 2]})  # a result can be reused
    assert out["result"] == 2


def test_calculated_answer_is_validated_and_cited(tools):
    _, sinks = tools.call("count_template", {"template": "sinks"})
    _, doors = tools.call("count_template", {"template": "doors"})
    _, total = tools.call("calculate", {"operation": "add", "operands": [2, 2]})
    ans = agent.validate("How many sinks and doors are there in total?", _final(value=4, explanation="4 in total."),
                   [sinks, doors, total], tools)
    assert ans.answer == 4 and ans.template is None and ans.confidence != "not_found"
    assert {c.id for c in ans.citations} == {"sink-0", "sink-1"}  # the items behind the operands
    # a sum the tool never produced is still rejected
    assert agent.validate("total?", _final(value=5, explanation="5."), [sinks, doors, total], tools).answer is None


def test_items_in_follows_the_room_outline():
    """L-shaped kitchen: its bounding box covers the notch where the WC is; the outline does not."""
    ell = [(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)]
    graph = PlanGraph(image="x.png", width=200, height=200, source="svg", items=[
        PlanItem(id="kitchen", category="room", cls="kitchen", bbox=(0, 0, 100, 100), polygon=ell),
        _item("sink-kitchen", "fixture", (10, 10, 20, 20), "sink"),
        _item("sink-wc", "fixture", (60, 60, 70, 70), "sink"),  # in the notch
        _item("sink-edge", "fixture", (34, 80, 44, 90), "sink"),  # against the wall: centre inside
        _item("window-top", "window", (30, -4, 70, 0)),  # in the wall above the kitchen
        _item("window-notch", "window", (60, 100, 90, 104)),  # in the wall below the notch
        _item("door-side", "door", (-12, 40, 0, 52)),
    ])
    t = PlanTools(graph)
    assert {r["id"] for r in t.items_in("kitchen", "fixture", "sink")[0]["items"]} == {"sink-kitchen", "sink-edge"}
    assert [r["id"] for r in t.items_in("kitchen", "window")[0]["items"]] == ["window-top"]
    assert t.items_in("kitchen", "door")[0]["count"] == 1
    graph.items[0].polygon = None  # no outline: the old bounding-box rule
    assert PlanTools(graph).items_in("kitchen", "fixture", "sink")[0]["count"] == 3
