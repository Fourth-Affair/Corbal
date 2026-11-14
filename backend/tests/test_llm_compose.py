import pytest

from app.answering import llm
from app.models import Answer, PlanGraph, PlanItem

GRAPH = PlanGraph(image="data/cubicasa5k/x.png", width=100, height=100, source="model")
DOORS = [PlanItem(id=f"d{i}", category="door", bbox=(i, i, i + 5, i + 5), score=0.9) for i in range(3)]
BASE = Answer(answer=3, unit="count", template="doors", confidence="medium",
              explanation="Estimated 3 doors.", citations=DOORS)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)


def _reply(monkeypatch, text):
    calls = []

    def fake(prompt):
        calls.append(prompt)
        if isinstance(text, Exception):
            raise text
        return text

    monkeypatch.setattr(llm, "_generate", fake)
    return calls


def test_explanation_replaced_value_kept(monkeypatch):
    _reply(monkeypatch, "Pohjapiirustuksessa on 3 ovea [1], [2] ja [3].")
    out = llm.compose("Kuinka monta ovea pohjapiirustuksessa on?", BASE, GRAPH)
    assert out.explanation.startswith("Pohjapiirustuksessa on 3 ovea")
    assert out.answered_by == "llm"
    assert (out.answer, out.confidence, out.citations) == (BASE.answer, BASE.confidence, BASE.citations)


@pytest.mark.parametrize("text", [
    "There are 4 doors [1].",       # different number
    "There are 13 doors.",          # number only as part of another number
    "There are 3 doors [4].",       # citation that does not exist
    "   ",
])
def test_bad_explanation_falls_back(monkeypatch, text):
    _reply(monkeypatch, text)
    assert llm.compose("How many doors?", BASE, GRAPH) == BASE


def test_api_error_falls_back(monkeypatch):
    _reply(monkeypatch, RuntimeError("quota"))
    assert llm.compose("How many doors?", BASE, GRAPH) == BASE


def test_no_key_or_no_answer_skips_llm(monkeypatch):
    calls = _reply(monkeypatch, "3 doors.")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert llm.compose("How many doors?", BASE, GRAPH) == BASE
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    missing = Answer(answer=None, confidence="not_found", explanation="Parser unavailable")
    assert llm.compose("How many doors?", missing, GRAPH) == missing
    assert calls == []


def test_cached(monkeypatch):
    calls = _reply(monkeypatch, "Found 3 doors [1].")
    first = llm.compose("How many doors?", BASE, GRAPH)
    second = llm.compose("How many doors?", BASE, GRAPH)
    assert first == second and first.answered_by == "llm" and len(calls) == 1
