"""Confidence of counting-model answers: classifier probability, detections near a rule's cut-off, else the old
evidence-match rule."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from app.cubicasa.answer import count
from app.cubicasa.detect import count_borderline, count_probability
from app.cubicasa.fit import RuleModel
from app.models import PlanGraph, PlanItem


def _graph(**kw) -> PlanGraph:
    items = [PlanItem(id="d1", category="door", bbox=(0, 0, 10, 10), score=0.9),
             PlanItem(id="k1", category="room", cls="kitchen", bbox=(0, 0, 50, 50), score=0.5)]
    return PlanGraph(image="x.png", width=100, height=100, source="model", items=items, **kw)


def test_probability_sets_the_tier_even_when_evidence_does_not_match():
    # two kitchens counted, one region highlighted: used to be "low" whatever the model's certainty
    g = _graph(counts={"kitchens": 2}, count_probability={"kitchens": 0.97})
    a = count(g, "kitchens")
    assert a.confidence == "high" and "97%" in a.explanation
    assert count(_graph(counts={"kitchens": 2}, count_probability={"kitchens": 0.85}), "kitchens").confidence == "medium"
    assert count(_graph(counts={"kitchens": 1}, count_probability={"kitchens": 0.5}), "kitchens").confidence == "low"


def test_borderline_detections_cap_threshold_rules():
    assert count(_graph(counts={"doors": 1}, count_borderline={"doors": 0}), "doors").confidence == "medium"
    a = count(_graph(counts={"doors": 1}, count_borderline={"doors": 2}), "doors")
    assert a.confidence == "low" and "2 detections close to the counting threshold" in a.explanation


def test_without_signals_the_old_rule_applies():
    assert count(_graph(counts={"doors": 1}), "doors").confidence == "medium"  # evidence matches, score 0.9
    assert count(_graph(counts={"doors": 3}), "doors").confidence == "low"


def test_count_probability_of_a_classifier():
    X = np.array([[0.0], [0.1], [1.0], [1.1]] * 10)
    y = np.array([0, 0, 1, 1] * 10)
    m = HistGradientBoostingClassifier(max_iter=20, min_samples_leaf=2).fit(X, y)
    p = count_probability({"kind": "hgb_c", "model": m}, np.array([[1.05]]), 1)
    assert p is not None and p > 0.9
    assert count_probability({"kind": "hgb_c", "model": m}, np.array([[1.05]]), 5) is None  # not a class
    assert count_probability({"kind": "hgb_p", "model": None}, np.array([[1.0]]), 1) is None


def test_count_borderline_counts_detections_near_the_cut_off():
    rule = RuleModel()
    rule.feature, rule.j, rule.a, rule.b = "det_window_c40", 0, 1.0, 0.0
    dets = [{"cls": "window", "conf": c} for c in (0.95, 0.30, 0.45, 0.20)] + [{"cls": "door", "conf": 0.4}]
    assert count_borderline({"kind": "rule", "model": rule}, dets) == 2  # 0.30 and 0.45; 0.20 and the door are out
    assert count_borderline({"kind": "rule", "model": rule}, None) is None
    rule.feature = "fl_n0.06"  # not a detection-count rule
    assert count_borderline({"kind": "rule", "model": rule}, dets) is None


def test_corrected_counts_lose_the_model_certainty():
    from app.main import _apply_corrections

    g = _graph(counts={"doors": 1, "kitchens": 1}, count_probability={"kitchens": 0.99},
               count_borderline={"doors": 0})
    out = _apply_corrections(g, [{"item_id": "d1", "corrected_item": None}])  # the user deletes the door
    assert out.counts["doors"] == 0 and "doors" not in out.count_borderline
    assert out.count_probability == {"kitchens": 0.99}  # untouched template keeps it
