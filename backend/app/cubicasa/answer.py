"""Template counters: PlanGraph + template -> count + the items counted (= citations)."""

from collections.abc import Callable

from app.models import Answer, PlanGraph, PlanItem

Pred = Callable[[PlanItem], bool]


def _cat(category: str) -> Pred:
    return lambda it: it.category == category


def _fixture(*classes: str) -> Pred:
    return lambda it: it.category == "fixture" and it.cls in classes


def _room(*classes: str) -> Pred:
    return lambda it: it.category == "room" and it.cls in classes


def _saunas(graph: PlanGraph) -> list[PlanItem]:
    """A sauna = a sauna stove, or a sauna room with no stove drawn inside it."""
    stoves = [it for it in graph.items if it.category == "fixture" and it.cls == "sauna_stove"]
    rooms = [it for it in graph.items if it.category == "room" and it.cls == "sauna"]

    def overlaps(a, b) -> bool:
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    return stoves + [r for r in rooms if not any(overlaps(r.bbox, s.bbox) for s in stoves)]


COUNTERS: dict[str, tuple[Pred, str]] = {
    "windows": (_cat("window"), "windows"),
    "doors": (_cat("door"), "doors"),
    "floors": (_cat("floor"), "floors"),
    "wc_seats": (_fixture("toilet"), "toilet bowls"),
    "sinks": (_fixture("sink", "double_sink"), "sinks/basins"),
    "showers_baths": (_fixture("shower", "bathtub"), "showers and bathtubs"),
    "bedrooms": (_room("bedroom"), "bedrooms"),
    "bathrooms": (_room("bathroom"), "bathrooms"),
    "wc_rooms": (_room("wc"), "WC rooms"),
    "kitchens": (_room("kitchen"), "kitchens"),
    "balconies": (_room("balcony"), "balconies"),
}


# Evidence shown for model-source answers (the count itself comes from graph.counts)
EVIDENCE: dict[str, Pred] = {
    **{t: p for t, (p, _) in COUNTERS.items()},
    "saunas": lambda it: (it.category == "fixture" and it.cls in ("sauna_bench", "sauna_stove"))
    or (it.category == "room" and it.cls == "sauna"),
    "showers_baths": _fixture("shower", "bathtub"),
}


# Classifier probability of the count -> confidence. Preview plans, refit out of fold: p >= 0.95 was right 96% of the
# time (65% of those answers), 0.8-0.95 74%, below 0.8 60%. The old tier ("medium") was right 91%.
HIGH_PROBABILITY = 0.95
MEDIUM_PROBABILITY = 0.80


def _model_confidence(graph: PlanGraph, template: str, n: int, ev: list[PlanItem]) -> tuple[str, str]:
    """(confidence, note) of a counting-model answer: the classifier's probability when it has one, the detections
    near the cut-off for threshold-count rules (never high: those counts are the least reliable), else whether the
    highlighted evidence matches the count."""
    p = graph.count_probability.get(template)
    if p is not None:
        tier = "high" if p >= HIGH_PROBABILITY else "medium" if p >= MEDIUM_PROBABILITY else "low"
        return tier, f" Counting model probability {p:.0%}."
    b = graph.count_borderline.get(template)
    if b is not None:
        return ("medium", "") if b == 0 else \
            ("low", f" {b} detection{'s' if b != 1 else ''} close to the counting threshold: check the highlights.")
    scores = [it.score for it in ev[:n] if it.score is not None]
    return ("medium" if len(ev) == n and (not scores or min(scores) >= 0.6) else "low"), ""


def count(graph: PlanGraph, template: str) -> Answer:
    if graph.counts and template in graph.counts:
        n = graph.counts[template]
        ev = sorted((it for it in graph.items if EVIDENCE[template](it)), key=lambda it: -(it.score or 0))
        confidence, why = _model_confidence(graph, template, n, ev)
        note = "" if len(ev) == n else f" ({len(ev)} candidate regions highlighted; count adjusted by the fusion model)"
        reviewed_note = " Includes user-reviewed corrections." if graph.user_correction_count or any(it.user_verified for it in ev) else ""
        return Answer(answer=n, unit="count", template=template, confidence=confidence,
                      explanation=f"Estimated {n} {COUNTERS.get(template, (None, template))[1]}{note}.{why}{reviewed_note}",
                      citations=ev)

    if template == "saunas":
        hits, noun = _saunas(graph), "saunas (sauna stoves, or sauna rooms without a drawn stove)"
    else:
        pred, noun = COUNTERS[template]
        hits = [it for it in graph.items if pred(it)]
    scores = [it.score for it in hits if it.score is not None]
    if graph.source == "svg":
        confidence = "high"
    elif scores and min(scores) < 0.5:
        confidence = "low"
    else:
        confidence = "medium"
    reviewed_note = " Includes user-reviewed corrections." if graph.user_correction_count or any(it.user_verified for it in hits) else ""
    return Answer(
        answer=len(hits),
        unit="count",
        template=template,
        confidence=confidence,
        explanation=f"Found {len(hits)} {noun} ({graph.source} parse); each is highlighted.{reviewed_note}",
        citations=hits,
    )
