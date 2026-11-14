"""Room type from combined evidence: OCR labels (strong), fixtures inside the room (strong when decisive) and the
2019 CubiCasa net's room-class pixels (weak).

Every piece of evidence adds score points to one or more room types; the type is the highest score. Weights are
hand-set (not fitted) and all live in EVIDENCE_WEIGHTS, so they can be tuned in one place. They are on a log-odds-like
scale: a readable label (+6) outweighs any fixture combination, a decisive fixture (+4..5) outweighs the net (<= +1.5),
so the net only decides when nothing else is known, and then only as a `weak` guess.

Tiers:
  strong   the winner has an OCR label or a decisive fixture, leads the runner-up by >= STRONG_MARGIN, no conflict
  weak     the winner rests on weak evidence only (net pixels, ambiguous fixtures), or strong evidence conflicts
  unknown  no evidence at all
A conflict = strong evidence (label or decisive fixture) for a different type than the winner, e.g. label "K" but a
toilet inside: the label wins, the conflict is recorded and the tier drops to weak.
"""

from dataclasses import dataclass

from app.models import RoomEvidence, RoomLabel

EVIDENCE_WEIGHTS = {
    # OCR purpose label inside the room (rooms_dict.room_purpose)
    "label": 6.0,  # OCR score >= LABEL_STRONG_SCORE
    "label_unsure": 3.0,  # LABEL_MIN_SCORE <= OCR score < LABEL_STRONG_SCORE
    # fixtures inside the room outline: {fixture: {type: points}}
    "fixtures": {
        "toilet": {"wc": 4.0, "bathroom": 2.0},
        "shower": {"bathroom": 4.0},
        "bathtub": {"bathroom": 4.0},
        "sauna_stove": {"sauna": 5.0},
        "sauna_bench": {"sauna": 5.0},
        "sink": {"kitchen": 1.0, "bathroom": 1.0, "wc": 0.5},
        "double_sink": {"kitchen": 1.0, "bathroom": 1.0},
        "appliance": {"kitchen": 3.0},  # CubiCasa "electrical appliance" icon: stove, fridge, dishwasher
        "fireplace": {"living": 2.0},
        "closet_small_room": {"closet": 2.0, "storage": 1.0},
        "closet_large_room": {"bedroom": 1.0},  # a wardrobe inside a room
    },
    # combinations on top of the single fixtures
    "toilet_with_wet_fixture": {"bathroom": 5.0, "wc": -4.0},  # toilet + shower/bathtub = bathroom, not a WC
    "appliance_with_sink": {"kitchen": 2.0},
    # 2019 net: points = weight x fraction of the room's pixels of that net class (fuse.ROOM names)
    "segmentation": {"kitchen": {"kitchen": 1.5}, "living": {"living": 1.5}, "bedroom": {"bedroom": 1.5},
                     "bath": {"bathroom": 1.5, "wc": 0.75}, "entry": {"entry": 1.5}, "storage": {"storage": 1.5},
                     "outdoor": {"balcony": 1.5}, "garage": {"garage": 1.5}},
}
LABEL_STRONG_SCORE = 0.7
LABEL_MIN_SCORE = 0.5
DECISIVE_FIXTURES = {"toilet", "shower", "bathtub", "sauna_stove", "sauna_bench", "appliance"}
SMALL_ROOM_FRAC = 0.015  # of the plan area: below this a closet icon means a closet, above it a wardrobe
STRONG_MARGIN = 2.0
WEAK_MIN_SCORE = 0.75
FIXTURE_WORDS = {"toilet": "toilet", "shower": "shower", "bathtub": "bathtub", "sauna_stove": "sauna stove",
                 "sauna_bench": "sauna bench", "sink": "sink", "double_sink": "double sink",
                 "appliance": "kitchen appliance", "fireplace": "fireplace"}


@dataclass
class Fixture:
    kind: str  # toilet, sink, double_sink, shower, bathtub, sauna_stove, sauna_bench, appliance, closet, fireplace
    item_id: str | None = None


@dataclass
class Classification:
    type: str
    tier: str
    purposes: list[str]
    scores: dict[str, float]
    evidence: list[RoomEvidence]
    conflicts: list[str]


def classify(labels: list[RoomLabel], fixtures: list[Fixture], seg_fractions: dict[str, float],
             area_frac: float) -> Classification:
    """Combine the evidence of one room. seg_fractions: net class name -> share of the room's pixels."""
    W = EVIDENCE_WEIGHTS
    ev: list[RoomEvidence] = []
    strong_types: set[str] = set()

    # OCR labels: one vote per distinct purpose (a label repeated by OCR does not count twice)
    seen: set[str] = set()
    for lb in sorted(labels, key=lambda lb: -(lb.score or 0)):
        if not lb.room_type or lb.room_type in seen or (lb.score or 0) < LABEL_MIN_SCORE:
            continue
        seen.add(lb.room_type)
        sure = (lb.score or 0) >= LABEL_STRONG_SCORE
        drawing = lb.source == "drawing"
        ev.append(RoomEvidence(source="drawing" if drawing else "ocr", type=lb.room_type,
                               weight=W["label"] if sure else W["label_unsure"],
                               detail=(f'drawing text "{lb.text}"' if drawing else f'label "{lb.text}"')
                               + ("" if sure else " (unclear reading)")))
        if sure:
            strong_types.add(lb.room_type)

    # fixtures (each instance counts once; several toilets do not make a room more of a WC)
    kinds = {f.kind for f in fixtures}
    first_id = {f.kind: f.item_id for f in reversed(fixtures)}
    for kind in sorted(kinds):
        table = W["fixtures"].get(kind)
        if kind == "closet":
            table = W["fixtures"]["closet_small_room" if area_frac < SMALL_ROOM_FRAC else "closet_large_room"]
        for rtype, pts in (table or {}).items():
            ev.append(RoomEvidence(source="fixture", type=rtype, weight=pts, item_id=first_id.get(kind),
                                   detail=f"{FIXTURE_WORDS.get(kind, kind)} inside"))
    if "toilet" in kinds and kinds & {"shower", "bathtub"}:
        for rtype, pts in W["toilet_with_wet_fixture"].items():
            ev.append(RoomEvidence(source="fixture", type=rtype, weight=pts, detail="toilet together with shower/bath"))
    if "appliance" in kinds and kinds & {"sink", "double_sink"}:
        for rtype, pts in W["appliance_with_sink"].items():
            ev.append(RoomEvidence(source="fixture", type=rtype, weight=pts, detail="appliance together with sink"))

    # the 2019 net: weak, capped by construction (fraction <= 1)
    for cls, frac in seg_fractions.items():
        for rtype, w in W["segmentation"].get(cls, {}).items():
            if frac >= 0.2:
                ev.append(RoomEvidence(source="segmentation", type=rtype, weight=round(w * frac, 2),
                                       detail=f"2019 model: {frac:.0%} of the room looks like {cls}"))

    scores: dict[str, float] = {}
    for e in ev:
        scores[e.type] = scores.get(e.type, 0.0) + e.weight
    # decisive fixtures make the type they point to strong (bathroom from toilet+shower, sauna from a stove, ...)
    fixture_best: dict[str, float] = {}
    for e in ev:
        if e.source == "fixture" and e.weight > 0:
            fixture_best[e.type] = fixture_best.get(e.type, 0.0) + e.weight
    fixture_top = None
    if kinds & DECISIVE_FIXTURES and fixture_best:
        top = max(fixture_best, key=fixture_best.get)
        if fixture_best[top] >= 4.0:
            strong_types.add(top)
            fixture_top = top

    ranked = sorted(((t, round(s, 2)) for t, s in scores.items() if s > 0), key=lambda x: -x[1])
    purposes = [t for t in dict.fromkeys(e.type for e in ev
                                         if e.source in ("ocr", "drawing") and e.weight >= W["label"])]
    if not ranked:
        return Classification("unknown", "unknown", [], {}, ev, [])
    best, best_score = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    # open plan: several clear labels in one space are not a conflict between each other. Decisive fixtures that
    # all fit the winner are not a conflict either: a toilet in a room labelled KPH points to "wc" more than to
    # "bathroom", but bathrooms have toilets. A toilet in a room labelled K does conflict (it gives kitchen nothing).
    fits = all(W["fixtures"].get(k, {}).get(best, 0) > 0 for k in kinds & DECISIVE_FIXTURES)
    conflicts = sorted(t for t in strong_types if t != best and t not in purposes and not (t == fixture_top and fits))
    if best in strong_types and best_score - runner >= STRONG_MARGIN and not conflicts:
        tier = "strong"
    elif best in purposes and len(purposes) > 1 and not conflicts:
        tier = "strong"  # open plan, e.g. living + kitchen both labelled
    elif best_score >= WEAK_MIN_SCORE:
        tier = "weak"
    else:
        tier = "unknown"
    return Classification(best if tier != "unknown" else "unknown", tier, purposes or ([best] if tier == "strong" else []),
                          dict(ranked[:3]), ev, conflicts)
