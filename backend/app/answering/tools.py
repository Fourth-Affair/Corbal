"""Deterministic tools over one PlanGraph. The LLM picks and combines them; every number comes from here.

Each call returns (result for the model, ToolTrace). The trace records which item ids and numbers the call
produced, so the agent can check that the final answer and its citations are backed by tool output.
"""

from dataclasses import dataclass, field
from typing import Any

from app.cubicasa.answer import count
from app.models import Answer, PlanGraph, PlanItem
from app.router import TEMPLATES

# What each count template counts (the dataset's counting rules). Shown to the model in the tool description.
TEMPLATE_DEFS = {
    "windows": "windows",
    "doors": "doors fitted with a door leaf (open doorways without a leaf do not count)",
    "floors": "floors / storeys / levels shown on the plan",
    "wc_seats": "toilet bowls (WC seats) as fixtures, wherever they are",
    "sinks": "sinks and wash basins anywhere (kitchen + bathroom combined; double-bowl sink = one)",
    "showers_baths": "showers and bathtubs together (one combined number)",
    "saunas": "saunas (sauna rooms / sauna stoves)",
    "kitchens": "kitchens incl. kitchenettes, open kitchens and utility kitchens",
    "bedrooms": "bedrooms (MH)",
    "bathrooms": "bathrooms / washrooms (KH, KPH, PH, PESUH, SH), not counting separate WC rooms",
    "wc_rooms": "separate WC rooms (toilet rooms marked WC)",
    "balconies": "balconies incl. glazed balconies (parveke)",
}
assert set(TEMPLATE_DEFS) == set(TEMPLATES)

CATEGORIES = ["room", "door", "window", "fixture", "floor"]
ROOM_TYPES = ["bedroom", "bathroom", "wc", "sauna", "kitchen", "living", "balcony", "entry", "storage", "garage", "other"]
FIXTURE_TYPES = ["toilet", "sink", "double_sink", "shower", "bathtub", "sauna_stove", "sauna_bench", "other"]
MAX_LIST = 50


@dataclass
class ToolTrace:
    item_ids: list[str] = field(default_factory=list)  # items the call returned (citable)
    numbers: list[int | float] = field(default_factory=list)  # numbers the call computed (answerable)
    template_answer: Answer | None = None  # set by count_template


def _kind(it: PlanItem) -> str:
    return "label" if it.raw == "ocr" else "region"


def _row(it: PlanItem) -> dict[str, Any]:
    row: dict[str, Any] = {"id": it.id, "category": it.category, "bbox": [round(v) for v in it.bbox]}
    if it.user_verified:
        row["evidence_source"] = "user-reviewed correction"
    if it.cls:
        row["type"] = it.cls
    if it.category == "room":
        row["kind"] = _kind(it)
    if it.text:
        row["text"] = it.text
    if it.score is not None:
        row["score"] = round(it.score, 2)
    return row


def _match(it: PlanItem, category: str, type_: str | None) -> bool:
    if it.category != category:
        return False
    if not type_:
        return True
    if type_ == "sink":  # a double sink is a sink for every question we know
        return it.cls in ("sink", "double_sink")
    return it.cls == type_


def _inside(inner: tuple, outer: tuple) -> bool:
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _area(b: tuple) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


class PlanTools:
    def __init__(self, graph: PlanGraph):
        self.graph = graph
        self.by_id = {it.id: it for it in graph.items}

    # --- tools -------------------------------------------------------------------------------------------------

    def plan_summary(self) -> tuple[dict, ToolTrace]:
        kinds: dict[str, int] = {}
        for it in self.graph.items:
            key = it.category + (f":{it.cls}" if it.cls else "") + (f" ({_kind(it)})" if it.category == "room" else "")
            kinds[key] = kinds.get(key, 0) + 1
        return {
            "image_size_px": [self.graph.width, self.graph.height],
            "source": "ground-truth labels" if self.graph.source == "svg" else "image analysis (detector, segmentation, OCR)",
            "items_by_kind": dict(sorted(kinds.items())),
            "user_reviewed_items": sum(it.user_verified for it in self.graph.items),
            "user_correction_events": self.graph.user_correction_count,
            "note": "No drawing scale is known: real-world lengths and areas (m, m²) cannot be computed.",
        }, ToolTrace()

    def count_template(self, template: str) -> tuple[dict, ToolTrace]:
        if template not in TEMPLATE_DEFS:
            return {"error": f"unknown template {template!r}"}, ToolTrace()
        ans = count(self.graph, template)
        ids = [it.id for it in ans.citations]
        return ({"template": template, "value": ans.answer, "confidence": ans.confidence, "note": ans.explanation,
                 "evidence_ids": ids},
                ToolTrace(item_ids=ids, numbers=[ans.answer] if ans.answer is not None else [], template_answer=ans))

    def list_items(self, category: str, type: str | None = None) -> tuple[dict, ToolTrace]:
        hits = [it for it in self.graph.items if _match(it, category, type)]
        labels_of: dict[str, list[PlanItem]] = {}
        if category == "room":  # a room read as a segmented region AND as a text label inside it = one room
            regions = [it for it in hits if _kind(it) == "region"]
            kept = []
            for it in hits:
                home = None if _kind(it) == "region" else next(
                    (r for r in sorted(regions, key=lambda r: _area(r.bbox))
                     if r.cls == it.cls and _inside(it.bbox, r.bbox)), None)
                if home is None:
                    kept.append(it)
                else:
                    labels_of.setdefault(home.id, []).append(it)
            hits = kept
        shown = sorted(hits, key=lambda it: -(it.score or 1))[:MAX_LIST]
        rows = []
        for it in shown:
            row = _row(it)
            if it.id in labels_of:
                row["labels"] = [lb.text for lb in labels_of[it.id]]
            rows.append(row)
        out = {"count": len(hits), "items": rows}
        if len(hits) > len(shown):
            out["truncated"] = f"showing {len(shown)} of {len(hits)}"
        if category == "room" and self.graph.source == "model":
            out["note"] = ("Room 'region' = segmented room extent, 'label' = room name read from the drawing (OCR) with "
                           "no matching region around it. Raw detections: not a calibrated count.")
        ids = [it.id for it in shown] + [lb.id for it in shown for lb in labels_of.get(it.id, [])]
        return out, ToolTrace(item_ids=ids, numbers=[len(hits)])

    def items_in(self, container_id: str, category: str, type: str | None = None) -> tuple[dict, ToolTrace]:
        box = self.by_id.get(container_id)
        if box is None:
            return {"error": f"no item {container_id!r}"}, ToolTrace()
        region = box
        if box.category == "room" and _kind(box) == "label":  # a label is a few words of text: use the room around it
            around = [it for it in self.graph.items
                      if it.category == "room" and _kind(it) == "region" and _inside(box.bbox, it.bbox)]
            if not around:
                return {"error": f"{container_id} is a text label and no segmented room region contains it"}, ToolTrace()
            region = min(around, key=lambda it: _area(it.bbox))
        hits = [it for it in self.graph.items
                if it.id != region.id and _match(it, category, type) and _inside(it.bbox, region.bbox)]
        out = {"container": _row(region), "count": len(hits), "items": [_row(it) for it in hits[:MAX_LIST]],
               "note": "Containment uses the container's bounding box (item centre inside), so it is approximate."}
        return out, ToolTrace(item_ids=[region.id] + [it.id for it in hits[:MAX_LIST]], numbers=[len(hits)])

    # --- dispatch ----------------------------------------------------------------------------------------------

    def call(self, name: str, args: dict) -> tuple[dict, ToolTrace]:
        fn = {"plan_summary": self.plan_summary, "count_template": self.count_template,
              "list_items": self.list_items, "items_in": self.items_in}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}, ToolTrace()
        try:
            return fn(**args)
        except TypeError as error:  # wrong / missing arguments from the model
            return {"error": str(error)}, ToolTrace()


DECLARATIONS = [
    {
        "name": "plan_summary",
        "description": "What the plan contains: image size, how it was parsed, and how many items of each kind exist.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "count_template",
        "description": ("Best whole-plan count of one thing, from the calibrated counting models. ALWAYS prefer this "
                        "over counting raw items when the question asks for one of these whole-plan counts. "
                        "Templates: " + "; ".join(f"{k} = {v}" for k, v in TEMPLATE_DEFS.items())),
        "parameters": {"type": "object", "properties": {"template": {"type": "string", "enum": TEMPLATES}},
                       "required": ["template"]},
    },
    {
        "name": "list_items",
        "description": ("Raw detected items of one category (optionally one type) with ids, bounding boxes (pixels) "
                        "and scores. Raw detections are not calibrated counts."),
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "type": {"type": "string", "enum": ROOM_TYPES + FIXTURE_TYPES,
                     "description": "room type (for category=room) or fixture type (for category=fixture)"},
        }, "required": ["category"]},
    },
    {
        "name": "items_in",
        "description": ("Items of a category (optionally a type) located inside a container item, e.g. fixtures in a "
                        "bathroom or windows in a bedroom. container_id is a room or floor id from list_items; "
                        "for a room label the segmented room region around it is used."),
        "parameters": {"type": "object", "properties": {
            "container_id": {"type": "string"},
            "category": {"type": "string", "enum": CATEGORIES},
            "type": {"type": "string", "enum": ROOM_TYPES + FIXTURE_TYPES},
        }, "required": ["container_id", "category"]},
    },
]
