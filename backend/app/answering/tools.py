"""Deterministic tools over one PlanGraph. The LLM picks and combines them; every number comes from here.

Each call returns (result for the model, ToolTrace). The trace records which item ids and numbers the call
produced, so the agent can check that the final answer and its citations are backed by tool output.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable

from app.cubicasa.answer import count
from app.models import Answer, PlanGeometry, PlanGraph, PlanItem, RoomShape, WallSegment
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


MAX_CITATIONS = 200
CONFIDENCE_ORDER = ["not_found", "low", "medium", "high"]


@dataclass
class ToolTrace:
    item_ids: list[str] = field(default_factory=list)  # items the call returned (citable)
    numbers: list[int | float] = field(default_factory=list)  # numbers the call computed (answerable)
    template_answer: Answer | None = None  # set by count_template (and other calibrated lookups)
    derived_from: list["ToolTrace"] | None = None  # set by calculate: the calls whose numbers it combined
    confidence: str | None = None  # the producing tool's own confidence; None = the toolset's raw_confidence
    items: list[PlanItem] = field(default_factory=list)  # citable items created by this call (registered in by_id)
    mentioned: list[float] = field(default_factory=list)  # numbers inside result strings (set by BaseTools.call)


# a number written in text: not part of a word or identifier ("IFC4", GUID digits), e.g. "37" in "C30/37"
NUMBER_IN_TEXT = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_ID_KEYS = {"id", "ids", "gid", "global_id", "evidence_ids", "item_ids", "container_id"}


def _decimals(v: float) -> int:
    text = repr(float(v))
    if "e" in text or text.endswith(".0"):
        return 0
    return len(text.split(".")[1]) if "." in text else 0


def written_decimals(text: str) -> int:
    """Decimal places as written: "87.0" has 1 (the float 87.0 alone would say 0)."""
    text = text.replace(",", ".")
    return len(text.split(".")[1]) if "." in text else 0


def number_matches(value: float, number: float, decimals: int | None = None) -> bool:
    """The value equals the tool number, or is that number rounded to the value's (at least one) decimals:
    45.2 matches 45.237, but an integer never matches a fraction (3 does not match 2.9). decimals = as written in
    text, when the value was read from text ("87.0" matches 87.012)."""
    if isinstance(value, bool) or isinstance(number, bool):
        return False
    if float(value) == float(number):
        return True
    d = _decimals(value) if decimals is None else decimals  # by distance, not round(): round(90.645, 2) is 90.64 in binary floats, people write 90.65
    return 0 < d < _decimals(number) and abs(float(value) - float(number)) <= 0.5 * 10 ** -d + 1e-9


def backed(value: float, numbers, decimals: int | None = None) -> bool:
    return any(number_matches(value, n, decimals) for n in numbers)


def _mentioned(result: Any, out: list[float]) -> list[float]:
    """Numbers written inside the strings of a tool result ("C30/37", "EI 60"), skipping identifiers. Numeric
    values are not included: a tool lists the ones that may be stated in its trace's numbers (bbox coordinates and
    scores must not back a stated number)."""
    if isinstance(result, dict):
        for k, v in result.items():
            if k not in _ID_KEYS:
                _mentioned(v, out)
    elif isinstance(result, list | tuple):
        for v in result:
            _mentioned(v, out)
    elif isinstance(result, str):
        out.extend(float(n) for n in NUMBER_IN_TEXT.findall(result))
    return out


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


def _shape(region: PlanItem):
    """The region's outline as a shapely polygon, or None when it only has a bounding box."""
    if not region.polygon or len(region.polygon) < 3:
        return None
    from shapely.geometry import Polygon

    poly = Polygon(region.polygon)
    poly = poly if poly.is_valid else poly.buffer(0)
    return None if poly.is_empty else poly


def _contains(region: PlanItem, shape, it: PlanItem) -> bool:
    """Is the item in the region? With an outline, an L-shaped room no longer claims what sits in its notch.

    Fixtures, rooms and labels: centre inside the outline, or most of the box inside it (a basin against the wall of
    a roughly segmented room). Doors and windows sit in the wall, just outside the room outline: they belong to
    the room when at least half of their length runs along it (so a door belongs to both rooms it connects). Without an outline: centre in the bounding box."""
    if shape is None:
        return _inside(it.bbox, region.bbox)
    from shapely.geometry import Point, box

    x0, y0, x1, y1 = it.bbox
    if it.category in ("door", "window"):
        pad = max(2.0, min(x1 - x0, y1 - y0))  # about one wall thickness
        hit = box(x0 - pad, y0 - pad, x1 + pad, y1 + pad).intersection(shape)
        if hit.is_empty:
            return False
        hx0, hy0, hx1, hy1 = hit.bounds
        w, h = x1 - x0, y1 - y0
        share_x, share_y = (hx1 - hx0) / (w + 2 * pad), (hy1 - hy0) / (h + 2 * pad)
        if max(w, h) >= 2 * min(w, h):  # a bar in the wall: measure along it
            return (share_x if w >= h else share_y) >= 0.5
        return max(share_x, share_y) >= 0.5  # squarish box (door with its swing): the wall may run either way
    if shape.contains(Point((x0 + x1) / 2, (y0 + y1) / 2)):
        return True
    area = (x1 - x0) * (y1 - y0)
    return area > 0 and box(x0, y0, x1, y1).intersection(shape).area >= 0.5 * area


def _area(b: tuple) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def _room_item(rm: RoomShape) -> PlanItem:
    """A geometry room as a citable item: the frontend highlights its polygon."""
    return PlanItem(id=rm.id, category="room", cls=rm.type, bbox=rm.bbox, polygon=rm.polygon, text=rm.name,
                    raw=f"geometry:{rm.tier}")


def _segment_item(sg: WallSegment) -> PlanItem:
    """A wall segment as a citable item: its centreline buffered by half its thickness."""
    from shapely.geometry import LineString

    poly = LineString([sg.p0, sg.p1]).buffer(max(1.0, sg.thickness_px / 2), cap_style=2)
    coords = [(round(x, 1), round(y, 1)) for x, y in list(poly.exterior.coords)[:-1]]
    x0, y0, x1, y1 = poly.bounds
    return PlanItem(id=sg.id, category="wall", bbox=(x0, y0, x1, y1), polygon=coords, raw="wall segment")


def _evidence_summary(rm: RoomShape) -> str:
    """The evidence for the winning type, strongest first, e.g. 'label "WC"; toilet inside'."""
    parts = [e for e in rm.evidence if e.type == rm.type and e.weight > 0]
    return "; ".join(e.detail for e in sorted(parts, key=lambda e: -e.weight)[:3])


def _simplify(points: list[tuple[float, float]], max_points: int = 24) -> list[list[float]]:
    from shapely.geometry import Polygon

    poly, tol = Polygon(points), 1.0
    while len(poly.exterior.coords) - 1 > max_points and tol < 1e4:
        poly, tol = Polygon(points).simplify(tol, preserve_topology=True), tol * 2
    return [[round(x), round(y)] for x, y in list(poly.exterior.coords)[:-1]]


def _room_row(rm: RoomShape, detail: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"id": rm.id, "tier": rm.tier}
    if rm.tier != "unknown":
        row["type"] = rm.type
        if rm.name:
            row["label"] = rm.name
        if len(rm.purposes) > 1:
            row["purposes"] = rm.purposes  # open plan: several labels in one enclosed space
        row["because"] = _evidence_summary(rm)
        others = [(t, v) for t, v in rm.scores.items() if t != rm.type][:2]
        if others:
            row["alternatives"] = {t: v for t, v in others}
        if rm.conflicts:
            row["conflicting_evidence_for"] = rm.conflicts
    if rm.area_m2 is not None:
        row["area_m2"] = rm.area_m2
    if rm.labeled_area_m2 is not None:
        row["area_m2_written_on_drawing"] = rm.labeled_area_m2
    if rm.area_m2_range is not None:
        row["rough_area_m2_range"] = list(rm.area_m2_range)  # from door / fixture sizes: bounds, not a measurement
    if rm.size_check:
        row["size_check"] = rm.size_check
    row["neighbours"] = rm.neighbours
    row["openings"] = len(rm.connections)
    if detail:
        row["text_inside"] = [lb.text for lb in rm.labels]
        row["bbox_px"] = [round(v) for v in rm.bbox]
        row["evidence"] = [{"for": e.type, "points": e.weight, "what": e.detail, **({"item": e.item_id} if e.item_id else {})}
                           for e in sorted(rm.evidence, key=lambda e: -e.weight)]
    return row


def _segment_row(sg: WallSegment) -> dict[str, Any]:
    row: dict[str, Any] = {"id": sg.id, "from_px": list(sg.p0), "to_px": list(sg.p1), "orientation": sg.orientation,
                           "length_px": sg.length_px, "thickness_px": sg.thickness_px}
    if sg.length_m is not None:
        row["length_m"] = sg.length_m
        row["thickness_m"] = sg.thickness_m
    row["side_a"] = sg.rooms_left
    row["side_b"] = sg.rooms_right
    row["exterior"] = sg.exterior
    if sg.openings:
        row["openings"] = sg.openings
    return row


class BaseTools:
    """What the agent needs from a toolset (image plan, PDF, IFC model): declarations, a dispatcher that records
    which numbers and items each call produced, calculate, and the toolset's prompt rules / summary / cache key /
    confidence defaults / deterministic fallback."""

    rules = ""  # toolset-specific prompt rules, placed between the shared prompt's head and tail
    summary_label = "Plan summary"
    uses_scale = False  # the user's drawing calibration is relevant to this toolset
    cross_check_router = False  # compare with the cubicasa regex router (image plans only)
    raw_confidence = "low"  # a value from a tool without its own confidence
    text_confidence = "low"  # a text (non-numeric) answer
    geometry: PlanGeometry | None = None

    def __init__(self):
        self.by_id: dict[str, PlanItem] = {}
        self.produced: dict[float, ToolTrace] = {}  # every number a tool returned so far -> the call it came from

    # --- to override -------------------------------------------------------------------------------------------

    def declarations(self) -> list[dict]:
        raise NotImplementedError

    def handlers(self) -> dict[str, Callable[..., tuple[dict, ToolTrace]]]:
        raise NotImplementedError

    def summary(self) -> dict:
        return {}

    def fingerprint(self) -> Any:
        """JSON-able identity of the evidence (part of the answer cache key)."""
        raise NotImplementedError

    def fallback(self, question: str) -> Answer:
        """The deterministic answerer, used when Gemini is unavailable."""
        raise NotImplementedError

    # --- shared ------------------------------------------------------------------------------------------------

    def item(self, item_id: str) -> PlanItem | None:
        return self.by_id.get(item_id)

    def source_of(self, number: float) -> ToolTrace | None:
        """The call that produced this number (exactly, else as a rounded form of it)."""
        if isinstance(number, bool) or not isinstance(number, int | float):
            return None
        hit = self.produced.get(float(number))
        if hit is not None:
            return hit
        return next((t for n, t in self.produced.items() if number_matches(number, n)), None)

    def calculate(self, operation: str, operands: list) -> tuple[dict, ToolTrace]:
        """Sum or difference of numbers earlier calls returned. The model supplies no numbers of its own."""
        if operation not in ("add", "subtract"):
            return {"error": "operation must be 'add' or 'subtract'"}, ToolTrace()
        if not isinstance(operands, list) or len(operands) < 2 or any(
                isinstance(v, bool) or not isinstance(v, int | float) for v in operands):
            return {"error": "operands must be a list of at least two numbers"}, ToolTrace()
        unknown = [v for v in operands if self.source_of(v) is None]
        if unknown:
            return {"error": f"{unknown} were not returned by a tool: every operand must be a number from an earlier "
                             "tool result (call the tool that provides it first)"}, ToolTrace()
        result = sum(operands) if operation == "add" else operands[0] - sum(operands[1:])
        result = int(result) if float(result).is_integer() else round(result, 6)
        sources = list({id(t): t for t in (self.source_of(v) for v in operands)}.values())
        ids = list(dict.fromkeys(i for t in sources for i in t.item_ids))
        return ({"operation": operation, "operands": operands, "result": result,
                 "note": "subtract = first operand minus the others" if operation == "subtract" else ""},
                ToolTrace(item_ids=ids, numbers=[result], derived_from=sources))

    def call(self, name: str, args: dict) -> tuple[dict, ToolTrace]:
        fn = self.handlers().get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}, ToolTrace()
        try:
            result, trace = fn(**args)
        except (TypeError, ValueError) as error:  # wrong / missing arguments from the model
            return {"error": str(error)}, ToolTrace()
        for it in trace.items:
            self.by_id.setdefault(it.id, it)
        trace.mentioned = _mentioned(result, [])
        for number in trace.numbers:  # a calibrated count keeps the claim on its number over raw item counts
            if isinstance(number, bool) or not isinstance(number, int | float):
                continue
            known = self.produced.get(float(number))
            if known is None or (known.template_answer is None and trace.template_answer is not None):
                self.produced[float(number)] = trace
        return result, trace


class PlanTools(BaseTools):
    """Image plans (cubicasa pipeline or SVG labels)."""

    summary_label = "Plan summary"
    uses_scale = True
    cross_check_router = True

    def __init__(self, graph: PlanGraph, geometry: PlanGeometry | None = None):
        super().__init__()
        self.graph = graph
        self.geometry = geometry
        self.by_id = {it.id: it for it in graph.items}
        self.rooms = {rm.id: rm for rm in geometry.rooms} if geometry else {}
        self.segments = {sg.id: sg for sg in geometry.wall_segments} if geometry else {}
        for rm in self.rooms.values():
            self.by_id.setdefault(rm.id, _room_item(rm))
        for sg in self.segments.values():
            self.by_id.setdefault(sg.id, _segment_item(sg))
        self.rules = IMAGE_RULES
        self.raw_confidence = "high" if graph.source == "svg" else "low"  # raw items, not a calibrated count
        self.text_confidence = "medium" if graph.source == "svg" else "low"

    def declarations(self) -> list[dict]:
        return DECLARATIONS + (ROOM_DECLARATIONS if self.geometry else [])

    def handlers(self) -> dict[str, Callable[..., tuple[dict, ToolTrace]]]:
        return {"plan_summary": self.plan_summary, "count_template": self.count_template,
                "list_items": self.list_items, "items_in": self.items_in, "calculate": self.calculate,
                **({"list_rooms": self.list_rooms, "room_details": self.room_details, "list_walls": self.list_walls,
                    "covering_quantity": self.covering_quantity} if self.geometry else {})}

    def summary(self) -> dict:
        return self.plan_summary()[0]

    def fingerprint(self) -> Any:
        parts: list[Any] = [self.graph.model_dump(mode="json")]
        if self.geometry is not None:
            parts.append(self.geometry.model_dump(mode="json"))
        return parts

    def fallback(self, question: str) -> Answer:
        from app.router import route

        template = route(question)
        if template is None:
            return Answer(answer=None, confidence="not_found", explanation="This question type is not supported yet.")
        return count(self.graph, template)

    # --- tools -------------------------------------------------------------------------------------------------

    def plan_summary(self) -> tuple[dict, ToolTrace]:
        kinds: dict[str, int] = {}
        for it in self.graph.items:
            key = it.category + (f":{it.cls}" if it.cls else "") + (f" ({_kind(it)})" if it.category == "room" else "")
            kinds[key] = kinds.get(key, 0) + 1
        scale = self.geometry.scale if self.geometry else None
        return {
            "image_size_px": [self.graph.width, self.graph.height],
            "source": "ground-truth labels" if self.graph.source == "svg" else "image analysis (detector, segmentation, OCR)",
            "items_by_kind": dict(sorted(kinds.items())),
            "user_reviewed_items": sum(it.user_verified for it in self.graph.items),
            "user_correction_events": self.graph.user_correction_count,
            "note": ("Drawing scale is available for metric measurements." if scale else
                     "No drawing scale is known: real-world lengths and areas (m, m²) cannot be computed."),
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
                      if it.category == "room" and _kind(it) == "region" and _contains(it, _shape(it), box)]
            if not around:
                return {"error": f"{container_id} is a text label and no segmented room region contains it"}, ToolTrace()
            region = min(around, key=lambda it: _area(it.bbox))
        shape = _shape(region)
        hits = [it for it in self.graph.items
                if it.id != region.id and _match(it, category, type) and _contains(region, shape, it)]
        out = {"container": _row(region), "count": len(hits), "items": [_row(it) for it in hits[:MAX_LIST]],
               "note": ("Containment uses the container's outline: fixtures by their centre, doors and windows when "
                        "they run along the outline." if shape is not None else
                        "Containment uses the container's bounding box (item centre inside), so it is approximate.")}
        return out, ToolTrace(item_ids=[region.id] + [it.id for it in hits[:MAX_LIST]], numbers=[len(hits)])

    def _scale_note(self) -> str:
        sc = self.geometry.scale if self.geometry else None
        sym = self.geometry.symbol_scale if self.geometry else None
        if sc is None:
            if sym is not None:
                return ("No drawing scale. rough_area_m2_range is a wide bound estimated from the typical sizes of the "
                        "detected doors and fixtures: give it only as a rough range ('roughly X-Y m²'), never as a "
                        "measurement, and suggest setting the scale for exact sizes.")
            return "No drawing scale: metric sizes are unknown except areas written on the drawing."
        warn = f" Warning: this scale {sc.warning}." if sc.warning else ""
        if sc.source == "calibration":
            return "Metric sizes use the user's scale calibration." + warn
        return (f"Metric sizes use a scale derived from the areas written in rooms {', '.join(sc.rooms)}: "
                "mention that they are derived, not measured." + warn)

    def list_rooms(self, include_uncertain: bool = False) -> tuple[dict, ToolTrace]:
        rooms = [rm for rm in self.rooms.values() if include_uncertain or rm.tier == "strong"]
        tiers = {t: sum(rm.tier == t for rm in self.rooms.values()) for t in ("strong", "weak", "unknown")}
        out = {"strong_rooms": tiers["strong"], "weak_rooms": tiers["weak"], "unknown_spaces": tiers["unknown"],
               "rooms": [_room_row(rm) for rm in rooms[:MAX_LIST]],
               "note": ("Rooms are spaces enclosed by the detected walls; the type combines a readable label inside "
                        "the room (strong), decisive fixtures such as a toilet or shower (strong) and the 2019 "
                        "segmentation model (weak). tier strong: state the type plainly. weak: say 'probably <type>' "
                        "and why. unknown: purpose not determined, never guess it. An open-plan space with several "
                        "labels is one enclosed space with several purposes. " + self._scale_note())}
        numbers: list[float] = [tiers["strong"], tiers["weak"], tiers["unknown"], len(rooms)]
        numbers += [rm.area_m2 for rm in rooms if rm.area_m2 is not None]
        numbers += [rm.labeled_area_m2 for rm in rooms if rm.labeled_area_m2 is not None]
        numbers += [v for rm in rooms if rm.area_m2_range for v in rm.area_m2_range]
        return out, ToolTrace(item_ids=[rm.id for rm in rooms[:MAX_LIST]], numbers=numbers)

    def room_details(self, room_id: str) -> tuple[dict, ToolTrace]:
        rm = self.rooms.get(room_id)
        if rm is None:
            return {"error": f"no room {room_id!r} (use ids from list_rooms)"}, ToolTrace()
        from shapely.geometry import Point, Polygon

        poly = Polygon(rm.polygon)
        inside = [it for it in self.graph.items if it.category in ("door", "window", "fixture") and poly.contains(
            Point((it.bbox[0] + it.bbox[2]) / 2, (it.bbox[1] + it.bbox[3]) / 2))]
        openings = [op for op in (self.geometry.openings if self.geometry else []) if op.id in rm.connections]
        walls = [self.segments[w] for w in rm.walls if w in self.segments]
        row = _room_row(rm, detail=True)
        row["outline_px"] = _simplify(rm.polygon)
        if self.geometry and self.geometry.scale:
            m = self.geometry.scale.m_per_px
            row["outline_m"] = [[round(x * m, 2), round(y * m, 2)] for x, y in row["outline_px"]]
        row["neighbours"] = [{"id": n, "tier": self.rooms[n].tier,
                              **({"type": self.rooms[n].type} if self.rooms[n].tier != "unknown" else {})}
                             for n in rm.neighbours if n in self.rooms]
        row["openings"] = [{"id": op.id, "leads_to": [r for r in op.rooms if r != rm.id] or ["(same room)"],
                            **({"symbol": op.symbol} if op.symbol else {})} for op in openings]
        row["walls"] = [_segment_row(sg) for sg in walls[:MAX_LIST]]
        row["items_inside"] = [_row(it) for it in inside[:MAX_LIST]]
        row["note"] = ("Items inside = detected doors/windows/fixtures whose centre lies in the room outline. "
                       "Walls = wall segments bounding the room ('outside' = exterior). " + self._scale_note())
        numbers: list[float] = [len(inside), len(openings), len(walls)]
        numbers += [v for v in (rm.area_m2, rm.labeled_area_m2) if v is not None]
        numbers += list(rm.area_m2_range or ())
        numbers += [v for sg in walls for v in (sg.length_m, sg.thickness_m, sg.length_px) if v is not None]
        return row, ToolTrace(item_ids=[rm.id] + [it.id for it in inside[:MAX_LIST]] + [sg.id for sg in walls[:MAX_LIST]],
                              numbers=numbers)

    def list_walls(self, room_id: str | None = None, exterior_only: bool = False) -> tuple[dict, ToolTrace]:
        segs = list(self.segments.values())
        if room_id:
            if room_id not in self.rooms:
                return {"error": f"no room {room_id!r} (use ids from list_rooms)"}, ToolTrace()
            segs = [sg for sg in segs if room_id in sg.rooms_left + sg.rooms_right]
        if exterior_only:
            segs = [sg for sg in segs if sg.exterior]
        segs.sort(key=lambda sg: -sg.length_px)
        shown = segs[:MAX_LIST]
        out = {"count": len(segs), "walls": [_segment_row(sg) for sg in shown],
               "note": ("Straight wall pieces (centrelines) from the wall detection; side_a / side_b = rooms on the two "
                        "faces, 'outside' = exterior. Coordinates are image pixels, origin top-left. "
                        + self._scale_note())}
        if len(segs) > len(shown):
            out["truncated"] = f"showing the {len(shown)} longest of {len(segs)}"
        numbers: list[float] = [len(segs)]
        numbers += [v for sg in shown for v in (sg.length_m, sg.thickness_m, sg.length_px, sg.thickness_px)
                    if v is not None]
        return out, ToolTrace(item_ids=[sg.id for sg in shown], numbers=numbers)

    def covering_quantity(self, room_type: str, unit_width_m: float, unit_height_m: float,
                          waste_percent: float = 10) -> tuple[dict, ToolTrace]:
        """Estimate count of rectangular covering units for the matching room floors."""
        if not self.geometry:
            return {"error": "Room geometry is unavailable."}, ToolTrace()
        if unit_width_m <= 0 or unit_height_m <= 0 or not 0 <= waste_percent <= 100:
            return {"error": "Unit dimensions must be positive and waste_percent must be 0..100."}, ToolTrace()
        rooms = [rm for rm in self.rooms.values() if rm.type == room_type and rm.tier != "unknown"]
        if not self.geometry.scale and any(rm.area_m2 is not None for rm in rooms):
            # area_m2 is normally populated only with a scale; keep this guard explicit for malformed geometry.
            return {"error": "A drawing scale is required to calculate room area."}, ToolTrace()
        missing = [rm.id for rm in rooms if rm.area_m2 is None and rm.labeled_area_m2 is None]
        usable = [rm for rm in rooms if rm.area_m2 is not None or rm.labeled_area_m2 is not None]
        if not usable:
            return {"error": f"No measured or labeled areas available for {room_type} rooms."}, ToolTrace()
        area = sum(rm.area_m2 if rm.area_m2 is not None else rm.labeled_area_m2 for rm in usable)
        unit_area = unit_width_m * unit_height_m
        base_count = math.ceil(area / unit_area)
        quantity = math.ceil(base_count * (1 + waste_percent / 100))
        ids = [rm.id for rm in usable]
        result = {"room_type": room_type, "room_count": len(usable), "area_m2": round(area, 3),
                  "unit_area_m2": round(unit_area, 5), "quantity_without_waste": base_count,
                  "waste_percent": waste_percent, "quantity_with_waste": quantity,
                  "area_sources": ["measured" if rm.area_m2 is not None else "written on drawing" for rm in usable]}
        if missing:
            result["rooms_without_area"] = missing
        return result, ToolTrace(item_ids=ids, numbers=[area, unit_area, base_count, waste_percent, quantity])


CALCULATE = {
    "name": "calculate",
    "description": ("Add or subtract numbers that earlier tool calls returned, e.g. two count_template values: "
                    "'how many X and Y in total' -> add, 'how many more X than Y' -> subtract (first operand "
                    "minus the others). Every operand must be a number from an earlier tool result; the result "
                    "may be given as the answer value."),
    "parameters": {"type": "object", "properties": {
        "operation": {"type": "string", "enum": ["add", "subtract"]},
        "operands": {"type": "array", "items": {"type": "number"}, "minItems": 2},
    }, "required": ["operation", "operands"]},
}

# the same tool for toolsets without count_template (PDF, IFC)
CALCULATE_ANY = {**CALCULATE, "description": (
    "Add or subtract numbers that earlier tool calls returned: 'how many X and Y in total' or 'total area of A and B' "
    "-> add, 'how many more X than Y' or 'how much larger' -> subtract (first operand minus the others). Every "
    "operand must be a number from an earlier tool result; the result may be given as the answer value.")}

# Prompt rules for image plans (placed between the shared prompt's head and tail by the agent).
IMAGE_RULES = """- For a whole-plan count that matches a count_template template, use count_template (it is calibrated); do not
  count raw items from list_items for those.
- "Is there a ...?" questions about something count_template covers: answer from count_template only (yes if its
  value > 0, no if 0), even if raw items suggest otherwise; mention low confidence if reported.
- For a question restricted to a room or area (e.g. "sinks in the kitchen"), find the room with
  list_items(category="room", type=...) and use items_in on it. If several such rooms exist and the question does not
  say which, ask one short, specific follow-up question asking which room they mean. Do not guess which room.
  For questions asking how many rooms there are and what types they are, call
  list_items(category="room") once: its count is the deduplicated room count and its items give the room types and
  labels. Answer both parts from that single result. Do not refuse merely because the rooms have different types.
"""

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
                        "for a room label the segmented room region around it is used. Uses the room's outline "
                        "when it has one, so it also works for rooms that are not rectangular."),
        "parameters": {"type": "object", "properties": {
            "container_id": {"type": "string"},
            "category": {"type": "string", "enum": CATEGORIES},
            "type": {"type": "string", "enum": ROOM_TYPES + FIXTURE_TYPES},
        }, "required": ["container_id", "category"]},
    },
    CALCULATE,
]


# Only offered when the plan has stored room geometry (uploaded plans); the benchmark path keeps the tools above.
ROOM_DECLARATIONS = [
    {
        "name": "covering_quantity",
        "description": ("Calculate how many rectangular covering units are needed for floors of rooms of a given type. "
                        "Uses detected room areas and drawing scale, or areas printed on the plan. Returns exact "
                        "rounded-up unit count before and after waste, with citations to rooms. Use when asked for "
                        "tiles, panels or similar area-covering materials; state the waste assumption."),
        "parameters": {"type": "object", "properties": {
            "room_type": {"type": "string", "enum": ROOM_TYPES},
            "unit_width_m": {"type": "number"}, "unit_height_m": {"type": "number"},
            "waste_percent": {"type": "number", "description": "waste allowance percentage; use 10 when not specified"},
        }, "required": ["room_type", "unit_width_m", "unit_height_m"]},
    },
    {
        "name": "list_rooms",
        "description": ("Rooms as spaces enclosed by the plan's walls. Each has a type from combined evidence (a label "
                        "read inside it, fixtures such as a toilet or shower, the segmentation model) with a tier: "
                        "strong, weak or unknown; why; its area in m² when a scale is known or an area is written in "
                        "it; and its neighbouring rooms. Use it for which rooms exist, what they are for, their sizes "
                        "and how they connect."),
        "parameters": {"type": "object", "properties": {
            "include_uncertain": {"type": "boolean",
                                  "description": "also list weak and unknown spaces (default: strong only)"},
        }},
    },
    {
        "name": "room_details",
        "description": ("One room from list_rooms: outline coordinates, all text read inside it, the full evidence for "
                        "its type, its wall segments with lengths, openings and where they lead, neighbouring rooms, "
                        "and the detected doors, windows and fixtures inside it."),
        "parameters": {"type": "object", "properties": {"room_id": {"type": "string"}}, "required": ["room_id"]},
    },
    {
        "name": "list_walls",
        "description": ("Wall segments (straight centrelines): endpoints, length and thickness (px, and m when a scale "
                        "is known), the rooms on each side, exterior or interior, and openings in them. Optionally only "
                        "the walls of one room or only exterior walls."),
        "parameters": {"type": "object", "properties": {
            "room_id": {"type": "string"},
            "exterior_only": {"type": "boolean"},
        }},
    },
]
