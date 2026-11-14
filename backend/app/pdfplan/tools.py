"""Deterministic tools over a parsed vector PDF plan, for the Gemini agent (app.answering.agent).

pdf_lookup exposes the calibrated templates of app.pdfplan.answer (the offline answerer) as one tool; the list tools
give the agent the rooms, apartments and drawn symbols themselves so it can answer what no template covers. Every
number comes from here; citations are PDF points of page 1 with the same ids as the offline answers.
"""

from __future__ import annotations

from typing import Any, Callable

from app.answering.tools import CALCULATE_ANY, MAX_LIST, BaseTools, ToolTrace
from app.models import Answer, PlanItem
from app.pdfplan.answer import (NEEDS_ARG, TEMPLATES, _CATEGORY, _room_item, answer_question, answer_template,
                                raster_symbols)
from app.pdfplan.parse import Plan, Room, norm, parse_plan, resolve_pdf

SYMBOL_KINDS = ["door", "window", "toilet", "washbasin", "bathtub", "shower", "kitchen", "stairs"]

PDF_RULES = """- The plan is a vector PDF drawing; rooms carry a number, usually a name and a net area (m²) printed on the
  drawing; apartments are tagged on the drawing (e.g. "Whg. A"). Areas are in m², as printed or measured from outlines.
- pdf_lookup answers the standard questions (counts of apartments / doors / windows / toilets / washbasins /
  bathtubs / showers, rooms with a given label and their area, the area of one room, of one apartment, of all numbered
  indoor rooms) with the plan's counting rules: use it when the question is one of these, in any language or wording.
- For anything else, combine list_rooms, list_apartments and list_symbols, and calculate for sums and differences.
  Room names are printed in the drawing's language (German, French, Italian ...): match the meaning of the question
  ("bedroom" = Zimmer / chambre / camera) and say which printed name you used.
- Copy numbers exactly as the tools return them.
- When the question asks for a list or breakdown (e.g. spaces with their areas, types with quantities), answer with
  text, not value: the total on the first line if there is one, then the complete list, one item per line with its
  numbers as the tools returned them (at most 40 lines; say how many more there are). Keep the explanation short.
"""


def _room_row(r: Room, apartment: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {"id": f"room-{r.number}", "number": r.number, "name": r.name}
    if r.area is not None:
        row["area_m2"] = r.area
        row["area_source"] = "printed on plan" if r.printed_area is not None else "measured from outline"
    row["outdoor"] = r.outdoor
    row["indoor_numbered"] = r.counts_as_indoor
    if apartment:
        row["apartment"] = apartment
    return row


class PdfTools(BaseTools):
    summary_label = "Plan summary"
    raw_confidence = "medium"
    text_confidence = "medium"

    def __init__(self, rel: str, use_dxf: bool = True):
        super().__init__()
        self.rel, self.use_dxf = rel, use_dxf
        self.plan: Plan = parse_plan(rel, use_dxf)
        self.rules = PDF_RULES
        self.apartment_of = {r.number: u.letter for u in self.plan.units for r in (u.rooms or [])}

    def declarations(self) -> list[dict]:
        return DECLARATIONS

    def handlers(self) -> dict[str, Callable[..., tuple[dict, ToolTrace]]]:
        return {"pdf_lookup": self.pdf_lookup, "list_rooms": self.list_rooms, "list_apartments": self.list_apartments,
                "list_symbols": self.list_symbols, "calculate": self.calculate}

    def summary(self) -> dict:
        p = self.plan
        symbols = ("vector drawing (DXF)" if p.symbols is not None else
                   "none in the PDF: door / window / fixture symbols come from an image detector (low confidence)")
        names = sorted({r.name for r in p.rooms if r.name})
        return {"page_size_pt": [round(v) for v in p.page_size], "rooms": len(p.rooms),
                "rooms_with_printed_area": sum(r.printed_area is not None for r in p.rooms),
                "room_names": names[:60], "apartments": [u.tag for u in p.units],
                "apartment_room_membership": "known (outlines)" if any(u.rooms is not None for u in p.units)
                else "unknown (no apartment outlines)",
                "symbol_source": symbols, "sources": p.sources}

    def fingerprint(self) -> Any:
        st = resolve_pdf(self.rel).stat()
        return ["pdf", self.rel, st.st_size, st.st_mtime, self.use_dxf]

    def fallback(self, question: str) -> Answer:
        return answer_question(self.rel, question, self.use_dxf)

    # --- tools -------------------------------------------------------------------------------------------------

    def pdf_lookup(self, template: str, argument: str | None = None) -> tuple[dict, ToolTrace]:
        ans = answer_template(self.plan, self.rel, template, argument)
        ids = [it.id for it in ans.citations]
        out = {"template": template, "value": ans.answer, "unit": ans.unit, "confidence": ans.confidence,
               "note": ans.explanation, "evidence_ids": ids[:MAX_LIST]}
        return out, ToolTrace(item_ids=ids, items=list(ans.citations), template_answer=ans,
                              numbers=[ans.answer] if isinstance(ans.answer, int | float) else [])

    def list_rooms(self, name: str | None = None, apartment: str | None = None,
                   include_outdoor: bool = True) -> tuple[dict, ToolTrace]:
        """Rooms, optionally only those whose printed name contains `name` or that belong to one apartment."""
        if apartment:
            unit = next((u for u in self.plan.units if u.letter.casefold() == apartment.strip().casefold()), None)
            if unit is None:
                return {"error": f"no apartment {apartment!r}; apartments: {[u.letter for u in self.plan.units]}"}, ToolTrace()
            if unit.rooms is None:
                return {"error": f"apartment {unit.letter} is tagged, but which rooms belong to it is unknown (the PDF "
                                 "has no apartment outlines)"}, ToolTrace()
            rooms = list(unit.rooms)
        else:
            rooms = list(self.plan.rooms)
        if name:
            rooms = [r for r in rooms if r.name and norm(name) in norm(r.name)]
        if not include_outdoor:
            rooms = [r for r in rooms if not r.outdoor]
        shown = rooms[:MAX_LIST]
        out: dict[str, Any] = {"count": len(rooms), "rooms": [_room_row(r, self.apartment_of.get(r.number)) for r in shown]}
        if len(rooms) > len(shown):
            out["truncated"] = f"showing {len(shown)} of {len(rooms)}"
        items = [_room_item(r) for r in rooms]
        numbers = [len(rooms)] + [r.area for r in rooms if r.area is not None]
        measured = any(r.area is not None and r.printed_area is None for r in rooms)
        return out, ToolTrace(item_ids=[it.id for it in items], items=items, numbers=numbers,
                              confidence="medium" if measured else "high")

    def list_apartments(self) -> tuple[dict, ToolTrace]:
        rows, items = [], []
        for u in self.plan.units:
            row: dict[str, Any] = {"id": f"unit-{u.letter}", "letter": u.letter, "tag": u.tag}
            if u.rooms is not None:
                row["rooms"] = [r.number for r in u.rooms]
                row["room_count"] = len(u.rooms)
            rows.append(row)
            items.append(PlanItem(id=f"unit-{u.letter}", category="unit", bbox=u.bbox, polygon=u.polygon, text=u.tag,
                                  raw="pdf text" + (" + outline" if u.polygon else "")))
        numbers = [len(rows)] + [r["room_count"] for r in rows if "room_count" in r]
        return ({"count": len(rows), "apartments": rows},
                ToolTrace(item_ids=[it.id for it in items], items=items, numbers=numbers, confidence="high"))

    def list_symbols(self, kind: str) -> tuple[dict, ToolTrace]:
        if kind not in SYMBOL_KINDS:
            return {"error": f"kind must be one of {SYMBOL_KINDS}"}, ToolTrace()
        vector = self.plan.symbols is not None
        syms = self.plan.by_kind(kind) if vector else [s for s in raster_symbols(self.rel) if s.kind == kind]
        items = [PlanItem(id=f"{kind}-{i}", category=_CATEGORY.get(kind, "fixture"),
                          cls="" if kind in _CATEGORY else kind, bbox=s.bbox,
                          raw="vector drawing" if vector else "image detector")
                 for i, s in enumerate(syms, 1)]
        shown = items[:MAX_LIST]
        out: dict[str, Any] = {"kind": kind, "count": len(items),
                               "source": "vector drawing (DXF)" if vector else "image detector on the rendered page",
                               "items": [{"id": it.id, "bbox_pt": [round(v) for v in it.bbox]} for it in shown]}
        if not vector:
            out["note"] = "Detector output, not drawn symbols: the count can be wrong; say it should be checked."
        return out, ToolTrace(item_ids=[it.id for it in items], items=items, numbers=[len(items)],
                              confidence="high" if vector else "low")


DECLARATIONS = [
    {
        "name": "pdf_lookup",
        "description": ("Standard question answered with the plan's counting rules (calibrated; prefer it when it fits). "
                        "Templates: count:apartments | count:doors | count:windows | count:toilets | count:washbasins | "
                        "count:bathtubs | count:showers = how many on the plan; label_count = rooms labelled exactly "
                        "<argument>; label_area = total area of the rooms labelled <argument>; room_area = area of room "
                        "number <argument> (e.g. 1.05); apartment_area = area of apartment <argument> (letter) without "
                        "outdoor rooms; floor_area = total area of the numbered indoor rooms (no outdoor rooms, shafts, "
                        "voids). Labels are matched exactly as printed (case-insensitive)."),
        "parameters": {"type": "object", "properties": {
            "template": {"type": "string", "enum": TEMPLATES},
            "argument": {"type": "string", "description": "room label / room number / apartment letter for: "
                                                          + ", ".join(sorted(NEEDS_ARG))},
        }, "required": ["template"]},
    },
    {
        "name": "list_rooms",
        "description": ("Rooms printed on the plan: number, printed name, area in m² (printed, or measured from the "
                        "outline), outdoor or not, counted as numbered indoor room or not, and the apartment when known. "
                        "Optionally only rooms whose name contains a text, or the rooms of one apartment."),
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "substring of the printed room name, e.g. 'Bad' or 'Zimmer'"},
            "apartment": {"type": "string", "description": "apartment letter, e.g. 'A'"},
            "include_outdoor": {"type": "boolean", "description": "default true"},
        }},
    },
    {
        "name": "list_apartments",
        "description": "Apartments tagged on the plan, with their rooms when apartment outlines are available.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "list_symbols",
        "description": ("Drawn symbols of one kind with their boxes (PDF points) and count: doors, windows and fixtures. "
                        "Says whether they come from the vector drawing or from an image detector."),
        "parameters": {"type": "object", "properties": {"kind": {"type": "string", "enum": SYMBOL_KINDS}},
                       "required": ["kind"]},
    },
    CALCULATE_ANY,
]
