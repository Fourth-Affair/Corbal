"""Answer floor-plan questions (EN / DE / FR / IT) about a vector PDF. Every value is computed here; citations are
PDF points of page 1.

Templates:
  count:<kind>   apartments, doors, windows, toilets, washbasins, bathtubs, showers
  label_count    rooms labelled '<name>'                 label_area      total area of rooms labelled '<name>'
  room_area      area of room <number>                   apartment_area  area of apartment <letter> without outdoor rooms
  floor_area     total area of numbered indoor rooms (without outdoor rooms, shafts and voids)
"""

import re
from functools import lru_cache

from app.models import Answer, PlanGraph, PlanItem
from app.pdfplan.parse import Plan, Room, Symbol, norm, parse_plan, resolve_pdf

# (template kind, pattern) in priority order: specific before generic ("Badewanne" before "Bad", labels before kinds)
_KINDS = [
    ("bathtubs", r"bathtub|badewanne|baignoire|vasch[ea] da bagno"),
    ("showers", r"shower|dusche|douche|docc[ei]a?"),
    ("washbasins", r"washbasin|wash basin|lavabo|lavabi"),
    ("toilets", r"toilet|\bwcs?\b"),
    ("windows", r"window|fenster|fen[eê]tre|finestr"),
    ("doors", r"\bdoors?\b|t[üu]ren?\b|\bportes?\b|\bport[ae]\b"),
    ("apartments", r"apartment|wohnung|appartement|logement|appartament|unit[àa] abitativ"),
]
_AREA = r"\barea\b|m²|m2|fl[äa]che|surface|superficie"
_FLOOR = r"numbered indoor|nummerierten innenr|int[ée]rieures num[ée]rot|interni numerati"
_LABEL = r"labell?ed|bezeichnung|d[ée]sign|denomina"
_QUOTES = [("«", "»"), ("„", "“"), ("“", "”"), ("‘", "’"), ("'", "'"), ('"', '"')]

_SYMBOL = {"doors": "door", "windows": "window", "toilets": "toilet", "washbasins": "washbasin",
           "bathtubs": "bathtub", "showers": "shower"}
_CATEGORY = {"door": "door", "window": "window"}
# image symbol detector classes -> our kinds (fallback when the PDF has no vector symbol source)
_DET_KIND = {"door": "door", "window": "window", "toilet": "toilet", "sink": "washbasin", "double_sink": "washbasin",
             "shower": "shower", "bathtub": "bathtub"}
_ZERO = (0.0, 0.0, 0.0, 0.0)


def _label(q: str) -> str | None:
    """Text between the outermost matching quotes ("'Cage d'escalier'" keeps its inner apostrophe)."""
    for a, b in _QUOTES:
        i, j = q.find(a), q.rfind(b)
        if i != -1 and j > i + 1:
            return q[i + 1:j].strip()
    return None


def parse_question(q: str) -> tuple[str, str | None] | None:
    """-> (template, argument) or None if the question is not a supported type."""
    low = q.casefold()
    if re.search(_FLOOR, low):
        return "floor_area", None
    if (m := re.search(r"(?:apartment|wohnung|appartement|appartamento)\s+([A-Z]{1,2})\b", q, re.I)) and re.search(_AREA, low):
        return "apartment_area", m.group(1).upper()
    if m := re.search(r"(?:room|raum|pi[eè]ce|locale)\s+([A-Z]{0,2}\d+\.\d+)", q, re.I):
        return "room_area", m.group(1)
    label = _label(q)
    if label and re.search(_LABEL, low):
        return ("label_area" if re.search(_AREA, low) else "label_count"), label
    for kind, pat in _KINDS:
        if re.search(pat, low):
            return f"count:{kind}", None
    return None


@lru_cache(maxsize=16)
def raster_symbols(rel: str) -> tuple[Symbol, ...]:
    """Fallback without vector symbols: the image symbol detector on page 1 rendered at 2x, boxes in PDF points."""
    import pypdfium2 as pdfium

    from app.cubicasa import symdet

    if not symdet.available():
        return ()
    from app.concurrency import pdfium_lock

    k = 2.0
    with pdfium_lock:  # pdfium is not thread-safe
        pdf = pdfium.PdfDocument(str(resolve_pdf(rel)))
        try:
            img = pdf[0].render(scale=k).to_pil().convert("RGB")
        finally:
            pdf.close()
    out = []
    for d in symdet.predict_image(img, conf=symdet.ITEM_CONF):
        kind = _DET_KIND.get(d["cls"])
        if kind:
            x0, y0, x1, y1 = d["bbox"]
            out.append(Symbol(kind, (x0 / k, y0 / k, x1 / k, y1 / k)))
    return tuple(out)


def _box(pts):
    return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))


def _room_item(r: Room) -> PlanItem:
    return PlanItem(id=f"room-{r.number}", category="room", cls="outdoor" if r.outdoor else "room",
                    bbox=_box(r.polygon) if r.polygon else r.bbox, polygon=r.polygon,
                    text=" ".join(x for x in (r.name, r.number, f"{r.area} m²" if r.area is not None else None) if x),
                    raw="pdf text" + (" + outline" if r.polygon else ""))


def _area_answer(rooms: list[Room], what: str, template: str) -> Answer:
    if not rooms:
        return Answer(answer=None, unit="m2", template=template, confidence="not_found",
                      explanation=f"No rooms found for {what} on this plan.")
    missing = [r.number for r in rooms if r.area is None]
    if missing:  # a partial sum would be a confident wrong answer
        return Answer(answer=None, unit="m2", template=template, confidence="not_found",
                      explanation=f"{what}: no area is printed for room(s) {', '.join(missing)} and there are no room "
                                  "outlines to measure them.", citations=[_room_item(r) for r in rooms])
    total = round(sum(r.area for r in rooms), 2)
    parts = " + ".join(f"{r.number} {r.name or '(unnamed)'} {r.area}" for r in rooms[:12])
    more = f" + {len(rooms) - 12} more" if len(rooms) > 12 else ""
    measured = any(r.printed_area is None for r in rooms)
    note = " Rooms without a printed area are measured from their outline." if measured else ""
    return Answer(answer=total, unit="m2", template=template, confidence="medium" if measured else "high",
                  explanation=f"{what}: {total} m² = {parts}{more} (net areas printed on the plan).{note}",
                  citations=[_room_item(r) for r in rooms])


def plan_graph(rel: str, image: str | None = None) -> PlanGraph:
    """The parsed plan as a PlanGraph (PDF points): room labels, apartments and any vector symbols, for the viewer."""
    plan = parse_plan(rel)
    items = [_room_item(r) for r in plan.rooms]
    items += [PlanItem(id=f"unit-{u.letter}", category="unit", bbox=u.bbox, polygon=u.polygon, text=u.tag)
              for u in plan.units]
    items += [PlanItem(id=f"{s.kind}-{i}", category=_CATEGORY.get(s.kind, "fixture"),
                       cls="" if s.kind in _CATEGORY else s.kind, bbox=s.bbox, raw="vector drawing")
              for i, s in enumerate(plan.symbols or [], 1) if s.kind not in ("other",)]
    w, h = plan.page_size
    return PlanGraph(image=image or rel, width=round(w), height=round(h), source="pdf", items=items)


def answer_question(rel: str, question: str, use_dxf: bool = True) -> Answer:
    parsed = parse_question(question)
    if parsed is None:
        return Answer(answer=None, confidence="not_found", explanation="This question type is not supported for PDF plans yet.")
    template, arg = parsed
    plan: Plan = parse_plan(rel, use_dxf)
    tid = f"pdf_{template.replace(':', '_')}"

    if template == "count:apartments":
        n = len(plan.units)
        items = [PlanItem(id=f"unit-{u.letter}", category="unit", bbox=u.bbox, polygon=u.polygon,
                          text=u.tag, raw="pdf text" + (" + outline" if u.polygon else "")) for u in plan.units]
        return Answer(answer=n, unit="count", template=tid, confidence="high" if n else "low",
                      explanation=f"{n} apartments: the plan tags " + (", ".join(u.tag for u in plan.units) or "none") + ".",
                      citations=items)

    if template.startswith("count:"):
        kind = _SYMBOL[template.split(":")[1]]
        vector = plan.symbols is not None
        syms = plan.by_kind(kind) if vector else [s for s in raster_symbols(rel) if s.kind == kind]
        items = [PlanItem(id=f"{kind}-{i}", category=_CATEGORY.get(kind, "fixture"), cls="" if kind in _CATEGORY else kind,
                          bbox=s.bbox, raw="vector drawing" if vector else "image detector")
                 for i, s in enumerate(syms, 1)]
        how = (f"one {kind} symbol each in the vector drawing ({plan.sources[-1]})" if vector
               else "image symbol detector on the rendered page; no vector symbols available")
        return Answer(answer=len(syms), unit="count", template=tid, confidence="high" if vector else "low",
                      explanation=f"{len(syms)} {template.split(':')[1]} drawn on the plan ({how}).", citations=items)

    if template in ("label_count", "label_area"):
        rooms = [r for r in plan.rooms if r.name and norm(r.name) == norm(arg)]
        if template == "label_count":
            return Answer(answer=len(rooms), unit="count", template=tid, confidence="high",
                          explanation=f"{len(rooms)} rooms are labelled '{arg}'"
                                      + (f": {', '.join(r.number for r in rooms)}." if rooms else "."),
                          citations=[_room_item(r) for r in rooms])
        return _area_answer(rooms, f"Rooms labelled '{arg}'", tid)

    if template == "room_area":
        return _area_answer([r for r in plan.rooms if r.number == arg], f"Room {arg}", tid)

    if template == "apartment_area":
        unit = next((u for u in plan.units if u.letter == arg), None)
        if unit is not None and unit.rooms is None:
            return Answer(answer=None, unit="m2", template=tid, confidence="not_found",
                          explanation=f"Apartment {arg} is tagged, but which rooms belong to it needs apartment outlines, "
                                      "which this PDF does not provide.")
        rooms = [r for r in unit.rooms if r.counts_as_indoor] if unit else []
        return _area_answer(rooms, f"Apartment {arg} (without outdoor rooms)", tid)

    if template == "floor_area":
        a = _area_answer([r for r in plan.rooms if r.counts_as_indoor],
                         "Numbered indoor rooms (without outdoor rooms, shafts and voids)", tid)
        if a.confidence == "high" and any(r.name is None and r.has_use is None for r in plan.rooms):
            a.confidence = "medium"  # unnamed rooms could not be classified without geometry
        return a

    return Answer(answer=None, confidence="not_found", explanation="Unsupported question.")
