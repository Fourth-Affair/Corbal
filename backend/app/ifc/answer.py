"""Answer natural-language questions about an IFC model, deterministically (no API key needed).

question -> Query (operation, element classes, storey, space, filters, quantity, grouping) -> tools -> Answer.
The parser is a small multilingual (EN / DE / FI / SV, some FR / NL) vocabulary over IFC concepts, not a list of
dataset phrasings: "how many <thing> [on <storey>]", "<area|volume|length> of <room|things> [on <storey>]",
"which <thing> types ... with quantities", "materials of <thing>", "largest <room|thing>", "fire rating of
<thing>", "software / schema", "project / location", "storey heights", "systems". Anything it cannot map to an
operation returns not_found instead of a guess.

Citations: one PlanItem per cited IFC entity, id = the IFC GlobalId, raw = the IFC class, text = its name,
category = room / door / window / wall / floor (storeys) / fixture (everything else), bbox = (0, 0, 0, 0): an IFC
element has no position on a 2D page, so the GlobalId is the address a viewer selects.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from app.ifc import tools as T
from app.ifc.model import Element, IfcModel, load_model
from app.models import Answer, PlanItem

MAX_CITATIONS = 200
MAX_ROWS = 40

_COUNT = r"how many|number of|count\b|wie ?viele|anzahl|kuinka (monta|paljon)|montako|hur m[aå]nga|antal|combien|nombre de|hoeveel"
_AREA = r"\barea\b|\bareas\b|how big|how large|size of|wie gro(ss|ß)|kuinka (iso|suuri)|hur stor|m²|\bm2\b|sq\.? ?m|square met|fl[aä]che|pinta-?ala|\byta\b|surface|superficie|oppervlak"
_VOLUME = r"volume|m³|\bm3\b|\bcu\.? ?m\b|\bcubic\b|volumen|tilavuus|kubik"
_LENGTH = r"\blength|\blong\b|l[aä]nge|pituus|l[aä]ngd|longueur|lengte"
_GROSS = r"\bgross\b|brutto|\bbgf\b|bruttoala|gfa\b"
_NET = r"\bnet\b|netto|\bnrf\b|\bnfa\b|nettoala|usable"
_TOTAL = r"\btotal|\ball\b|\bsum\b|combined|overall|altogether|gesamt|summe|insgesamt|yhteens[aä]|kaikki|totalt|sammanlagt"
_EXTERNAL = r"exterior|external|outside|outer|fa[cç]ade|au(ss|ß)en|ulko|yttervägg|ytter"
_INTERNAL = r"interior|internal|inside|inner|partition|innen|sis[aä]|inner"
_LOADBEARING = r"load[- ]?bearing|structural walls?|tragend|kantava|b[aä]rande"
_PER_STOREY = r"(each|per|every|by|for all|across)\s+(the\s+)?(building\s+)?(floor|level|storey|story)s?|by (floor|level|storey)|floor[- ]by[- ]floor|level[- ]by[- ]level|geschossweise|je geschoss|kerroksittain"
_TYPES = r"\btypes?\b|\bkinds?\b|\bsorts?\b|typen|tyypi\w*|\btyp\b|varieties"
_LIST = r"\blist\b|\bwhich\b|\bwhat are\b|\bshow\b|\bnames?\b|\bwelche\b|\bmitkä\b|\bvilka\b|\bgive me\b|breakdown|enumerate"
_MATERIAL = r"materials?|made of|constructed (of|from|with)|werkstoff|baustoff|materiaali\w*|\bmaterial\w*"
_FIRE = r"fire[- ]?(rating|resistance|protection|class)|brandschutz|feuerwiderstand|palonkest\w*|paloluokka|brandklass"
_META = r"software|application|authoring|originating|ifc (version|schema|standard)|\bschema\b|modell?ing standards?|created (with|by|in)|exported|which (tool|program)|file format|mvd|model view"
_PROJECT = r"project (info\w*|name|number|data|details?)|\blocation\b|\baddress\b|latitude|longitude|\bsite\b|where is|geo-?ref|coordinates|standort|sijainti"
_GUID = r"\bguid\b|globalid|global id|\bifc id\b"
_SYSTEM = r"\bsystems?\b|anlagen?|j[aä]rjestelm\w*"
_EXTREME = {
    "largest": r"\blargest|\bbiggest|\bmaximum\b|\bmax\b|\bmost spacious|gr[oö](ss|ß)te[nrs]?|suurin|st[oö]rsta",
    "smallest": r"\bsmallest|\bminimum\b|\bmin\b|\btiniest|kleinste[nrs]?|pienin|minsta",
    "widest": r"\bwidest|\bbroadest|breiteste[nrs]?|levein",
    "narrowest": r"\bnarrowest|schmalste[nrs]?|kapein",
    "tallest": r"\btallest|\bhighest|h[oö]chste[nrs]?|korkein",
    "lowest": r"\blowest|\bshortest|niedrigste[nrs]?|matalin",
    "longest": r"\blongest|l[aä]ngste[nrs]?|pisin",
}
_QUOTES = [("'", "'"), ('"', '"'), ("„", "“"), ("“", "”"), ("‘", "’"), ("«", "»")]


def _quoted(q: str) -> list[str]:
    out = []
    for a, b in _QUOTES:
        pat = re.escape(a) + r"(.+?)" + re.escape(b)
        if a == "'":  # apostrophes inside words ("building's") are not quotes
            pat = r"(?<![A-Za-z])'(.+?)'(?![A-Za-z])"
        out += [x.strip() for x in re.findall(pat, q) if x.strip()]
    return out


def _has(pat: str, text: str) -> bool:
    return re.search(pat, text, re.I) is not None


# ------------------------------------------------------------------------------------------------ citations

_CATEGORY = {"IfcSpace": "room", "IfcDoor": "door", "IfcWindow": "window", "IfcWall": "wall", "IfcBuildingStorey": "floor"}


def citation(m: IfcModel, gid: str) -> PlanItem:
    e = m.get(gid)
    if e is None:  # project / site / building / system records
        rec = next((x for x in [m.project, *m.sites, *m.buildings] if x.get("gid") == gid), None)
        grp = next((g for g in m.groups if g.gid == gid), None)
        name = (rec or {}).get("name") or (grp.name if grp else None)
        raw = "IfcProject" if rec is m.project else "IfcSite" if rec in m.sites else "IfcBuilding" if rec else (grp.ifc_class if grp else "Ifc")
        return PlanItem(id=gid, category="unit" if raw == "IfcBuilding" else "fixture", cls=raw, bbox=(0.0, 0.0, 0.0, 0.0),
                        text=name, raw=raw)
    cat = next((c for k, c in _CATEGORY.items() if k in e.classes), "fixture")
    return PlanItem(id=gid, category=cat, cls=e.kind, bbox=(0.0, 0.0, 0.0, 0.0), text=T.label(e), raw=e.ifc_class)


def _answer(m: IfcModel, value, unit, template, confidence, explanation, ids) -> Answer:
    ids = list(dict.fromkeys(i for i in ids if i))
    if len(ids) > MAX_CITATIONS:
        explanation += f" (citing the first {MAX_CITATIONS} of {len(ids)} elements)"
    cites = [citation(m, g) for g in ids[:MAX_CITATIONS]]
    if value is not None and not cites and confidence in ("high", "medium"):
        confidence = "low"  # uncited answers are flagged
    return Answer(answer=value, unit=unit, template=template, confidence=confidence, explanation=explanation,
                  citations=cites)


def _not_found(template: str | None, why: str, unit: str | None = None) -> Answer:
    return Answer(answer=None, unit=unit, template=template, confidence="not_found", explanation=why)


def _fmt(v: float | None, nd: int = 2) -> str:
    if v is None:
        return "?"
    return f"{v:,.{nd}f}".replace(",", " ")


def _dms(deg: float) -> str:
    sign = "-" if deg < 0 else ""
    deg = abs(deg)
    d = int(deg)
    mnt = int((deg - d) * 60)
    sec = (deg - d - mnt / 60) * 3600
    return f"{sign}{d}° {mnt}' {sec:.2f}\""


def _rows_text(lines: list[str]) -> str:
    if len(lines) > MAX_ROWS:
        return "\n".join(lines[:MAX_ROWS] + [f"... and {len(lines) - MAX_ROWS} more"])
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ query


@dataclass
class Query:
    q: str
    low: str
    quoted: list[str]
    storey: Element | None = None
    storey_how: str = ""
    classes: tuple[str, ...] = ()
    predefined: tuple[str, ...] | None = None
    name_pattern: str | None = None
    words: str = ""  # the words that named the classes
    function: tuple[str, str] | None = None  # (words, space-name regex) for room functions ("bedrooms")
    external: bool | None = None
    load_bearing: bool | None = None
    per_storey: bool = False
    extra: dict = field(default_factory=dict)


def parse(m: IfcModel, q: str) -> Query:
    low = q.casefold()
    qq = Query(q=q, low=low, quoted=_quoted(q))
    for text in [*qq.quoted, q]:
        s, how = T.resolve_storey(m, text)
        if s is not None:
            qq.storey, qq.storey_how = s, how
            break
    qq.per_storey = _has(_PER_STOREY, low) and qq.storey is None
    # strip what is not an element word before class matching: storey names, "floor area", "floor plan", ordinals
    rest = low
    for t in [s for s in (qq.storey.name if qq.storey else None, qq.storey.long_name if qq.storey else None) if s]:
        rest = rest.replace(t.casefold(), " ")
    for x in qq.quoted:
        rest = rest.replace(x.casefold(), " ")
    rest = re.sub(_PER_STOREY, " ", rest)
    rest = re.sub(r"floor (area|plan|space|finish|height)s?|(ground|basement|first|second|third|fourth|fifth|top|upper|lower|\d+(st|nd|rd|th)?)\s+(floor|level|storey|story)s?"
                  r"|\b(on|at|of|in) (the )?(floor|level|storey|story)\b|\b(level|floor|storey|ebene|kerros)\s+-?\d+\b|in the model|of the model|this model|the model", " ", rest)
    mc = T.match_classes(rest)
    if mc:
        qq.classes, qq.predefined, qq.name_pattern, qq.words, cpos = mc
    fn = T.match_space_function(rest)
    if fn and (not mc or qq.classes == ("IfcSpace",) or fn[2] <= cpos):  # "bedrooms", "stairwells": rooms by name
        qq.function = fn[:2]
        qq.classes, qq.predefined, qq.name_pattern, qq.words = ("IfcSpace",), None, fn[1], fn[0]
    elif qq.classes == ("IfcSpace",) and not qq.name_pattern:
        # "how many X-RAY rooms": the word in front of rooms / spaces names them
        mw = re.search(r"([a-z0-9][a-z0-9\-/]*)\s+(?:rooms?|spaces?)\b", rest)
        stop = {"many", "the", "all", "of", "largest", "smallest", "biggest", "which", "individual", "available", "total", "are", "total", "number", "these", "which", "what", "any", "individual",
                "different", "other", "and", "with", "how", "its", "their", "each", "every", "a", "an", "in", "on", "s"}
        if mw and mw.group(1) not in stop and not re.fullmatch(r"\d+|\w+est", mw.group(1)):
            qq.name_pattern = re.escape(mw.group(1)).replace("\\-", "[- ]?")
            qq.function = (mw.group(1), qq.name_pattern)
    if _has(_EXTERNAL, low):
        qq.external = True
    elif _has(_INTERNAL, low):
        qq.external = False
    if _has(_LOADBEARING, low):
        qq.load_bearing = not _has(r"non[- ]?load|not load|nicht tragend|ei kantava", low)
    return qq


def _selection(m: IfcModel, qq: Query, storey: bool = True) -> list[Element]:
    """Elements of the asked classes. When the model has none of that class at all, fall back to elements named like
    it (exporters often write generic IfcBuildingElementProxy / IfcDistributionElement), then to spaces named like it
    ('toilets' in a model without sanitary terminals). qq.extra['fallback'] says which fallback was used."""
    st = qq.storey if storey else None
    els = T.select(m, qq.classes, st, predefined=qq.predefined, name_pattern=qq.name_pattern,
                   external=qq.external, load_bearing=qq.load_bearing)
    if els or not qq.classes or qq.classes == ("IfcSpace",) or T.select(m, qq.classes, predefined=qq.predefined):
        return els
    hint = qq.name_pattern or next((T.NAME_HINTS[c] for c in qq.classes if c in T.NAME_HINTS), None)
    if hint:
        els = [e for e in T.select_by_name(m, hint, st) if qq.external is None or T.is_external(e) is qq.external]
        if els:
            qq.extra["fallback"] = (f"the model has no {qq.classes[0]}; counted elements of other classes whose name or "
                                    f"type matches '{qq.words}'")
            return els
    if qq.name_pattern or qq.words:
        pat = qq.name_pattern or re.escape(qq.words.rstrip("s")[:6])
        els = T.select_by_name(m, pat, st, spaces_only=True)
        if els:
            qq.extra["fallback"] = f"the model has no {qq.classes[0]}; these are the spaces named like '{qq.words}'"
    return els


def _what(qq: Query) -> str:
    s = T.class_label(qq.classes[0]) if qq.classes else "elements"
    if qq.name_pattern and qq.classes == ("IfcSpace",) and qq.function:
        s = f"spaces named like '{qq.function[0]}'"
    elif qq.name_pattern:
        s = f"{qq.words} ({s})"
    if qq.external is True:
        s = "exterior " + s
    elif qq.external is False:
        s = "interior " + s
    if qq.load_bearing is not None:
        s = ("load-bearing " if qq.load_bearing else "non-load-bearing ") + s
    return s


def _where(qq: Query) -> str:
    return f" on storey '{qq.storey.name}'" if qq.storey else ""


def _filter_note(m: IfcModel, qq: Query) -> str:
    """When IsExternal / LoadBearing filters are used, say how many candidates had no such property."""
    if qq.external is None and qq.load_bearing is None:
        return ""
    base = T.select(m, qq.classes, qq.storey, predefined=qq.predefined, name_pattern=qq.name_pattern)
    unknown = [e for e in base if (qq.external is not None and T.is_external(e) is None)
               or (qq.load_bearing is not None and T.is_load_bearing(e) is None)]
    note = ""
    if qq.external is not None:
        srcs = Counter(T.external_evidence(e)[1] for e in base if T.external_evidence(e)[0] is not None)
        if srcs and set(srcs) != {"IsExternal"}:
            note += " Exterior/interior from: " + ", ".join(f"{k} ({v})" for k, v in srcs.most_common(4)) + "."
    if unknown:
        note += f" {len(unknown)} of {len(base)} candidates say nothing about being exterior/load-bearing and are left out."
    return note


# ------------------------------------------------------------------------------------------------ operations


def _op_meta(m: IfcModel, qq: Query) -> Answer:
    r = T.metadata(m)
    h = m.header
    apps = "; ".join(" ".join(str(x) for x in (a.get("developer"), a.get("name"), a.get("version")) if x) for a in m.applications)
    lines = [f"IFC schema: {m.schema} (Industry Foundation Classes, ISO 16739)"]
    if apps:
        lines.append(f"Authoring application (IfcApplication): {apps}")
    if h.get("originating_system"):
        lines.append(f"Originating system (file header): {h['originating_system']}")
    if h.get("preprocessor_version"):
        lines.append(f"Preprocessor (IFC exporter): {h['preprocessor_version']}")
    if h.get("description"):
        lines.append(f"File description / model view definition: {h['description']}")
    if h.get("time_stamp"):
        lines.append(f"File time stamp: {h['time_stamp']}")
    text = "\n".join(lines)
    return _answer(m, text, None, "ifc:metadata", "high", text, r.ids)


def _op_project(m: IfcModel, qq: Query) -> Answer:
    r = T.metadata(m)
    p = m.project
    lines = []
    if p:
        lines.append(f"Project: {p.get('name') or '-'}" + (f" ({p['long_name']})" if p.get("long_name") else "")
                     + (f", description: {p['description']}" if p.get("description") else "")
                     + (f", phase: {p['phase']}" if p.get("phase") else ""))
    for b in m.buildings:
        occ = T.prop(T.Element(gid="", ifc_class="", kind="", classes=[], props=b.get("props", {})), "OccupancyType")
        lines.append(f"Building: {b.get('name') or '-'}" + (f" ({b['long_name']})" if b.get("long_name") else "")
                     + (f", type: {b['object_type']}" if b.get("object_type") else "")
                     + (f", occupancy: {occ}" if occ else "") + (f", address: {b['address']}" if b.get("address") else ""))
    for s in m.sites:
        loc = []
        if s.get("latitude") is not None and s.get("longitude") is not None:
            loc.append(f"latitude {s['latitude']:.6f}° ({_dms(s['latitude'])}), longitude {s['longitude']:.6f}° ({_dms(s['longitude'])})")
        if s.get("elevation") is not None:
            loc.append(f"elevation {s['elevation']:.2f} m")
        if s.get("address"):
            loc.append(f"address: {s['address']}")
        lines.append(f"Site: {s.get('name') or '-'}" + (f" ({'; '.join(loc)})" if loc else " (no location data)"))
    if not lines:
        return _not_found("ifc:project", "The model has no project, site or building records.")
    text = "\n".join(lines)
    return _answer(m, text, None, "ifc:project", "high", text, r.ids)


def _op_storeys(m: IfcModel, qq: Query, count_only: bool) -> Answer:
    r = T.storeys(m)
    if not r.rows:
        return _not_found("ifc:storeys", "The model has no IfcBuildingStorey.", "count" if count_only else None)
    lines = [f"- {x['name']}: elevation {_fmt(x['elevation'])} m" + (f", height {_fmt(x['height'])} m" if x["height"] is not None else "")
             for x in r.rows]
    text = f"{len(r.rows)} storeys (bottom-up):\n" + "\n".join(lines) + "\nHeights = elevation difference to the next storey."
    if count_only:
        return _answer(m, len(r.rows), "count", "ifc:count_storeys", "high", text, r.ids)
    if qq.storey is not None:
        row = next(x for x in r.rows if x["gid"] == qq.storey.gid)
        if _has(r"elevation|niveau|korko|h[oö]henlage|level of", qq.low) and row["elevation"] is not None:
            return _answer(m, round(row["elevation"], 3), "m", "ifc:storey_elevation", "high",
                           f"Storey '{row['name']}' has elevation {_fmt(row['elevation'])} m (IfcBuildingStorey.Elevation).", [row["gid"]])
        if row["height"] is not None:
            return _answer(m, round(row["height"], 3), "m", "ifc:storey_height", "high",
                           f"Storey '{row['name']}' is {_fmt(row['height'])} m high ({row['height_source']}).", [row["gid"]])
    return _answer(m, text, None, "ifc:storeys", "high", text, r.ids)


def _op_count(m: IfcModel, qq: Query) -> Answer:
    if qq.classes == ("IfcBuildingStorey",):
        return _op_storeys(m, qq, count_only=True)
    if not qq.classes and _has(r"how many (\w+ )?buildings|number of (\w+ )?buildings|wie ?viele geb[aä]ude|montako rakennusta|kuinka monta rakennusta", qq.low):
        bs = m.buildings
        text = f"The model contains {len(bs)} IfcBuilding record(s)" + (": " + ", ".join(str(b.get("name") or "-") for b in bs[:20]) if bs else "") + "."
        return _answer(m, len(bs), "count", "ifc:count_buildings", "high", text, [b["gid"] for b in bs])
    if not qq.classes:
        return _not_found("ifc:count", "Could not tell which kind of element to count.", "count")
    els = _selection(m, qq)
    what, where = _what(qq), _where(qq)
    if qq.per_storey:
        g = T.group_count(m, els, "storey")
        lines = [f"- {row['key']}: {row['count']} ({row['share']:.1f}%)" for row in g.rows]
        text = f"{len(els)} {what} in total; by storey:\n" + "\n".join(lines)
        if qq.classes == ("IfcSpace",):
            for row in g.rows:
                text += f"\n{row['key']}: " + ", ".join(
                    f"{T.label(m.get(x))} {_fmt(T.space_area(m.get(x))[0])} m²" for x in row["ids"][:30])
        return _answer(m, len(els), "count", "ifc:count_by_storey", "high", text, [e.gid for e in els])
    conf = "high" if not qq.name_pattern and "fallback" not in qq.extra else "medium"
    if not els:
        n_all = len(T.select(m, qq.classes))
        text = f"No {what}{where} in this model" + (f" ({n_all} in the whole model)." if qq.storey and n_all else ".")
        text += _filter_note(m, qq)
        return _answer(m, 0, "count", "ifc:count", "medium", text, [qq.storey.gid] if qq.storey else [])
    g = T.group_count(m, els, "type")
    detail = ", ".join(f"{row['key']}: {row['count']}" for row in g.rows[:8])
    text = f"There are {len(els)} {what}{where} (IFC class {', '.join(qq.classes)}" \
           + (f", names matching '{qq.words}'" if qq.name_pattern and qq.classes != ("IfcSpace",) else "") + f"). By type: {detail}" \
           + (" ..." if len(g.rows) > 8 else "") + "." + _filter_note(m, qq)
    if "fallback" in qq.extra:
        text += f" Note: {qq.extra['fallback']}."
    if qq.classes == ("IfcSpace",) and qq.function:
        text += " Names: " + ", ".join(T.label(e) for e in els[:20]) + ("..." if len(els) > 20 else "")
    return _answer(m, len(els), "count", "ifc:count", conf, text, [e.gid for e in els])


def _space_query_text(qq: Query) -> str:
    """The part of the question that names a room: quoted text that is not the storey, else the words after 'room'."""
    for x in qq.quoted:
        if not (qq.storey and T.norm(x) in (T.norm(qq.storey.name), T.norm(qq.storey.long_name))):
            return x
    m = re.search(r"\b(?:room|space|raum|huone|tila|rum)\s+(?:number\s+|no\.?\s+|nr\.?\s+)?([\w.\-/]+(?:\s+[\w.\-/]+){0,3})", qq.q, re.I)
    return m.group(1) if m else ""


def _op_space_value(m: IfcModel, qq: Query, what: str) -> Answer | None:
    """Area / volume / height of one named space ('net floor area of the room JAN. (1A11)')."""
    text = _space_query_text(qq)
    if text:
        hits = T.find_spaces(m, text, qq.storey) or T.find_spaces(m, text)
    else:  # no quotes, no "room X": accept only spaces whose number appears in the question ("kitchen A103")
        text = qq.q
        bare = re.sub(r"\b(level|floor|storey|story|ebene|kerros|geschoss|etage)\s+\S+", " ", qq.q, flags=re.I)
        hits = [e for e in (T.find_spaces(m, bare, qq.storey) or T.find_spaces(m, bare))
                if e.name and re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]|\d{3,}", e.name) and f" {T.norm(e.name)} " in f" {T.norm(bare)} "]
        if qq.storey and any(T.norm(e.name) == T.norm(qq.storey.name) for e in hits):
            hits = []
    if not hits:
        return None
    gross = _has(_GROSS, qq.low)
    unit = {"area": "m2", "volume": "m3"}.get(what, "m")
    vals = []
    for e in hits:
        if what == "area":
            v, src = T.space_area(e, "gross" if gross else "net")
        else:
            v, src = T.quantity(e, what)
        vals.append((v, src, e))
    known = [(v, s, e) for v, s, e in vals if v is not None]
    if not known:
        return _not_found(f"ifc:space_{what}", f"Space {T.label(hits[0])} has no {what} in the model.", unit)
    distinct = {round(v, 2) for v, _, _ in known}
    v, src, e = known[0]
    st = m.storey(e.storey)
    if len(distinct) > 1:
        lines = [f"- {T.label(x)} ({m.storey(x.storey).name if m.storey(x.storey) else '-'}): {_fmt(val)} {unit}" for val, _, x in known]
        return _answer(m, None, unit, f"ifc:space_{what}", "low",
                       f"{len(known)} spaces match '{text}' with different values; which one?\n" + "\n".join(lines),
                       [x.gid for _, _, x in known]) .model_copy(update={"clarification_needed": True, "confidence": "not_found"})
    label = {"area": ("gross" if gross else "net") + " floor area"}.get(what, what)
    conf = "high" if src in ("NetFloorArea", "GrossFloorArea", "NetVolume", "GrossVolume", "ClearHeight", "Height") else "medium"
    expl = f"The {label} of {T.label(e)}" + (f" on '{st.name}'" if st else "") + f" is {_fmt(v)} {unit} (source: {src}"
    if src == "geometry footprint":
        expl += ", the plan area of the IfcSpace body shape"
        other, osrc = next(((val, k) for k, val in e.props.items() if e.dims.get(k) == "area" and isinstance(val, float)), (None, None))
        if other:
            expl += f"; the authoring tool's property {osrc} says {_fmt(other)} {unit}"
    expl += ")."
    if len(hits) > 1:
        expl += f" {len(hits)} spaces share this name and value."
    return _answer(m, round(v, 2), unit, f"ifc:space_{what}", conf, expl, [x.gid for _, _, x in known])


def _op_space_total(m: IfcModel, qq: Query) -> Answer:
    gross = _has(_GROSS, qq.low)
    which = "gross" if gross else "net"
    name_pat = qq.name_pattern if qq.classes == ("IfcSpace",) else None
    if qq.per_storey or (qq.storey is None and _has(r"by (level|floor|storey)|each (level|floor|storey)|per (level|floor|storey)", qq.low)):
        lines, ids, grand = [], [], 0.0
        for s in m.storeys:
            r = T.space_table(m, s, which, name_pat)
            if not r.rows:
                continue
            lines.append(f"- {s.name}: {_fmt(r.value)} m² ({len(r.rows)} spaces)")
            ids += r.ids
            grand += r.value or 0
        if not lines:
            return _not_found("ifc:area_by_storey", "The model has no spaces with areas.", "m2")
        text = f"Total {which} floor area of the spaces: {_fmt(grand)} m²; by storey:\n" + "\n".join(lines)
        return _answer(m, round(grand, 2), "m2", "ifc:area_by_storey", "medium", text, ids)
    if _has(r"function|usage|\buse\b|utili[sz]|allocat|program|nutzung|k[aä]ytt", qq.low):
        return _op_space_by_function(m, qq, which)
    r = T.space_table(m, qq.storey, which, name_pat)
    if not r.rows or r.value is None:
        named = f" named like '{qq.function[0]}'" if name_pat and qq.function else ""
        return _not_found("ifc:space_area_total", f"No spaces{named} with an area{_where(qq)}.", "m2")
    whom = f"the {len(r.rows)} {'spaces named like ' + repr(qq.function[0]) if name_pat and qq.function else 'spaces'}{_where(qq) or ' in the model'}"
    text = f"Total {which} floor area of {whom}: {_fmt(r.value)} m² (sum of each space's area; sources: {r.source}). {r.note}".strip()
    if len(r.rows) <= MAX_ROWS:
        text += "\n" + "\n".join(f"- {row['name']}: {_fmt(row['area'])} m²" for row in r.rows)
    building_qto = next((b["qto"].get("GrossFloorArea") for b in m.buildings if b.get("qto", {}).get("GrossFloorArea")), None)
    if gross and qq.storey is None and building_qto:
        text += f"\nThe building record itself states GrossFloorArea = {_fmt(building_qto)} m²."
    conf = "high" if "geometry" not in r.source and not r.note else "medium"
    return _answer(m, r.value, "m2", "ifc:space_area_total", conf, text, r.ids)


def _op_space_by_function(m: IfcModel, qq: Query, which: str = "net") -> Answer:
    """Spaces grouped by their name / function (LongName, else Name) with summed areas and shares."""
    groups: dict[str, list[tuple[float, str]]] = {}
    for e in T.spaces(m, qq.storey):
        a, _ = T.space_area(e, which)
        key = (e.long_name or e.name or "(unnamed)").strip()
        groups.setdefault(key, []).append((a or 0.0, e.gid))
    if not groups:
        return _not_found("ifc:space_by_function", f"No spaces{_where(qq)} in this model.")
    total = sum(a for v in groups.values() for a, _ in v)
    rows = sorted(groups.items(), key=lambda kv: -sum(a for a, _ in kv[1]))
    lines = [f"- {k}: {len(v)} space(s), {_fmt(sum(a for a, _ in v))} m² ({100 * sum(a for a, _ in v) / total:.1f}%)" for k, v in rows]
    text = (f"{sum(len(v) for v in groups.values())} spaces{_where(qq)}, {len(groups)} distinct names/functions, "
            f"total {which} area {_fmt(total)} m²:\n" + _rows_text(lines))
    return _answer(m, text, None, "ifc:space_by_function", "high", text, [g for v in groups.values() for _, g in v])


def _op_space_list(m: IfcModel, qq: Query) -> Answer:
    gross = _has(_GROSS, qq.low)
    name_pat = qq.name_pattern if qq.classes == ("IfcSpace",) else None
    storeys = [qq.storey] if qq.storey else [s for s in m.storeys if T.spaces(m, s)] or [None]
    parts, ids, grand, n = [], [], 0.0, 0
    for s in storeys:
        r = T.space_table(m, s, "gross" if gross else "net", name_pat)
        if not r.rows:
            continue
        head = f"{s.name if s else 'Spaces'}: {len(r.rows)} spaces, total {_fmt(r.value)} m²"
        lines = [f"- {row['name']}: {_fmt(row['area'])} m²" for row in r.rows]
        parts.append(head + "\n" + _rows_text(lines))
        ids += r.ids
        grand += r.value or 0
        n += len(r.rows)
    if not parts:
        return _not_found("ifc:space_list", f"No spaces{_where(qq)} in this model.")
    text = "\n\n".join(parts)
    if len(parts) > 1:
        text += f"\n\nAll storeys: {n} spaces, {_fmt(grand)} m²"
    return _answer(m, text, None, "ifc:space_list", "high", text, ids)


def _op_quantity_total(m: IfcModel, qq: Query, what: str) -> Answer:
    """Sum of a quantity over a selection: 'total volume of all floor slabs', 'length of exterior walls'."""
    if not qq.classes and _has(r"concrete|beton", qq.low):
        els = [e for e in m.elements if any(re.search(r"concrete|beton|betoni|stb\b|c\d\d/\d\d", x, re.I) for x in e.materials)
               and e.kind != "IfcSpace"]
        what_s = "elements with a concrete material"
    else:
        if not qq.classes:
            return _not_found(f"ifc:sum_{what}", f"Could not tell which elements to sum the {what} of.")
        els = _selection(m, qq)
        what_s = _what(qq)
        if _has(r"concrete|beton", qq.low):
            conc = [e for e in els if any(re.search(r"concrete|beton|betoni|stb\b|c\d\d/\d\d", x, re.I) for x in e.materials)]
            if conc:
                els, what_s = conc, f"{what_s} with a concrete material"
            elif els:
                qq.extra["fallback"] = f"no {what_s} has a material assignment naming concrete, so this is the volume of all of them"
    unit = {"area": "m2", "volume": "m3"}.get(what, "m")
    if not els:
        return _not_found(f"ifc:sum_{what}", f"No {what_s}{_where(qq)} in this model." + _filter_note(m, qq), unit)
    r = T.sum_quantity(m, els, what)
    if r.value is None:
        return _not_found(f"ifc:sum_{what}", f"None of the {len(els)} {what_s}{_where(qq)} carries a {what} quantity or property.", unit)
    conf = "high" if not r.note and "geometry" not in r.source else "medium"
    text = f"Total {what} of the {len(r.ids)} {what_s}{_where(qq)}: {_fmt(r.value)} {unit} (sources: {r.source}). {r.note}".strip()
    text += _filter_note(m, qq)
    if "fallback" in qq.extra:
        text += f" Note: {qq.extra['fallback']}."
        conf = "medium"
    if qq.per_storey or _has(r"by type|per type|each type|breakdown", qq.low):
        by_t: dict[str, float] = {}
        for row in r.rows:
            e = m.get(row["gid"])
            k = T.type_label(e) if e else "?"
            by_t[k] = by_t.get(k, 0) + row["value"]
        text += "\nBy type:\n" + "\n".join(f"- {k}: {_fmt(v)} {unit} ({100 * v / r.value:.1f}%)" for k, v in
                                           sorted(by_t.items(), key=lambda kv: -kv[1])[:MAX_ROWS])
    if qq.per_storey or _has(r"by (floor|level|storey)|per (floor|level|storey)|each (floor|level|storey)", qq.low):
        by = {}
        for row in r.rows:
            e = m.get(row["gid"])
            s = m.storey(e.storey) if e else None
            by[s.name if s else "(no storey)"] = by.get(s.name if s else "(no storey)", 0) + row["value"]
        text += "\n" + "\n".join(f"- {k}: {_fmt(v)} {unit}" for k, v in by.items())
    return _answer(m, round(r.value, 2), unit, f"ifc:sum_{what}", conf, text, r.ids)


def _op_types(m: IfcModel, qq: Query) -> Answer:
    if not qq.classes:
        return _not_found("ifc:types", "Could not tell which kind of element the question is about.")
    els = _selection(m, qq)
    what = _what(qq)
    if not els:
        return _answer(m, f"No {what}{_where(qq)} found in this model.", None, "ifc:types", "medium",
                       f"No {what}{_where(qq)} found in this model." + _filter_note(m, qq), [])
    r = T.type_table(m, els)
    lines = []
    for row in r.rows:
        dims = []
        for q in ("width", "height", "thickness"):
            if q in row and not (q == "thickness" and "IfcWindow" in qq.classes + ("",) or q == "thickness" and "IfcDoor" in qq.classes):
                vals = row[q]
                dims.append(f"{q} {', '.join(_fmt(v * 1000, 0) for v in vals[:3])} mm" + ("..." if len(vals) > 3 else ""))
        mat = f"; materials: {', '.join(row['materials'])}" if row.get("materials") and _has(_MATERIAL, qq.low) else ""
        share = 100 * row["count"] / len(els)
        lines.append(f"- {row['type']}: {row['count']} ({share:.1f}%)" + (f" ({'; '.join(dims)})" if dims else "") + mat)
        if _has(r"composition|layers?|build-?up|aufbau|kerros", qq.low):
            e0 = m.get(row["ids"][0])
            if e0 is not None and e0.layers:
                lines += [f"    {n}: {_fmt((t or 0) * 1000, 0)} mm" for n, t in e0.layers]
    by_storey = ""
    if qq.per_storey or _has(r"distribut|by (level|floor|storey)|per (level|floor|storey)", qq.low):
        g = T.group_count(m, els, "storey")
        by_storey = "\nBy storey:\n" + "\n".join(f"- {row['key']}: {row['count']} ({row['share']:.1f}%)" for row in g.rows)
    by_class = ""
    if len(qq.classes) > 1:
        g = T.group_count(m, els, "class")
        by_class = "By class:\n" + "\n".join(f"- {T.class_label(row['key'])}: {row['count']} ({row['share']:.1f}%)" for row in g.rows) + "\nBy type:\n"
    text = f"{len(els)} {what}{_where(qq)} of {len(r.rows)} types:\n" + by_class + _rows_text(lines) + by_storey + _filter_note(m, qq)
    if "fallback" in qq.extra:
        text += f"\nNote: {qq.extra['fallback']}."
    return _answer(m, text, None, "ifc:types", "high" if "fallback" not in qq.extra else "medium", text, [e.gid for e in els])


def _op_materials(m: IfcModel, qq: Query) -> Answer:
    els = _selection(m, qq) if qq.classes else [e for e in m.elements if e.kind != "IfcSpace"]
    what = _what(qq)
    if not els:
        return _answer(m, f"No {what}{_where(qq)} found in this model.", None, "ifc:materials", "medium",
                       f"No {what}{_where(qq)} found in this model." + _filter_note(m, qq), [])
    r = T.materials(m, els)
    if not r.rows:
        text = f"The {len(els)} {what}{_where(qq)} have no material assignments in the model."
        return _answer(m, text, None, "ifc:materials", "medium", text, [e.gid for e in els])
    lines = []
    for row in r.rows:
        types = sorted({T.type_label(m.get(g)) for g in row["ids"] if m.get(g)})
        lines.append(f"- {row['material']}: {row['count']} element(s)" + (f"; types: {', '.join(types[:5])}" if types else ""))
    layered = [e for e in els if e.layers]
    if layered and (_has(r"layer|insulation|d[aä]mm|eriste|thick|composition|each|per|by type|types?\b", qq.low)
                    or len({T.type_label(e) for e in layered}) <= 12):
        seen = {}
        for e in layered:
            key = T.type_label(e)
            total = sum(t or 0 for _, t in e.layers)
            seen.setdefault(key, " | ".join(f"{n} {_fmt((t or 0) * 1000, 0)} mm" for n, t in e.layers)
                            + f" (total {_fmt(total * 1000, 0)} mm)")
        lines.append("Layer build-ups:")
        lines += [f"- {k}: {v}" for k, v in seen.items()]
    text = f"{len(r.rows)} material(s) on {len(els)} {what}{_where(qq)}:\n" + _rows_text(lines) + _filter_note(m, qq)
    return _answer(m, text, None, "ifc:materials", "high", text, r.ids)


def _op_thickness(m: IfcModel, qq: Query) -> Answer:
    if _has(r"insulation|d[aä]mm|eriste|isolering|isolation", qq.low):
        els = _selection(m, qq) if qq.classes else [e for e in m.elements if e.layers]
        found = {}
        for e in els:
            for n, t in e.layers:
                if t and re.search(r"insul|d[aä]mm|eriste|isoler|wool|wolle|villa|eps|xps|pir|pur|mineral", n, re.I):
                    found.setdefault((n, round(t, 4)), []).append(e.gid)
        if not found:
            return _not_found("ifc:insulation", f"No insulation layer in the material layers of the {_what(qq)}.")
        lines = [f"- {n}: {_fmt(t * 1000, 0)} mm ({len(ids)} element(s))" for (n, t), ids in found.items()]
        vals = {t for (_, t) in found}
        value = round(next(iter(vals)) * 1000, 1) if len(vals) == 1 else None
        text = "Insulation layers:\n" + "\n".join(lines)
        return _answer(m, value if value is not None else text, "mm" if value is not None else None, "ifc:insulation",
                       "high", text, [g for ids in found.values() for g in ids])
    if not qq.classes:
        qq.classes = ("IfcWall",)
    return _op_types(m, qq)


def _op_extreme(m: IfcModel, qq: Query, kind: str) -> Answer:
    what = {"largest": "area", "smallest": "area", "widest": "width", "narrowest": "width", "tallest": "height",
            "lowest": "height", "longest": "length"}[kind]
    for dim, pat in (("width", r"width|wide|breite|leveys"), ("height", r"height|tall|high|h[oö]he|korkeus|ceiling"),
                     ("length", r"length|long\b|l[aä]nge|pituus"), ("volume", r"volume|volumen|tilavuus")):
        if _has(pat, qq.low):
            what = dim
            break
    if what == "area" and not qq.classes:
        qq.classes = ("IfcSpace",)
    if what == "height" and (not qq.classes or _has(r"ceiling", qq.low)) and T.spaces(m):
        qq.classes, qq.predefined = ("IfcSpace",), None
    if what == "area" and qq.classes and qq.classes != ("IfcSpace",) and _has(r"volume", qq.low):
        what = "volume"
    if not qq.classes:
        return _not_found("ifc:extreme", "Could not tell which elements to compare.")
    els = _selection(m, qq)
    if not els:
        return _not_found("ifc:extreme", f"No {_what(qq)}{_where(qq)} in this model.")
    r = T.extreme(m, els, what, largest=kind in ("largest", "widest", "tallest", "longest"))
    if r.value is None:
        return _not_found("ifc:extreme", f"None of the {_what(qq)} has a {what}.")
    unit = r.unit
    names = "; ".join(f"{row['name']}" + (f" ({row['storey']})" if row.get("storey") else "") for row in r.rows[:10])
    text = f"{kind.capitalize()} {_what(qq).rstrip('s')} by {what}: {names} with {_fmt(r.value)} {unit} (source: {r.source})." + (f" {r.note}" if r.note else "")
    if len(els) > 1:
        ranked = sorted(((T.quantity(e, what)[0] or 0, e) for e in els), key=lambda x: -x[0] if kind in ("largest", "widest", "tallest", "longest") else x[0])
        text += "\nTop 5:\n" + "\n".join(f"- {T.label(e)}: {_fmt(v)} {unit}" for v, e in ranked[:5])
    return _answer(m, text, None, "ifc:extreme", "high" if "geometry" not in r.source else "medium", text, r.ids)


def _op_fire(m: IfcModel, qq: Query) -> Answer:
    els = _selection(m, qq) if qq.classes else [e for e in m.elements if e.kind != "IfcSpace"]
    what = _what(qq) if qq.classes else "elements"
    rated: dict[tuple[str, str], list[str]] = {}
    for e in els:
        fr = T.fire_rating(e)
        if fr:
            rated.setdefault((T.type_label(e), fr), []).append(e.gid)
    if not rated:
        text = f"No fire rating information (FireRating property) on the {len(els)} {what}{_where(qq)} in this model."
        return _answer(m, text, None, "ifc:fire_rating", "medium", text, [e.gid for e in els][:20])
    lines = [f"- {t}: {fr} ({len(ids)} element(s))" for (t, fr), ids in sorted(rated.items(), key=lambda kv: -len(kv[1]))]
    n = sum(len(v) for v in rated.values())
    text = f"Fire rating information is available for {n} of {len(els)} {what}{_where(qq)} ({len(rated)} type/rating combinations):\n" + _rows_text(lines)
    return _answer(m, text, None, "ifc:fire_rating", "high", text, [g for v in rated.values() for g in v])


def _op_load_bearing(m: IfcModel, qq: Query) -> Answer:
    if not qq.classes:
        qq.classes = ("IfcWall",)
    qq.load_bearing = None
    els = _selection(m, qq)
    if not els:
        return _not_found("ifc:load_bearing", f"No {_what(qq)} in this model.")
    yes = [e for e in els if T.is_load_bearing(e) is True]
    no = [e for e in els if T.is_load_bearing(e) is False]
    unk = len(els) - len(yes) - len(no)
    if not yes and not no:
        return _not_found("ifc:load_bearing", f"None of the {len(els)} {_what(qq)} has a LoadBearing property.")
    pct = lambda k: 100 * k / len(els)  # noqa: E731
    text = (f"Of {len(els)} {_what(qq)}{_where(qq)}: {len(yes)} load-bearing ({pct(len(yes)):.1f}%), "
            f"{len(no)} non-load-bearing ({pct(len(no)):.1f}%)" + (f", {unk} without a LoadBearing property ({pct(unk):.1f}%)" if unk else "")
            + " (property LoadBearing).")
    if _has(r"^(are|is|do|does)\b", qq.low.strip()):
        verdict = "Yes" if yes and not no else "No" if no and not yes else "Partly"
        text = f"{verdict}. " + text
    return _answer(m, text, None, "ifc:load_bearing", "high", text, [e.gid for e in yes + no])


_SYSTEM_KINDS = {
    "heating": (r"heat|heiz|hzg|l[aä]mm|radiator", ("IfcSpaceHeater", "IfcBoiler", "IfcHeatExchanger", "IfcUnitaryEquipment")),
    "cooling": (r"cool|chill|k[uü]hl|j[aä][aä]hd", ("IfcChiller", "IfcCoolingTower", "IfcCoil", "IfcEvaporator")),
    "ventilation": (r"vent|air|luft|lft|ilma|hvac|exhaust|supply", ("IfcDuctSegment", "IfcAirTerminal", "IfcFan", "IfcDuctFitting", "IfcFilter")),
    "plumbing": (r"plumb|water|wasser|san|drain|sewage|vesi|domestic", ("IfcPipeSegment", "IfcSanitaryTerminal", "IfcValve", "IfcPipeFitting")),
    "electrical": (r"elec|power|strom|light|lighting|s[aä]hk", ("IfcLightFixture", "IfcOutlet", "IfcSwitchingDevice", "IfcCableSegment",
                                                               "IfcElectricDistributionBoard", "IfcProtectiveDevice")),
}


def _op_systems(m: IfcModel, qq: Query) -> Answer:
    kind = next((k for k in _SYSTEM_KINDS if _has(k[:4], qq.low)), None)
    pat, classes = _SYSTEM_KINDS[kind] if kind else (None, ())
    if kind == "electrical" and _has(r"lighting", qq.low):
        classes = ("IfcLightFixture",)
    r = T.systems(m, pat)
    lines = [f"- {row['name'] or '(unnamed)'} [{row['class']}{', ' + row['type'] if row['type'] else ''}]: {row['members']} member(s)" for row in r.rows]
    els = T.select(m, classes) if classes else T.select(m, ("IfcDistributionElement",))
    g = T.group_count(m, els, "class")
    comp = [f"- {row['key']}: {row['count']}" for row in g.rows]
    label = f"{kind} " if kind else "MEP "
    if not r.rows and not els:
        text = f"No {label}systems or {label}components found in this model."
        return _answer(m, text, None, "ifc:systems", "medium", text, [])
    text = ""
    if r.rows:
        text += f"{len(r.rows)} {label}system(s) (IfcSystem):\n" + _rows_text(lines)
    else:
        text += f"No explicit {label}systems (IfcSystem) in this model."
    if els:
        text += f"\n{len(els)} {label}component(s) by class:\n" + _rows_text(comp)
    return _answer(m, text, None, "ifc:systems", "high", text, r.ids + [e.gid for e in els])


def _op_guid(m: IfcModel, qq: Query) -> Answer:
    els = []
    if qq.classes:
        els = _selection(m, qq)
    text_q = _space_query_text(qq)
    if qq.classes == ("IfcSpace",) or (not els and text_q):
        els = T.find_spaces(m, text_q or qq.q, qq.storey) or els
    if not els:
        return _not_found("ifc:guid", "Could not find the element the question refers to.")
    if len(els) > 10:
        return _not_found("ifc:guid", f"{len(els)} elements match; please name one.")
    lines = [f"- {T.label(e)} [{e.ifc_class}]: {e.gid}" for e in els]
    text = "\n".join(lines)
    return _answer(m, els[0].gid if len(els) == 1 else text, None, "ifc:guid", "high" if len(els) == 1 else "medium", text,
                   [e.gid for e in els])


def _op_element_list(m: IfcModel, qq: Query) -> Answer:
    if qq.classes == ("IfcSpace",) and not qq.name_pattern:
        if _has(_AREA, qq.low) or not _has(r"names?\b", qq.low):
            return _op_space_list(m, qq)
    els = _selection(m, qq)
    what = _what(qq)
    if not els:
        text = f"No {what}{_where(qq)} found in this model." + _filter_note(m, qq)
        return _answer(m, text, None, "ifc:list", "medium", text, [])
    if qq.classes == ("IfcSpace",):
        lines = [f"- {T.label(e)}" + (f": {_fmt(T.space_area(e)[0])} m²" if T.space_area(e)[0] else "") for e in els]
        text = f"{len(els)} {what}{_where(qq)}:\n" + _rows_text(lines)
        return _answer(m, text, None, "ifc:list", "high", text, [e.gid for e in els])
    return _op_types(m, qq)


def _op_property(m: IfcModel, qq: Query) -> Answer | None:
    """'What is the <property> of <named space/element>' -> the property value, if the model has one with that name."""
    target = _space_query_text(qq)
    hits = T.find_spaces(m, target or qq.q, qq.storey) if target else []
    if not hits:
        return None
    e = hits[0]
    words = [w for w in re.findall(r"[a-zäöüß]{4,}", qq.low) if w not in {"what", "which", "room", "space", "there", "with", "does", "have", "this", "that", "from", "model"}]
    best = None
    for k, v in e.props.items():
        kn = k.rsplit(".", 1)[-1].casefold()
        score = sum(1 for w in words if w in kn)
        if score and (best is None or score > best[0]):
            best = (score, k, v)
    if best is None:
        return None
    _, k, v = best
    unit = {"length": "m", "area": "m2", "volume": "m3"}.get(e.dims.get(k, ""))
    text = f"{T.label(e)}: {k} = {v}" + (f" {unit}" if unit else "")
    return _answer(m, round(v, 3) if isinstance(v, float) else v, unit, "ifc:property", "medium", text, [e.gid])


# attribute words -> property-name regex. Asked about selected elements, these are answered from the properties
# that carry such a name (any property set), or not_found: never from the type list.
_ATTRIBUTES = {
    "colour": r"colou?r|farbe|v[aä]ri\b|f[aä]rg",
    "cost": r"\bcost|price|kosten|preis|hinta|kostnad",
    "manufacturer": r"manufactur|hersteller|valmistaja|tillverkare|vendor|supplier",
    "u-value": r"u-?value|thermal transmittance|w[aä]rmedurchgang|u-arvo|lämmönläpäisy",
    "acoustic": r"acoustic|sound|schall|ääni",
    "finish": r"\bfinish|surface treatment|oberfl[aä]che",
    "model number": r"model (number|no)|article|artikel",
}
_ATTRIBUTE_PROPS = {
    "colour": r"colou?r|farbe|v[aä]ri|f[aä]rg", "cost": r"cost|price|kosten|preis|hinta", "manufacturer": r"manufactur|hersteller|valmistaja",
    "u-value": r"thermaltransmittance|u-?value|u-wert|uvalue", "acoustic": r"acoustic|schall", "finish": r"finish|oberfl",
    "model number": r"model|article|artikel",
}


def _op_attribute(m: IfcModel, qq: Query, attr: str) -> Answer:
    els = _selection(m, qq)
    what = _what(qq)
    pat = _ATTRIBUTE_PROPS[attr]
    vals: dict[tuple[str, str], list[str]] = {}
    for e in els:
        for k, v in e.props.items():
            short = k.rsplit(".", 1)[-1]
            if re.search(pat, short.replace(" ", ""), re.I) and v not in (None, "", 0, 0.0) and T.norm(v) != T.norm(short):
                vals.setdefault((k.rsplit(".", 1)[-1], str(v)), []).append(e.gid)
    if not vals:
        return _not_found("ifc:attribute", f"None of the {len(els)} {what}{_where(qq)} has a property for {attr} in this model.")
    lines = [f"- {k} = {v}: {len(ids)} element(s)" for (k, v), ids in sorted(vals.items(), key=lambda kv: -len(kv[1]))]
    text = f"{attr.capitalize()} of the {what}{_where(qq)} (from their properties):\n" + _rows_text(lines)
    nums = {}
    for (k, v), ids in vals.items():
        try:
            nums[(k, v)] = float(v) * len(ids)
        except ValueError:
            pass
    if attr == "cost" and nums:
        total = sum(nums.values())
        n = sum(len(vals[k]) for k in nums)
        text += f"\nTotal over the {n} element(s) with a numeric value: {_fmt(total)} (currency as entered in the model)."
        if n < len(els):
            text += f" {len(els) - n} of {len(els)} {what} have no cost."
    return _answer(m, text, None, "ifc:attribute", "high", text, [g for ids in vals.values() for g in ids])


# ------------------------------------------------------------------------------------------------ dispatch


def answer_model(m: IfcModel, question: str) -> Answer:
    qq = parse(m, question)
    low = qq.low
    if _has(_GUID, low):
        return _op_guid(m, qq)
    if _has(_META, low) and not _has(_COUNT, low):
        return _op_meta(m, qq)
    if _has(_PROJECT, low) and not _has(_COUNT, low) and not qq.classes:
        return _op_project(m, qq)
    if _has(_FIRE, low):
        return _op_fire(m, qq)
    if _has(r"percent|share|proportion|ratio|anteil|osuus|compared|versus|\bvs\.?\b|\bor\b", low) and _has(_LOADBEARING, low) or \
            _has(r"^(are|is)\b", low.strip()) and _has(_LOADBEARING, low):
        return _op_load_bearing(m, qq)
    if _has(_SYSTEM, low) and (not qq.classes or _has(r"mep|hvac|heating|cooling|ventilation|plumbing|electrical|lighting", low)):
        return _op_systems(m, qq)
    if _has(r"thickness|thick\b|dicke|st[aä]rke|paksuus|tjocklek", low):
        return _op_thickness(m, qq)
    if (qq.classes == ("IfcBuildingStorey",) or qq.storey is not None and not qq.classes) and \
            _has(r"height|elevation|h[oö]he|korkeus|niveau|korko", low):
        return _op_storeys(m, qq, count_only=False)
    if _has(_MATERIAL, low) and not _has(_COUNT, low):
        return _op_materials(m, qq)
    for kind, pat in _EXTREME.items():
        if _has(pat, low):
            return _op_extreme(m, qq, kind)
    if qq.classes and _has(_TYPES, low) and _has(r"of each|each type|per type|by type|types?\b.*\b(quantit|count|how many|number)|(quantit|count|how many|number).*\btypes?\b", low):
        return _op_types(m, qq)
    if _has(_COUNT, low) and not (_has(_AREA, low) and _has(r"\b(sq|m²|m2|square|area)\b", low) and not _has(r"how many (rooms|spaces)", low)):
        return _op_count(m, qq)
    if _has(_VOLUME, low):
        if (a := _op_space_value(m, qq, "volume")) is not None:
            return a
        return _op_quantity_total(m, qq, "volume")
    if _has(_LENGTH, low):
        return _op_quantity_total(m, qq, "length")
    if _has(r"footprint", low):
        return _op_footprint(m, qq)
    if _has(_AREA, low):
        if (not qq.classes or qq.classes == ("IfcSpace",)) and (a := _op_space_value(m, qq, "area")) is not None:
            return a
        if not qq.classes or qq.classes == ("IfcSpace",):
            if _has(_LIST, low) and _has(r"\bwith\b|\band\b|\bincluding\b|breakdown|each|individual|all (the )?(spaces|rooms)", low):
                return _op_space_list(m, qq)
            return _op_space_total(m, qq)
        return _op_quantity_total(m, qq, "area")
    if _has(r"height|h[oö]he|korkeus", low) and (a := _op_space_value(m, qq, "height")) is not None:
        return a
    if qq.classes and (attr := next((k for k, pat in _ATTRIBUTES.items() if _has(pat, low)), None)):
        return _op_attribute(m, qq, attr)
    if _has(_TYPES, low) and qq.classes:
        return _op_types(m, qq)
    if qq.classes and _has(_LIST + r"|\bwhat\b|are there|present|included|installed|located|exist", low):
        return _op_element_list(m, qq)
    if (a := _op_property(m, qq)) is not None:
        return a
    if _has(r"summary|overview|describe|what is (in )?this model|contents", low):
        return _op_summary(m)
    return _not_found(None, "This question could not be mapped to the IFC model's data (supported: counts, areas, "
                            "volumes, lengths, types, materials, storeys, fire ratings, systems, metadata).")


def _op_footprint(m: IfcModel, qq: Query) -> Answer:
    for b in m.buildings:
        for k in ("FootprintArea", "GrossPlannedArea"):
            if b.get("qto", {}).get(k):
                v = b["qto"][k]
                return _answer(m, round(v, 2), "m2", "ifc:footprint", "high", f"Building {b.get('name')}: {k} = {_fmt(v)} m².", [b["gid"]])
    base = T.select(m, ("IfcSlab",), predefined=("BASESLAB",))
    if base:
        r = T.sum_quantity(m, base, "footprint")
        if r.value:
            return _answer(m, round(r.value, 2), "m2", "ifc:footprint", "medium",
                           f"Footprint estimated as the area of the {len(r.ids)} base slab(s) (IfcSlab BASESLAB): {_fmt(r.value)} m² "
                           f"(sources: {r.source}). {r.note}".strip(), r.ids)
    return _not_found("ifc:footprint", "The model states no footprint area (no building quantity, no base slab with an area).", "m2")


def _op_summary(m: IfcModel) -> Answer:
    r = T.summary(m)
    row = r.rows[0]
    counts = ", ".join(f"{T.class_label(k)}: {v}" for k, v in list(row["counts"].items())[:15])
    text = (f"Project {row['project'] or '-'}" + (f" ({row['project_long_name']})" if row["project_long_name"] else "")
            + f", IFC {row['schema']}, {len(row['storeys'])} storeys ({', '.join(s['name'] or '?' for s in row['storeys'])}). "
            + f"Elements: {counts}.")
    return _answer(m, text, None, "ifc:summary", "high", text, r.ids)


def answer_question(rel: str, question: str) -> Answer:
    """Dataset-relative path of an .ifc file + question -> Answer (same shape as app.pdfplan.answer)."""
    try:
        m = load_model(rel)
    except FileNotFoundError:
        raise
    except Exception as error:  # noqa: BLE001  unreadable / unsupported IFC
        return _not_found(None, f"Could not read the IFC model: {type(error).__name__}: {error}")
    return answer_model(m, question)
