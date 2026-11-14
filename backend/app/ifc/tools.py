"""General, deterministic tools over an IfcModel. Every result carries the GlobalIds it was computed from.

The tools know IFC, not the dataset: classes come from the IFC schema (with supertypes, and the type object for
IFC2X3 generic classes), quantities from quantity sets / properties / geometry in a documented priority order, and
all values are SI (m, m², m³). They are written so that an LLM agent could call them directly (plain arguments,
JSON-able results); app/ifc/answer.py composes them for the offline path.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.ifc.model import Element, IfcModel


@dataclass
class Result:
    value: Any
    unit: str | None = None
    ids: list[str] = field(default_factory=list)  # GlobalIds behind the value
    rows: list[dict[str, Any]] = field(default_factory=list)  # structured detail (one row per element / group)
    source: str = ""  # where the numbers come from, e.g. "Qto_SpaceBaseQuantities.NetFloorArea"
    note: str = ""


def norm(s: Any) -> str:
    """Casefolded, accent-free, punctuation-insensitive text for matching names."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip()


# ------------------------------------------------------------------------------------------------ vocabulary
# word pattern (EN / DE / FI / SV / FR / NL stems) -> (IFC classes, predefined types or None, name pattern or None).
# Order matters: more specific entries first ("curtain wall" before "wall", "light switch" before "light").
CLASS_WORDS: list[tuple[str, tuple[str, ...], tuple[str, ...] | None, str | None]] = [
    (r"curtain ?walls?|vorhangfassade", ("IfcCurtainWall",), None, None),
    (r"electrical (components?|elements?|devices?|equipment|fixtures?|installations?)|electric(al)? items?",
     ("IfcLightFixture", "IfcOutlet", "IfcSwitchingDevice", "IfcJunctionBox", "IfcElectricDistributionBoard",
      "IfcElectricAppliance", "IfcCableSegment", "IfcCableCarrierSegment", "IfcCableFitting", "IfcCableCarrierFitting",
      "IfcProtectiveDevice", "IfcElectricFlowStorageDevice", "IfcTransformer", "IfcElectricGenerator", "IfcElectricMotor"), None, None),
    (r"structural (elements?|members?|components?)|load[- ]bearing elements?|tragwerk\w*",
     ("IfcBeam", "IfcColumn", "IfcSlab", "IfcFooting", "IfcPile", "IfcMember"), None, None),
    (r"fire extinguishers?|feuerl[oö]scher|sammutin\w*", ("IfcElement",), None, r"extinguish|feuerl|sammutin"),
    (r"junction (boxes|box)|abzweigdosen?", ("IfcJunctionBox",), None, None),
    (r"waste terminals?|drains?|gullys?|floor drains?", ("IfcWasteTerminal",), None, None),
    (r"(flow )?controllers?|dampers?", ("IfcFlowController", "IfcController", "IfcDamper"), None, None),
    (r"receptacles?", ("IfcOutlet",), None, None),
    (r"light switch(es)?|switch(es)?|schalter|kytkim", ("IfcSwitchingDevice",), None, None),
    (r"light(ing)? fixtures?|luminaires?|lamps?|lights?\b|leuchten?|valaisin|valaisimet|armatur", ("IfcLightFixture",), None, None),
    (r"sanitary (fixtures?|terminals?|objects?|equipment)|sanit[aä]r\w*|saniteetti\w*|plumbing fixtures?", ("IfcSanitaryTerminal",), None, None),
    (r"toilets? (seats?|pans?|bowls?)|wc (seats?|pans?)|toilet fixtures?|toilets?(?! rooms?)|wcs?(?! rooms?)|urinals?|klosett\w*|wc-istuin\w*",
     ("IfcSanitaryTerminal",), None, r"toilet|\bwc\b|klosett|water ?closet|urinal|wc-?istuin"),
    (r"showers?|duschen?|suihku\w*|bathtubs?|badewannen?|kylpyamme\w*", ("IfcSanitaryTerminal",), None, r"shower|dusch|suihku|\bbath|badewanne|kylpyamme"),
    (r"wash ?basins?|sinks?|lavatory|waschbecken|waschtisch|lavabo|pesuallas|pesualtaat", ("IfcSanitaryTerminal",), None, r"sink|basin|lavator|waschbecken|waschtisch|lavabo|pesuallas"),
    (r"air terminals?|diffusers?|grilles?", ("IfcAirTerminal",), None, None),
    (r"radiators?|heizk[oö]rper|patteri|space heaters?", ("IfcSpaceHeater",), None, None),
    (r"(electrical )?outlets?|sockets?|steckdosen?|pistorasia", ("IfcOutlet",), None, None),
    (r"pipe fittings?", ("IfcPipeFitting",), None, None),
    (r"duct fittings?", ("IfcDuctFitting",), None, None),
    (r"pipes?|piping|rohre?|putki\w*", ("IfcPipeSegment",), None, None),
    (r"ducts?|ductwork|kan[aä]le?|kanava\w*", ("IfcDuctSegment",), None, None),
    (r"cables?|kabel\w*|kaapeli\w*", ("IfcCableSegment", "IfcCableCarrierSegment"), None, None),
    (r"valves?|ventile?|venttiili\w*", ("IfcValve",), None, None),
    (r"pumps?|pumpen?|pumppu\w*", ("IfcPump",), None, None),
    (r"fans?\b|ventilator\w*|puhallin\w*", ("IfcFan",), None, None),
    (r"filters?", ("IfcFilter",), None, None),
    (r"boilers?|kessel", ("IfcBoiler",), None, None),
    (r"sensors?|detectors?|f[uü]hler", ("IfcSensor",), None, None),
    (r"fire suppression|sprinklers?", ("IfcFireSuppressionTerminal",), None, None),
    (r"elevators?|lifts?|aufz[uü]ge?|hissi\w*", ("IfcTransportElement",), None, None),
    (r"doors?|t[uü]ren?|ovi|ove[nta]|ovia|ovien|ovissa|d[oö]rr(ar)?|portes?", ("IfcDoor",), None, None),
    (r"windows?|fenster|ikkuna\w*|f[oö]nster|fen[eê]tres?", ("IfcWindow",), None, None),
    (r"walls?|w[aä]nde?|seinä\w*|seina\w*|v[aä]gg(ar)?|murs?", ("IfcWall",), None, None),
    (r"columns?|st[uü]tzen?|pilari\w*|pelare|poteaux?", ("IfcColumn",), None, None),
    (r"beams?|tr[aä]ger|palkki\w*|balk(ar)?|poutres?", ("IfcBeam",), None, None),
    (r"ceilings?|unterdecken?|alakatto\w*", ("IfcCovering",), ("CEILING",), None),
    (r"floor(ing)? (finish(es)?|coverings?)|flooring", ("IfcCovering",), ("FLOORING",), None),
    (r"coverings?|bekleidung\w*", ("IfcCovering",), None, None),
    (r"(floor )?slabs?|decken?platten?|bodenplatten?|laatta\w*|bj[aä]lklag|dalles?", ("IfcSlab",), None, None),
    (r"roofs?|d[aä]cher|dach|katto|kattoja|tak\b", ("IfcRoof",), None, None),
    (r"stair ?flights?|treppenl[aä]ufe?", ("IfcStairFlight",), None, None),
    (r"stairs?|staircases?|stairways?|treppen?|portaat|portaita|porras|trapp(a|or)", ("IfcStair",), None, None),
    (r"ramps?|rampen?|luiska\w*", ("IfcRamp",), None, None),
    (r"railings?|handrails?|gel[aä]nder|kaide\w*|r[aä]cke", ("IfcRailing",), None, None),
    (r"footings?|foundations?|fundamente?|perustu\w*|anturat?", ("IfcFooting", "IfcPile"), None, None),
    (r"piles?|pf[aä]hle?|paalu\w*", ("IfcPile",), None, None),
    (r"members?|mullions?|purlins?", ("IfcMember",), None, None),
    (r"plates?|panels?", ("IfcPlate",), None, None),
    (r"furniture|furnishings?|m[oö]bel|huonekalu\w*|kalusteet|m[oö]bler", ("IfcFurnishingElement",), None, None),
    (r"proxies|proxy elements?", ("IfcBuildingElementProxy",), None, None),
    (r"rooms?|spaces?|r[aä]ume?|raum|huone\w*|tiloja|tilat|tila|rum\b|pi[eè]ces?", ("IfcSpace",), None, None),
    (r"(building )?storeys?|stories|levels?|floors?|geschosse?|etagen?|kerros\w*|v[aå]ningar?", ("IfcBuildingStorey",), None, None),
    (r"mep (components?|elements?)|distribution elements?", ("IfcDistributionElement",), None, None),
]

# Room functions that people count or sum ("how many bedrooms"): matched against space names / long names
SPACE_FUNCTIONS: list[tuple[str, str]] = [
    (r"bed ?rooms?|schlafzimmer|makuuhuone\w*|sovrum", r"bed|schlaf|makuu|sovrum|\bmh\b"),
    (r"bath ?rooms?|badezimmer|b[aä]der|kylpyhuone\w*|badrum", r"bath|bad\b|bad |badezimmer|kylpy|\bkph\b|badrum"),
    (r"restrooms?|toilet rooms?|wc rooms?|toiletten?räume", r"toilet|\bwc\b|restroom|men|women"),
    (r"kitchens?|k[uü]chen?|keitti\w*|k[oö]k\b", r"kitchen|k[uü]che|keitti|kok\b|teek"),
    (r"living rooms?|wohnzimmer|olohuone\w*|vardagsrum", r"living|wohn|olohuone|vardags"),
    (r"offices?|b[uü]ros?|toimisto\w*|kontor", r"office|b[uü]ro|toimisto|kontor"),
    (r"corridors?|hallways?|fl[uü]re?|k[aä]yt[aä]v\w*|korridor", r"corridor|hall|flur|kaytava|korridor"),
    (r"storage rooms?|lager|varasto\w*|f[oö]rr[aå]d", r"storage|lager|varasto|forrad|store"),
    (r"meeting rooms?|conference rooms?|besprechung\w*|neuvottelu\w*", r"meeting|conference|besprech|neuvottelu|seminar"),
    (r"stairwells?|staircases?|treppenh[aä]user?|porrashuone\w*", r"stair|treppe|porras"),
    (r"saunas?", r"sauna"),
    (r"balcon(y|ies)|balkone?|parvekke\w*", r"balcon|balkon|parvek"),
]

_GENERIC_CLASS_LABEL = {"IfcSpace": "spaces", "IfcBuildingStorey": "storeys"}


def class_label(cls: str) -> str:
    return _GENERIC_CLASS_LABEL.get(cls, re.sub(r"(?<!^)(?=[A-Z])", " ", cls.removeprefix("Ifc")).lower())


def match_classes(text: str) -> tuple[tuple[str, ...], tuple[str, ...] | None, str | None, str, int] | None:
    """First vocabulary entry found in the text -> (classes, predefined, name pattern, matched words, position).
    The earliest mention wins ("doors on the floor plan" -> doors); at equal position the longer match."""
    low = text.casefold()
    best = None
    for pat, classes, predefined, name_pat in CLASS_WORDS:
        m = re.search(rf"(?<![a-z])(?:{pat})(?![a-z])", low)
        if m and (best is None or (m.start(), -len(m.group(0))) < (best[4], -len(best[3]))):
            best = (classes, predefined, name_pat, m.group(0), m.start())
    return best


def match_space_function(text: str) -> tuple[str, str, int] | None:
    """Room function words ("bedrooms") -> (matched words, space-name regex, position)."""
    low = text.casefold()
    best = None
    for pat, name_pat in SPACE_FUNCTIONS:
        m = re.search(rf"(?<![a-z])(?:{pat})(?![a-z])", low)
        if m and (best is None or m.start() < best[2]):
            best = (m.group(0), name_pat, m.start())
    return best


# ------------------------------------------------------------------------------------------------ selection


def label(e: Element) -> str:
    """Human-readable element label: 'Living Room (A102)' for spaces, the name otherwise."""
    if e.kind == "IfcSpace":
        if e.long_name and e.name and norm(e.long_name) != norm(e.name):
            return f"{e.long_name} ({e.name})"
        return e.long_name or e.name or e.gid
    return e.name or e.type_name or e.gid


def type_label(e: Element) -> str:
    """Type name; Revit writes 'Family:Type' into ObjectType and only 'Type' into the type object, so the fuller
    ObjectType wins when it contains the type name."""
    if e.type_name and e.object_type and e.type_name in e.object_type and len(e.object_type) > len(e.type_name):
        return e.object_type
    return e.type_name or e.object_type or e.name or "(no type)"


# Name hints for element kinds that authoring tools often export as generic classes (IfcBuildingElementProxy,
# IfcDistributionElement, IfcFlowTerminal without a type): used only when the IFC class itself finds nothing.
NAME_HINTS = {
    "IfcSanitaryTerminal": r"toilet|\bwc\b|sink|basin|urinal|shower|bath ?tub|lavat|wasch|bidet|klosett|dusch|pesuallas|suihku",
    "IfcLightFixture": r"light|lamp|luminaire|leuchte|valaisin|downlight|spot",
    "IfcSwitchingDevice": r"switch|schalter|kytkin|dimmer",
    "IfcOutlet": r"outlet|socket|steckdose|pistorasia|receptacle",
    "IfcSpaceHeater": r"radiator|heizk[oö]rper|heater|patteri|convector",
    "IfcAirTerminal": r"air terminal|diffuser|grille|luftauslass|ventiel|tuloilma|poistoilma",
    "IfcFurnishingElement": r"furniture|chair|table|desk|\bbed\b|sofa|cabinet|shelf|wardrobe|m[oö]bel|stuhl|tisch",
    "IfcValve": r"valve|ventil|venttiili|hahn|klappe",
    "IfcPump": r"pump",
    "IfcFan": r"\bfan\b|ventilator|puhallin",
    "IfcTransportElement": r"elevator|\blift\b|aufzug|hissi",
    "IfcFooting": r"footing|foundation|fundament|antura|perustus",
    "IfcPile": r"\bpile|pfahl|paalu|fundatiepaal",
}


def prop(e: Element, *names: str) -> Any:
    """First property whose name (after the pset prefix) matches one of `names`, casefold; occurrence before type."""
    want = [n.casefold() for n in names]
    for w in want:
        for k, v in e.props.items():
            if k.rsplit(".", 1)[-1].casefold() == w:
                return v
    return None


def _truthy(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        t = v.strip().casefold()
        if t in ("true", "yes", "ja", "kyllä", ".t.", "1"):
            return True
        if t in ("false", "no", "nein", "ei", ".f.", "0"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


_EXT_WORDS = r"exterior|external|outside|au(ss|ß)en|ulko|ytter|fa[cç]ade|fassade|gevel"
_INT_WORDS = r"interior|internal|inside|innen|sis[aä]|inner|partition|trennwand|v[aä]li"


def external_evidence(e: Element) -> tuple[bool | None, str]:
    """IsExternal, else what the authoring tool's own classification says (type name 'Exterior - Brick', layer
    'Außenwände', category / function property) -> (value, source)."""
    v = _truthy(prop(e, "IsExternal"))
    if v is not None:
        return v, "IsExternal"
    for label_, text in (("type name", type_label(e)), ("name", e.name)):
        t = str(text or "")
        if re.search(_EXT_WORDS, t, re.I):
            return True, label_
        if re.search(_INT_WORDS, t, re.I):
            return False, label_
    for k, val in e.props.items():
        if isinstance(val, str) and re.search(r"layer|ebene|category|kategorie|function|funktion|usage|position|lage\b|taso", k.rsplit(".", 1)[-1], re.I):
            if re.search(_EXT_WORDS, val, re.I):
                return True, k
            if re.search(_INT_WORDS, val, re.I):
                return False, k
    return None, ""


def is_external(e: Element) -> bool | None:
    return external_evidence(e)[0]


def is_load_bearing(e: Element) -> bool | None:
    return _truthy(prop(e, "LoadBearing"))


def fire_rating(e: Element) -> str | None:
    v = prop(e, "FireRating", "Fire Rating", "Feuerwiderstandsklasse", "Brandschutzklasse", "Feuerwiderstand")
    if v in (None, "", "-", "None", "none", "FireRating"):
        return None
    return str(v)


def select(m: IfcModel, classes: Iterable[str] | None = None, storey: Element | None = None,
           predefined: Iterable[str] | None = None, name_pattern: str | None = None,
           external: bool | None = None, load_bearing: bool | None = None) -> list[Element]:
    """Elements of any of the classes (IFC supertypes included), optionally on one storey, with a predefined type,
    a name / type name matching a regex, IsExternal or LoadBearing set to the given value."""
    classes = tuple(classes or ())
    out = []
    for e in m.elements:
        if classes and not any(c in e.classes for c in classes):
            continue
        if storey is not None and e.storey != storey.gid:
            continue
        if predefined and (e.predefined or "") not in predefined:
            continue
        if name_pattern:
            hay = " ".join(str(x) for x in (e.name, e.long_name, e.type_name, e.object_type, e.predefined) if x).casefold()
            if not re.search(name_pattern, hay):
                continue
        if external is not None and is_external(e) is not external:
            continue
        if load_bearing is not None and is_load_bearing(e) is not load_bearing:
            continue
        out.append(e)
    return out


def count(m: IfcModel, classes: Iterable[str], storey: Element | None = None, **kw) -> Result:
    els = select(m, classes, storey, **kw)
    return Result(len(els), "count", [e.gid for e in els], [{"gid": e.gid, "class": e.ifc_class, "name": label(e)} for e in els])


def group_count(m: IfcModel, elements: list[Element], key: str = "type") -> Result:
    """Counts by 'type' (type object name), 'storey', 'class', 'predefined', 'material' or 'function' (space names)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for e in elements:
        if key == "type":
            ks = [type_label(e)]
        elif key == "storey":
            s = m.storey(e.storey)
            ks = [s.name if s else "(no storey)"]
        elif key == "class":
            ks = [e.kind]
        elif key == "predefined":
            ks = [e.predefined or "NOTDEFINED"]
        elif key == "material":
            ks = e.materials or ["(no material)"]
        elif key == "function":
            ks = [e.long_name or e.name or "(unnamed)"]
        else:
            raise ValueError(key)
        for k in ks:
            groups[str(k)].append(e.gid)
    n = max(1, len(elements))
    rows = [{"key": k, "count": len(v), "share": 100 * len(v) / n, "ids": v}
            for k, v in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    return Result(len(rows), "groups", [e.gid for e in elements], rows)


def select_by_name(m: IfcModel, pattern: str, storey: Element | None = None, spaces_only: bool = False) -> list[Element]:
    """Elements (or spaces) whose name / type name / object type matches a regex (casefold)."""
    out = []
    for e in m.elements:
        if spaces_only != (e.kind == "IfcSpace"):
            continue
        if storey is not None and e.storey != storey.gid:
            continue
        hay = " ".join(str(x) for x in (e.name, e.long_name, e.type_name, e.object_type) if x).casefold()
        if re.search(pattern, hay):
            out.append(e)
    return out


# ------------------------------------------------------------------------------------------------ storeys


def storeys(m: IfcModel) -> Result:
    """Storeys bottom-up with elevation and height (elevation difference to the next storey, else a height quantity)."""
    rows = []
    ss = m.storeys
    for i, s in enumerate(ss):
        h = None
        src = ""
        if i + 1 < len(ss) and s.elevation is not None and ss[i + 1].elevation is not None:
            h, src = ss[i + 1].elevation - s.elevation, "elevation difference to the next storey"
        else:
            for q in ("GrossHeight", "NetHeight", "Height"):
                if q in s.qto:
                    h, src = s.qto[q], f"quantity {q}"
                    break
        n = sum(1 for e in m.elements if e.storey == s.gid)
        rows.append({"gid": s.gid, "name": s.name, "long_name": s.long_name, "elevation": s.elevation, "height": h,
                     "height_source": src, "elements": n})
    return Result(len(ss), "count", [s.gid for s in ss], rows)


_ORDINALS = {"ground": 0, "erdgeschoss": 0, "eg": 0, "maantaso": 0, "basement": -1, "keller": -1, "kellari": -1,
             "untergeschoss": -1, "ug": -1, "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
             "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "top": 99, "roof": 99, "attic": 99, "dachgeschoss": 99}


def resolve_storey(m: IfcModel, text: str) -> tuple[Element | None, str]:
    """Find the storey a question refers to -> (storey, how it was matched). Exact / contained storey names win;
    then ordinal words ('ground floor', 'second floor', 'level 2') against storeys sorted by elevation."""
    if not m.storeys:
        return None, ""
    t = f" {norm(text)} "
    named = [(len(norm(s.name)), s) for s in m.storeys if s.name and f" {norm(s.name)} " in t]
    named += [(len(norm(s.long_name)), s) for s in m.storeys if s.long_name and f" {norm(s.long_name)} " in t]
    if named:
        named.sort(key=lambda x: -x[0])
        return named[0][1], "name"
    ss = [s for s in m.storeys if s.elevation is not None]
    if not ss:
        return None, ""
    ground_i = min(range(len(ss)), key=lambda i: abs(ss[i].elevation))
    mm = re.search(r"\b(ground|basement|first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|top|roof|attic)\s+(floor|level|storey|story)\b"
                   r"|\b(erdgeschoss|keller|untergeschoss|dachgeschoss|kellari|maantaso\w*)\b", t)
    idx = None
    if mm:
        word = mm.group(1) or mm.group(3)
        word = "maantaso" if word.startswith("maantaso") else word
        k = _ORDINALS.get(word)
        if k == 99:
            idx = len(ss) - 1
        elif k is not None:
            # "first floor": British/German usage = one above ground. Storey names decide when they say it.
            idx = ground_i + k
    elif mm2 := re.search(r"\b(?:level|floor|storey|story|ebene|kerros|geschoss)\s+(-?\d+)\b|\b(-?\d+)\.?\s*(?:og|obergeschoss|kerros|floor)\b", t):
        k = int(mm2.group(1) or mm2.group(2))
        idx = ground_i + k
    if idx is not None and 0 <= idx < len(ss):
        return ss[idx], "ordinal"
    return None, ""


# ------------------------------------------------------------------------------------------------ spaces


def spaces(m: IfcModel, storey: Element | None = None) -> list[Element]:
    return select(m, ("IfcSpace",), storey)


def find_spaces(m: IfcModel, query: str, storey: Element | None = None) -> list[Element]:
    """Spaces whose number and/or name appear in the query ('JAN. (1A11)', 'room A203', 'Living Room').
    Best score first; ties keep all (several rooms can share a name)."""
    q = f" {norm(query)} "
    scored = []
    for e in spaces(m, storey):
        s = 0
        n, ln = norm(e.name), norm(e.long_name)
        if n and f" {n} " in q:
            s += 2 + len(n) / 100
        if ln and f" {ln} " in q:
            s += 1 + len(ln) / 100
        if s:
            scored.append((s, e))
    if not scored:
        return []
    best = max(s for s, _ in scored)
    return [e for s, e in scored if s == best]


def space_area(e: Element, which: str = "net") -> tuple[float | None, str]:
    """Floor area of a space (m²) and where it came from. net: NetFloorArea quantity > body footprint (geometry) >
    GrossFloorArea quantity > an area property. gross: GrossFloorArea > footprint > NetFloorArea > property."""
    q = e.qto
    geo = e.geom.get("footprint_area")
    if which == "gross":
        order = [("GrossFloorArea", q.get("GrossFloorArea")), ("geometry footprint", geo), ("NetFloorArea", q.get("NetFloorArea"))]
    else:
        order = [("NetFloorArea", q.get("NetFloorArea")), ("geometry footprint", geo), ("GrossFloorArea", q.get("GrossFloorArea"))]
    for src, v in order:
        if v:
            return float(v), src
    for k, v in e.props.items():
        if e.dims.get(k) == "area" and isinstance(v, (int, float)) and re.search(r"area|fl[aä]che|ala\b|pinta", k, re.I) \
                and not re.search(r"wall|wand|ceiling|decke|door|t[uü]r|window|fenster|abzug|deduct", k, re.I):
            return float(v), k
    return None, ""


def space_height(e: Element) -> tuple[float | None, str]:
    for q in ("ClearHeight", "FinishCeilingHeight", "Height", "NetHeight", "GrossHeight"):
        if e.qto.get(q):
            return float(e.qto[q]), q
    for k, v in e.props.items():
        if e.dims.get(k) == "length" and isinstance(v, (int, float)) and re.search(r"height|h[oö]he|korkeus", k, re.I):
            return float(v), k
    if e.geom.get("height"):
        return float(e.geom["height"]), "geometry height"
    return None, ""


def space_table(m: IfcModel, storey: Element | None = None, which: str = "net", name_pattern: str | None = None) -> Result:
    rows, ids, total, missing, srcs = [], [], 0.0, [], Counter()
    for e in spaces(m, storey):
        if name_pattern and not re.search(name_pattern, f"{e.name or ''} {e.long_name or ''}".casefold()):
            continue
        a, src = space_area(e, which)
        s = m.storey(e.storey)
        rows.append({"gid": e.gid, "name": label(e), "number": e.name, "long_name": e.long_name,
                     "storey": s.name if s else None, "area": a, "source": src})
        ids.append(e.gid)
        if a is None:
            missing.append(e.gid)
        else:
            total += a
            srcs[src] += 1
    rows.sort(key=lambda r: -(r["area"] or 0))
    return Result(round(total, 2) if rows and len(missing) < len(rows) else None, "m2", ids, rows,
                  source=", ".join(f"{k} ({v})" for k, v in srcs.most_common()),
                  note=f"{len(missing)} space(s) without an area" if missing else "")


# ------------------------------------------------------------------------------------------------ quantities

_QTY = {  # quantity -> (quantity names in priority order, property name regex, dimension)
    "length": (("Length", "NetLength", "GrossLength"), r"(^|\.)(length|l[aä]nge|pituus)$", "length"),
    "width": (("Width", "OverallWidth", "NominalWidth"), r"(^|\.)(width|breite|leveys)$", "length"),
    "height": (("Height", "OverallHeight", "NominalHeight", "Depth"), r"(^|\.)(height|h[oö]he|korkeus)$", "length"),
    "thickness": (("Thickness", "Width"), r"(^|\.)(thickness|dicke|st[aä]rke|paksuus)$", "length"),
    "volume": (("NetVolume", "GrossVolume", "Volume"), r"(^|\.)(volume|volumen|tilavuus|netto-volumen)$", "volume"),
    "area": (("NetArea", "GrossArea", "NetSideArea", "GrossSideArea", "NetSurfaceArea", "OuterSurfaceArea",
              "NetFootprintArea", "GrossFootprintArea", "Area"), r"(^|\.)(area|fl[aä]che|pinta-ala)$", "area"),
    "footprint": (("NetFootprintArea", "GrossFootprintArea", "NetArea", "GrossArea"), r"$^", "area"),
}


def quantity(e: Element, what: str) -> tuple[float | None, str]:
    """One element's length / width / height / thickness / volume / area (SI) and its source."""
    if e.kind == "IfcSpace" and what in ("area", "footprint"):
        return space_area(e)
    if e.kind == "IfcSpace" and what == "height":
        return space_height(e)
    if e.kind == "IfcSpace" and what == "volume":
        for q in ("NetVolume", "GrossVolume"):
            if e.qto.get(q):
                return float(e.qto[q]), q
        if e.geom.get("volume"):
            return float(e.geom["volume"]), "geometry volume"
    names, prop_re, dim = _QTY[what]
    if what == "thickness" and e.layers and all(t for _, t in e.layers):
        return float(sum(t for _, t in e.layers)), "material layers"
    for n in names:
        v = e.qto.get(n)
        if isinstance(v, (int, float)) and v > 0:
            return float(v), n
    for k, v in e.props.items():
        if e.dims.get(k) == dim and isinstance(v, (int, float)) and v > 0 and re.search(prop_re, k, re.I):
            return float(v), k
    if what == "length" and e.axis_length:
        return float(e.axis_length), "wall axis geometry"
    return None, ""


def sum_quantity(m: IfcModel, elements: list[Element], what: str) -> Result:
    total, ids, missing, srcs = 0.0, [], [], Counter()
    rows = []
    for e in elements:
        v, src = quantity(e, what)
        if v is None:
            missing.append(e.gid)
            continue
        total += v
        ids.append(e.gid)
        srcs[src] += 1
        rows.append({"gid": e.gid, "name": label(e), "value": v, "source": src})
    unit = {"area": "m2", "footprint": "m2", "volume": "m3"}.get(what, "m")
    note = f"{len(missing)} of {len(elements)} element(s) have no {what}" if missing else ""
    return Result(round(total, 3) if ids else None, unit, ids, rows,
                  source=", ".join(f"{k} ({v})" for k, v in srcs.most_common()), note=note)


def extreme(m: IfcModel, elements: list[Element], what: str, largest: bool = True) -> Result:
    vals = [(quantity(e, what), e) for e in elements]
    vals = [(v, src, e) for (v, src), e in vals if v is not None]
    if not vals:
        return Result(None, None)
    best = max(v for v, _, _ in vals) if largest else min(v for v, _, _ in vals)
    tied = [(v, src, e) for v, src, e in vals if abs(v - best) <= 1e-6 * max(1.0, abs(best))]
    unit = {"area": "m2", "volume": "m3"}.get(what, "m")
    rows = [{"gid": e.gid, "name": label(e), "value": v, "source": src,
             "storey": (m.storey(e.storey).name if m.storey(e.storey) else None)} for v, src, e in tied]
    return Result(round(best, 3), unit, [e.gid for _, _, e in tied], rows, source=tied[0][1],
                  note=f"{len(tied)} element(s) share this value" if len(tied) > 1 else "")


# ------------------------------------------------------------------------------------------------ descriptive tools


def element_details(m: IfcModel, gid: str) -> Result:
    e = m.get(gid)
    if e is None:
        return Result(None, note=f"no element {gid}")
    s = m.storey(e.storey)
    row = {"gid": e.gid, "class": e.ifc_class, "kind": e.kind, "name": e.name, "long_name": e.long_name,
           "type": e.type_name, "predefined": e.predefined, "storey": s.name if s else None,
           "materials": e.materials, "layers": e.layers, "quantities": e.qto, "properties": e.props}
    return Result(label(e), None, [e.gid], [row])


def materials(m: IfcModel, elements: list[Element]) -> Result:
    by: dict[str, list[str]] = defaultdict(list)
    for e in elements:
        for name in e.materials:
            by[name].append(e.gid)
    rows = [{"material": k, "count": len(v), "ids": v} for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))]
    return Result(len(rows), "count", [g for v in by.values() for g in v], rows)


def type_table(m: IfcModel, elements: list[Element]) -> Result:
    """Types with counts and typical width / height / thickness (doors, windows, walls, slabs, columns)."""
    by: dict[str, list[Element]] = defaultdict(list)
    for e in elements:
        by[type_label(e)].append(e)
    rows = []
    for k, es in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        row: dict[str, Any] = {"type": k, "count": len(es), "ids": [e.gid for e in es]}
        for q in ("width", "height", "thickness"):
            vals = sorted({round(v, 3) for v in (quantity(e, q)[0] for e in es) if v})
            if vals:
                row[q] = vals
        mats = Counter(mat for e in es for mat in e.materials)
        if mats:
            row["materials"] = [x for x, _ in mats.most_common(4)]
        rows.append(row)
    return Result(len(rows), "types", [e.gid for e in elements], rows)


def metadata(m: IfcModel) -> Result:
    """Authoring application, IFC schema, file header, and the project / site / building records."""
    row = {"schema": m.schema, "header": m.header, "applications": m.applications, "project": m.project,
           "sites": m.sites, "buildings": m.buildings, "units": m.units}
    ids = [x["gid"] for x in [m.project, *m.sites, *m.buildings] if x.get("gid")]
    return Result(m.schema, None, ids, [row])


def systems(m: IfcModel, pattern: str | None = None) -> Result:
    rows = []
    for g in m.groups:
        hay = " ".join(str(x) for x in (g.ifc_class, g.name, g.long_name, g.object_type, g.predefined) if x).casefold()
        if pattern and not re.search(pattern, hay):
            continue
        rows.append({"gid": g.gid, "class": g.ifc_class, "name": g.name, "type": g.predefined or g.object_type,
                     "members": len(g.members)})
    return Result(len(rows), "count", [r["gid"] for r in rows], rows)


def summary(m: IfcModel) -> Result:
    kinds = Counter(e.kind for e in m.elements)
    s = storeys(m).rows
    row = {"project": m.project.get("name"), "project_long_name": m.project.get("long_name"),
           "buildings": [b.get("name") for b in m.buildings], "schema": m.schema, "storeys": s,
           "counts": dict(kinds.most_common())}
    ids = [x["gid"] for x in [m.project, *m.buildings] if x.get("gid")]
    return Result(len(m.elements), "elements", ids, [row])
