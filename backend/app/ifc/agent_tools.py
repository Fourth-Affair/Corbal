"""The IFC tools (app.ifc.tools) as agent tools for Gemini (app.answering.agent).

Thin wrappers, no new IFC logic: the model picks IFC class names (the question's language is its job, so Danish or
Estonian wordings need no word lists), filters and quantities; the functions in app.ifc.tools compute every value
in SI units and return the GlobalIds behind it. Citations are PlanItems with id = GlobalId (app.ifc.answer.citation).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.answering.tools import CALCULATE_ANY, MAX_LIST, BaseTools, ToolTrace
from app.ifc import tools as T
from app.ifc.answer import answer_model, citation
from app.ifc.model import Element, IfcModel, resolve_ifc
from app.models import Answer, PlanItem

QUANTITIES = sorted(T._QTY)  # length, width, height, thickness, volume, area, footprint
GROUP_KEYS = ["type", "storey", "class", "predefined", "material", "function"]
_CLASS = re.compile(r"^Ifc[A-Za-z0-9]+$")
MAX_PROPS = 80

IFC_RULES = """- The "plan" here is a BIM model (IFC file). Tools take IFC class names (IfcDoor, IfcWall, IfcSpace,
  IfcSanitaryTerminal ...): translate the question's words, in any language, into classes; the model summary lists the
  classes present and their counts. Supertypes work (IfcWall includes IfcWallStandardCase). A class absent from the
  model has 0 elements. Rooms are IfcSpace; storeys / floors / levels are IfcBuildingStorey (use list_storeys).
- Narrow a class with name_contains (e.g. toilets among IfcSanitaryTerminal: ["toilet", "wc"]), predefined_types,
  storey, external or load_bearing. Say which filter you used when it matters.
- Values are SI: lengths m, areas m², volumes m³. Room areas: space_areas / find_spaces (net floor area by default).
  Copy numbers exactly as returned, or rounded to fewer decimals; say where a quantity comes from when the tool
  reports a geometry fallback or missing values.
- When the question asks for a list or breakdown (e.g. spaces with their areas, types with quantities), answer with
  text, not value: the total on the first line if there is one, then the complete list, one item per line with its
  numbers as the tools returned them (at most 40 lines; say how many more there are). Keep the explanation short.
"""

_FILTERS = {
    "ifc_classes": {"type": "array", "items": {"type": "string"},
                    "description": "IFC class names, e.g. [\"IfcDoor\"]; several = any of them"},
    "storey": {"type": "string", "description": "storey name as in list_storeys, or e.g. 'ground floor', 'level 2'"},
    "predefined_types": {"type": "array", "items": {"type": "string"}, "description": "e.g. [\"CEILING\"]"},
    "name_contains": {"type": "array", "items": {"type": "string"},
                      "description": "keep elements whose name / type name contains any of these texts"},
    "external": {"type": "boolean", "description": "only external (true) or internal (false) elements"},
    "load_bearing": {"type": "boolean", "description": "only load-bearing (true) or non-load-bearing (false)"},
}


def _round(v: Any) -> Any:
    return round(v, 3) if isinstance(v, float) else v


def _numbers(v: Any, out: list) -> None:
    """Every number in a row value, nested ones included (quantities and properties of element_details)."""
    if isinstance(v, bool):
        return
    if isinstance(v, int | float):
        out.append(_round(v))
    elif isinstance(v, dict):
        for k, x in v.items():
            if k not in ("ids", "gid"):
                _numbers(x, out)
    elif isinstance(v, list | tuple):
        for x in v:
            _numbers(x, out)


class IfcTools(BaseTools):
    summary_label = "Model summary"
    raw_confidence = "high"  # deterministic values from the model
    text_confidence = "medium"

    def __init__(self, model: IfcModel, rel: str | None = None):
        super().__init__()
        self.m, self.rel = model, rel
        self.rules = IFC_RULES

    def declarations(self) -> list[dict]:
        return DECLARATIONS

    def handlers(self) -> dict[str, Callable[..., tuple[dict, ToolTrace]]]:
        return {"count_elements": self.count_elements, "group_elements": self.group_elements,
                "list_storeys": self.list_storeys, "find_spaces": self.find_spaces, "space_areas": self.space_areas,
                "sum_quantity": self.sum_quantity, "extreme": self.extreme, "element_details": self.element_details,
                "materials": self.materials, "types": self.types, "metadata": self.metadata,
                "systems": self.systems, "calculate": self.calculate}

    def item(self, item_id: str) -> PlanItem | None:
        if item_id not in self.by_id:
            known = self.m.get(item_id) is not None or any(
                x.get("gid") == item_id for x in [self.m.project, *self.m.sites, *self.m.buildings]) \
                or any(g.gid == item_id for g in self.m.groups)
            if not known:
                return None
            self.by_id[item_id] = citation(self.m, item_id)
        return self.by_id[item_id]

    def summary(self) -> dict:
        row = T.summary(self.m).rows[0]
        return {"schema": self.m.schema, "applications": [a.get("name") or a for a in self.m.applications][:3],
                "project": row["project"], "buildings": row["buildings"],
                "storeys": [{"name": s["name"], "elevation_m": _round(s["elevation"])} for s in row["storeys"]],
                "element_classes": row["counts"]}

    def fingerprint(self) -> Any:
        if self.rel is None:
            return ["ifc", self.m.path]
        st = resolve_ifc(self.rel).stat()
        return ["ifc", self.rel, st.st_size, st.st_mtime]

    def fallback(self, question: str) -> Answer:
        return answer_model(self.m, question)

    # --- helpers -----------------------------------------------------------------------------------------------

    def _storey(self, storey: str | None) -> Element | None:
        if not storey:
            return None
        s, _ = T.resolve_storey(self.m, storey)
        if s is None:
            s = next((x for x in self.m.storeys if T.norm(x.name) == T.norm(storey)), None)
        if s is None:
            raise ValueError(f"no storey {storey!r}; storeys: {[x.name for x in self.m.storeys]}")
        return s

    def _select(self, ifc_classes: list[str] | None = None, storey: str | None = None,
                predefined_types: list[str] | None = None, name_contains: list[str] | None = None,
                external: bool | None = None, load_bearing: bool | None = None) -> tuple[list[Element], dict]:
        classes = [c.strip() for c in (ifc_classes or []) if c and c.strip()]
        bad = [c for c in classes if not _CLASS.match(c)]
        if bad:
            raise ValueError(f"{bad} are not IFC class names (use e.g. IfcDoor, IfcWall, IfcSpace)")
        s = self._storey(storey)
        pattern = "|".join(re.escape(t.casefold()) for t in name_contains or [] if t.strip()) or None
        predefined = [p.upper() for p in predefined_types or []] or None
        elements = T.select(self.m, classes or None, s, predefined, pattern, external, load_bearing)
        applied = {k: v for k, v in {"ifc_classes": classes, "storey": s.name if s else None,
                                     "predefined_types": predefined, "name_contains": name_contains,
                                     "external": external, "load_bearing": load_bearing}.items() if v not in (None, [])}
        return elements, applied

    def _out(self, r: T.Result, extra: dict | None = None) -> tuple[dict, ToolTrace]:
        """A Result as the model sees it (rows capped, numbers rounded) + the trace (every id, every number)."""
        rows, ids, numbers = [], list(r.ids), []
        if isinstance(r.value, int | float) and not isinstance(r.value, bool):
            numbers.append(_round(r.value))
        for raw in r.rows:
            row = {}
            for k, v in raw.items():
                if k == "gid":
                    row["id"] = v
                    ids.append(v)
                elif k == "ids":
                    ids.extend(v)
                    row["ids"] = v[:10] + (["..."] if len(v) > 10 else [])
                else:
                    row[k] = _round(v)
                    _numbers(v, numbers)
            rows.append(row)
        out: dict[str, Any] = {"value": _round(r.value), "unit": r.unit, **(extra or {})}
        if r.source:
            out["source"] = r.source
        if r.note:
            out["note"] = r.note
        out["rows"] = rows[:MAX_LIST]
        if len(rows) > MAX_LIST:
            out["truncated"] = f"showing {MAX_LIST} of {len(rows)} rows"
        geometry = "geometry" in r.source or any("geometry" in str(x.get("source", "")) for x in r.rows)
        return out, ToolTrace(item_ids=list(dict.fromkeys(i for i in ids if i)), numbers=numbers,
                              confidence="medium" if geometry else "high")

    # --- tools -------------------------------------------------------------------------------------------------

    def count_elements(self, ifc_classes: list[str], **filters) -> tuple[dict, ToolTrace]:
        els, applied = self._select(ifc_classes, **filters)
        rows = [{"gid": e.gid, "class": e.kind, "name": T.label(e)} for e in els]
        extra = {"filters": applied}
        if not els:
            extra["note"] = "no matching elements in this model"
        return self._out(T.Result(len(els), "count", [e.gid for e in els], rows), extra)

    def group_elements(self, ifc_classes: list[str], by: str, **filters) -> tuple[dict, ToolTrace]:
        if by not in GROUP_KEYS:
            raise ValueError(f"by must be one of {GROUP_KEYS}")
        els, applied = self._select(ifc_classes, **filters)
        r = T.group_count(self.m, els, by)
        return self._out(T.Result(r.value, "groups", r.ids, r.rows), {"filters": applied, "elements": len(els)})

    def list_storeys(self) -> tuple[dict, ToolTrace]:
        return self._out(T.storeys(self.m))

    def find_spaces(self, query: str, storey: str | None = None) -> tuple[dict, ToolTrace]:
        """Spaces whose number / name appear in the query, with their areas, height and volume."""
        found = T.find_spaces(self.m, query, self._storey(storey))
        if not found:  # a plain name fragment ("kitchen") rather than a full name
            found = T.select_by_name(self.m, re.escape(query.casefold()), self._storey(storey), spaces_only=True)
        rows = []
        for e in found[:MAX_LIST]:
            net, net_src = T.space_area(e, "net")
            gross, _ = T.space_area(e, "gross")
            h, _ = T.space_height(e)
            vol, _ = T.quantity(e, "volume")
            s = self.m.storey(e.storey)
            rows.append({"gid": e.gid, "number": e.name, "name": e.long_name, "storey": s.name if s else None,
                         "net_area_m2": net, "gross_area_m2": gross, "area_source": net_src,
                         "height_m": h, "volume_m3": vol})
        return self._out(T.Result(len(found), "count", [e.gid for e in found], rows))

    def space_areas(self, storey: str | None = None, name_contains: list[str] | None = None,
                    which: str = "net") -> tuple[dict, ToolTrace]:
        if which not in ("net", "gross"):
            raise ValueError("which must be 'net' or 'gross'")
        pattern = "|".join(re.escape(t.casefold()) for t in name_contains or [] if t.strip()) or None
        r = T.space_table(self.m, self._storey(storey), which, pattern)
        return self._out(r, {"what": f"total {which} floor area of {len(r.rows)} space(s)", "spaces": len(r.rows)})

    def sum_quantity(self, ifc_classes: list[str], quantity: str, **filters) -> tuple[dict, ToolTrace]:
        if quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        els, applied = self._select(ifc_classes, **filters)
        return self._out(T.sum_quantity(self.m, els, quantity), {"filters": applied, "elements": len(els)})

    def extreme(self, ifc_classes: list[str], quantity: str, largest: bool = True, **filters) -> tuple[dict, ToolTrace]:
        if quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        els, applied = self._select(ifc_classes, **filters)
        return self._out(T.extreme(self.m, els, quantity, largest), {"filters": applied, "elements": len(els)})

    def element_details(self, global_id: str) -> tuple[dict, ToolTrace]:
        r = T.element_details(self.m, global_id)
        if r.value is None:
            raise ValueError(r.note)
        row = dict(r.rows[0])
        props = row.get("properties") or {}
        if len(props) > MAX_PROPS:
            row["properties"] = dict(list(props.items())[:MAX_PROPS])
            row["properties_truncated"] = f"showing {MAX_PROPS} of {len(props)}"
        return self._out(T.Result(r.value, None, r.ids, [row]))

    def materials(self, ifc_classes: list[str] | None = None, **filters) -> tuple[dict, ToolTrace]:
        els, applied = self._select(ifc_classes, **filters)
        return self._out(T.materials(self.m, els), {"filters": applied, "elements": len(els)})

    def types(self, ifc_classes: list[str], **filters) -> tuple[dict, ToolTrace]:
        els, applied = self._select(ifc_classes, **filters)
        return self._out(T.type_table(self.m, els), {"filters": applied, "elements": len(els)})

    def metadata(self) -> tuple[dict, ToolTrace]:
        return self._out(T.metadata(self.m))

    def systems(self, name_contains: str | None = None) -> tuple[dict, ToolTrace]:
        pattern = re.escape(name_contains.casefold()) if name_contains else None
        return self._out(T.systems(self.m, pattern))


def _decl(name: str, description: str, props: dict, required: list[str] | None = None, filters: bool = False) -> dict:
    properties = {**props, **({k: v for k, v in _FILTERS.items() if k not in props} if filters else {})}
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []}}


_CLASSES = {"ifc_classes": _FILTERS["ifc_classes"]}
_QTY_ARG = {"quantity": {"type": "string", "enum": QUANTITIES,
                         "description": "area = surface / floor area, footprint = plan area"}}

DECLARATIONS = [
    _decl("count_elements", "How many elements of the given IFC classes, optionally filtered; lists them with GlobalIds.",
          _CLASSES, ["ifc_classes"], filters=True),
    _decl("group_elements", "Counts of the selected elements grouped by type, storey, class, predefined type, "
          "material or function (space names).",
          {**_CLASSES, "by": {"type": "string", "enum": GROUP_KEYS}}, ["ifc_classes", "by"], filters=True),
    _decl("list_storeys", "Storeys bottom-up with name, elevation (m), height (m) and number of elements.", {}),
    _decl("find_spaces", "Spaces (rooms) whose number or name appears in the query, e.g. 'A203', 'Living Room', "
          "'kitchen': net and gross floor area (m²), height (m), volume (m³), storey.",
          {"query": {"type": "string"}, "storey": _FILTERS["storey"]}, ["query"]),
    _decl("space_areas", "Floor areas (m²) of all spaces, or those on one storey / whose name contains a text, "
          "largest first, with the total.",
          {"storey": _FILTERS["storey"], "name_contains": _FILTERS["name_contains"],
           "which": {"type": "string", "enum": ["net", "gross"], "description": "default net"}}),
    _decl("sum_quantity", "Total length / width / height / thickness (m), area (m²) or volume (m³) of the selected "
          "elements, with each element's value and source.", {**_CLASSES, **_QTY_ARG},
          ["ifc_classes", "quantity"], filters=True),
    _decl("extreme", "The element(s) with the largest (or smallest) quantity among the selected elements.",
          {**_CLASSES, **_QTY_ARG, "largest": {"type": "boolean", "description": "default true; false = smallest"}},
          ["ifc_classes", "quantity"], filters=True),
    _decl("element_details", "Everything about one element: class, type, storey, materials and layer thicknesses, "
          "quantities and properties (e.g. fire rating, IsExternal).", {"global_id": {"type": "string"}}, ["global_id"]),
    _decl("materials", "Materials used by the selected elements (all elements if no class) with element counts.",
          _CLASSES, filters=True),
    _decl("types", "Type objects of the selected elements with counts, typical width / height / thickness and "
          "materials.", _CLASSES, ["ifc_classes"], filters=True),
    _decl("metadata", "IFC schema, authoring application, file header, project / site / building records "
          "(name, address, coordinates) and units.", {}),
    _decl("systems", "Systems and zones (IfcSystem / IfcZone / distribution systems) with member counts.",
          {"name_contains": {"type": "string"}}),
    CALCULATE_ANY,
]
