"""IFC file -> IfcModel: everything the answer tools need, as plain addressable evidence.

Every element keeps its IFC GlobalId (the citation), its class, the storey it belongs to (through containment,
aggregation, nesting and door/window -> opening -> wall), its type object, materials (with layer thicknesses),
every single-value property ("Pset.Prop") and every quantity. Lengths, areas and volumes are converted to metres,
m² and m³ here, from the project units (or a quantity's own unit), so nothing downstream guesses units.

Parsing a large model takes seconds to minutes; the result is cached as gzipped JSON in CACHE_DIR/ifc/, keyed by
file name, size and mtime (and EXTRACT_VERSION).
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import CACHE_DIR, DATASET_DIR

log = logging.getLogger(__name__)

EXTRACT_VERSION = 6  # bump when the extracted content changes

# Measure types -> which project unit converts them to SI
_MEASURE_UNIT = {
    "IfcLengthMeasure": "LENGTHUNIT", "IfcPositiveLengthMeasure": "LENGTHUNIT", "IfcNonNegativeLengthMeasure": "LENGTHUNIT",
    "IfcAreaMeasure": "AREAUNIT", "IfcVolumeMeasure": "VOLUMEUNIT",
}
_QUANTITY_UNIT = {"IfcQuantityLength": ("LengthValue", "LENGTHUNIT"), "IfcQuantityArea": ("AreaValue", "AREAUNIT"),
                  "IfcQuantityVolume": ("VolumeValue", "VOLUMEUNIT"), "IfcQuantityCount": ("CountValue", None),
                  "IfcQuantityWeight": ("WeightValue", "MASSUNIT"), "IfcQuantityTime": ("TimeValue", None)}
# Quantity dimension by name, used to tag quantities and numeric properties: "area" / "volume" / "length"
QUANTITY_DIM = {"LENGTHUNIT": "length", "AREAUNIT": "area", "VOLUMEUNIT": "volume", "MASSUNIT": "mass"}

# Generic IFC2X3 classes whose real kind is only told by the type object (IfcFlowTerminal + IfcSanitaryTerminalType)
_GENERIC = {"IfcFlowTerminal", "IfcFlowSegment", "IfcFlowFitting", "IfcFlowController", "IfcFlowMovingDevice",
            "IfcFlowStorageDevice", "IfcFlowTreatmentDevice", "IfcEnergyConversionDevice", "IfcDistributionFlowElement",
            "IfcDistributionControlElement", "IfcBuildingElementProxy", "IfcFurnishingElement", "IfcDistributionElement"}


@dataclass
class Element:
    gid: str
    ifc_class: str  # entity as written, e.g. IfcWallStandardCase, IfcFlowTerminal
    kind: str  # effective class: IfcWall, IfcSanitaryTerminal (from the type object for IFC2X3 generic classes)
    classes: list[str]  # kind + ifc_class and all their supertypes (for class queries)
    name: str | None = None
    long_name: str | None = None
    description: str | None = None
    object_type: str | None = None
    predefined: str | None = None
    type_gid: str | None = None
    type_name: str | None = None
    type_class: str | None = None
    storey: str | None = None  # GlobalId of the storey
    building: str | None = None
    materials: list[str] = field(default_factory=list)
    layers: list[tuple[str, float | None]] = field(default_factory=list)  # (material, thickness m)
    props: dict[str, Any] = field(default_factory=dict)  # "Pset.Prop" -> value (lengths/areas/volumes in SI)
    qto: dict[str, float] = field(default_factory=dict)  # quantity name -> SI value (first set wins, Qto_* first)
    dims: dict[str, str] = field(default_factory=dict)  # prop / quantity name -> "length" | "area" | "volume"
    axis_length: float | None = None  # walls: length of the axis polyline (m)
    elevation: float | None = None  # storeys: elevation (m)
    geom: dict[str, float] = field(default_factory=dict)  # spaces: footprint_area (m²), volume (m³), height (m) of the body


@dataclass
class Group:
    """IfcSystem / IfcZone / IfcDistributionSystem with its members."""

    gid: str
    ifc_class: str
    name: str | None
    long_name: str | None
    object_type: str | None
    predefined: str | None
    members: list[str]


@dataclass
class IfcModel:
    path: str
    schema: str
    header: dict[str, Any]
    applications: list[dict[str, Any]]
    units: dict[str, Any]
    project: dict[str, Any]
    sites: list[dict[str, Any]]
    buildings: list[dict[str, Any]]
    storeys: list[Element]
    elements: list[Element]  # every IfcElement (not openings) and every IfcSpace
    groups: list[Group]

    _by_gid: dict[str, Element] = field(default_factory=dict, repr=False, compare=False)

    def get(self, gid: str) -> Element | None:
        if not self._by_gid:
            self._by_gid = {e.gid: e for e in [*self.storeys, *self.elements]}
        return self._by_gid.get(gid)

    def storey(self, gid: str | None) -> Element | None:
        return next((s for s in self.storeys if s.gid == gid), None) if gid else None


# --------------------------------------------------------------------------------------------- extraction helpers


def _clean(v: Any) -> Any:
    if isinstance(v, str):
        v = v.strip()
        return v[:300] if v else None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (tuple, list)):
        return ", ".join(str(_clean(x)) for x in v if x is not None)
    return v


def _wrapped(value) -> tuple[Any, str | None]:
    """IfcValue (entity_instance wrapper) -> (python value, measure type)."""
    if value is None:
        return None, None
    try:
        return value.wrappedValue, value.is_a()
    except AttributeError:
        return value, None


def _degrees(compound) -> float | None:
    """IfcCompoundPlaneAngleMeasure (deg, min, sec, millionths) -> decimal degrees."""
    if not compound:
        return None
    parts = list(compound) + [0] * (4 - len(compound))
    sign = -1 if any(p < 0 for p in parts) else 1
    d, m, s, u = (abs(p) for p in parts[:4])
    return sign * (d + m / 60 + (s + u / 1e6) / 3600)


def _address(a) -> str | None:
    if a is None:
        return None
    bits = []
    for attr in ("AddressLines", "PostalBox", "Town", "Region", "PostalCode", "Country"):
        v = getattr(a, attr, None)
        if v:
            bits.extend(v if isinstance(v, (list, tuple)) else [v])
    return ", ".join(str(b) for b in bits if b) or None


class _Extractor:
    def __init__(self, f):
        import ifcopenshell
        import ifcopenshell.util.unit as uu

        self.f = f
        self.uu = uu
        self.schema = ifcopenshell.ifcopenshell_wrapper.schema_by_name(f.schema_identifier)
        self.scale = {}
        length = uu.calculate_unit_scale(f, "LENGTHUNIT")
        self.scale["LENGTHUNIT"] = length
        assigned = set()
        if (projects := f.by_type("IfcProject")) and projects[0].UnitsInContext:
            assigned = {getattr(u, "UnitType", None) for u in projects[0].UnitsInContext.Units}
        for ut, power in (("AREAUNIT", 2), ("VOLUMEUNIT", 3)):
            self.scale[ut] = uu.calculate_unit_scale(f, ut) if ut in assigned else length**power
        self.scale["MASSUNIT"] = uu.calculate_unit_scale(f, "MASSUNIT") if "MASSUNIT" in assigned else 1.0
        self.unit_names = {}
        for ut in ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT"):
            try:
                u = uu.get_project_unit(f, ut)
                self.unit_names[ut] = uu.get_full_unit_name(u) if u else None
            except Exception:  # noqa: BLE001
                self.unit_names[ut] = None
        self._supers: dict[str, list[str]] = {}
        self._index()

    # ---- relationship indexes (one pass over the relationship entities instead of per-element inverse lookups)
    def _index(self):
        f = self.f
        self.parent: dict[int, Any] = {}
        for rel in f.by_type("IfcRelContainedInSpatialStructure"):
            for e in rel.RelatedElements or []:
                self.parent.setdefault(e.id(), rel.RelatingStructure)
        for rel in f.by_type("IfcRelAggregates"):
            for e in rel.RelatedObjects or []:
                self.parent.setdefault(e.id(), rel.RelatingObject)
        for rel in f.by_type("IfcRelNests"):
            for e in rel.RelatedObjects or []:
                self.parent.setdefault(e.id(), rel.RelatingObject)
        opening_host = {rel.RelatedOpeningElement.id(): rel.RelatingBuildingElement for rel in f.by_type("IfcRelVoidsElement")}
        for rel in f.by_type("IfcRelFillsElement"):
            host = opening_host.get(rel.RelatingOpeningElement.id())
            if host is not None:
                self.parent.setdefault(rel.RelatedBuildingElement.id(), host)
        self.type_of: dict[int, Any] = {}
        for rel in f.by_type("IfcRelDefinesByType"):
            for e in rel.RelatedObjects or []:
                self.type_of[e.id()] = rel.RelatingType
        self.psets_of: dict[int, list] = {}
        for rel in f.by_type("IfcRelDefinesByProperties"):
            d = rel.RelatingPropertyDefinition
            if d is None or not hasattr(d, "is_a"):
                continue
            for e in rel.RelatedObjects or []:
                self.psets_of.setdefault(e.id(), []).append(d)
        self.material_of: dict[int, Any] = {}
        for rel in f.by_type("IfcRelAssociatesMaterial"):
            for e in rel.RelatedObjects or []:
                self.material_of.setdefault(e.id(), rel.RelatingMaterial)

    def supers(self, cls: str) -> list[str]:
        if cls not in self._supers:
            out = []
            try:
                d = self.schema.declaration_by_name(cls)
                while d is not None:
                    out.append(d.name())
                    d = d.supertype()
            except Exception:  # noqa: BLE001  unknown entity name
                out = [cls]
            self._supers[cls] = out
        return self._supers[cls]

    def storey_and_building(self, e) -> tuple[str | None, str | None]:
        storey = building = None
        x, seen = e, set()
        while x is not None and x.id() not in seen:
            seen.add(x.id())
            if storey is None and x.is_a("IfcBuildingStorey"):
                storey = x.GlobalId
            if x.is_a("IfcBuilding"):
                building = x.GlobalId
                break
            x = self.parent.get(x.id())
        return storey, building

    def quantity_value(self, q) -> tuple[str, float, str | None] | None:
        spec = _QUANTITY_UNIT.get(q.is_a())
        if spec is None:
            return None
        attr, ut = spec
        v = getattr(q, attr, None)
        if v is None:
            return None
        scale = 1.0
        if ut:
            unit = getattr(q, "Unit", None)
            scale = self.uu.get_unit_scale(unit) if unit is not None else self.scale.get(ut, 1.0)
        return q.Name, float(v) * scale, QUANTITY_DIM.get(ut) if ut else None

    def read_definitions(self, defs, props: dict, qto: dict, dims: dict, override: bool):
        # Qto_* (standard) sets first, then BaseQuantities, then the rest: the first value of a quantity name wins
        def order(d):
            n = getattr(d, "Name", "") or ""
            return (0 if n.startswith("Qto_") else 1 if n == "BaseQuantities" else 2)

        for d in sorted(defs, key=order):
            pname = getattr(d, "Name", None) or d.is_a()
            if d.is_a("IfcElementQuantity"):
                for q in d.Quantities or []:
                    r = self.quantity_value(q)
                    if r is None:
                        continue
                    name, v, dim = r
                    key = f"{pname}.{name}"
                    if override or key not in props:
                        props[key] = v
                    if dim:
                        dims[key] = dim
                        dims.setdefault(name, dim)
                    if override or name not in qto:
                        qto[name] = v
            elif d.is_a("IfcPropertySet"):
                for p in d.HasProperties or []:
                    key = f"{pname}.{p.Name}"
                    if not override and key in props:
                        continue
                    if p.is_a("IfcPropertySingleValue"):
                        v, mtype = _wrapped(p.NominalValue)
                        ut = _MEASURE_UNIT.get(mtype or "")
                        if ut and isinstance(v, (int, float)):
                            v = float(v) * self.scale[ut]
                            dims[key] = QUANTITY_DIM[ut]
                        v = _clean(v)
                    elif p.is_a("IfcPropertyEnumeratedValue"):
                        v = _clean([_wrapped(x)[0] for x in (p.EnumerationValues or [])])
                    elif p.is_a("IfcPropertyListValue"):
                        v = _clean([_wrapped(x)[0] for x in (p.ListValues or [])])
                    else:
                        continue
                    if v is not None:
                        props[key] = v

    def materials(self, select) -> tuple[list[str], list[tuple[str, float | None]]]:
        names, layers = [], []
        if select is None:
            return names, layers
        k = select.is_a()
        ls = self.scale["LENGTHUNIT"]
        if k == "IfcMaterial":
            names.append(select.Name)
        elif k == "IfcMaterialList":
            names += [m.Name for m in select.Materials or []]
        elif k in ("IfcMaterialLayerSetUsage", "IfcMaterialLayerSet"):
            s = select.ForLayerSet if k == "IfcMaterialLayerSetUsage" else select
            for layer in s.MaterialLayers or []:
                m = layer.Material.Name if layer.Material else None
                t = float(layer.LayerThickness) * ls if layer.LayerThickness is not None else None
                layers.append((m or "?", t))
                if m:
                    names.append(m)
        elif k in ("IfcMaterialProfileSetUsage", "IfcMaterialProfileSet"):
            s = select.ForProfileSet if k == "IfcMaterialProfileSetUsage" else select
            names += [p.Material.Name for p in s.MaterialProfiles or [] if p.Material]
        elif k == "IfcMaterialConstituentSet":
            names += [c.Material.Name for c in select.MaterialConstituents or [] if c.Material]
        elif k == "IfcMaterialLayer" and select.Material:
            names.append(select.Material.Name)
        elif k == "IfcMaterialProfile" and select.Material:
            names.append(select.Material.Name)
        seen = []
        for n in names:
            if n and n not in seen:
                seen.append(n)
        return seen, layers

    def axis_length(self, e) -> float | None:
        """Length of a wall's 'Axis' representation (polyline / trimmed line), in metres."""
        rep = getattr(e, "Representation", None)
        if rep is None:
            return None
        for r in rep.Representations or []:
            if (r.RepresentationIdentifier or "") != "Axis":
                continue
            total = 0.0
            for item in r.Items or []:
                if item.is_a("IfcPolyline"):
                    pts = [p.Coordinates for p in item.Points]
                    total += sum(math.dist(a[:2], b[:2]) for a, b in zip(pts, pts[1:]))
                elif item.is_a("IfcTrimmedCurve") and item.BasisCurve.is_a("IfcLine"):
                    t1, t2 = item.Trim1, item.Trim2
                    p1 = next((t for t in t1 if hasattr(t, "is_a") and t.is_a("IfcCartesianPoint")), None)
                    p2 = next((t for t in t2 if hasattr(t, "is_a") and t.is_a("IfcCartesianPoint")), None)
                    if p1 is not None and p2 is not None:
                        total += math.dist(p1.Coordinates[:2], p2.Coordinates[:2])
                    else:
                        v1 = next((t.wrappedValue for t in t1 if hasattr(t, "wrappedValue")), None)
                        v2 = next((t.wrappedValue for t in t2 if hasattr(t, "wrappedValue")), None)
                        if v1 is not None and v2 is not None:
                            mag = item.BasisCurve.Dir.Magnitude or 1.0
                            total += abs(v2 - v1) * mag
                else:
                    return None
            return total * self.scale["LENGTHUNIT"] if total > 0 else None
        return None

    def element(self, e) -> Element:
        cls = e.is_a()
        t = self.type_of.get(e.id())
        kind = cls
        for suffix in ("StandardCase", "ElementedCase"):
            if kind.endswith(suffix) and kind != "Ifc" + suffix:
                kind = kind[: -len(suffix)]
        if t is not None and cls in _GENERIC:
            tc = t.is_a()
            cand = tc[:-4] if tc.endswith("Type") else tc[:-5] if tc.endswith("Style") else None
            if cand and cand.startswith("Ifc"):  # IFC4 name even when the file is IFC2X3 (IfcSanitaryTerminal)
                kind = cand
        classes = list(dict.fromkeys([*self.supers(kind), *self.supers(cls)]))
        props: dict[str, Any] = {}
        qto: dict[str, float] = {}
        dims: dict[str, str] = {}
        self.read_definitions(self.psets_of.get(e.id(), []), props, qto, dims, override=False)
        if t is not None:  # type properties fill what the occurrence does not say
            self.read_definitions(list(getattr(t, "HasPropertySets", None) or []), props, qto, dims, override=False)
        mats, layers = self.materials(self.material_of.get(e.id()))
        if not mats and t is not None:
            mats, layers = self.materials(self.material_of.get(t.id()))
        storey, building = self.storey_and_building(e)
        predefined = getattr(e, "PredefinedType", None)
        if predefined in (None, "NOTDEFINED", "USERDEFINED") and t is not None:
            predefined = getattr(t, "PredefinedType", None) or predefined
        el = Element(
            gid=e.GlobalId, ifc_class=cls, kind=kind, classes=classes, name=_clean(e.Name),
            long_name=_clean(getattr(e, "LongName", None)), description=_clean(e.Description),
            object_type=_clean(getattr(e, "ObjectType", None)), predefined=predefined if isinstance(predefined, str) else None,
            type_gid=t.GlobalId if t is not None else None, type_name=_clean(t.Name) if t is not None else None,
            type_class=t.is_a() if t is not None else None, storey=storey, building=building,
            materials=mats, layers=layers, props=props, qto=qto, dims=dims)
        for attr in ("OverallWidth", "OverallHeight"):  # doors / windows
            v = getattr(e, attr, None)
            if isinstance(v, (int, float)):
                el.qto.setdefault(attr, float(v) * self.scale["LENGTHUNIT"])
                el.dims[attr] = "length"
        if "IfcWall" in classes:
            try:
                el.axis_length = self.axis_length(e)
            except Exception:  # noqa: BLE001
                el.axis_length = None
        return el


def _space_geometry(spaces) -> dict[int, dict[str, float]]:
    """Footprint area, volume and height of each space's body shape (world coordinates, SI units)."""
    if not spaces:
        return {}
    import ifcopenshell.geom as geom
    import ifcopenshell.util.shape as us

    settings = geom.settings()
    try:
        settings.set("convert-back-units", False)  # SI metres
    except Exception:  # noqa: BLE001
        pass
    out = {}
    for sp in spaces:
        if getattr(sp, "Representation", None) is None:
            continue
        try:
            g = geom.create_shape(settings, sp).geometry
            area = _floor_area(g)
            if area <= 0:
                area = float(us.get_footprint_area(g))
            vol = abs(float(us.get_volume(g)))
            height = float(us.get_z(g))
        except Exception:  # noqa: BLE001  broken / unsupported geometry: no geometric values
            continue
        if area > 0:
            d = {"footprint_area": area, "height": height}
            if vol > 0.5 * area * height:  # open surface models give meaningless volumes
                d["volume"] = vol
            out[sp.id()] = d
    return out


def _floor_area(g) -> float:
    """Plan area of the triangles lying at the bottom of the shape (within 1 cm of its lowest point). Works for
    solids and surface models alike, whatever the face orientation."""
    import numpy as np

    v = np.asarray(g.verts, dtype=float).reshape(-1, 3)
    f = np.asarray(g.faces, dtype=int).reshape(-1, 3)
    if not len(v) or not len(f):
        return 0.0
    zmin = v[:, 2].min()
    tri = v[f]
    low = np.all(np.abs(tri[:, :, 2] - zmin) < 0.01, axis=1)
    t = tri[low]
    if not len(t):
        return 0.0
    a = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
    return float(a.sum())


def extract(path: str | Path, rel: str | None = None) -> IfcModel:
    import ifcopenshell

    f = ifcopenshell.open(str(path))
    x = _Extractor(f)
    h = f.header
    fn, fd = h.file_name, h.file_description
    header = {
        "file_name": _clean(fn.name), "time_stamp": _clean(fn.time_stamp), "author": _clean(fn.author),
        "organization": _clean(fn.organization), "preprocessor_version": _clean(fn.preprocessor_version),
        "originating_system": _clean(fn.originating_system), "authorization": _clean(fn.authorization),
        "description": _clean(fd.description), "schema": f.schema_identifier,
    }
    apps = []
    for a in f.by_type("IfcApplication"):
        apps.append({"name": _clean(a.ApplicationFullName), "version": _clean(a.Version),
                     "identifier": _clean(a.ApplicationIdentifier),
                     "developer": _clean(a.ApplicationDeveloper.Name) if a.ApplicationDeveloper else None})
    proj = f.by_type("IfcProject")
    project = {}
    if proj:
        p = proj[0]
        project = {"gid": p.GlobalId, "name": _clean(p.Name), "long_name": _clean(p.LongName),
                   "description": _clean(p.Description), "phase": _clean(p.Phase), "object_type": _clean(p.ObjectType)}
        props: dict = {}
        x.read_definitions(x.psets_of.get(p.id(), []), props, {}, {}, override=False)
        project["props"] = props
    sites = []
    for s in f.by_type("IfcSite"):
        props = {}
        x.read_definitions(x.psets_of.get(s.id(), []), props, {}, {}, override=False)
        sites.append({"gid": s.GlobalId, "name": _clean(s.Name), "long_name": _clean(s.LongName),
                      "description": _clean(s.Description),
                      "latitude": _degrees(s.RefLatitude), "longitude": _degrees(s.RefLongitude),
                      "elevation": float(s.RefElevation) * x.scale["LENGTHUNIT"] if s.RefElevation is not None else None,
                      "address": _address(getattr(s, "SiteAddress", None)), "land_title": _clean(s.LandTitleNumber),
                      "props": props})
    buildings = []
    for b in f.by_type("IfcBuilding"):
        props = {}
        qto: dict = {}
        x.read_definitions(x.psets_of.get(b.id(), []), props, qto, {}, override=False)
        buildings.append({"gid": b.GlobalId, "name": _clean(b.Name), "long_name": _clean(b.LongName),
                          "description": _clean(b.Description), "object_type": _clean(b.ObjectType),
                          "address": _address(getattr(b, "BuildingAddress", None)), "props": props, "qto": qto})
    storeys = []
    for s in f.by_type("IfcBuildingStorey"):
        el = x.element(s)
        if s.Elevation is not None:
            el.elevation = float(s.Elevation) * x.scale["LENGTHUNIT"]
        else:
            try:
                import ifcopenshell.util.placement as up

                el.elevation = float(up.get_local_placement(s.ObjectPlacement)[2][3]) * x.scale["LENGTHUNIT"]
            except Exception:  # noqa: BLE001
                el.elevation = None
        storeys.append(el)
    storeys.sort(key=lambda s: (s.elevation is None, s.elevation or 0.0))
    elements = []
    for e in f.by_type("IfcElement"):
        if e.is_a("IfcFeatureElement"):  # openings, projections: holes, not things
            continue
        elements.append(x.element(e))
    spaces = f.by_type("IfcSpace")
    geoms = _space_geometry(spaces)
    for e in spaces:
        el = x.element(e)
        el.geom = geoms.get(e.id(), {})
        elements.append(el)
    groups = []
    for g in f.by_type("IfcGroup"):
        if not g.is_a("IfcSystem") and not g.is_a("IfcZone"):
            continue
        members = []
        for grouping in getattr(g, "IsGroupedBy", None) or []:
            members += [o.GlobalId for o in grouping.RelatedObjects or [] if hasattr(o, "GlobalId")]
        groups.append(Group(gid=g.GlobalId, ifc_class=g.is_a(), name=_clean(g.Name),
                            long_name=_clean(getattr(g, "LongName", None)), object_type=_clean(g.ObjectType),
                            predefined=getattr(g, "PredefinedType", None), members=members))
    units = {"length_scale": x.scale["LENGTHUNIT"], "area_scale": x.scale["AREAUNIT"],
             "volume_scale": x.scale["VOLUMEUNIT"], "names": x.unit_names}
    return IfcModel(path=rel or str(path), schema=f.schema_identifier, header=header, applications=apps, units=units,
                    project=project, sites=sites, buildings=buildings, storeys=storeys, elements=elements, groups=groups)


# --------------------------------------------------------------------------------------------- cache


def _to_dict(m: IfcModel) -> dict:
    d = asdict(m)
    d.pop("_by_gid", None)
    return d


def _from_dict(d: dict) -> IfcModel:
    d = dict(d)
    d.pop("_by_gid", None)
    d["storeys"] = [Element(**e) for e in d["storeys"]]
    d["elements"] = [Element(**e) for e in d["elements"]]
    d["groups"] = [Group(**g) for g in d["groups"]]
    return IfcModel(**d)


def resolve_ifc(rel: str) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = (DATASET_DIR / rel).resolve()
        if not p.is_relative_to(DATASET_DIR.resolve()):
            raise FileNotFoundError(rel)
    if not p.is_file():
        raise FileNotFoundError(rel)
    return p


def _cache_file(p: Path) -> Path:
    st = p.stat()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem)[:80]
    return CACHE_DIR / "ifc" / f"{safe}-{st.st_size:x}-{int(st.st_mtime):x}-v{EXTRACT_VERSION}.json.gz"


@lru_cache(maxsize=8)
def _load(path_str: str, rel: str, mtime: float) -> IfcModel:
    p = Path(path_str)
    cf = _cache_file(p)
    if cf.is_file():
        try:
            with gzip.open(cf, "rt", encoding="utf-8") as fh:
                m = _from_dict(json.load(fh))
            m.path = rel
            return m
        except Exception:  # noqa: BLE001  stale / broken cache: re-extract
            log.warning("ignoring broken IFC cache %s", cf)
    m = extract(p, rel)
    try:
        cf.parent.mkdir(parents=True, exist_ok=True)
        tmp = cf.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(_to_dict(m), fh, ensure_ascii=False, default=str)
        tmp.replace(cf)
    except OSError:
        log.warning("could not write IFC cache %s", cf)
    return m


def load_model(rel: str) -> IfcModel:
    """Dataset-relative (or absolute) path -> IfcModel, cached on disk and in memory."""
    p = resolve_ifc(rel)
    return _load(str(p), rel, p.stat().st_mtime)
