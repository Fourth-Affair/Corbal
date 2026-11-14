"""IFC answering on a tiny model built here with ifcopenshell (no dataset needed). Project units are millimetres,
so every area / volume / length also checks the unit conversion."""

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from app.ifc import model as model_mod  # noqa: E402
from app.ifc import tools as T  # noqa: E402
from app.ifc.answer import answer_model, answer_question  # noqa: E402


def _guid():
    import ifcopenshell.guid

    return ifcopenshell.guid.new()


def build_ifc(path):
    f = ifcopenshell.file(schema="IFC4")
    o = f.createIfcOwnerHistory()
    app_ = f.createIfcApplication(f.createIfcOrganization(Name="TestSoft Oy"), "1.2", "TestCAD", "TCAD")
    o.OwningApplication = app_
    mm = f.createIfcSIUnit(UnitType="LENGTHUNIT", Prefix="MILLI", Name="METRE")
    m2 = f.createIfcSIUnit(UnitType="AREAUNIT", Name="SQUARE_METRE")
    m3 = f.createIfcSIUnit(UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
    units = f.createIfcUnitAssignment([mm, m2, m3])
    origin = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, 0.0)))
    ctx = f.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, origin, None)
    body_ctx = f.createIfcGeometricRepresentationSubContext("Body", "Model", None, None, None, None, ctx, None, "MODEL_VIEW", None)
    project = f.createIfcProject(_guid(), o, "Testitalo", None, None, "Test house", None, [ctx], units)
    site = f.createIfcSite(_guid(), o, "Tontti", None, None, None, None, None, "ELEMENT", (60, 10, 0), (24, 56, 0), 12000.0)
    building = f.createIfcBuilding(_guid(), o, "Talo A", None, None, None, None, None, "ELEMENT")
    place = f.createIfcLocalPlacement(None, origin)
    s0 = f.createIfcBuildingStorey(_guid(), o, "1. kerros", None, None, place, None, None, "ELEMENT", 0.0)
    s1 = f.createIfcBuildingStorey(_guid(), o, "2. kerros", None, None, place, None, None, "ELEMENT", 3000.0)
    f.createIfcRelAggregates(_guid(), o, None, None, project, [site])
    f.createIfcRelAggregates(_guid(), o, None, None, site, [building])
    f.createIfcRelAggregates(_guid(), o, None, None, building, [s0, s1])

    def pset(el, name, props):
        vals = []
        for k, v in props.items():
            if isinstance(v, bool):
                vals.append(f.createIfcPropertySingleValue(k, None, f.createIfcBoolean(v), None))
            else:
                vals.append(f.createIfcPropertySingleValue(k, None, f.createIfcLabel(v), None))
        ps = f.createIfcPropertySet(_guid(), o, name, None, vals)
        f.createIfcRelDefinesByProperties(_guid(), o, None, None, [el], ps)

    def qto(el, name, area=None, volume=None, length=None):
        qs = []
        if area is not None:
            qs.append(f.createIfcQuantityArea("NetFloorArea", None, None, area))
        if volume is not None:
            qs.append(f.createIfcQuantityVolume("NetVolume", None, None, volume))
        if length is not None:
            qs.append(f.createIfcQuantityLength("Length", None, None, length))
        eq = f.createIfcElementQuantity(_guid(), o, name, None, None, qs)
        f.createIfcRelDefinesByProperties(_guid(), o, None, None, [el], eq)

    def box_rep(x, y, h):  # extruded rectangle, sizes in mm
        prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(f.createIfcCartesianPoint((x / 2, y / 2))), x, y)
        solid = f.createIfcExtrudedAreaSolid(prof, origin, f.createIfcDirection((0.0, 0.0, 1.0)), h)
        rep = f.createIfcShapeRepresentation(body_ctx, "Body", "SweptSolid", [solid])
        return f.createIfcProductDefinitionShape(None, None, [rep])

    # spaces: one with a quantity set (m² in the file), one with only geometry (4 m x 2.5 m = 10 m²)
    kitchen = f.createIfcSpace(_guid(), o, "101", None, None, place, None, "Keittiö", "ELEMENT", None, None)
    qto(kitchen, "Qto_SpaceBaseQuantities", area=12.5)
    bath = f.createIfcSpace(_guid(), o, "102", None, None, place, box_rep(4000, 2500, 2600), "Kylpyhuone", "ELEMENT", None, None)
    bed = f.createIfcSpace(_guid(), o, "201", None, None, place, None, "Bedroom", "ELEMENT", None, None)
    qto(bed, "Qto_SpaceBaseQuantities", area=14.0)
    f.createIfcRelAggregates(_guid(), o, None, None, s0, [kitchen, bath])
    f.createIfcRelAggregates(_guid(), o, None, None, s1, [bed])

    # walls: 2 exterior (with layers) + 1 interior; length quantity in mm
    wall_type = f.createIfcWallType(_guid(), o, "Ulkoseinä 300", None, None, None, None, None, None, "STANDARD")
    layers = f.createIfcMaterialLayerSet([
        f.createIfcMaterialLayer(f.createIfcMaterial("Betoni"), 150.0, None, None, None, None, None),
        f.createIfcMaterialLayer(f.createIfcMaterial("Mineraalivilla insulation"), 150.0, None, None, None, None, None),
    ], "UW300", None)
    walls = []
    for i, (ext, length) in enumerate([(True, 5000.0), (True, 4000.0), (False, 2500.0)]):
        w = f.createIfcWall(_guid(), o, f"Wall {i}", None, None, place, None, None, "STANDARD")
        pset(w, "Pset_WallCommon", {"IsExternal": ext, "LoadBearing": ext, "FireRating": "EI60" if ext else ""})
        qto(w, "Qto_WallBaseQuantities", length=length)
        walls.append(w)
    f.createIfcRelDefinesByType(_guid(), o, None, None, walls[:2], wall_type)
    f.createIfcRelAssociatesMaterial(_guid(), o, None, None, [wall_type], f.createIfcMaterialLayerSetUsage(layers, "AXIS2", "POSITIVE", 0.0, None))
    doors = [f.createIfcDoor(_guid(), o, f"D{i}", None, None, place, None, None, 2100.0, 900.0 + 100 * i) for i in range(3)]
    toilet_type = f.createIfcSanitaryTerminalType(_guid(), o, "WC-istuin", None, None, None, None, None, None, "TOILETPAN")
    toilet = f.createIfcSanitaryTerminal(_guid(), o, "WC 1", None, None, place, None, None, None)
    f.createIfcRelDefinesByType(_guid(), o, None, None, [toilet], toilet_type)
    slab = f.createIfcSlab(_guid(), o, "Floor slab", None, None, place, None, None, "FLOOR")
    qto(slab, "Qto_SlabBaseQuantities", volume=6.0)
    f.createIfcRelAssociatesMaterial(_guid(), o, None, None, [slab], f.createIfcMaterial("Concrete C30/37"))
    f.createIfcRelContainedInSpatialStructure(_guid(), o, None, None, [*walls, doors[0], doors[1], toilet, slab], s0)
    f.createIfcRelContainedInSpatialStructure(_guid(), o, None, None, [doors[2]], s1)
    f.write(str(path))
    return {"kitchen": kitchen.GlobalId, "bath": bath.GlobalId, "walls": [w.GlobalId for w in walls],
            "toilet": toilet.GlobalId, "s0": s0.GlobalId, "doors": [d.GlobalId for d in doors]}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("ifc") / "tiny.ifc"
    ids = build_ifc(path)
    return model_mod.extract(path, "tiny.ifc"), ids, path


def ask(m, q):
    return answer_model(m, q)


def test_extract(tiny):
    m, ids, _ = tiny
    assert m.schema == "IFC4"
    assert [s.name for s in m.storeys] == ["1. kerros", "2. kerros"]
    assert m.storeys[1].elevation == pytest.approx(3.0)  # mm -> m
    wall = m.get(ids["walls"][0])
    assert wall.qto["Length"] == pytest.approx(5.0)
    assert wall.materials == ["Betoni", "Mineraalivilla insulation"]  # inherited from the wall type
    assert m.get(ids["toilet"]).kind == "IfcSanitaryTerminal"
    assert m.get(ids["bath"]).geom["footprint_area"] == pytest.approx(10.0, rel=1e-3)
    assert m.get(ids["doors"][2]).storey == m.storeys[1].gid


@pytest.mark.parametrize("question,value", [
    ("How many doors are there in the model?", 3),
    ("How many doors are shown on the floor plan of '1. kerros'?", 2),
    ("Kuinka monta ovea on 2. kerros?", 1),
    ("Wie viele Wände enthält das Modell?", 3),
    ("How many exterior walls are there?", 2),
    ("How many toilets are there?", 1),
    ("How many rooms (spaces) are there in the model?", 3),
    ("How many storeys does the building have?", 2),
])
def test_counts(tiny, question, value):
    m, _, _ = tiny
    a = ask(m, question)
    assert a.answer == value, a.explanation
    assert a.unit == "count"


def test_room_area_from_quantity_and_geometry(tiny):
    m, ids, _ = tiny
    a = ask(m, "What is the net floor area of the room 'Keittiö (101)' on '1. kerros', in m²?")
    assert a.answer == pytest.approx(12.5) and a.confidence == "high"
    assert [c.id for c in a.citations] == [ids["kitchen"]]
    assert a.citations[0].category == "room" and a.citations[0].raw == "IfcSpace"
    b = ask(m, "What is the area of room 102?")
    assert b.answer == pytest.approx(10.0, rel=1e-3) and "geometry" in b.explanation


def test_totals(tiny):
    m, _, _ = tiny
    a = ask(m, "What is the total net floor area of all rooms (spaces) on '1. kerros', in m²?")
    assert a.answer == pytest.approx(22.5, rel=1e-3)
    assert len(a.citations) == 2
    b = ask(m, "What is the total length of exterior walls?")
    assert b.answer == pytest.approx(9.0) and b.unit == "m"
    c = ask(m, "What is the total volume of all floor slabs combined?")
    assert c.answer == pytest.approx(6.0) and c.unit == "m3"


def test_text_answers_cite_evidence(tiny):
    m, ids, _ = tiny
    a = ask(m, "Which software and IFC schema were used?")
    assert "TestCAD" in a.answer and "IFC4" in a.answer
    b = ask(m, "Which wall types are used, with their quantities?")
    assert "Ulkoseinä 300: 2" in b.answer
    assert {c.id for c in b.citations} == set(ids["walls"])
    c = ask(m, "Which wall types have fire rating information?")
    assert "EI60" in c.answer
    d = ask(m, "What is the thickness of the insulation of the external walls?")
    assert d.answer == pytest.approx(150.0) and d.unit == "mm"
    e = ask(m, "What are the heights of the levels?")
    assert "3.00 m" in e.answer
    f = ask(m, "What are all the spaces on level '1. kerros' with their areas?")
    assert "Keittiö (101): 12.50 m²" in f.answer


def test_not_found(tiny):
    m, _, _ = tiny
    assert ask(m, "What is the weather like tomorrow?").confidence == "not_found"
    a = ask(m, "What is the net floor area of the room 'Sauna (999)'?")
    assert a.answer is None or a.confidence != "high"


def test_tools_directly(tiny):
    m, ids, _ = tiny
    r = T.count(m, ["IfcWall"], external=True)
    assert r.value == 2 and set(r.ids) == set(ids["walls"][:2])
    s, how = T.resolve_storey(m, "on the ground floor")
    assert s is not None and s.name == "1. kerros" and how == "ordinal"
    r = T.sum_quantity(m, T.select(m, ["IfcWall"]), "length")
    assert r.value == pytest.approx(11.5) and len(r.ids) == 3


def test_answer_question_caches(tiny, tmp_path, monkeypatch):
    _, _, path = tiny
    monkeypatch.setattr(model_mod, "CACHE_DIR", tmp_path)
    model_mod._load.cache_clear()
    a = answer_question(str(path), "How many doors are there?")
    assert a.answer == 3
    assert list((tmp_path / "ifc").glob("*.json.gz"))
    model_mod._load.cache_clear()
    assert answer_question(str(path), "How many doors are there?").answer == 3  # from the disk cache


def test_api_routes_ifc(tiny, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.cubicasa.graph as graph
    from app.main import app

    _, _, path = tiny
    data = tmp_path / "data"
    data.mkdir()
    (data / "tiny.ifc").write_bytes(path.read_bytes())
    monkeypatch.setattr(graph, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(model_mod, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(model_mod, "CACHE_DIR", tmp_path / "cache")
    r = TestClient(app).post("/api/ask", json={"path": "data/tiny.ifc", "question": "How many windows are there?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == 0 and body["unit"] == "count"
    r = TestClient(app).post("/api/ask", json={"path": "data/missing.ifc", "question": "How many doors?"})
    assert r.status_code == 404


# --- agent toolset ------------------------------------------------------------------------------------------------


def test_agent_tools(tiny):
    from app.ifc.agent_tools import IfcTools

    m, ids, _ = tiny
    t = IfcTools(m)
    out, trace = t.call("count_elements", {"ifc_classes": ["IfcDoor"]})
    assert out["value"] == 3 and set(trace.item_ids) == set(ids["doors"]) and trace.numbers == [3]
    assert t.call("count_elements", {"ifc_classes": ["IfcColumn"]})[0]["value"] == 0  # absent class: 0, not an error
    assert "error" in t.call("count_elements", {"ifc_classes": ["door"]})[0]  # not a class name
    assert "error" in t.call("count_elements", {"ifc_classes": ["IfcDoor"], "storey": "Mars"})[0]
    assert "error" in t.call("count_elements", {"ifc_classes": ["IfcDoor"], "colour": "red"})[0]
    out, trace = t.call("space_areas", {})
    assert out["value"] == pytest.approx(36.5) and 36.5 in trace.numbers
    out, _ = t.call("sum_quantity", {"ifc_classes": ["IfcWall"], "quantity": "length"})
    assert out["value"] == pytest.approx(11.5)
    assert t.item(ids["toilet"]).id == ids["toilet"] and t.item("nope") is None


def test_agent_validate_with_ifc_tools(tiny):
    from app.answering import agent
    from app.ifc.agent_tools import IfcTools

    m, ids, _ = tiny
    t = IfcTools(m)
    _, doors = t.call("count_elements", {"ifc_classes": ["IfcDoor"]})
    final = {"not_found": False, "value": 3, "explanation": "There are 3 doors."}
    a = agent.validate("How many doors are there?", final, [doors], t)
    # the cubicasa router would say "doors" with no template: that must not lower an IFC answer
    assert (a.answer, a.confidence) == (3, "high") and {c.id for c in a.citations} == set(ids["doors"])
    _, areas = t.call("space_areas", {})
    a = agent.validate("Total floor area?", {"not_found": False, "value": 36.5, "unit": "m2",
                                             "explanation": "36.5 m² in total."}, [doors, areas], t)
    assert a.answer == 36.5 and a.confidence == "medium" and len(a.citations) == 3  # one area is from geometry
    a = agent.validate("Total floor area?", {"not_found": False, "value": 40, "explanation": "40 m²."}, [areas], t)
    assert a.confidence == "not_found"
