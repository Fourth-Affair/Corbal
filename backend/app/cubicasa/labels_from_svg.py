"""Original CubiCasa5K model.svg -> PlanGraph (ground truth for training, validation and the oracle eval).

Challenge file `data/cubicasa5k/<variant>-<id>.png` is byte-identical to the original
`<variant>/<id>/F1_original.png`. SVG coordinates are in `F1_scaled.png` pixel space, so they are
rescaled by original/scaled size to land on the challenge image.
"""

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from app.config import CUBICASA_ORIG_DIR
from app.models import PlanGraph, PlanItem

SVG_NS = "{http://www.w3.org/2000/svg}"
Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1, 0, 0, 1, 0, 0)
_NUM = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")


def svg_path_for(rel: str) -> Path:
    stem = Path(rel).stem  # e.g. high_quality_architectural-1
    variant, _, num = stem.rpartition("-")
    return CUBICASA_ORIG_DIR / variant / num / "model.svg"


# ---------- geometry helpers ----------

def _mul(m: Matrix, n: Matrix) -> Matrix:
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2,
            a * e2 + c * f2 + e, b * e2 + d * f2 + f)


def _parse_transform(s: str | None) -> Matrix:
    m = IDENTITY
    if not s:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
        v = [float(x) for x in _NUM.findall(args)]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            r = math.radians(v[0])
            t = (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)
            if len(v) == 3:
                t = _mul(_mul((1, 0, 0, 1, v[1], v[2]), t), (1, 0, 0, 1, -v[1], -v[2]))
        else:
            continue
        m = _mul(m, t)
    return m


def _apply(m: Matrix, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    a, b, c, d, e, f = m
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in pts]


_PATH_TOKEN = re.compile(r"[MmLlHhVvQqCcSsTtAaZz]|" + _NUM.pattern)
_PATH_ARITY = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1, "Q": 4, "S": 4, "C": 6, "A": 7, "Z": 0}


def _path_points(d: str) -> list[tuple[float, float]]:
    """End points and control points of an SVG path (a superset of its extent, good enough for bboxes)."""
    pts: list[tuple[float, float]] = []
    x = y = 0.0
    cmd = None
    toks = _PATH_TOKEN.findall(d)
    i = 0
    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                continue
        if cmd is None:
            break
        n = _PATH_ARITY[cmd.upper()]
        chunk = toks[i:i + n]
        if len(chunk) < n or any(t.isalpha() for t in chunk):
            break  # malformed / compact arc flags: keep what we have
        v = [float(t) for t in chunk]
        i += n
        rel = cmd.islower()
        if cmd.upper() == "H":
            x = v[0] + (x if rel else 0)
        elif cmd.upper() == "V":
            y = v[0] + (y if rel else 0)
        elif cmd.upper() == "A":
            x, y = v[5] + (x if rel else 0), v[6] + (y if rel else 0)
        else:
            for k in range(0, n, 2):
                px, py = v[k] + (x if rel else 0), v[k + 1] + (y if rel else 0)
                pts.append((px, py))
            x, y = pts[-1]
        pts.append((x, y))
        if cmd == "M":
            cmd = "L"
        elif cmd == "m":
            cmd = "l"
    return pts


def _shape_points(el: ET.Element) -> list[tuple[float, float]]:
    tag = el.tag.removeprefix(SVG_NS)
    g = lambda k: float(el.get(k, 0) or 0)  # noqa: E731
    if tag in ("polygon", "polyline"):
        v = [float(x) for x in _NUM.findall(el.get("points", ""))]
        return list(zip(v[0::2], v[1::2]))
    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if tag == "path":
        return _path_points(el.get("d", ""))
    if tag == "line":
        return [(g("x1"), g("y1")), (g("x2"), g("y2"))]
    if tag == "circle":
        cx, cy, r = g("cx"), g("cy"), g("r")
        return [(cx - r, cy - r), (cx + r, cy + r)]
    return []


def _collect(el: ET.Element, m: Matrix, out: list, skip_hidden: bool = True) -> None:
    """All shape points under el (absolute coordinates), skipping display:none subtrees."""
    if skip_hidden and "display: none" in (el.get("style") or ""):
        return
    m = _mul(m, _parse_transform(el.get("transform")))
    out.extend(_apply(m, _shape_points(el)))
    for ch in el:
        _collect(ch, m, out, skip_hidden)


def _bbox(pts) -> tuple[float, float, float, float] | None:
    if not pts:
        return None
    xs, ys = zip(*pts)
    return (min(xs), min(ys), max(xs), max(ys))


def _first_polygon(el: ET.Element, m: Matrix) -> list[tuple[float, float]] | None:
    for ch in el:
        if ch.tag == SVG_NS + "polygon":
            return _apply(_mul(m, _parse_transform(el.get("transform"))), _shape_points(ch))
    return None


def _name_label(el: ET.Element) -> str:
    for d in el.iter():
        if "NameLabel" in (d.get("class") or ""):
            return " ".join("".join(d.itertext()).split())
    return ""


# ---------- canonical class mapping (fitted against questions.jsonl, see app.cubicasa.fit_rules) ----------

def classify_room(svg_class: str, name: str) -> str:
    """Fitted on the train split: every rule matches questions.jsonl on >=99.9% of images.

    The drawn room name is only used where the question semantics need it (WC vs bathroom,
    balcony vs other outdoor space). This derives *training labels*; at runtime the model
    predicts the room class from pixels.
    """
    toks = svg_class.split()[1:]  # drop "Space"
    n = name.upper().replace(" ", "")
    main = toks[0] if toks else ""
    if main == "Bedroom":
        return "bedroom"
    if main == "Bath":
        return "wc" if n.startswith("WC") else "bathroom"
    if main == "Sauna":
        return "sauna"
    if main == "Kitchen":
        return "kitchen"
    if main == "Outdoor" and "PARV" in n:
        return "balcony"
    return "other"


def classify_fixture(svg_class: str) -> str:
    k = svg_class.split()[-1]
    if k == "Toilet":
        return "toilet"
    if k.startswith("DoubleSink"):
        return "double_sink"
    if "Sink" in k:  # Sink, RoundSink, CornerSink, SideSink
        return "sink"
    if k == "Shower":  # ShowerCab/ShowerScreen/ShowerPlatform are not counted by the questions
        return "shower"
    if k.startswith("Bathtub"):
        return "bathtub"
    if k in ("SaunaStove", "SaunaStoveRound"):
        return "sauna_stove"
    return "other"


# ---------- main ----------

def graph_from_svg(rel: str) -> PlanGraph:
    svg = svg_path_for(rel)
    if not svg.is_file():
        raise FileNotFoundError(f"original CubiCasa label not found: {svg}")
    from PIL import Image

    from app.config import DATASET_DIR

    with Image.open(DATASET_DIR / rel) as im:
        width, height = im.size
    with Image.open(svg.parent / "F1_scaled.png") as im:
        sw, sh = im.size
    scale: Matrix = (width / sw, 0, 0, height / sh, 0, 0)

    root = ET.parse(svg).getroot()
    items: list[PlanItem] = []

    def add(category, cls, el, m, raw, polygon=None, text=None):
        pts: list = []
        _collect(el, m, pts)
        bb = _bbox(pts)
        if bb is None:
            return
        items.append(PlanItem(id=f"{category}-{len(items)}", category=category, cls=cls, bbox=bb,
                              polygon=polygon, text=text or None, raw=raw))

    def walk(el: ET.Element, m: Matrix, in_floor: bool) -> None:
        cls_attr = el.get("class") or ""
        m2 = _mul(m, _parse_transform(el.get("transform")))
        head = cls_attr.split()[0] if cls_attr else ""
        if head == "Floorplan":
            start = len(items)
            for ch in el:
                walk(ch, m2, True)
            inner = [it.bbox for it in items[start:]]
            if inner:  # floor extent = union of its rooms/openings/fixtures
                bb = (min(b[0] for b in inner), min(b[1] for b in inner),
                      max(b[2] for b in inner), max(b[3] for b in inner))
                items.append(PlanItem(id=f"floor-{len(items)}", category="floor", bbox=bb, raw=cls_attr))
            return
        elif head == "Space":
            name = _name_label(el)
            poly = _first_polygon(el, m)
            # bbox from the room polygon only (labels/dimensions can stick out)
            if poly:
                items.append(PlanItem(id=f"room-{len(items)}", category="room", cls=classify_room(cls_attr, name),
                                      bbox=_bbox(poly), polygon=poly, text=name or None, raw=cls_attr))
            return
        elif head == "Door":
            add("door", "", el, m, cls_attr)
            return
        elif head == "Window":
            add("window", "", el, m, cls_attr)
            return
        elif head == "FixedFurniture":
            poly = None
            for ch in el:  # BoundaryPolygon holds the outline
                if "BoundaryPolygon" in (ch.get("class") or ""):
                    poly = _first_polygon(ch, m2)
            k = classify_fixture(cls_attr)
            if poly:
                items.append(PlanItem(id=f"fixture-{len(items)}", category="fixture", cls=k,
                                      bbox=_bbox(poly), raw=cls_attr))
            else:
                add("fixture", k, el, m, cls_attr)
            return
        for ch in el:
            walk(ch, m2, in_floor)

    walk(root, scale, False)
    for it in items:  # some door arcs / outdoor areas run past the image edge: clip them
        x0, y0, x1, y1 = it.bbox
        c = (max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1))
        # Items entirely outside the image are kept unclipped: the questions still count them
        # (~0.1% of answers). export_yolo skips them since they are invisible.
        if c[2] > c[0] and c[3] > c[1]:
            it.bbox = c
    return PlanGraph(image=rel, width=width, height=height, source="svg", items=items)
