"""Parse a vector PDF floor plan deterministically. All output coordinates are PDF points of page 1, origin top-left.

1. PDF text layer (always). Rooms are labelled with centred stacks "name / number / area", e.g.
   "Zimmer / 2.01 / 15.61 m²"; rooms too small for a name show only the number. Apartment tags: "Whg. A",
   "App. B". This gives room labels, printed net areas and apartment tags for any PDF that prints them.
2. A DXF with the same name next to the PDF (optional: the same drawing in a CAD format). It adds what the PDF
   text cannot: room and apartment polygons, and exact door / window / fixture geometry. Layers are recognised
   by keyword (door, Tür, porte, ...). DXF and PDF are joined through text printed in both (room numbers,
   apartment tags), and DXF geometry is mapped to PDF points with an affine transform fitted on those texts.
Without a DXF, symbol counts fall back to the image symbol detector on the rendered page (answer.py), with
low confidence.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, box

from app.config import DATASET_DIR
from app.pdfplan.text import Seg, page_segments

NUM = re.compile(r"^[A-Z]{0,2}\d+\.\d+$")  # room number: 2.05, U1.15
AREA = re.compile(r"^(\d+[.,]\d+)\s*m")  # "18.36 m²", "15,04 m²"
UNIT_TAG = re.compile(r"^(Whg|Wohnung|App|Appartement|Apt|Appartamento|Apartment|Unit)\.?\s+([A-Z]{1,2})$", re.I)
OUTDOOR = re.compile(r"balkon|balcon|loggia|terrass|sitzplatz|veranda|garten|jardin|giardino|outdoor|patio", re.I)
SHAFT = re.compile(r"schacht|luftraum|gaine|vide|cavedi|vuoto|shaft|void", re.I)

# DXF layer name keywords, checked in this order
LAYER_KINDS = [
    ("unit", r"unit|apart|wohnung|logement|apartment"),
    ("door", r"door|(?<![a-z])t(?:ü|ue|u)r|porte|porta"),  # not the "tur" inside "FIXTURE"
    ("window", r"window|fenster|fen[eê]tre|finestr"),
    ("fixture", r"fixture|sanit|furn|equip|m[öo]bel|mobil"),
    ("room", r"room|raum|space|pi[eè]ce|locale|zimmer"),
]
INSUNITS_M = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}  # DXF $INSUNITS -> metres per unit


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", " ".join(s.split())).casefold()


@dataclass
class Room:
    number: str
    name: str | None  # None: no name printed (room too small)
    printed_area: float | None  # m², as printed on the plan
    bbox: tuple[float, float, float, float]  # the label stack
    anchor: tuple[float, float]  # centre of the number
    polygon: list[tuple[float, float]] | None = None  # outline (from the DXF)
    geom_area: float | None = None  # m², from the outline
    has_use: bool | None = None  # contains a fixture or touches a door; None = unknown (no geometry)

    @property
    def area(self) -> float | None:
        return self.printed_area if self.printed_area is not None else self.geom_area

    @property
    def outdoor(self) -> bool:
        return bool(self.name and OUTDOOR.search(self.name))

    @property
    def counts_as_indoor(self) -> bool:
        """Numbered indoor room for net-area sums: named and not outdoor/shaft, or unnamed but in use."""
        if self.name is None:
            return bool(self.has_use)
        return not OUTDOOR.search(self.name) and not SHAFT.search(self.name)


@dataclass
class Unit:
    letter: str
    tag: str
    bbox: tuple[float, float, float, float]  # the tag text, replaced by the outline's box when known
    polygon: list[tuple[float, float]] | None = None
    rooms: list[Room] | None = None  # None = membership unknown (no outlines)


@dataclass
class Symbol:
    kind: str  # door, window, toilet, washbasin, bathtub, shower, kitchen, stairs, other
    bbox: tuple[float, float, float, float]


@dataclass
class Plan:
    rel: str
    page_size: tuple[float, float]
    rooms: list[Room]
    units: list[Unit]
    symbols: list[Symbol] | None  # None = no vector symbol source (no DXF)
    sources: list[str] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Symbol]:
        return [s for s in self.symbols or [] if s.kind == kind]


# ------------------------------------------------------------------ PDF text

def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _neighbour(s: Seg, segs: list[Seg], above: bool) -> Seg | None:
    """Closest text run directly above/below s in the same centred column."""
    tol = max(2.0, 0.5 * s.h)
    best, gap_best = None, 1.2 * s.h
    for t in segs:
        if t is s or abs(t.cx - s.cx) > tol:
            continue
        gap = s.bbox[1] - t.bbox[3] if above else t.bbox[1] - s.bbox[3]
        if -0.5 <= gap < gap_best:
            best, gap_best = t, gap
    return best


def rooms_from_text(segs: list[Seg]) -> list[Room]:
    rooms = []
    for s in segs:
        if not NUM.match(s.text):
            continue
        up, down = _neighbour(s, segs, True), _neighbour(s, segs, False)
        name = up.text if up and not (NUM.match(up.text) or AREA.match(up.text) or UNIT_TAG.match(up.text)) else None
        m = AREA.match(down.text) if down else None
        area = float(m.group(1).replace(",", ".")) if m else None
        parts = [s.bbox] + ([up.bbox] if name else []) + ([down.bbox] if m else [])
        rooms.append(Room(s.text, name, area, _union(parts), (s.cx, s.cy)))
    return rooms


def units_from_text(segs: list[Seg]) -> list[Unit]:
    return [Unit(m.group(2).upper(), s.text, s.bbox) for s in segs if (m := UNIT_TAG.match(s.text))]


# ------------------------------------------------------------------ DXF geometry

def _ebox(e):
    if e.dxftype() in ("CIRCLE", "ARC"):
        c, r = e.dxf.center, e.dxf.radius
        return (c.x - r, c.y - r, c.x + r, c.y + r)
    pts = list(e.get_points("xy"))
    return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))


def _inside(b, r, tol=1e-3) -> bool:
    return b[0] >= r[0] - tol and b[1] >= r[1] - tol and b[2] <= r[2] + tol and b[3] <= r[3] + tol


def _is_rect(e) -> bool:
    return e.dxftype() == "LWPOLYLINE" and e.closed and len(e) == 4


def _fixture_kind(marks: Counter) -> str:
    """Fixture type from the marks inside its outline rectangle (radii in metres).

    drain = small circle (r <= 5 cm). Bathtub: drain + tub outline (> 48 points). Washbasin: drain + 48-point oval.
    WC: a line (cistern) + 48-point bowl. Shower: drain only. Kitchen: >= 4 burners (r ~8 cm). Stairs: >= 5 lines.
    """
    drain = sum(v for k, v in marks.items() if k[0] == "C" and k[1] <= 0.05)
    tub = sum(v for k, v in marks.items() if k[0] == "P" and k[1] > 48)
    burners = sum(v for k, v in marks.items() if k[0] == "C" and 0.06 <= k[1] <= 0.1)
    if burners >= 4:
        return "kitchen"
    if marks[("L", 0)] >= 5:
        return "stairs"
    if drain and tub:
        return "bathtub"
    if drain and marks[("P", 48)]:
        return "washbasin"
    if marks[("L", 0)] and marks[("P", 48)]:
        return "toilet"
    if drain:
        return "shower"
    return "other"


def _fixtures(ents, u: float) -> list[tuple[str, tuple]]:
    area = lambda e: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(_ebox(e))  # noqa: E731
    rects = sorted((e for e in ents if _is_rect(e)), key=area)
    used: set[int] = set()
    out = []
    for r in rects:  # smallest first: a mark belongs to the tightest outline around it
        rb = _ebox(r)
        marks: Counter = Counter()
        for e in ents:
            if e is r or id(e) in used or _is_rect(e) or not _inside(_ebox(e), rb):
                continue
            used.add(id(e))
            if e.dxftype() == "CIRCLE":
                marks[("C", round(e.dxf.radius * u, 2))] += 1
            elif e.dxftype() == "LWPOLYLINE":
                marks[("L", 0) if len(e) == 2 else ("P", len(e))] += 1
        out.append((_fixture_kind(marks), rb))
    return out


def _doors(ents, u: float) -> list[tuple]:
    """One door per 5-point frame polyline, its box extended by the swing quadrant of the arc hinged on it."""
    swings = []
    for e in ents:
        if e.dxftype() == "ARC":
            pts = [e.dxf.center, e.start_point, e.end_point]
            swings.append((e.dxf.center, (min(p.x for p in pts), min(p.y for p in pts),
                                          max(p.x for p in pts), max(p.y for p in pts))))
    out = []
    for e in ents:
        if e.dxftype() == "LWPOLYLINE" and len(e) == 5:
            b = _ebox(e)
            near = [(Point(c.x, c.y).distance(box(*b)), s) for c, s in swings]
            if near and (best := min(near, key=lambda t: t[0]))[0] < 0.05 / u:
                s = best[1]
                b = (min(b[0], s[0]), min(b[1], s[1]), max(b[2], s[2]), max(b[3], s[3]))
            out.append(b)
    return out


def _text_point(e) -> tuple[float, float]:
    """Anchor of a DXF TEXT/MTEXT; for aligned TEXT the alignment point (centre for middle-centre text)."""
    if e.dxftype() == "TEXT" and (e.dxf.get("halign", 0) or e.dxf.get("valign", 0)) and e.dxf.hasattr("align_point"):
        p = e.dxf.align_point
    else:
        p = e.dxf.insert
    return (p.x, p.y)


def _text_of(e) -> str:
    raw = e.plain_text() if e.dxftype() == "MTEXT" else e.dxf.text
    return " ".join(raw.split())


def _fit_affine(src: list, dst: list):
    """3x2 matrix M with [x y 1] @ M = PDF point; None if too few pairs or a poor fit (> 2 pt median error)."""
    if len(src) < 3:
        return None
    a = np.c_[np.array(src, dtype=float), np.ones(len(src))]
    b = np.array(dst, dtype=float)
    m, *_ = np.linalg.lstsq(a, b, rcond=None)
    if np.linalg.matrix_rank(a) < 3 or float(np.median(np.linalg.norm(a @ m - b, axis=1))) > 2.0:
        return None
    return m


def _apply(m, pts) -> list[tuple[float, float]]:
    out = np.c_[np.array(pts, dtype=float), np.ones(len(pts))] @ m
    return [(float(x), float(y)) for x, y in out]


def _apply_box(m, b) -> tuple[float, float, float, float]:
    return _union([(x, y, x, y) for x, y in _apply(m, [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])])])


def _outlines(ents) -> list[Polygon]:
    """Valid polygons from polylines that are closed, or whose ends meet (outlines are not always flagged closed)."""
    out = []
    for e in ents:
        if e.dxftype() != "LWPOLYLINE" or len(e) < 3:
            continue
        pts = list(e.get_points("xy"))
        if e.closed or Point(pts[0]).distance(Point(pts[-1])) < 1e-6 or len(pts) >= 4:
            p = Polygon(pts)
            if p.is_valid and p.area > 0:
                out.append(p)
    return out


def _layer_kind(name: str) -> str | None:
    return next((k for k, pat in LAYER_KINDS if re.search(pat, name, re.I)), None)


def enrich_from_dxf(plan: Plan, dxf_path: Path) -> bool:
    """Add room/apartment outlines and door/window/fixture symbols from the DXF. False if it cannot be aligned."""
    import ezdxf

    doc = ezdxf.readfile(str(dxf_path))
    u = INSUNITS_M.get(doc.header.get("$INSUNITS", 0), 1.0)
    msp = doc.modelspace()
    by_kind: dict[str, list] = {}
    texts: dict[str, list[tuple[float, float]]] = {}
    for e in msp:
        if e.dxftype() in ("TEXT", "MTEXT"):
            texts.setdefault(_text_of(e), []).append(_text_point(e))
        elif kind := _layer_kind(e.dxf.layer):
            by_kind.setdefault(kind, []).append(e)

    # join through texts printed exactly once in both files
    pdf_pts = {r.number: r.anchor for r in plan.rooms}
    pairs = [(texts[t][0], p) for t, p in pdf_pts.items() if len(texts.get(t, [])) == 1
             and sum(r.number == t for r in plan.rooms) == 1]
    m = _fit_affine([a for a, _ in pairs], [b for _, b in pairs])
    if m is None:
        return False

    room_polys, unit_polys = _outlines(by_kind.get("room", [])), _outlines(by_kind.get("unit", []))
    door_boxes = _doors(by_kind.get("door", []), u)
    window_boxes = [_ebox(e) for e in by_kind.get("window", []) if _is_rect(e)]
    fixtures = _fixtures(by_kind.get("fixture", []), u)
    use = [box(*b) for k, b in fixtures if k != "other"]  # stairs too: an unnamed stairwell is still a room
    doors_g = [box(*b) for b in door_boxes]

    dxf_poly: dict[int, Polygon] = {}
    for r in plan.rooms:
        if len(texts.get(r.number, [])) != 1:
            continue
        pt = Point(texts[r.number][0])
        poly = min((p for p in room_polys if p.contains(pt)), key=lambda p: p.area, default=None)
        if poly is None:
            continue
        dxf_poly[id(r)] = poly
        r.polygon = _apply(m, list(poly.exterior.coords))
        r.geom_area = round(poly.area * u * u, 2)
        r.has_use = any(poly.contains(b.centroid) for b in use) or any(poly.buffer(0.05 / u).intersects(b) for b in doors_g)

    for un in plan.units:
        if len(texts.get(un.tag, [])) != 1:
            continue
        pt = Point(texts[un.tag][0])
        poly = min((p for p in unit_polys if p.contains(pt)), key=lambda p: p.area, default=None)
        if poly is None:
            continue
        un.polygon = _apply(m, list(poly.exterior.coords))
        un.bbox = _union([(x, y, x, y) for x, y in un.polygon])
        un.rooms = [r for r in plan.rooms if (id(r) in dxf_poly and
                    poly.intersection(dxf_poly[id(r)]).area > 0.5 * dxf_poly[id(r)].area)]

    plan.symbols = ([Symbol("door", _apply_box(m, b)) for b in door_boxes]
                    + [Symbol("window", _apply_box(m, b)) for b in window_boxes]
                    + [Symbol(k, _apply_box(m, b)) for k, b in fixtures])
    plan.sources.append(f"dxf:{dxf_path.name}")
    return True


# ------------------------------------------------------------------ entry point

def resolve_pdf(rel: str) -> Path:
    """A dataset-relative path, or an absolute path (e.g. an uploaded PDF in the cache)."""
    p = Path(rel)
    return p if p.is_absolute() else DATASET_DIR / rel


@lru_cache(maxsize=64)
def parse_plan(rel: str, use_dxf: bool = True) -> Plan:
    """rel: path of a PDF, dataset-relative or absolute."""
    pdf = resolve_pdf(rel)
    if not pdf.is_file():
        raise FileNotFoundError(rel)
    size, segs = page_segments(pdf)
    plan = Plan(rel, size, rooms_from_text(segs), units_from_text(segs), None, ["pdf-text"])
    dxf = pdf.with_suffix(".dxf")  # the same drawing in CAD form, if it was delivered alongside
    if use_dxf and dxf.is_file():
        enrich_from_dxf(plan, dxf)
    return plan
