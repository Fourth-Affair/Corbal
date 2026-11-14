"""Plan geometry: rooms as polygons enclosed by walls, typed from combined evidence, plus walls, wall segments,
openings and a scale derived from areas written in the rooms.

  python -m app.cubicasa.rooms data/cubicasa5k/colorful-10052.png      # print a summary (writes no cache)

Rooms are the connected free-space regions of the wall map (layers.wall_classes: wall segmenter, else the CubiCasa
net). Openings count as wall, so a doorway separates two rooms; gaps narrower than half a wall are closed first.
Regions touching the image border are outside. A room's type combines (app.cubicasa.room_types) the OCR labels
inside it (strong), the fixtures inside it (strong when decisive: toilet, shower, sauna stove, ...) and the 2019 net's
room-class pixels (weak), with a tier: strong / weak / unknown. Wall segments come from app.cubicasa.wallvec.
Openings (or detected doors/windows when the wall model has no opening class) link the rooms they touch.
Everything is in image pixels; the scale (m/px) is derived from written room areas when they agree, and the user's
calibration replaces it when the geometry is served.
"""

import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from skimage.measure import find_contours

from app.cubicasa.room_types import Fixture, classify
from app.cubicasa.rooms_dict import room_purpose
from app.models import PlanGeometry, PlanGraph, PlanScale, RoomEvidence, RoomLabel, RoomShape, WallShape
from app.profiling import steps

# CubiCasa net classes by index (fuse.ROOM / fuse.ICON order)
_SEG_NAMES = {1: "outdoor", 3: "kitchen", 4: "living", 5: "bedroom", 6: "bath", 7: "entry", 9: "storage", 10: "garage"}
_ICON_KINDS = {3: "closet", 4: "appliance", 8: "fireplace"}  # icon classes the detector / graph do not carry
_FIXTURE_KINDS = {"toilet", "sink", "double_sink", "shower", "bathtub", "sauna_stove", "sauna_bench"}
SPLIT_BY_LABELS = True  # cut spaces holding two labels that should not share a space (app.cubicasa.room_split)
CUT_MIN_ROOM_FRAC = 0.004  # each side of such a cut keeps at least this share of the image
SCALE_AGREE = 0.15  # written areas must agree within this (relative m/px) to set the plan scale
SCALE_CROSS_CHECK = 0.10  # dimension-line and written-area scales further apart than this get a warning
_AREA_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|m\^2|sqm|㎡)", re.IGNORECASE)
_BARE_AREA_RE = re.compile(r"^\s*(\d{1,3}[.,]\d{1,2})\s*$")
BARE_AREA_MIN_SCORE = 0.8  # OCR confidence for a unit-less area (a misread digit would skew the scale)
MIN_ROOM_FRAC = 1e-3  # of the image area
MAX_ROOM_FRAC = 0.5
MAX_POINTS = 400


def _rings(mask: np.ndarray, x0: int, y0: int) -> list[Polygon]:
    """Contour rings of a binary crop as shapely polygons in image coordinates (largest first)."""
    padded = np.pad(mask.astype(np.uint8), 1)  # find_contours cannot close a boundary on the array edge
    rings = []
    for c in find_contours(padded, 0.5):
        if len(c) < 4:
            continue
        p = Polygon([(float(x - 1 + x0), float(y - 1 + y0)) for y, x in c])
        if p.is_valid and p.area > 0:
            rings.append(p)
    return sorted(rings, key=lambda p: -p.area)


def _coords(ring, tol: float) -> list[tuple[float, float]]:
    pts = list(ring.simplify(tol, preserve_topology=True).coords)[:-1]
    if len(pts) > MAX_POINTS:
        pts = pts[:: (len(pts) + MAX_POINTS - 1) // MAX_POINTS]
    return [(round(x, 1), round(y, 1)) for x, y in pts]


def _shape(mask: np.ndarray, sl: tuple[slice, slice], tol: float, holes: bool = False):
    """(exterior polygon, holes) of the region `mask[sl]`, simplified; None if degenerate."""
    ys, xs = sl
    rings = _rings(mask[sl], xs.start, ys.start)
    if not rings:
        return None
    outer = rings[0]
    inner = [r for r in rings[1:] if outer.contains(r.representative_point())] if holes else []
    ext = _coords(outer.exterior, tol)
    if len(ext) < 3:
        return None
    return ext, [h for h in (_coords(r.exterior, tol) for r in inner) if len(h) >= 3]


def _box(sl) -> tuple[float, float, float, float]:
    ys, xs = sl
    return float(xs.start), float(ys.start), float(xs.stop), float(ys.stop)


def _grow(sl, r: int, shape) -> tuple[slice, slice]:
    ys, xs = sl
    return slice(max(0, ys.start - r), min(shape[0], ys.stop + r)), slice(max(0, xs.start - r), min(shape[1], xs.stop + r))


def _touching(lab: np.ndarray, mask: np.ndarray, sl, r: int) -> list[int]:
    """Labels of `lab` within r px of the region `mask[sl]` (region given in sl coordinates)."""
    g = _grow(sl, r + 1, lab.shape)
    sub = np.zeros(lab[g].shape, bool)
    oy, ox = sl[0].start - g[0].start, sl[1].start - g[1].start
    sub[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
    near = ndimage.binary_dilation(sub, iterations=r) & ~sub
    return [int(v) for v in np.unique(lab[g][near]) if v]


def _labeled_area(labels: list[RoomLabel]) -> float | None:
    """Area written in the room: a number with a unit ("12,5 m2"), else bare decimals ("11.5" under "MH", the usual
    Finnish convention) when the room also carries a purpose label. Bare decimals in one space are summed: an open
    plan ("OH 26.5" + "KEITTIÖ 17.0") or two rooms the wall map merged. Integers are not areas ("20" is usually a
    misread "2.0"); a wrong number is caught by the agreement test in scale_from_written_areas."""
    m = _AREA_RE.search(" ".join(lb.text for lb in labels))
    if m:
        return float(m.group(1).replace(",", "."))
    if not any(lb.room_type for lb in labels):
        return None
    vals = [float(b.group(1).replace(",", ".")) for lb in labels
            if (lb.score is None or lb.score >= BARE_AREA_MIN_SCORE) and (b := _BARE_AREA_RE.match(lb.text))]
    vals = [v for v in vals if 0.5 <= v <= 300]
    return round(sum(vals), 2) if vals else None


def geometry_from_image(rel: str, path: Path, graph: PlanGraph | None = None) -> PlanGeometry:
    """Rooms, walls and openings of an image plan. `graph` (optional) supplies detected doors/windows."""
    with steps() as step:  # stage timings for app.profiling (no-op without an active profile)
        return _geometry_from_image(rel, path, graph, step)


def _geometry_from_image(rel: str, path: Path, graph: PlanGraph | None, step) -> PlanGeometry:
    from app.cubicasa import layers

    step("walls")
    cls, model = layers.wall_classes(rel, path)
    step("thickness")
    H, W = cls.shape
    barrier = cls > 0
    # wall thickness: twice the median distance-to-background along the wall skeleton
    from skimage.morphology import skeletonize

    edt = ndimage.distance_transform_edt(barrier)
    skel = skeletonize(barrier)
    t = float(2 * np.median(edt[skel])) if skel.any() else 4.0
    t = min(max(t, 2.0), 0.05 * max(H, W))
    closed = ndimage.binary_closing(barrier, iterations=max(1, round(t / 4))) | barrier
    step("text")
    text = layers.text(rel, path)
    step("spaces")
    lab, n = ndimage.label(~closed)
    room_cuts = []
    if SPLIT_BY_LABELS:  # a bedroom open to the living room: cut between the two labels at the narrowest line
        from app.cubicasa.room_split import split_by_labels
        from app.cubicasa.room_types import LABEL_STRONG_SCORE

        label_seeds = [((tk.bbox[0] + tk.bbox[2]) / 2, (tk.bbox[1] + tk.bbox[3]) / 2, purpose) for tk in text.tokens
                       if tk.score >= LABEL_STRONG_SCORE and (purpose := room_purpose(tk.text))]
        lab, room_cuts = split_by_labels(lab, label_seeds, t, CUT_MIN_ROOM_FRAC * H * W)
        n = int(lab.max())
    outside = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.int32), lab, np.arange(1, n + 1))
    slices = ndimage.find_objects(lab)
    keep = [i + 1 for i in range(n) if i + 1 not in outside and MIN_ROOM_FRAC * H * W <= sizes[i] <= MAX_ROOM_FRAC * H * W]
    # reading order: top-to-bottom, then left-to-right by bbox centre
    keep.sort(key=lambda k: (round((slices[k - 1][0].start + slices[k - 1][0].stop) / 2 / max(1.0, 3 * t)),
                             (slices[k - 1][1].start + slices[k - 1][1].stop) / 2))
    room_id = {k: f"space-{j + 1}" for j, k in enumerate(keep)}

    # OCR tokens -> rooms (token centre, else the most common room within one wall thickness)
    labels_of: dict[int, list[RoomLabel]] = {k: [] for k in keep}
    r = max(2, round(t))
    for tok in text.tokens:
        cx, cy = int((tok.bbox[0] + tok.bbox[2]) / 2), int((tok.bbox[1] + tok.bbox[3]) / 2)
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        k = int(lab[cy, cx])
        if k not in labels_of:
            win = lab[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].ravel()
            win = win[np.isin(win, keep)]
            k = int(np.bincount(win).argmax()) if win.size else 0
        if k in labels_of:
            labels_of[k].append(RoomLabel(text=tok.text, bbox=tok.bbox, score=round(tok.score, 3),
                                          room_type=room_purpose(tok.text)))

    step("fixtures")
    seg, icons = _net_maps(rel, (H, W))
    fixtures_of = _fixtures_by_room(lab, graph, icons)
    step("rooms")
    tol = max(1.0, 0.002 * max(H, W))
    rooms: list[RoomShape] = []
    for k in keep:
        sl = slices[k - 1]
        shape = _shape(lab == k, sl, tol)
        if shape is None:
            continue
        labels = labels_of[k]
        fractions = {}
        if seg is not None:
            vals = seg[sl][lab[sl] == k]
            for idx, name in _SEG_NAMES.items():
                frac = float(np.mean(vals == idx)) if vals.size else 0.0
                if frac > 0:
                    fractions[name] = frac
        c = classify(labels, fixtures_of.get(k, []), fractions, float(sizes[k - 1]) / (H * W))
        name = " / ".join(dict.fromkeys(lb.text for lb in labels if lb.room_type and lb.room_type in c.purposes)) or None
        neighbours = [room_id[v] for v in _touching(lab, lab[sl] == k, sl, round(1.5 * t) + 2) if v in room_id]
        rooms.append(RoomShape(id=room_id[k], type=c.type, tier=c.tier, purposes=c.purposes, scores=c.scores,
                               evidence=c.evidence, conflicts=c.conflicts, name=name, labels=labels,
                               polygon=shape[0], bbox=_box(sl), area_px=float(sizes[k - 1]),
                               labeled_area_m2=_labeled_area(labels), neighbours=neighbours))

    step("walls_openings")
    walls = _walls(cls == 1, "wall", tol)
    openings = _openings(cls, lab, room_id, outside, t, graph, tol)
    by_id = {rm.id: rm for rm in rooms}
    for op in openings:
        for rid in op.rooms:
            if rid in by_id:
                by_id[rid].connections.append(op.id)

    from app.cubicasa.wallvec import vectorise

    def to_id(v: int) -> str | None:
        return room_id.get(v) or ("outside" if v in outside else None)

    step("wall_segments")
    segments = vectorise(cls, lab, to_id, t, openings)
    for sg in segments:
        for rid in set(sg.rooms_left) | set(sg.rooms_right):
            if rid in by_id:
                by_id[rid].walls.append(sg.id)
    from app.cubicasa.size_priors import symbol_scale

    step("scale")
    sym = symbol_scale(graph)
    return PlanGeometry(image=rel, width=W, height=H, model=model, wall_thickness_px=round(t, 1), rooms=rooms,
                        walls=walls, openings=openings, wall_segments=segments, room_cuts=room_cuts,
                        scale=plan_scale(rooms, _dimension_scale(path, text.tokens, sym)), symbol_scale=sym)


def _dimension_scale(path: Path, tokens, sym: PlanScale | None) -> PlanScale | None:
    """Scale from dimension lines, dropped when it is far outside the door / fixture size band (figures in cm or
    inches, or a misread chain that happens to agree with itself)."""
    from PIL import Image

    from app.cubicasa.dimensions import dimension_scale
    from app.cubicasa.size_priors import SCALE_DISAGREE

    try:
        dims = dimension_scale(Image.open(path), tokens)
    except Exception:  # noqa: BLE001  optional evidence
        return None
    if dims and sym and sym.low and sym.high and not sym.low / SCALE_DISAGREE <= dims.m_per_px <= sym.high * SCALE_DISAGREE:
        return None
    return dims


def plan_scale(rooms: list[RoomShape], dims: PlanScale | None) -> PlanScale | None:
    """The plan's own scale: dimension lines (a direct length measurement) when they agree, else written room areas
    (measured through the room outlines, so segmentation errors leak in). When both exist and differ by more than
    SCALE_CROSS_CHECK, the chosen one carries a warning."""
    areas = scale_from_written_areas(rooms)
    if dims and areas and abs(dims.m_per_px - areas.m_per_px) > SCALE_CROSS_CHECK * areas.m_per_px:
        dims.warning = (f"dimension lines give {dims.m_per_px * 1000:.2f} mm per pixel, the written room areas "
                        f"{areas.m_per_px * 1000:.2f}: check the scale")
    return dims or areas


def scale_from_written_areas(rooms: list[RoomShape]) -> PlanScale | None:
    """m/px from rooms whose area is written inside them: sqrt(m² / px) per room, the median when >= 2 rooms agree
    within SCALE_AGREE. One room alone is not trusted (an OCR misread would silently scale everything)."""
    est = [(rm.id, (rm.labeled_area_m2 / rm.area_px) ** 0.5) for rm in rooms
           if rm.labeled_area_m2 and rm.labeled_area_m2 > 0 and rm.area_px > 0]
    if len(est) < 2:
        return None
    med = float(np.median([v for _, v in est]))
    agree = [(rid, v) for rid, v in est if abs(v - med) <= SCALE_AGREE * med]
    if len(agree) < 2:
        return None
    return PlanScale(m_per_px=float(np.median([v for _, v in agree])), source="written_areas",
                     rooms=[rid for rid, _ in agree])


def _fixtures_by_room(lab: np.ndarray, graph: PlanGraph | None, icons: np.ndarray | None) -> dict[int, list[Fixture]]:
    """Fixtures per free-space label: graph fixture items (detector / icon blobs) by their box centre, plus the net's
    appliance / closet / fireplace icon blobs (not in the graph) by their centroid."""
    H, W = lab.shape
    out: dict[int, list[Fixture]] = {}

    def at(x: float, y: float) -> int:
        u, v = int(x), int(y)
        return int(lab[v, u]) if 0 <= u < W and 0 <= v < H else 0

    for it in (graph.items if graph else []):
        if it.category == "fixture" and it.cls in _FIXTURE_KINDS and (it.score is None or it.score >= 0.4):
            k = at((it.bbox[0] + it.bbox[2]) / 2, (it.bbox[1] + it.bbox[3]) / 2)
            if k:
                out.setdefault(k, []).append(Fixture(it.cls, it.id))
    if icons is not None:
        for idx, kind in _ICON_KINDS.items():
            blobs, n = ndimage.label(icons == idx)
            if not n:
                continue
            sizes = ndimage.sum(np.ones_like(blobs, dtype=np.int32), blobs, np.arange(1, n + 1))
            for (cy, cx), size in zip(ndimage.center_of_mass(blobs > 0, blobs, np.arange(1, n + 1)), sizes):
                if size >= 20:
                    k = at(cx, cy)
                    if k:
                        out.setdefault(k, []).append(Fixture(kind))
    return out


def _net_maps(rel: str, shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """CubiCasa net room and icon class maps at image resolution, if the perception features are cached."""
    try:
        from app.cubicasa.features import feature_paths

        npz = feature_paths(rel)[0]
        if not npz.is_file():
            return None, None
        z = np.load(npz)
        rooms, icons = z["rooms"], z["icons"]
        return (rooms if rooms.shape == shape else None), (icons if icons.shape == shape else None)
    except Exception:  # noqa: BLE001  optional evidence
        return None, None


def _walls(mask: np.ndarray, kind: str, tol: float, min_px: int = 50) -> list[WallShape]:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.int32), lab, np.arange(1, n + 1))
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sizes[i] < min_px:
            continue
        shape = _shape(lab == i + 1, sl, tol, holes=True)
        if shape is None:
            continue
        out.append(WallShape(id=f"{kind}-{len(out) + 1}", kind=kind, polygon=shape[0], holes=shape[1], bbox=_box(sl),
                             area_px=float(sizes[i])))
    return out


def _openings(cls, lab, room_id, outside, t, graph, tol) -> list[WallShape]:
    """Openings from the wall model's opening class, else the detected doors/windows; each joins the rooms it touches."""
    symbols = [it for it in (graph.items if graph else []) if it.category in ("door", "window")]
    r = round(t) + 2
    out: list[WallShape] = []
    if (cls == 2).any():
        cand = _walls(cls == 2, "opening", tol, min_px=20)
        olab, _ = ndimage.label(cls == 2, structure=np.ones((3, 3)))
    else:  # no opening class: the symbol boxes are the openings
        cand, olab = [], None
        for it in symbols:
            x0, y0, x1, y1 = it.bbox
            cand.append(WallShape(id=f"opening-{len(cand) + 1}", kind="opening", bbox=it.bbox,
                                  polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], area_px=(x1 - x0) * (y1 - y0)))
    H, W = lab.shape
    for op in cand:
        x0, y0, x1, y1 = (int(v) for v in op.bbox)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, max(x1, x0 + 1)), min(H, max(y1, y0 + 1))
        sl = (slice(y0, y1), slice(x0, x1))
        if olab is not None:
            centre = olab[(y0 + y1) // 2, (x0 + x1) // 2] or np.bincount(olab[sl].ravel()).argmax()
            mask = olab[sl] == centre if centre else np.ones((y1 - y0, x1 - x0), bool)
        else:
            mask = np.ones((y1 - y0, x1 - x0), bool)
        touched = _touching(lab, mask, sl, r)
        op.rooms = sorted({room_id[v] for v in touched if v in room_id}) + (["outside"] if set(touched) & outside else [])
        best, best_a = None, 0.0
        for it in symbols:
            ix = min(x1, it.bbox[2]) - max(x0, it.bbox[0])
            iy = min(y1, it.bbox[3]) - max(y0, it.bbox[1])
            if ix > 0 and iy > 0 and ix * iy > best_a:
                best, best_a = it.category, ix * iy
        op.symbol = best
        if op.rooms:
            out.append(op)
    for j, op in enumerate(out):
        op.id = f"opening-{j + 1}"
    return out


def geometry_from_graph(graph: PlanGraph) -> PlanGeometry:
    """PDF plans: rooms come from the drawing itself (DXF outlines / text stacks), no wall map."""
    rooms = []
    for it in graph.items:
        if it.category != "room" or not it.polygon:
            continue
        x0, y0, x1, y1 = it.bbox
        area = abs(Polygon(it.polygon).area) if len(it.polygon) >= 3 else (x1 - x0) * (y1 - y0)
        labels = [RoomLabel(text=it.text, bbox=it.bbox, room_type=room_purpose(it.text.split()[0]) if it.text else None)]             if it.text else []
        purpose = labels[0].room_type if labels else None
        rtype = purpose or (it.cls if it.cls not in ("", "room", "other") else None)
        evidence = [RoomEvidence(source="drawing", type=rtype, weight=6.0, detail=f'drawing text "{it.text}"')] \
            if rtype and it.text else []
        rooms.append(RoomShape(id=it.id, type=rtype or "unknown", tier="strong" if rtype else "unknown",
                               purposes=[rtype] if rtype else [], evidence=evidence, name=it.text,
                               labels=labels, polygon=list(it.polygon), bbox=it.bbox, area_px=area,
                               labeled_area_m2=_labeled_area(labels)))
    return PlanGeometry(image=graph.image, width=graph.width, height=graph.height, model="pdf", rooms=rooms,
                        scale=scale_from_written_areas(rooms))


def main() -> None:
    from app.cubicasa.symdet import image_path

    rel = sys.argv[1]
    g = geometry_from_image(rel, image_path(rel))
    print(f"{g.model}: {len(g.rooms)} rooms, {len(g.wall_segments)} wall segments, {len(g.openings)} openings, "
          f"wall {g.wall_thickness_px} px, scale {g.scale}")
    for rm in g.rooms:
        print(f"  {rm.id:<10}{rm.tier:<8}{rm.type:<10}{rm.scores}  evidence={[e.detail for e in rm.evidence]}  "
              f"conflicts={rm.conflicts}  {rm.area_px:>9.0f} px  name={rm.name!r} "
              f"labels={[lb.text for lb in rm.labels]} m2={rm.labeled_area_m2} nb={rm.neighbours} via={rm.connections}")


if __name__ == "__main__":
    main()
