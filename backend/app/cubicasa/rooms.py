"""Plan geometry: rooms as polygons enclosed by walls, with the OCR text inside each room, plus walls and openings.

  python -m app.cubicasa.rooms data/cubicasa5k/colorful-10052.png      # print a summary (writes no cache)

Confidence: a room is `labeled` only when a clear purpose label (rooms_dict.room_purpose, OCR score >= LABEL_MIN_SCORE)
lies inside it; every other enclosed space is `rough` (its shape may be a closet, shaft or segmentation artefact, its
type at best a segmentation guess). Overlays show labeled rooms; rough ones only on request.

Rooms are the connected free-space regions of the wall map (layers.wall_classes: wall segmenter, else the CubiCasa
net). Openings count as wall, so a doorway separates two rooms; gaps narrower than half a wall are closed first.
Regions touching the image border are outside. A room's type comes from the room labels read inside it (OCR +
rooms_dict, e.g. "MH" -> bedroom); without a label, from the CubiCasa room segmentation if it is cached.
Openings (or detected doors/windows when the wall model has no opening class) link the rooms they touch.
Everything is in image pixels; areas in m² need the plan's scale and are computed by the caller.
"""

import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from skimage.measure import find_contours

from app.cubicasa.rooms_dict import room_purpose
from app.models import PlanGeometry, PlanGraph, RoomLabel, RoomShape, WallShape

# CubiCasa net room classes -> our room types (fuse.ROOM order); others carry no type
_SEG_TYPES = {1: "balcony", 3: "kitchen", 4: "living", 5: "bedroom", 6: "bathroom", 7: "entry", 9: "storage",
              10: "garage"}
_AREA_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|m\^2|sqm|㎡)", re.IGNORECASE)
MIN_ROOM_FRAC = 1e-3  # of the image area
MAX_ROOM_FRAC = 0.5
MAX_POINTS = 400
LABEL_MIN_SCORE = 0.7  # OCR confidence for a label to make a room `labeled`


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


def _room_type(labels: list[RoomLabel]) -> tuple[str | None, str | None, list[str]]:
    """(type, name, purposes) from the clear purpose labels inside a room. Several purposes (open plan: "OH" + "K")
    keep the room's type as the most frequent one and list them all."""
    typed = [lb for lb in labels if lb.room_type and (lb.score or 0) >= LABEL_MIN_SCORE]
    if not typed:
        return None, None, []
    counts: dict[str, int] = {}
    for lb in typed:
        counts[lb.room_type] = counts.get(lb.room_type, 0) + 1
    purposes = list(dict.fromkeys(lb.room_type for lb in typed))
    best = max(purposes, key=lambda t: (counts[t], -purposes.index(t)))
    return best, " / ".join(dict.fromkeys(lb.text for lb in typed)), purposes


def _labeled_area(labels: list[RoomLabel]) -> float | None:
    m = _AREA_RE.search(" ".join(lb.text for lb in labels))
    return float(m.group(1).replace(",", ".")) if m else None


def geometry_from_image(rel: str, path: Path, graph: PlanGraph | None = None) -> PlanGeometry:
    """Rooms, walls and openings of an image plan. `graph` (optional) supplies detected doors/windows."""
    from app.cubicasa import layers

    cls, model = layers.wall_classes(rel, path)
    H, W = cls.shape
    barrier = cls > 0
    # wall thickness: twice the median distance-to-background along the wall skeleton
    from skimage.morphology import skeletonize

    edt = ndimage.distance_transform_edt(barrier)
    skel = skeletonize(barrier)
    t = float(2 * np.median(edt[skel])) if skel.any() else 4.0
    t = min(max(t, 2.0), 0.05 * max(H, W))
    closed = ndimage.binary_closing(barrier, iterations=max(1, round(t / 4))) | barrier

    lab, n = ndimage.label(~closed)
    outside = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.int32), lab, np.arange(1, n + 1))
    slices = ndimage.find_objects(lab)
    keep = [i + 1 for i in range(n) if i + 1 not in outside and MIN_ROOM_FRAC * H * W <= sizes[i] <= MAX_ROOM_FRAC * H * W]
    # reading order: top-to-bottom, then left-to-right by bbox centre
    keep.sort(key=lambda k: (round((slices[k - 1][0].start + slices[k - 1][0].stop) / 2 / max(1.0, 3 * t)),
                             (slices[k - 1][1].start + slices[k - 1][1].stop) / 2))
    room_id = {k: f"space-{j + 1}" for j, k in enumerate(keep)}

    # OCR tokens -> rooms (token centre, else the most common room within one wall thickness)
    text = layers.text(rel, path)
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

    seg = _room_segmentation(rel, (H, W))
    tol = max(1.0, 0.002 * max(H, W))
    rooms: list[RoomShape] = []
    for k in keep:
        sl = slices[k - 1]
        shape = _shape(lab == k, sl, tol)
        if shape is None:
            continue
        labels = labels_of[k]
        rtype, name, purposes = _room_type(labels)
        source = "label" if rtype else "unknown"
        confidence = "labeled" if rtype else "rough"
        if rtype is None and seg is not None:
            vals = seg[sl][lab[sl] == k]
            vals = vals[np.isin(vals, list(_SEG_TYPES))]
            if vals.size >= 0.3 * sizes[k - 1]:
                rtype, source = _SEG_TYPES[int(np.bincount(vals).argmax())], "segmentation"
        neighbours = [room_id[v] for v in _touching(lab, lab[sl] == k, sl, round(1.5 * t) + 2) if v in room_id]
        rooms.append(RoomShape(id=room_id[k], type=rtype or "unknown", type_source=source, confidence=confidence,
                               purposes=purposes, name=name, labels=labels,
                               polygon=shape[0], bbox=_box(sl), area_px=float(sizes[k - 1]),
                               labeled_area_m2=_labeled_area(labels), neighbours=neighbours))

    walls = _walls(cls == 1, "wall", tol)
    openings = _openings(cls, lab, room_id, outside, t, graph, tol)
    by_id = {rm.id: rm for rm in rooms}
    for op in openings:
        for rid in op.rooms:
            if rid in by_id:
                by_id[rid].connections.append(op.id)
    return PlanGeometry(image=rel, width=W, height=H, model=model, wall_thickness_px=round(t, 1), rooms=rooms,
                        walls=walls, openings=openings)


def _room_segmentation(rel: str, shape: tuple[int, int]) -> np.ndarray | None:
    """CubiCasa net room classes at image resolution, if the perception features are cached."""
    try:
        from app.cubicasa.features import feature_paths

        npz = feature_paths(rel)[0]
        if not npz.is_file():
            return None
        rooms = np.load(npz)["rooms"]
        return rooms if rooms.shape == shape else None
    except Exception:  # noqa: BLE001  optional evidence
        return None


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
        rooms.append(RoomShape(id=it.id, type=rtype or "unknown", type_source="drawing",
                               confidence="labeled" if rtype or it.text else "rough",
                               purposes=[rtype] if rtype else [], name=it.text,
                               labels=labels, polygon=list(it.polygon), bbox=it.bbox, area_px=area,
                               labeled_area_m2=_labeled_area(labels)))
    return PlanGeometry(image=graph.image, width=graph.width, height=graph.height, model="pdf", rooms=rooms)


def main() -> None:
    from app.cubicasa.symdet import image_path

    rel = sys.argv[1]
    g = geometry_from_image(rel, image_path(rel))
    print(f"{g.model}: {len(g.rooms)} rooms, {len(g.walls)} walls, {len(g.openings)} openings, wall {g.wall_thickness_px} px")
    for rm in g.rooms:
        print(f"  {rm.id:<10}{rm.confidence:<9}{rm.type:<10}{rm.type_source:<13}{rm.area_px:>9.0f} px  name={rm.name!r} "
              f"labels={[lb.text for lb in rm.labels]} m2={rm.labeled_area_m2} nb={rm.neighbours} via={rm.connections}")


if __name__ == "__main__":
    main()
