"""Wall map -> straight wall segments (centrelines) with length, thickness and the rooms on either side.

The wall + opening mask is skeletonised (openings included so a wall stays one line through a doorway), the skeleton's
pixel graph is traced between nodes (pixels whose neighbour count is not 2) and every traced path is simplified with
Douglas-Peucker into straight pieces. Per piece: endpoints, length, thickness (2 x median distance to the background
along it), angle, the rooms sampled just beyond each face ("outside" = exterior) and the openings it crosses.
Coordinates are image pixels (y down); "left" is the left-hand side walking p0 -> p1 as seen on screen.
"""

import math

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, box

from app.models import WallSegment, WallShape

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _trace(skel: np.ndarray) -> list[tuple[list[tuple[float, float]], tuple[int, int]]]:
    """Paths between skeleton junctions as ([(x, y), ...], (junction degree at start, at end)).

    Node pixels (neighbour count != 2) that touch each other form one junction, placed at their centroid, so a
    T- or X-crossing is one point and not a cluster of 1-2 px stubs. Degree 1 = a free end (spur candidate)."""
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    nb[~skel] = 0
    nodes = skel & (nb != 2)
    cluster, n_clusters = ndimage.label(nodes, structure=np.ones((3, 3)))
    centre = {i + 1: (float(cx), float(cy)) for i, (cy, cx) in
              enumerate(ndimage.center_of_mass(nodes, cluster, np.arange(1, n_clusters + 1)))} if n_clusters else {}
    H, W = skel.shape
    visited = np.zeros_like(skel, bool)
    raw: list[tuple[list[tuple[int, int]], int, int]] = []  # pixels, start cluster, end cluster (0 = loop)

    def neighbours(y, x):
        for dy, dx in _N8:
            v, u = y + dy, x + dx
            if 0 <= v < H and 0 <= u < W and skel[v, u]:
                yield v, u

    def walk(start, nxt):
        path = [start, nxt]
        prev, cur = start, nxt
        while not nodes[cur]:
            visited[cur] = True
            step = next(((v, u) for v, u in neighbours(*cur) if (v, u) != prev and (v, u) not in path[-3:]
                         and (nodes[v, u] or not visited[v, u])), None)
            if step is None:
                break
            prev, cur = cur, step
            path.append(cur)
        return path

    for y, x in zip(*np.nonzero(nodes)):
        c0 = int(cluster[y, x])
        for v, u in neighbours(y, x):
            if nodes[v, u]:
                c1 = int(cluster[v, u])
                if c1 != c0 and (y, x) < (v, u):  # two junctions touching directly
                    raw.append(([(y, x), (v, u)], c0, c1))
                continue
            if not visited[v, u]:
                path = walk((y, x), (v, u))
                end = path[-1]
                raw.append((path, c0, int(cluster[end]) if nodes[end] else 0))
    # closed loops with no junction (a free-standing wall ring): start anywhere, both ends are the same point
    rest = skel & ~visited & ~nodes
    while rest.any():
        y, x = map(int, np.argwhere(rest)[0])
        nodes[y, x] = True
        nxt = next(((v, u) for v, u in neighbours(y, x) if not visited[v, u] and not nodes[v, u]), None)
        if nxt is not None:
            raw.append((walk((y, x), nxt) + [(y, x)], 0, 0))
        nodes[y, x] = False
        visited[y, x] = True
        rest = skel & ~visited & ~nodes

    degree: dict[int, int] = {}
    for _, c0, c1 in raw:
        for c in (c0, c1):
            if c:
                degree[c] = degree.get(c, 0) + 1
    out = []
    for pix, c0, c1 in raw:
        if c0 and c0 == c1 and len(pix) <= 3:
            continue  # inside one junction
        pts = [(float(x), float(y)) for y, x in pix]
        if c0:
            pts[0] = centre[c0]
        if c1:
            pts[-1] = centre[c1]
        out.append((pts, (degree.get(c0, 2) if c0 else 2, degree.get(c1, 1) if c1 else (2 if c0 == 0 else 1))))
    return out


def _side(lab: np.ndarray, to_id, pts: list[tuple[float, float]], normal: tuple[float, float],
          offsets: tuple[float, ...]) -> list[str]:
    """Room ids found beyond one face of a segment (majority over the sample points, nearest offset that hits)."""
    H, W = lab.shape
    found: dict[str, int] = {}
    for x, y in pts:
        for off in offsets:
            u, v = int(round(x + normal[0] * off)), int(round(y + normal[1] * off))
            if not (0 <= u < W and 0 <= v < H):
                found["outside"] = found.get("outside", 0) + 1
                break
            rid = to_id(int(lab[v, u]))
            if rid:
                found[rid] = found.get(rid, 0) + 1
                break
    need = max(1, len(pts) // 3)
    return sorted(r for r, n in found.items() if n >= need)


def vectorise(cls: np.ndarray, lab: np.ndarray, to_id, thickness: float,
              openings: list[WallShape] | None = None) -> list[WallSegment]:
    """Wall segments of a wall-class map (0 background, 1 wall, 2 opening).

    lab: free-space label map (rooms.py); to_id(label) -> room id, "outside" or None (unknown / tiny region)."""
    from skimage.morphology import skeletonize

    mask = cls > 0
    if not mask.any():
        return []
    skel = skeletonize(mask)
    edt = ndimage.distance_transform_edt(mask)
    paths = _trace(skel)
    H, W = mask.shape
    tol = max(1.0, thickness / 2)
    min_spur = max(4.0, 1.5 * thickness)
    offsets = tuple(thickness / 2 + d for d in (3, 3 + thickness / 2, 3 + thickness))
    ops = [(op, box(*op.bbox).buffer(2)) for op in (openings or [])]
    segs: list[WallSegment] = []
    for coords, end_deg in paths:
        line = LineString(coords) if len(coords) >= 2 else None
        if line is None or line.length == 0:
            continue
        if line.length < min_spur and 1 in end_deg:
            continue  # spur: a skeleton whisker at a wall end or corner
        on = [edt[min(H - 1, int(round(y))), min(W - 1, int(round(x)))] for x, y in coords]
        t_med = float(2 * np.median(on))
        simple = list(line.simplify(tol).coords)
        for (x0, y0), (x1, y1) in zip(simple, simple[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            if length < 1:
                continue
            dx, dy = (x1 - x0) / length, (y1 - y0) / length
            angle = math.degrees(math.atan2(dy, dx)) % 180
            orient = "horizontal" if min(angle, 180 - angle) <= 10 else "vertical" if abs(angle - 90) <= 10 else "diagonal"
            samples = [(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f) for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
            left = _side(lab, to_id, samples, (dy, -dx), offsets)
            right = _side(lab, to_id, samples, (-dy, dx), offsets)
            seg_line = LineString([(x0, y0), (x1, y1)])
            crossing = [op.id for op, shape in ops if seg_line.intersects(shape)]
            segs.append(WallSegment(id=f"seg-{len(segs) + 1}", p0=(round(x0, 1), round(y0, 1)),
                                    p1=(round(x1, 1), round(y1, 1)), length_px=round(length, 1),
                                    thickness_px=round(t_med, 1), angle_deg=round(angle, 1), orientation=orient,
                                    rooms_left=left, rooms_right=right,
                                    exterior="outside" in left or "outside" in right, openings=crossing))
    return segs
