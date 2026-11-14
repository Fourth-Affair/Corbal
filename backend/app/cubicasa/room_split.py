"""Split spaces that hold two rooms: a bedroom open to the living room, two bedrooms sharing one flood-fill region.

The wall map closes rooms that have doors; rooms joined by a wide opening or only a wall stub stay one space. When a
space carries two strong room labels that should not share one space, it is cut along the straight horizontal or
vertical line between the labels that crosses the least free space (through wall stubs and the narrowest neck).
Open plans (kitchen, living, dining, hall, stairs) stay together. Measured with app.cubicasa.rooms_eval.
(Closing doorway-sized gaps between wall ends was tried and removed: corridors are as wide as passages, so it cut
halls into pieces.)
"""

import numpy as np
from scipy import ndimage

OPEN_PLAN = {"kitchen", "living", "dining", "entry", "stairs"}  # these share one space without walls

CUT_MAX_SPAN = 0.85  # the cut may cross at most this share of the region's extent along it


def _separate(a: str, b: str) -> bool:
    """Two labelled purposes that should not share one space."""
    if a in OPEN_PLAN and b in OPEN_PLAN:
        return False
    return a != b or a in ("bedroom", "study")


def split_by_labels(lab: np.ndarray, seeds: list[tuple[float, float, str]], t: float, min_area: float
                    ) -> tuple[np.ndarray, list[tuple[tuple[float, float], tuple[float, float]]]]:
    """Cut spaces holding two room labels that should not share a space (a bedroom open to the living room).
    The cut is the straight horizontal or vertical line between the two labels that crosses the least free space,
    so it runs through wall stubs and the narrowest neck. Both sides must keep >= min_area px and the cut may
    cross at most CUT_MAX_SPAN of the region; otherwise the space stays whole. Repeats while cuts are found."""
    out = lab.copy()
    cuts: list[tuple[tuple[float, float], tuple[float, float]]] = []
    H, W = lab.shape
    nxt = int(lab.max()) + 1
    for _ in range(12):  # each pass makes at most one cut per space
        made = False
        by_region: dict[int, list[tuple[float, float, str]]] = {}
        for x, y, p in seeds:
            xi, yi = int(x), int(y)
            if 0 <= xi < W and 0 <= yi < H and out[yi, xi]:
                by_region.setdefault(int(out[yi, xi]), []).append((x, y, p))
        for k, ss in by_region.items():
            pairs = [(a, b) for i, a in enumerate(ss) for b in ss[i + 1:]
                     if _separate(a[2], b[2]) and max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 3 * t]
            if not pairs:
                continue
            ys, xs = np.nonzero(out == k)
            y0, x0 = ys.min(), xs.min()
            region = out[y0:ys.max() + 1, x0:xs.max() + 1] == k
            best = None
            for a, b in pairs:
                for axis in (1, 0):  # 1: vertical cut line (x = c), 0: horizontal (y = c)
                    lo, hi = sorted((a[0], b[0]) if axis == 1 else (a[1], b[1]))
                    lo, hi = int(lo - (x0 if axis == 1 else y0)), int(hi - (x0 if axis == 1 else y0))
                    if hi - lo < 2 * t:
                        continue
                    span = region.sum(axis=0) if axis == 1 else region.sum(axis=1)
                    m = int(t)
                    cand = range(lo + m, hi - m + 1)
                    if not len(cand):
                        continue
                    c = min(cand, key=lambda v: span[v])
                    extent = region.shape[0] if axis == 1 else region.shape[1]
                    if span[c] > CUT_MAX_SPAN * extent:
                        continue
                    side1 = region[:, :c].sum() if axis == 1 else region[:c].sum()
                    side2 = region.sum() - side1 - span[c]
                    if min(side1, side2) < min_area:
                        continue
                    if best is None or span[c] < best[0]:
                        best = (span[c], axis, c)
            if best is None:
                continue
            _, axis, c = best
            sub = out[y0:y0 + region.shape[0], x0:x0 + region.shape[1]]
            if axis == 1:
                sub[:, c:][region[:, c:]] = nxt
                col = region[:, c]
                yy = np.nonzero(col)[0]
                cuts.append(((float(x0 + c), float(y0 + yy.min())), (float(x0 + c), float(y0 + yy.max()))))
            else:
                sub[c:][region[c:]] = nxt
                row = region[c]
                xx = np.nonzero(row)[0]
                cuts.append(((float(x0 + xx.min()), float(y0 + c)), (float(x0 + xx.max()), float(y0 + c))))
            nxt += 1
            made = True
        if not made:
            break
    # a straight cut can leave a far piece of the region disconnected on the wrong side: relabel components
    final = np.zeros_like(out)
    nid = 1
    for k in np.unique(out):
        if k == 0:
            continue
        comp, nc = ndimage.label(out == k)
        for c in range(1, nc + 1):
            final[comp == c] = nid
            nid += 1
    return final, cuts
