"""Virtual walls across passages: close doorway-sized gaps that have no door, so rooms that only meet through an
open passage (a bedroom off the living room, a hall into a kitchen) become separate spaces.

A wall that stops in the middle of nowhere usually stops at a passage. For each free end of the wall skeleton we
take the wall's direction just before the end (snapped to horizontal / vertical: plans are drawn on a grid), cast a
ray forward through free space and, when it hits another wall within a passage width, draw a virtual wall of the
plan's wall thickness across the gap. Passage width: PASSAGE_M metres when a scale estimate exists (the door /
fixture size band of size_priors is enough), else PASSAGE_T x the wall thickness.
Deterministic; the virtual walls are returned so they can be drawn and checked.
"""

import numpy as np
from scipy import ndimage

PASSAGE_M = (0.6, 1.6)  # passage widths that get closed, metres
PASSAGE_T = (3.0, 9.0)  # the same in wall thicknesses, when no scale is known
SNAP_DEG = 20  # wall ends within this of horizontal / vertical are used; others (diagonal walls) are skipped
TRACE = 2.0  # how far back along the skeleton (x wall thickness) the end direction is measured


def _endpoints(skel: np.ndarray) -> np.ndarray:
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skel
    return np.argwhere(skel & (nb == 1))


def _direction(skel: np.ndarray, y: int, x: int, steps: int) -> tuple[int, int] | None:
    """Unit axis direction pointing out of the wall end at (y, x), or None for diagonal ends."""
    H, W = skel.shape
    seen = {(y, x)}
    frontier = [(y, x)]
    far = (y, x)
    for _ in range(steps):
        nxt = []
        for cy, cx in frontier:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if (ny, nx) not in seen and 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                        seen.add((ny, nx))
                        nxt.append((ny, nx))
        if not nxt:
            break
        frontier = nxt
        far = nxt[0]
    vy, vx = y - far[0], x - far[1]
    if vy == 0 and vx == 0:
        return None
    ang = np.degrees(np.arctan2(abs(vy), abs(vx)))
    if ang <= SNAP_DEG:
        return 0, int(np.sign(vx))
    if ang >= 90 - SNAP_DEG:
        return int(np.sign(vy)), 0
    return None


def close_passages(barrier: np.ndarray, skel: np.ndarray, t: float, m_per_px: float | None = None
                   ) -> tuple[np.ndarray, list[tuple[tuple[float, float], tuple[float, float]]]]:
    """(virtual wall mask, [(p0, p1) in image x, y]) for the passages found."""
    H, W = barrier.shape
    if m_per_px:
        lo, hi = PASSAGE_M[0] / m_per_px, PASSAGE_M[1] / m_per_px
    else:
        lo, hi = PASSAGE_T[0] * t, PASSAGE_T[1] * t
    half = max(1, int(round(t / 2)))
    virtual = np.zeros_like(barrier)
    walls: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for y, x in _endpoints(skel):
        d = _direction(skel, int(y), int(x), max(3, int(TRACE * t)))
        if d is None:
            continue
        dy, dx = d
        # leave the wall body first (the skeleton end lies inside the wall), then march through free space
        k = 1
        while k < 3 * t and 0 <= y + dy * k < H and 0 <= x + dx * k < W and barrier[y + dy * k, x + dx * k]:
            k += 1
        start = k
        hit = None
        while k <= start + hi + 1:
            py, px = y + dy * k, x + dx * k
            if not (0 <= py < H and 0 <= px < W):
                break
            # the corridor the virtual wall would occupy must be free (no grazing along a parallel wall)
            if barrier[py, px]:
                hit = k
                break
            k += 1
        if hit is None or not lo <= hit - start <= hi:
            continue
        y1, x1 = y + dy * hit, x + dx * hit
        ys, xs = sorted((y, y1)), sorted((x, x1))
        if dy == 0:
            virtual[max(0, y - half):y + half + 1, xs[0]:xs[1] + 1] = True
        else:
            virtual[ys[0]:ys[1] + 1, max(0, x - half):x + half + 1] = True
        walls.append(((float(x), float(y)), (float(x1), float(y1))))
    return virtual, walls


def split_by_evidence(lab0: np.ndarray, virtual: np.ndarray, seeds: list[tuple[float, float]], r: int) -> np.ndarray:
    """Regions of `lab0` that the virtual walls cut into pieces stay split only where >= 2 pieces carry their own
    evidence (a room label, a decisive fixture: `seeds`, image x, y). Pieces without evidence join the evidence
    piece they touch most; the virtual wall pixels go to the nearest piece. Returns the new label image."""
    lab1, _ = ndimage.label((lab0 > 0) & ~virtual)
    H, W = lab0.shape
    seed_parts: dict[int, set[int]] = {}
    for x, y in seeds:
        xi, yi = int(x), int(y)
        if 0 <= xi < W and 0 <= yi < H and lab1[yi, xi]:
            seed_parts.setdefault(int(lab0[yi, xi]), set()).add(int(lab1[yi, xi]))
    out = lab0.copy()
    nxt = int(lab0.max()) + 1
    for k, seeded in seed_parts.items():
        if len(seeded) < 2:
            continue
        region = lab0 == k
        parts = [int(v) for v in np.unique(lab1[region]) if v]
        group = {p: p for p in seeded}
        pending = [p for p in parts if p not in group]
        for _ in range(len(parts)):  # unseeded pieces join the seeded group they touch most
            if not pending:
                break
            still = []
            for p in pending:
                ring = ndimage.binary_dilation(lab1 == p, iterations=r + 2) & region
                touch = [(int(np.sum(lab1[ring] == q)), group[q]) for q in group if np.any(lab1[ring] == q)]
                if touch:
                    group[p] = max(touch)[1]
                else:
                    still.append(p)
            if len(still) == len(pending):
                break
            pending = still
        biggest = max(seeded, key=lambda p: int(np.sum(lab1 == p)))
        ids = {g: (k if g == biggest else None) for g in set(group.values())}
        for g in ids:
            if ids[g] is None:
                ids[g] = nxt
                nxt += 1
        new = np.zeros_like(lab0)
        for p in parts:
            new[lab1 == p] = ids[group.get(p, biggest)]
        # virtual wall pixels (and anything unassigned) inside the region: nearest piece
        hole = region & (new == 0)
        if hole.any():
            idx = ndimage.distance_transform_edt(new == 0, return_distances=False, return_indices=True)
            filled = new[idx[0], idx[1]]
            new[hole] = filled[hole]
        out[region] = new[region]
    return out


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
