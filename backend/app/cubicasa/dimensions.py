"""Plan scale from dimension lines: the lengths printed along the walls (millimetres) and the pixel distance between
the ticks of the segment each one labels.

  python -m app.cubicasa.dimensions data/cubicasa5k/high_quality_architectural-5640.png

Per numeric OCR token:
1. Read it. Finnish drawings print millimetres with a thousands space ("3 100"); OCR splits vertical figures into
   pieces ("3" + "100") or drops one, so pieces in line are joined and the joined box, extended along the reading
   direction, is recognised again as one crop. Both readings are kept as alternatives.
2. Find its dimension line: the nearest thin dark run parallel to the text, within 1.5 text heights, covering the
   text's extent (a run thicker than half a text height is a wall, not a dimension line).
3. Find the segment's end ticks: walking along the line from the text centre, the first place on each side where
   strokes cross the line (an extension line, a slash or a dot), or where the line ends in a stroke.
4. m/px = length / span. The plan scale is the median of the readings that agree within DIM_AGREE, when at least
   MIN_AGREE different figures agree. A misread figure or a wrong segment agrees with nothing and is dropped.
Everything is deterministic and in image pixels; the figures used are returned so they can be shown and checked.
"""

import re
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app.models import PlanScale, ScaleDimension, TextToken

INK = 140  # grey level below which a pixel is ink
MIN_MM, MAX_MM = 100, 60000
DIM_AGREE = 0.04  # relative m/px agreement between figures
MIN_AGREE = 2
LINE_COVER = 0.9  # share of the text's extent the dimension line must cover
_DIGITS = re.compile(r"^\d[\d ]*$")


def _ink(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L")) < INK


def _vertical(b) -> bool:
    return (b[3] - b[1]) > 1.3 * (b[2] - b[0])


def _groups(tokens: list[TextToken]) -> list[list[TextToken]]:
    """Numeric tokens, with pieces of one figure joined (in line, same orientation, gap < 0.7 x text height)."""
    nums = [t for t in tokens if _DIGITS.match(t.text.replace(" ", " ").strip())]
    parent = list(range(len(nums)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(nums):
        for j in range(i + 1, len(nums)):
            b = nums[j]
            va, vb = _vertical(a.bbox), _vertical(b.bbox)
            # a 1-digit piece is square-ish: take the orientation of the longer piece
            vert = va if (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]) >= (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) else vb
            (p0, p1), (q0, q1) = ((0, 2), (1, 3)) if vert else ((1, 3), (0, 2))  # across axis, along axis
            ax0, ax1, bx0, bx1 = a.bbox[p0], a.bbox[p1], b.bbox[p0], b.bbox[p1]
            overlap = min(ax1, bx1) - max(ax0, bx0)
            short = min(ax1 - ax0, bx1 - bx0)
            gap = max(a.bbox[q0], b.bbox[q0]) - min(a.bbox[q1], b.bbox[q1])
            if short > 0 and overlap >= 0.5 * short and gap <= 0.7 * max(ax1 - ax0, bx1 - bx0):
                parent[find(i)] = find(j)
    out: dict[int, list[TextToken]] = {}
    for i, t in enumerate(nums):
        out.setdefault(find(i), []).append(t)
    return list(out.values())


def _union(group: list[TextToken]) -> tuple[float, float, float, float]:
    return (min(t.bbox[0] for t in group), min(t.bbox[1] for t in group),
            max(t.bbox[2] for t in group), max(t.bbox[3] for t in group))


def _readings(img: Image.Image, group: list[TextToken], box, vert: bool) -> list[tuple[str, float]]:
    """Alternative (text, value mm) readings of one figure: the pieces joined in reading order, and the joined box
    recognised again (extended along the text so a dropped leading / trailing digit is included)."""
    from app.cubicasa.ocr import recognise

    # vertical dimension text reads bottom-to-top (rotated 90° counter-clockwise), horizontal left-to-right
    order = sorted(group, key=lambda t: -t.bbox[3]) if vert else sorted(group, key=lambda t: t.bbox[0])
    alts = ["".join(t.text for t in order)]
    x0, y0, x1, y1 = box
    ext = 0.7 * ((x1 - x0) if vert else (y1 - y0))
    for e in (0.0, ext):
        b = (x0, y0 - e, x1, y1 + e) if vert else (x0 - e, y0, x1 + e, y1)
        text, score = recognise(img, b, vert)
        if score >= 0.7:
            alts.append(text)
    out = []
    for a in dict.fromkeys(re.sub(r"[\s ]", "", a) for a in alts):
        if a.isdigit() and 3 <= len(a) <= 5 and MIN_MM <= int(a) <= MAX_MM:
            out.append((a, float(a)))
    return out


def _segment(ink: np.ndarray, box) -> tuple[float, float, float] | None:
    """(left tick x, right tick x, line y) of the dimension segment below / above a horizontal text box, in the
    frame of `ink` (transpose both for vertical text)."""
    H, W = ink.shape
    x0, y0, x1, y1 = box
    h = y1 - y0
    xa, xb = int(x0 + 0.15 * (x1 - x0)), int(x1 - 0.15 * (x1 - x0))
    if xb - xa < 3 or h < 3:
        return None
    def cover(y: int) -> float:  # share of the text's extent with ink within one row of y (lines are skewed)
        return float(ink[max(0, y - 1):y + 2, xa:xb].any(axis=0).mean())

    line = None
    # nearest row to the text first; OCR boxes are padded, so the line can lie just inside the box
    for dy in sorted(range(-int(0.35 * h), int(1.5 * h) + 1), key=abs):
        for y in (int(y1) + dy, int(y0) - dy):
            if 0 < y < H - 1 and cover(y) >= LINE_COVER:
                line = y
                break
        if line is not None:
            break
    if line is None:
        return None
    line = max(range(max(0, line - 1), min(H, line + 2)), key=lambda y: ink[y, xa:xb].mean())
    ya = yb = line
    while ya > 0 and ink[ya - 1, xa:xb].mean() >= 0.5:
        ya -= 1
    while yb < H - 1 and ink[yb + 1, xa:xb].mean() >= 0.5:
        yb += 1
    if yb - ya + 1 > max(3, 0.5 * h):
        return None  # a wall, not a dimension line
    r = max(3, round(0.3 * h))
    m = 2  # the line's band: a scanned line drifts by a pixel or two
    reach = max(2, r // 3)  # a crossing stroke reaches at least this many rows beyond the band
    above = ink[max(0, ya - m - r):max(0, ya - m)].sum(axis=0) >= reach
    below = ink[yb + m + 1:yb + m + 1 + r].sum(axis=0) >= reach
    on = ink[max(0, ya - m):yb + m + 1].any(axis=0)
    w = max(1, r // 2)  # slashes cross the line diagonally: look a little to the sides

    def win(a, x):
        return a[max(0, x - w):x + w + 1].any()

    def walk(start: int, step: int) -> float | None:
        x, gap, last_on = start, 0, start
        while 0 <= x < W:
            outside_text = x < x0 or x > x1
            if outside_text and win(above, x) and win(below, x):  # stroke across the line
                xs = [x]
                while 0 <= xs[-1] + step < W and win(above, xs[-1] + step) and win(below, xs[-1] + step):
                    xs.append(xs[-1] + step)
                return (xs[0] + xs[-1]) / 2
            if on[x]:
                gap, last_on = 0, x
            else:
                gap += 1
                if gap > r:  # line ended (scanned lines have gaps of a few pixels): a stroke on one side at its end is the tick (extension line)
                    for e in range(last_on, last_on - step * (r + 1), -step):
                        if 0 <= e < W and (e < x0 or e > x1) and (win(above, e) or win(below, e)):
                            return float(e)
                    return None
            x += step
        return None

    cx = int((x0 + x1) / 2)
    a, b = walk(cx, -1), walk(cx, 1)
    if a is None or b is None or b - a < 0.8 * (x1 - x0):
        return None
    return a, b, (ya + yb) / 2


def dimension_scale(img: Image.Image, tokens: list[TextToken]) -> PlanScale | None:
    """m/px from the plan's dimension figures; None when fewer than MIN_AGREE figures agree."""
    ink = _ink(img)
    cands = []  # (figure index, ScaleDimension, m/px)
    for gi, group in enumerate(_groups(tokens)):
        box = _union(group)
        vert = _vertical(box)
        if vert:
            seg = _segment(ink.T, (box[1], box[0], box[3], box[2]))
        else:
            seg = _segment(ink, box)
        if seg is None:
            continue
        a, b, c = seg
        p0, p1 = ((c, a), (c, b)) if vert else ((a, c), (b, c))
        for text, mm in _readings(img, group, box, vert):
            cands.append((gi, ScaleDimension(text=text, value_mm=mm, bbox=box, p0=p0, p1=p1, length_px=round(b - a, 1)),
                          mm / 1000 / (b - a)))
    if not cands:
        return None

    def support(m):
        return len({gi for gi, _, v in cands if abs(v - m) <= DIM_AGREE * m})

    centre = max((v for _, _, v in cands), key=lambda m: (support(m), -statistics.median(
        abs(v - m) for _, _, v in cands)))
    if support(centre) < MIN_AGREE:
        return None
    used: dict[int, tuple[ScaleDimension, float]] = {}
    for gi, d, v in cands:
        if abs(v - centre) <= DIM_AGREE * centre and (gi not in used or abs(v - centre) < abs(used[gi][1] - centre)):
            used[gi] = (d, v)
    return PlanScale(m_per_px=float(statistics.median(v for _, v in used.values())), source="dimensions",
                     dimensions=[d for d, _ in used.values()])


def main() -> None:
    from app.cubicasa import layers
    from app.cubicasa.symdet import image_path

    rel = sys.argv[1]
    path: Path = image_path(rel)
    img = Image.open(path)
    tokens = layers.text(rel, path).tokens
    for g in _groups(tokens):
        box = _union(g)
        vert = _vertical(box)
        seg = _segment(_ink(img).T, (box[1], box[0], box[3], box[2])) if vert else _segment(_ink(img), box)
        print(f"{[t.text for t in g]} {'V' if vert else 'H'} {[round(v) for v in box]} readings="
              f"{[r for r, _ in _readings(img, g, box, vert)]} segment={seg and tuple(round(v, 1) for v in seg)}")
    print(dimension_scale(img, tokens))


if __name__ == "__main__":
    main()
