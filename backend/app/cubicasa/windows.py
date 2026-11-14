"""Split window detections that cover several adjacent windows.

The symbol detector often returns one box for a run of windows separated only by a mullion: one thin line crossing
the window symbol (CubiCasa draws a window as 2-3 long lines along the wall, closed by short jambs at the ends). The
wall segmenter cannot help: it paints the whole run as one opening. The drawing itself can: along the box's long
axis, a column whose ink spans (almost) the whole depth of the window, away from the ends, is a mullion or a jamb
between two windows. Pieces shorter than MIN_PIECE x the window depth are not split off (sash marks, handles).

  python -m app.cubicasa.windows data/cubicasa5k/high_quality_architectural-7731.png   # print the splits
"""

import sys

import numpy as np
from PIL import Image

from app.models import PlanItem

INK = 160
COVER = 0.45  # share of the empty strips between the window's long lines a mullion must bridge
GAP_ROW_MAX = 0.35  # rows (along the window) inked less than this are the strips between the lines
LINE_ROW_MIN = 0.7  # rows inked more than this are the window's long lines (at least 2 are required)
MIN_PIECE = 1.0  # minimum piece length, x the window depth
REQUIRE_DOUBLE = True  # cut only where two crossing lines meet (two jambs), see split_box
DOUBLE_MAX = 0.8  # max gap between the two jamb lines, x the window depth
MIN_ASPECT = 2.0  # only boxes at least this elongated are candidates (a window spans a wall, it is long and thin)


def _ink(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L")) < INK


def split_box(ink: np.ndarray, box) -> list[tuple[float, float, float, float]]:
    """The box itself, or its pieces when interior crossing lines split it into windows."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    H, W = ink.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return [tuple(box)]
    horizontal = (x1 - x0) >= (y1 - y0)
    a = ink[y0:y1, x0:x1] if horizontal else ink[y0:y1, x0:x1].T  # rows = depth, columns = along the window
    depth, length = a.shape
    if length < MIN_ASPECT * depth:
        return [tuple(box)]
    # the window's long lines are rows with ink almost everywhere; the strips between them are (nearly) empty rows.
    # A mullion bridges those strips: score each column by the share of gap rows it inks (+-1 column, lines slant)
    row_cov = a.mean(axis=1)
    gap_rows = row_cov < GAP_ROW_MAX
    if gap_rows.sum() < 2 or (row_cov >= LINE_ROW_MIN).sum() < 2:
        return [tuple(box)]  # not drawn as parallel lines: leave it alone
    g = a[gap_rows]
    near = g | np.roll(g, 1, axis=1) | np.roll(g, -1, axis=1)
    cover = near.mean(axis=0)
    min_piece = max(3, MIN_PIECE * depth)
    # crossing lines = runs of columns with cover >= COVER
    runs: list[tuple[int, int]] = []
    k = 0
    while k < length:
        if cover[k] >= COVER:
            j = k
            while j + 1 < length and cover[j + 1] >= COVER:
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1
    # two windows side by side meet with two jambs: two crossing lines close together, with a gap between them.
    # A single line is usually a sash division inside one window (the ground truth keeps those as one window).
    candidates: list[float] = []
    if REQUIRE_DOUBLE:
        for (a0, a1), (b0, b1) in zip(runs, runs[1:]):
            if 2 <= b0 - a1 <= DOUBLE_MAX * depth:
                candidates.append((a1 + b0) / 2)
    else:
        candidates = [(a0 + a1) / 2 for a0, a1 in runs]
    cuts: list[float] = []
    for c in candidates:
        if min_piece <= c <= length - min_piece and (not cuts or c - cuts[-1] >= min_piece):
            cuts.append(c)
    if not cuts:
        return [tuple(box)]
    edges = [0.0] + cuts + [float(length)]
    out = []
    for s, e in zip(edges, edges[1:]):
        if horizontal:
            out.append((x0 + s, float(y0), x0 + e, float(y1)))
        else:
            out.append((float(x0), y0 + s, float(x1), y0 + e))
    return out


def split_windows(items: list[PlanItem], img: Image.Image) -> list[PlanItem]:
    """Window items with merged runs split; other items unchanged. Split pieces keep the score and get ids
    "<id>.<k>" (raw "detector+split") so a citation still points back to the detection."""
    ink = None
    out: list[PlanItem] = []
    for it in items:
        if it.category != "window":
            out.append(it)
            continue
        if ink is None:
            ink = _ink(img)
        pieces = split_box(ink, it.bbox)
        if len(pieces) == 1:
            out.append(it)
            continue
        for k, b in enumerate(pieces):
            out.append(it.model_copy(update={"id": f"{it.id}.{k}", "bbox": tuple(round(v, 1) for v in b),
                                             "raw": f"{it.raw or 'detector'}+split"}))
    return out


def main() -> None:
    from app.cubicasa.graph import get_graph
    from app.cubicasa.symdet import image_path

    rel = sys.argv[1]
    g = get_graph(rel, "model")
    img = Image.open(image_path(rel))
    before = [it for it in g.items if it.category == "window"]
    after = [it for it in split_windows(g.items, img) if it.category == "window"]
    print(f"windows: {len(before)} boxes -> {len(after)} after splitting (counting model: {g.counts.get('windows')})")
    for it in after:
        if "+split" in (it.raw or ""):
            print(f"  {it.id} {[round(v) for v in it.bbox]}")


if __name__ == "__main__":
    main()
