"""SVG-derived PlanGraphs -> Ultralytics YOLO segmentation dataset.

  python -m app.cubicasa.export_yolo --out ../dataset/yolo-cubicasa

Uses the original CubiCasa5K train/val split; test images are never exported.
Rooms are exported as polygons, everything else as box-shaped polygons.
"""

import argparse
import os
from pathlib import Path

from app.config import DATASET_DIR
from app.cubicasa.classes import CLASS_INDEX, YOLO_CLASSES
from app.cubicasa.graph import get_graph


MIN_SIDE = 16  # px; windows/doors-in-wall can be only a few pixels thick


def _label_lines(graph) -> list[str]:
    w, h = graph.width, graph.height
    lines = []
    for it in graph.items:
        k = CLASS_INDEX.get((it.category, it.cls))
        if k is None:
            continue
        x0, y0, x1, y1 = it.bbox
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:  # annotated outside the visible image
            continue
        if it.polygon and len(it.polygon) >= 3:
            pts = it.polygon
        else:
            x0, y0, x1, y1 = it.bbox
            if x1 - x0 < MIN_SIDE:
                cx = (x0 + x1) / 2
                x0, x1 = cx - MIN_SIDE / 2, cx + MIN_SIDE / 2
            if y1 - y0 < MIN_SIDE:
                cy = (y0 + y1) / 2
                y0, y1 = cy - MIN_SIDE / 2, cy + MIN_SIDE / 2
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        coords = " ".join(f"{min(max(x / w, 0), 1):.5f} {min(max(y / h, 0), 1):.5f}" for x, y in pts)
        lines.append(f"{k} {coords}")
    return lines


DET_CLASSES = [("door", ""), ("window", ""), ("fixture", "toilet"), ("fixture", "sink"), ("fixture", "double_sink"),
               ("fixture", "shower"), ("fixture", "bathtub"), ("fixture", "sauna_stove")]
DET_NAMES = ["door", "window", "toilet", "sink", "double_sink", "shower", "bathtub", "sauna_stove"]


def _det_lines(graph) -> list[str]:
    """Detection format (class cx cy w h) for the symbol classes only."""
    w, h = graph.width, graph.height
    idx = {k: i for i, k in enumerate(DET_CLASSES)}
    lines = []
    for it in graph.items:
        k = idx.get((it.category, it.cls))
        if k is None:
            continue
        x0, y0, x1, y1 = it.bbox
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        bw, bh = max(x1 - x0, MIN_SIDE), max(y1 - y0, MIN_SIDE)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        lines.append(f"{k} {cx / w:.5f} {cy / h:.5f} {min(bw / w, 1):.5f} {min(bh / h, 1):.5f}")
    return lines


def main() -> None:
    from app.eval import load_questions, split_of

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATASET_DIR.parent / "yolo-cubicasa")
    ap.add_argument("--limit", type=int, default=0, help="max images per split (0 = all), for smoke tests")
    ap.add_argument("--det", action="store_true", help="detection labels for symbol classes only")
    args = ap.parse_args()

    paths = sorted({q["path"] for q in load_questions()})
    for split in ("train", "val"):
        sel = [p for p in paths if split_of(p) == split]
        if args.limit:
            sel = sel[: args.limit]
        img_dir, lbl_dir = args.out / "images" / split, args.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for rel in sel:
            stem = Path(rel).stem
            link = img_dir / f"{stem}.png"
            if not link.exists():
                os.symlink((DATASET_DIR / rel).resolve(), link)
            g = get_graph(rel, "svg")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(_det_lines(g) if args.det else _label_lines(g)) + "\n")
        print(f"{split}: {len(sel)} images")

    name_list = DET_NAMES if args.det else [n for n, _, _ in YOLO_CLASSES]
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(name_list))
    (args.out / "data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
