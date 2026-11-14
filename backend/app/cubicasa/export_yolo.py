"""SVG ground truth -> Ultralytics YOLO detection dataset for the symbol detector (symdet.py).

  python -m app.cubicasa.export_yolo --holdout ../dataset/junction-dataset-preview/questions.jsonl

Uses the original CubiCasa5K train/val split (refuses to run without it); test images and any --holdout
images are never exported. Images come from the challenge PNGs or, if absent, the original release.
Labels are boxes (class cx cy w h) for the classes in symdet.DET_NAMES, in that order.
"""

import argparse
import json
import os
from pathlib import Path

from app.config import CACHE_DIR, CUBICASA_ORIG_DIR, DATASET_DIR
from app.cubicasa.graph import get_graph
from app.cubicasa.symdet import DET_ITEMS, DET_NAMES

MIN_SIDE = 16  # px; windows/doors-in-wall can be only a few pixels thick
OUT_DIR = DATASET_DIR.parent / "yolo-cubicasa-det"


DET_CLASSES = [DET_ITEMS[n] for n in DET_NAMES]  # same order = class ids the runtime detector reports


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


def _image_source(rel: str) -> Path | None:
    """Challenge PNG if present, else the byte-identical original `<variant>/<id>/F1_original.png`."""
    p = DATASET_DIR / rel
    if p.is_file():
        return p
    variant, _, num = Path(rel).stem.rpartition("-")
    p = CUBICASA_ORIG_DIR / variant / num / "F1_original.png"
    return p if p.is_file() else None


def _holdout(files: list[Path]) -> set[str]:
    """Image paths listed in question files (e.g. the challenge preview = our test data): never trained on."""
    out = set()
    for f in files:
        for line in f.read_text().splitlines():
            path = json.loads(line)["path"]
            if path.startswith("data/cubicasa5k/"):
                out.add(path)
    return out


def main() -> None:
    from app.eval import _orig_split, split_of

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=0, help="max images per split (0 = all), for smoke tests")
    ap.add_argument("--holdout", type=Path, nargs="*", default=[],
                    help="questions.jsonl files whose images must stay out of training (e.g. the challenge preview)")
    args = ap.parse_args()

    if not _orig_split():
        # without the official split, split_of() falls back to a hash and could leak CubiCasa test images
        raise SystemExit(f"official CubiCasa split files not found under {CUBICASA_ORIG_DIR} (train.txt/val.txt/test.txt)")
    held = _holdout(args.holdout)
    # every image with ground-truth labels: parsed SVGs in the cache (see build_cache)
    paths = sorted(f"data/cubicasa5k/{f.stem}.png" for f in (CACHE_DIR / "svg").glob("*.json"))
    for split in ("train", "val"):
        in_split = [p for p in paths if split_of(p) == split]
        sel = [p for p in in_split if p not in held]
        if args.limit:
            sel = sel[: args.limit]
        img_dir, lbl_dir = args.out / "images" / split, args.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for rel in sel:
            src = _image_source(rel)
            if src is None:
                continue
            stem = Path(rel).stem
            link = img_dir / f"{stem}.png"
            if not link.exists():
                os.symlink(src.resolve(), link)
            g = get_graph(rel, "svg")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(_det_lines(g)) + "\n")
            n += 1
        print(f"{split}: {n} images ({len(sel) - n} without an image file, {len(in_split) - len(sel)} held out)")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(DET_NAMES))
    (args.out / "data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
