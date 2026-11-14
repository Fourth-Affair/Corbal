"""SVG ground truth -> Ultralytics YOLO detection dataset for the symbol detector (symdet.py).

  python -m app.cubicasa.export_yolo --holdout ../dataset/junction-dataset-preview/questions.jsonl
  python -m app.cubicasa.export_yolo --rot90 --tile 1024 --out ../dataset/yolo-det-t1024   # native-res tiles

Uses the original CubiCasa5K train/val split (refuses to run without it); test images and any --holdout
images are never exported. Images come from the challenge PNGs or, if absent, the original release.
Labels are boxes (class cx cy w h) for the classes in symdet.DET_NAMES, in that order.

--rot90 adds a 90-degree rotated copy of every *train* image: with the flips Ultralytics applies, that covers all
8 axis-aligned orientations (its `degrees` augmentation rotates by arbitrary angles and blurs line art).
--tile N cuts images into N x N native-resolution tiles (overlap symdet.TILE_OVERLAP), matching tiled inference in
symdet.detect. Rotated copies and tiles are written as files; plain exports symlink the source images.
"""

import argparse
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from app.config import CACHE_DIR, CUBICASA_ORIG_DIR, DATASET_DIR
from app.cubicasa.graph import get_graph
from app.cubicasa.symdet import DET_ITEMS, DET_NAMES, TILE_OVERLAP, image_path, tile_starts

MIN_SIDE = 16  # px; windows/doors-in-wall can be only a few pixels thick
OUT_DIR = DATASET_DIR.parent / "yolo-cubicasa-det"
KEEP_EMPTY_TILES = 0.25  # fraction of symbol-free tiles kept as negatives (deterministic per tile)
MIN_VISIBLE = 0.5  # a box cut by a tile border is kept if this fraction of it lies inside the tile

DET_CLASSES = [DET_ITEMS[n] for n in DET_NAMES]  # same order = class ids the runtime detector reports
Box = tuple[int, float, float, float, float]  # class id, x0, y0, x1, y1 in pixels


def _boxes(graph) -> list[Box]:
    """Ground-truth symbol boxes of the classes in DET_CLASSES, clipped to the image, at least MIN_SIDE wide."""
    w, h = graph.width, graph.height
    idx = {k: i for i, k in enumerate(DET_CLASSES)}
    out = []
    for it in graph.items:
        k = idx.get((it.category, it.cls))
        if k is None:
            continue
        x0, y0, x1, y1 = it.bbox
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = max(x1 - x0, MIN_SIDE), max(y1 - y0, MIN_SIDE)
        out.append((k, max(0, cx - bw / 2), max(0, cy - bh / 2), min(w, cx + bw / 2), min(h, cy + bh / 2)))
    return out


def _lines(boxes: list[Box], w: float, h: float) -> list[str]:
    """YOLO format: class cx cy w h, normalised."""
    return [f"{k} {(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} {min((x1 - x0) / w, 1):.5f} {min((y1 - y0) / h, 1):.5f}"
            for k, x0, y0, x1, y1 in boxes]


def _det_lines(graph) -> list[str]:
    """Detection format (class cx cy w h) for the symbol classes only."""
    return _lines(_boxes(graph), graph.width, graph.height)


def _rot90(boxes: list[Box], w: float) -> list[Box]:
    """Boxes after PIL `Image.ROTATE_90` (counter-clockwise) of a w-wide image: (x, y) -> (y, w - x)."""
    return [(k, y0, w - x1, y1, w - x0) for k, x0, y0, x1, y1 in boxes]


def _tile_boxes(boxes: list[Box], tx: int, ty: int, tw: int, th: int) -> list[Box]:
    """Boxes visible in the tile at (tx, ty), clipped and shifted to tile coordinates."""
    out = []
    for k, x0, y0, x1, y1 in boxes:
        cx0, cy0, cx1, cy1 = max(x0, tx), max(y0, ty), min(x1, tx + tw), min(y1, ty + th)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < MIN_VISIBLE * (x1 - x0) * (y1 - y0):
            continue
        out.append((k, cx0 - tx, cy0 - ty, cx1 - tx, cy1 - ty))
    return out


def _write(img: Image.Image, boxes: list[Box], stem: str, img_dir: Path, lbl_dir: Path, tile: int) -> int:
    """Write one image (or its tiles) and the labels. Returns the number of image files written."""
    if not tile:
        img.save(img_dir / f"{stem}.png")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(_lines(boxes, img.width, img.height)) + "\n")
        return 1
    n = 0
    for ty in tile_starts(img.height, tile):
        for tx in tile_starts(img.width, tile):
            tw, th = min(tile, img.width - tx), min(tile, img.height - ty)
            tb = _tile_boxes(boxes, tx, ty, tw, th)
            name = f"{stem}_{tx}_{ty}"
            if not tb and zlib.crc32(name.encode()) % 1000 >= KEEP_EMPTY_TILES * 1000:
                continue
            img.crop((tx, ty, tx + tw, ty + th)).save(img_dir / f"{name}.png")
            (lbl_dir / f"{name}.txt").write_text("\n".join(_lines(tb, tw, th)) + "\n")
            n += 1
    return n


def _export_one(job: tuple) -> int:
    rel, src, img_dir, lbl_dir, tile, rot90 = job
    g = get_graph(rel, "svg")
    boxes = _boxes(g)
    stem = Path(rel).stem
    n = 0
    if not tile:  # whole images: link the source image instead of copying it
        link = img_dir / f"{stem}.png"
        if not link.exists():
            os.symlink(src.resolve(), link)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(_lines(boxes, g.width, g.height)) + "\n")
        n = 1
        if not rot90:
            return n
    img = Image.open(src).convert("RGB")
    if img.size != (g.width, g.height):  # labels are in challenge-PNG pixels (= the original image)
        raise ValueError(f"{rel}: image {img.size} != labels {(g.width, g.height)}")
    if tile:
        n += _write(img, boxes, stem, img_dir, lbl_dir, tile)
    if rot90:
        n += _write(img.transpose(Image.ROTATE_90), _rot90(boxes, img.width), f"{stem}_r90", img_dir, lbl_dir, tile)
    return n


def _holdout(files: list[Path]) -> set[str]:
    """Image paths listed in question files (e.g. the challenge preview = our test data): never trained on."""
    out = set()
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
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
    ap.add_argument("--tile", type=int, default=0, help="native-resolution tile side (0 = whole images)")
    ap.add_argument("--rot90", action="store_true", help="add a 90-degree rotated copy of each train image")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
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
        jobs = [(rel, image_path(rel), img_dir, lbl_dir, args.tile, args.rot90 and split == "train") for rel in sel]
        jobs = [j for j in jobs if j[1].is_file()]
        with ProcessPoolExecutor(args.workers) as ex:
            files = sum(ex.map(_export_one, jobs, chunksize=4))
        print(f"{split}: {len(jobs)} images -> {files} files ({len(sel) - len(jobs)} without an image file, "
              f"{len(in_split) - len(sel)} held out)", flush=True)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(DET_NAMES))
    (args.out / "data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
