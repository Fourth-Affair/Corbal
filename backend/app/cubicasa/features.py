"""Cache per-image perception outputs: CubiCasa network segmentation maps + OCR tokens.

  python -m app.cubicasa.features --splits val test train --train-limit 1500

Writes CACHE_DIR/features/<stem>.npz (rooms/icons argmax + icon confidence, uint8) and <stem>.json (OCR).
Resumable: existing files are skipped.
"""

import argparse
import json
import random
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from app.config import CACHE_DIR, DATASET_DIR
from app.profiling import stage

FEAT_DIR = CACHE_DIR / "features"
MAX_SIDE = 1536


ICON_PROB_CLASSES = (1, 2, 5, 6, 7, 9)  # window, door, toilet, sink, sauna bench, bathtub
ROOM_PROB_CLASSES = (1, 3, 5, 6)  # outdoor, kitchen, bedroom, bath


def _npz_ok(npz: Path) -> bool:
    if not npz.exists():
        return False
    try:
        with np.load(npz) as z:
            return "iprob" in z.files
    except Exception:  # truncated file from an interrupted run
        return False


def feature_paths(rel: str) -> tuple[Path, Path]:
    stem = Path(rel).stem
    return FEAT_DIR / f"{stem}.npz", FEAT_DIR / f"{stem}.json"


def ocr_tokens(rel: str, img: Image.Image | None = None) -> dict:
    """{"width", "height", "ocr": [{"text", "bbox", "score"}]}: OCR only, cached in the same JSON as compute()."""
    from app.cubicasa.ocr import read_text

    _, js = feature_paths(rel)
    if js.exists():
        try:
            return json.loads(js.read_text(encoding="utf-8"))
        except ValueError:  # truncated by an interrupted run: recompute
            pass
    js.parent.mkdir(parents=True, exist_ok=True)
    img = (img or Image.open(DATASET_DIR / rel)).convert("RGB")
    toks = [{"text": t, "bbox": [round(float(v), 1) for v in b], "score": round(float(s), 3)} for t, b, s in read_text(img)]
    out = {"width": img.width, "height": img.height, "ocr": toks}
    tmp = js.with_name(f"{js.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        tmp.replace(js)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def compute(rel: str, img: Image.Image | None = None) -> None:
    from app.cubicasa import cubinet

    npz, js = feature_paths(rel)
    npz.parent.mkdir(parents=True, exist_ok=True)  # not part of the share bundle
    img = img or Image.open(DATASET_DIR / rel)
    img = img.convert("RGB")
    if not _npz_ok(npz):
        with stage("cubinet"):
            pred = cubinet.predict(img, max_side=MAX_SIDE)[0]
            rooms = torch.softmax(pred[21:33], 0)
            icons = torch.softmax(pred[33:44], 0)
        with stage("maps"):  # argmax / quantisation at full resolution + compressed npz
            tmp = npz.with_name(f"{npz.name}.{uuid4().hex}.tmp.npz")
            try:
                np.savez_compressed(
                    tmp,
                    heat=(pred[:21].clamp(0, 1) * 255).to(torch.uint8).numpy()[:, ::2, ::2],  # junction heatmaps, half res
                    rooms=rooms.argmax(0).to(torch.uint8).numpy(),
                    rooms_conf=(rooms.max(0).values * 255).to(torch.uint8).numpy(),
                    icons=icons.argmax(0).to(torch.uint8).numpy(),
                    icons_conf=(icons.max(0).values * 255).to(torch.uint8).numpy(),
                    # per-class probabilities: argmax loses symbols that sit on walls / touch each other
                    iprob=(icons[list(ICON_PROB_CLASSES)] * 255).to(torch.uint8).numpy(),
                    rprob=(rooms[list(ROOM_PROB_CLASSES)] * 255).to(torch.uint8).numpy(),
                )
                tmp.replace(npz)
            finally:
                tmp.unlink(missing_ok=True)
    with stage("ocr"):
        ocr_tokens(rel, img)


def main() -> None:
    from app.eval import load_questions, split_of

    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--train-limit", type=int, default=1500)
    args = ap.parse_args()
    torch.set_num_threads(8)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = sorted({q["path"] for q in load_questions()})
    todo = []
    for sp in args.splits:
        ps = [p for p in all_paths if split_of(p) == sp]
        if sp == "train":
            random.Random(0).shuffle(ps)
            ps = ps[: args.train_limit]
        todo += ps
    t0 = time.time()
    for i, rel in enumerate(todo, 1):
        npz, js = feature_paths(rel)
        if js.exists() and _npz_ok(npz):
            continue
        try:
            compute(rel)
        except Exception as e:
            print("ERR", rel, type(e).__name__, e, flush=True)
        if i % 25 == 0:
            print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("done", len(todo), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
