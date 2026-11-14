"""Dedicated wall segmenter: U-Net (pretrained encoder) -> background / solid wall / opening, at native resolution.

  python -m app.cubicasa.wallseg prepare --holdout ../dataset/junction-dataset-preview/questions.jsonl   # CPU, masks
  python -m app.cubicasa.wallseg train --device cuda --epochs 40          # GPU (CUDA or ROCm)
  python -m app.cubicasa.walls --split test --limit 60 --model wallseg    # compare with the CubiCasa net

Why: the 2019 CubiCasa net sees plans downscaled to 1536 px, shares capacity across 44 outputs and paints door/window
openings as wall. This model only does walls, keeps native resolution (512 px crops, sliding-window inference), and
separates solid wall from openings, so `wall | opening` reproduces the old definition and `wall` alone is solid wall.

Labels come from the CubiCasa SVG wall polygons (`walls.gt_walls`). Floors whose walls the SVG does not annotate are
set to IGNORE, so missing annotations do not teach the model to skip real walls. Splits: official CubiCasa train for
training, val for model selection; test and any --holdout images are never used.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import CACHE_DIR, CUBICASA_ORIG_DIR, MODELS_DIR

MASK_DIR = CACHE_DIR / "wallmasks"
WEIGHTS = Path(os.environ.get("WALLSEG_WEIGHTS", MODELS_DIR / "wallseg.pt"))
BG, WALL, OPENING, IGNORE = 0, 1, 2, 255
CLASSES = ["background", "wall", "opening"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE, OVERLAP = 512, 128


def image_path(stem: str) -> Path:
    variant, _, num = stem.rpartition("-")
    return CUBICASA_ORIG_DIR / variant / num / "F1_original.png"


# ---------------------------------------------------------------- labels (CPU)

def make_mask(rel: str) -> np.ndarray:
    from app.cubicasa.walls import annotated_region, gt_walls, rasterize

    walls, openings, size = gt_walls(rel)
    wall = rasterize(walls, size)
    opening = rasterize(openings, size) & wall
    m = np.full(wall.shape, BG, dtype=np.uint8)
    m[wall] = WALL
    m[opening] = OPENING
    m[~annotated_region(rel, size, walls)] = IGNORE
    return m


def _prepare_one(args):
    stem, split = args
    out = MASK_DIR / f"{stem}.png"
    if not out.exists():
        try:
            Image.fromarray(make_mask(f"data/cubicasa5k/{stem}.png")).save(out, optimize=True)
        except Exception as e:  # noqa: BLE001
            return stem, split, f"{type(e).__name__}: {e}"
    return stem, split, None


def prepare(holdout_files: list[Path], workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from app.eval import _orig_split

    split = _orig_split()
    if not split:
        raise SystemExit(f"official split files not found under {CUBICASA_ORIG_DIR}")
    held = set()
    for f in holdout_files:
        held |= {Path(json.loads(line)["path"]).stem for line in f.read_text(encoding="utf-8").splitlines()}
    from app.config import DATASET_DIR

    # label geometry is read against the challenge PNG (same pixels as F1_original.png)
    todo = [(s, sp) for s, sp in sorted(split.items()) if sp in ("train", "val") and s not in held
            and image_path(s).is_file() and (DATASET_DIR / "data" / "cubicasa5k" / f"{s}.png").is_file()]
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    index, errors = {"train": [], "val": []}, 0
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, (stem, sp, err) in enumerate(ex.map(_prepare_one, todo, chunksize=8), 1):
            if err:
                errors += 1
                print("ERR", stem, err, flush=True)
            else:
                index[sp].append(stem)
            if i % 500 == 0:
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    (MASK_DIR / "index.json").write_text(json.dumps(index))
    print(f"masks: train={len(index['train'])} val={len(index['val'])} errors={errors} (holdout excluded: {len(held)})")


# ---------------------------------------------------------------- model

def build_model(encoder: str = "resnet34", pretrained: bool = True):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, classes=len(CLASSES))


def to_tensor(img: np.ndarray):
    import torch

    x = (img.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


# ---------------------------------------------------------------- training data

class CropDataset:
    """Random 512 px crops at a random scale, with flips / 90° rotations / photometric noise (scans vs. clean CAD)."""

    def __init__(self, stems: list[str], crops_per_image: int = 2, train: bool = True, seed: int = 0):
        self.stems, self.k, self.train = stems, crops_per_image, train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.stems) * self.k

    def _load(self, stem):
        img = Image.open(image_path(stem)).convert("RGB")
        mask = Image.open(MASK_DIR / f"{stem}.png")
        return img, mask

    def __getitem__(self, i):
        import torch

        rng = random.Random(self.rng.random() + i) if self.train else random.Random(i)
        img, mask = self._load(self.stems[i // self.k])
        if self.train and rng.random() < 0.8:  # resolution varies a lot between plans
            s = math.exp(rng.uniform(math.log(0.6), math.log(1.4)))
            size = (max(64, round(img.width * s)), max(64, round(img.height * s)))
            img, mask = img.resize(size, Image.BILINEAR), mask.resize(size, Image.NEAREST)
        a, m = np.asarray(img), np.asarray(mask)
        H, W = m.shape
        # pad small plans to a full tile: white paper, ignored label
        if H < TILE or W < TILE:
            ph, pw = max(0, TILE - H), max(0, TILE - W)
            a = np.pad(a, ((0, ph), (0, pw), (0, 0)), constant_values=255)
            m = np.pad(m, ((0, ph), (0, pw)), constant_values=IGNORE)
            H, W = m.shape
        # prefer crops that contain wall (most of a plan is paper)
        for _ in range(8):
            y, x = rng.randint(0, H - TILE), rng.randint(0, W - TILE)
            mc = m[y:y + TILE, x:x + TILE]
            if not self.train or ((mc == WALL) | (mc == OPENING)).mean() > 0.01:
                break
        ac = a[y:y + TILE, x:x + TILE].copy()
        if self.train:
            k = rng.randint(0, 3)
            ac, mc = np.rot90(ac, k), np.rot90(mc, k)
            if rng.random() < 0.5:
                ac, mc = ac[:, ::-1], mc[:, ::-1]
            ac = ac.astype(np.float32)
            if rng.random() < 0.3:  # greyscale (scans) and paper tint
                ac = np.repeat(ac.mean(2, keepdims=True), 3, 2)
            ac = ac * rng.uniform(0.8, 1.15) + rng.uniform(-20, 20)
            if rng.random() < 0.3:
                ac = ac + np.random.default_rng(i).normal(0, rng.uniform(2, 10), ac.shape)
            ac = ac.clip(0, 255).astype(np.uint8)
        return to_tensor(np.ascontiguousarray(ac)), torch.from_numpy(np.ascontiguousarray(mc).astype(np.int64))


# ---------------------------------------------------------------- inference

def predict(img: Image.Image, model=None, device: str = "cpu", batch: int = 8, scale: float = 1.0) -> np.ndarray:
    """Class probabilities (3, H, W) float32 at the image's native resolution (sliding window, overlap blending)."""
    import torch
    import torch.nn.functional as F

    model = model or load(device)
    img = img.convert("RGB")
    W0, H0 = img.size
    if scale != 1.0:
        img = img.resize((max(1, round(W0 * scale)), max(1, round(H0 * scale))), Image.BILINEAR)
    a = np.asarray(img)
    H, W = a.shape[:2]
    ph, pw = max(0, TILE - H), max(0, TILE - W)
    a = np.pad(a, ((0, ph), (0, pw), (0, 0)), constant_values=255)
    Hp, Wp = a.shape[:2]
    step = TILE - OVERLAP
    ys = sorted({min(y, Hp - TILE) for y in range(0, Hp, step)})
    xs = sorted({min(x, Wp - TILE) for x in range(0, Wp, step)})
    # cosine window: tile centres count more than their edges
    w1 = np.sin(np.linspace(0, np.pi, TILE, dtype=np.float32)) ** 2 + 1e-3
    win = torch.from_numpy(np.outer(w1, w1))
    acc = torch.zeros((len(CLASSES), Hp, Wp))
    wsum = torch.zeros((Hp, Wp))
    coords = [(y, x) for y in ys for x in xs]
    with torch.inference_mode():
        for i in range(0, len(coords), batch):
            cs = coords[i:i + batch]
            xb = torch.stack([to_tensor(a[y:y + TILE, x:x + TILE]) for y, x in cs]).to(device)
            pb = F.softmax(model(xb).float(), 1).cpu()
            for (y, x), p in zip(cs, pb):
                acc[:, y:y + TILE, x:x + TILE] += p * win
                wsum[y:y + TILE, x:x + TILE] += win
    prob = (acc / wsum)[:, :H, :W]
    if scale != 1.0:
        prob = F.interpolate(prob[None], size=(H0, W0), mode="bilinear", align_corners=False)[0]
    return prob.numpy()


_MODEL = {}


def available() -> bool:
    return WEIGHTS.is_file()


def load(device: str = "cpu"):
    import torch

    if device not in _MODEL:
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"wall segmenter weights not found at {WEIGHTS} (train: app.cubicasa.wallseg train)")
        ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        m = build_model(ck.get("encoder", "resnet34"), pretrained=False)
        m.load_state_dict(ck["model"])
        _MODEL[device] = m.eval().to(device)
    return _MODEL[device]


def predict_walls(img: Image.Image, device: str = "cpu") -> dict:
    """Same shape as walls.predict_walls: {"mask": wall incl. openings, "solid": solid wall, "prob": P(wall|opening)}."""
    p = predict(img, device=device)
    cls = p.argmax(0)
    return {"mask": cls != BG, "solid": cls == WALL, "opening": cls == OPENING, "prob": p[WALL] + p[OPENING],
            "polygons": []}


# ---------------------------------------------------------------- training

def _dice_loss(logits, target, ignore=IGNORE):
    import torch.nn.functional as F

    valid = (target != ignore)
    p = F.softmax(logits, 1)
    t = F.one_hot(target.clamp(0, len(CLASSES) - 1), len(CLASSES)).permute(0, 3, 1, 2).float()
    v = valid[:, None].float()
    dims = (0, 2, 3)
    inter = (p * t * v).sum(dims)
    den = (p * v).sum(dims) + (t * v).sum(dims)
    dice = (2 * inter + 1) / (den + 1)
    return 1 - dice[1:].mean()  # wall + opening


@np.errstate(all="ignore")
def _iou(conf: np.ndarray) -> np.ndarray:
    return np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf))


def train(args) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    idx = json.loads((MASK_DIR / "index.json").read_text(encoding="utf-8"))
    tr, va = idx["train"], idx["val"]
    if args.limit:
        tr, va = tr[: args.limit], va[: max(8, args.limit // 8)]
    dev = torch.device(args.device)
    model = build_model(args.encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dl_tr = DataLoader(CropDataset(tr, args.crops, True, seed=0), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(CropDataset(va, 2, False), batch_size=args.batch, num_workers=args.workers)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    cw = torch.tensor([0.5, 1.0, 1.5], device=dev)  # paper is easy; openings are rare
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    log = WEIGHTS.with_suffix(".log.csv")
    log.write_text("epoch,seconds,train_loss,val_iou_wall,val_iou_opening,val_iou_wall_or_opening\n")
    best = -1.0
    print(f"train crops/epoch={len(dl_tr) * args.batch} val crops={len(dl_va.dataset)} device={dev}", flush=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(x)
            out = out.float()
            loss = F.cross_entropy(out, y, weight=cw, ignore_index=IGNORE) + _dice_loss(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        # validation: confusion over val crops (fixed crops, no augmentation)
        model.eval()
        conf = np.zeros((3, 3), dtype=np.int64)
        wo = np.zeros(3, dtype=np.int64)  # inter, pred, true for wall|opening
        with torch.inference_mode():
            for x, y in dl_va:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                    pr = model(x.to(dev)).float().argmax(1).cpu().numpy()
                t = y.numpy()
                v = t != IGNORE
                conf += np.bincount(t[v] * 3 + pr[v], minlength=9).reshape(3, 3)
                pw, tw = (pr != BG) & v, (t != BG) & v
                wo += [(pw & tw).sum(), pw.sum(), tw.sum()]
        iou = _iou(conf)
        iou_wo = wo[0] / max(wo[1] + wo[2] - wo[0], 1)
        dt = time.time() - t0
        line = f"{ep},{dt:.0f},{tot / max(n, 1):.4f},{iou[WALL]:.4f},{iou[OPENING]:.4f},{iou_wo:.4f}"
        with log.open("a") as f:
            f.write(line + "\n")
        print(f"epoch {ep}/{args.epochs} {dt:.0f}s loss {tot / max(n, 1):.4f}  val IoU wall {iou[WALL]:.3f} "
              f"opening {iou[OPENING]:.3f} wall|opening {iou_wo:.3f}", flush=True)
        score = iou[WALL] + iou[OPENING]
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "encoder": args.encoder, "epoch": ep, "val_iou": iou.tolist(),
                        "val_iou_wall_or_opening": float(iou_wo)}, WEIGHTS)
    print(f"best val (wall+opening IoU) {best:.4f}; saved {WEIGHTS}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--holdout", type=Path, nargs="*", default=[])
    p.add_argument("--workers", type=int, default=12)
    p = sub.add_parser("train")
    p.add_argument("--device", default="cuda")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--crops", type=int, default=2, help="random crops per training image per epoch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="use only N training images (smoke test)")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.holdout, args.workers)
    else:
        train(args)


if __name__ == "__main__":
    main()
