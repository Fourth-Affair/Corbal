"""Fine-tune the symbol detector (YOLO11 / RT-DETR via Ultralytics) on the export from export_yolo.

  python -m app.cubicasa.train --device 0 --onnx                          # GPU -> dataset/models/symbol-det.{pt,onnx}
  python -m app.cubicasa.train --base rtdetr-l.pt --epochs 60 --batch 8   # stronger, slower alternative
  python -m app.cubicasa.train --epochs 1 --imgsz 640 --fraction 0.05     # CPU smoke test

For a rented GPU use scripts/verda_train.py, which runs this module on a Verda VM.
"""

import argparse
import json
import shutil
from pathlib import Path

from app.config import DATASET_DIR, MODELS_DIR

DET_WEIGHTS = MODELS_DIR / "symbol-det.pt"
DATA_YAML = DATASET_DIR.parent / "yolo-cubicasa-det" / "data.yaml"

# Plans are axis-aligned line drawings: flips are valid, colour jitter mostly isn't. Symbols are small, so keep
# scale augmentation mild.
AUG = dict(fliplr=0.5, flipud=0.5, degrees=0.0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.2, mosaic=0.5, scale=0.3,
           close_mosaic=15)


def per_class_metrics(model, data: Path, imgsz: int, batch: int, device: str) -> dict:
    """Validation metrics per class (on the val split of `data`, i.e. tiles for a tiled export)."""
    m = model.val(data=str(data), imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False)
    names = m.names
    out = {"map50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4), "classes": {}}
    for i, c in enumerate(m.box.ap_class_index):
        out["classes"][names[int(c)]] = {"ap50": round(float(m.box.ap50[i]), 4), "p": round(float(m.box.p[i]), 4),
                                         "r": round(float(m.box.r[i]), 4)}
    return out


def train(data: Path, base: str, epochs: int, imgsz: int, batch: int, device: str, fraction: float = 1.0,
          workers: int = 4, project: Path = DATASET_DIR.parent / "runs", name: str = "symdet",
          onnx: bool = False, degrees: float = 0.0, scale: float = 0.3) -> tuple[Path, Path | None]:
    """Train, then return (best .pt, ONNX export or None)."""
    from ultralytics import RTDETR, YOLO

    cls = RTDETR if "rtdetr" in base else YOLO
    model = cls(base)
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device, fraction=fraction,
                workers=workers, project=str(project), name=name, exist_ok=True,
                **(AUG | {"degrees": degrees, "scale": scale}))
    best = Path(model.trainer.best)
    onnx_path = Path(cls(str(best)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)) if onnx else None
    return best, onnx_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_YAML)
    ap.add_argument("--base", default="yolo11s.pt", help="pretrained checkpoint to fine-tune (yolo11s.pt, rtdetr-l.pt)")
    ap.add_argument("--out", type=Path, default=DET_WEIGHTS, help="where to copy the best weights")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280, help="must match SYMDET_IMGSZ at inference")
    ap.add_argument("--batch", type=float, default=16, help="images per batch, or a fraction (0-1) of GPU memory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train images to use")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--degrees", type=float, default=0.0, help="random rotation range (+/-), for rotated plans")
    ap.add_argument("--scale", type=float, default=0.3, help="random scale augmentation (+/- fraction)")
    ap.add_argument("--tile", type=int, default=0, help="tile side the --data export was cut into (0 = whole images); "
                    "recorded in the sidecar so symdet runs tiled inference")
    ap.add_argument("--name", default="symdet", help="run name under the project dir")
    ap.add_argument("--onnx", action="store_true", help="also export ONNX next to --out (CPU runtime)")
    args = ap.parse_args()

    batch = int(args.batch) if args.batch >= 1 else args.batch  # Ultralytics: int = images, float < 1 = memory share
    best, onnx_path = train(args.data, args.base, args.epochs, args.imgsz, batch, args.device, args.fraction,
                            args.workers, name=args.name, onnx=args.onnx, degrees=args.degrees, scale=args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    print(f"copied {best} -> {args.out}")
    if onnx_path:
        shutil.copy(onnx_path, args.out.with_suffix(".onnx"))
        print(f"copied {onnx_path} -> {args.out.with_suffix('.onnx')}")
    # sidecar read by symdet.weights_meta(): inference must use the training resolution / tiling
    from ultralytics import RTDETR, YOLO

    metrics = per_class_metrics((RTDETR if "rtdetr" in args.base else YOLO)(str(best)), args.data, args.imgsz,
                                max(1, batch // 2) if isinstance(batch, int) else 8, args.device)
    meta = {"imgsz": args.imgsz, "tile": args.tile, "base": args.base, "epochs": args.epochs, "val": metrics}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("val per class:", json.dumps(metrics))


if __name__ == "__main__":
    main()
