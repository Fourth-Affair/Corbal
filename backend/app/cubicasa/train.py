"""Train the cubicasa segmentation/detection model and copy the best weights to CUBICASA_WEIGHTS.

  python -m app.cubicasa.train --epochs 80 --imgsz 1280 --device 0        # GPU
  python -m app.cubicasa.train --epochs 1 --imgsz 640 --fraction 0.05     # CPU smoke test
"""

import argparse
import shutil
from pathlib import Path

from app.config import DATASET_DIR, MODEL_WEIGHTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATASET_DIR.parent / "yolo-cubicasa" / "data.yaml")
    ap.add_argument("--base", default="yolo11s-seg.pt", help="pretrained checkpoint to fine-tune")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train images to use")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        fraction=args.fraction, workers=args.workers, project=str(DATASET_DIR.parent / "runs"), name="cubicasa",
        exist_ok=True,
        # Plans are axis-aligned line drawings: rotations by 90° and flips are valid, colour jitter mostly isn't.
        fliplr=0.5, flipud=0.5, degrees=0.0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.2, mosaic=0.5,
    )
    best = Path(model.trainer.best)
    MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, MODEL_WEIGHTS)
    print(f"copied {best} -> {MODEL_WEIGHTS}")


if __name__ == "__main__":
    main()
