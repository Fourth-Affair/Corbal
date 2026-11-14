"""CubiCasa5K multi-task network (Kalervo et al., SCIA 2019; weights trained on the original train split).

One forward pass gives: 21 junction heatmaps, 12-class room/wall segmentation, 11-class icon segmentation.
`get_polygons` (vendored post-processing) turns them into wall polygons, door/window openings, icon
polygons and room polygons. Vendored code + weights: CC BY-NC 4.0.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import DATASET_DIR, MODELS_DIR

_VENDOR = Path(__file__).parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from floortrans.models.hg_furukawa_original import hg_furukawa_original  # noqa: E402
from floortrans.post_prosessing import get_polygons, split_prediction  # noqa: E402

# 208 MB, too big for git: fetched by scripts/fetch_models.py. The old share-bundle location still works.
_LEGACY = DATASET_DIR.parent / "cubicasa-pretrained" / "model_best_val_loss_var.pkl"
WEIGHTS = Path(os.environ.get("CUBINET_WEIGHTS") or (
    MODELS_DIR / "cubicasa-net.pkl" if (MODELS_DIR / "cubicasa-net.pkl").is_file() or not _LEGACY.is_file() else _LEGACY))
ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room", "Bath", "Entry",
                "Railing", "Storage", "Garage", "Undefined"]
ICON_CLASSES = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet", "Sink", "Sauna Bench",
                "Fire Place", "Bathtub", "Chimney"]
SPLIT = [21, 12, 11]


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    model = hg_furukawa_original(n_classes=51)
    model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
    # weights_only=True: never unpickle arbitrary objects from a downloaded file
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict(img: Image.Image, max_side: int = 0, tta: bool = False) -> torch.Tensor:
    """Return the 44-channel prediction at the input image resolution (1 x 44 x H x W)."""
    img = img.convert("RGB")
    W, H = img.size
    work = img
    if max_side and max(W, H) > max_side:
        s = max_side / max(W, H)
        work = img.resize((round(W * s), round(H * s)), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(work, dtype=np.float32)).permute(2, 0, 1)[None]
    x = 2 * (x / 255.0) - 1
    h, w = x.shape[2:]
    # network downsamples by 4 twice internally; pad to a multiple of 16
    ph, pw = (-h) % 16, (-w) % 16
    x = F.pad(x, (0, pw, 0, ph), value=1.0)
    model = load_model()
    with torch.inference_mode():
        pred = model(x)
        if tta:  # average room/icon logits over 90° rotations (heatmaps stay from the unrotated pass)
            acc = pred[:, 21:].clone()
            for k in (1, 2, 3):
                pr = model(torch.rot90(x, k, dims=(2, 3)))
                acc += torch.rot90(pr[:, 21:], -k, dims=(2, 3))
            pred[:, 21:] = acc / 4
        pred = pred[:, :, :h, :w]
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)
    return pred


def parse(pred: torch.Tensor, threshold: float = 0.2):
    """Run the original post-processing. Returns (polygons, types, room_polygons, room_types, rooms_seg, icons_seg)."""
    H, W = pred.shape[2:]
    heatmaps, rooms, icons = split_prediction(pred, (H, W), SPLIT)
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), threshold, [1, 2])
    return polygons, types, room_polygons, room_types, np.argmax(rooms, 0).astype(np.uint8), np.argmax(icons, 0).astype(np.uint8)
