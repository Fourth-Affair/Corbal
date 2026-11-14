"""Turn cached perception outputs (CubiCasa net maps + OCR) into per-image features, and per-template counts.

Features are explicit evidence counts (connected components of each icon / room class at several size
thresholds, OCR room-label hits by dictionary, apartment-type headers, storey blocks), so every predicted
number can be traced back to highlighted regions.
"""

import json
import re
from functools import lru_cache

import numpy as np
from scipy import ndimage

from app.cubicasa.features import feature_paths
from app.cubicasa.rooms_dict import apartment_type, room_type

ICON = {"window": 1, "door": 2, "closet": 3, "appliance": 4, "toilet": 5, "sink": 6, "sauna_bench": 7,
        "fireplace": 8, "bathtub": 9, "chimney": 10}
ROOM = {"background": 0, "outdoor": 1, "wall": 2, "kitchen": 3, "living": 4, "bedroom": 5, "bath": 6,
        "entry": 7, "railing": 8, "storage": 9, "garage": 10, "undefined": 11}
ICON_PROB = (1, 2, 5, 6, 7, 9)  # same order as features.ICON_PROB_CLASSES
OCR_TYPES = ["bedroom", "bathroom", "wc", "sauna", "kitchen", "living", "balcony", "other"]


def load(rel: str):
    npz, js = feature_paths(rel)
    z = np.load(npz)
    meta = json.loads(js.read_text(encoding="utf-8"))
    return z, meta


def components(mask: np.ndarray, conf: np.ndarray | None = None):
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, np.zeros(0), np.zeros(0), []
    # Same numbers as ndimage.sum / ndimage.mean over labels 1..n (which bincount every pixel of the map, four passes
    # per call: ~11 s on a 12 MP plan), counting only the labelled pixels. Each label's sum still accumulates its own
    # pixels in the same order, so the means are bit-identical (the fusion models depend on these features).
    flat = lab.ravel()
    on = flat > 0
    labs = flat[on]
    counts = np.bincount(labs, minlength=n + 1)[1:]
    sizes = counts.astype(np.float64)
    if conf is not None:
        mconf = np.bincount(labs, weights=conf.ravel()[on].astype(np.float32), minlength=n + 1)[1:] / counts / 255
    else:
        mconf = np.ones(n)
    boxes = ndimage.find_objects(lab)
    return lab, sizes, mconf, boxes


def image_features(rel: str) -> dict:
    z, meta = load(rel)
    rooms, icons = z["rooms"], z["icons"]
    rconf, iconf = z["rooms_conf"], z["icons_conf"]
    H, W = icons.shape
    area = float(H * W)
    f: dict[str, float] = {"W": W, "H": H, "log_area": np.log(area), "aspect": W / H}
    # No features from the file name: the dataset names encode the CubiCasa variant ("high_quality-8033.png"), an
    # upload's name does not. The fusion models fitted before 2026-11-15 still list var_* keys; they now always read 0,
    # as they did for uploads (measured on the 241 preview plans: 2 of 2892 counts change, accuracy equal).

    # ---- icons
    for name, k in ICON.items():
        _, sizes, mconf, _ = components(icons == k, iconf)
        f[f"ic_{name}_px"] = float(sizes.sum()) / area * 1e3
        for t in (5, 20, 60, 150, 400):
            f[f"ic_{name}_n{t}"] = float((sizes >= t).sum())
        for r in (1e-5, 5e-5, 2e-4):
            f[f"ic_{name}_r{r:g}"] = float((sizes >= r * area).sum())
        for c in (0.5, 0.6, 0.7, 0.8, 0.9):
            for t in (10, 20, 60):
                f[f"ic_{name}_c{int(c * 10)}_s{t}"] = float(((sizes >= t) & (mconf >= c)).sum())
        f[f"ic_{name}_c7"] = f[f"ic_{name}_c7_s20"]
        if name in ("window", "door", "sink", "toilet"):  # split touching symbols
            m = (icons == k) & (iconf >= 0.6 * 255)
            for it in (1, 2):
                _, es, _, _ = components(ndimage.binary_erosion(m, iterations=it))
                f[f"ic_{name}_ero{it}"] = float((es >= 10).sum())
        if len(sizes):
            big = sizes[sizes >= 20]
            med = float(np.median(big)) if len(big) else 1.0
            f[f"ic_{name}_area_over_med"] = float(big.sum() / med) if len(big) else 0.0
        else:
            f[f"ic_{name}_area_over_med"] = 0.0

    # ---- split merged openings: adjacent windows/doors along a wall fuse into one blob
    for name in ("window", "door"):
        k = ICON[name]
        for src, mask in (("am", (icons == k) & (iconf >= 0.5 * 255)),
                          ("p35", (z["iprob"][list(ICON_PROB).index(k)] >= 0.35 * 255) if "iprob" in z.files else None)):
            if mask is None:
                continue
            _, sizes, _, boxes = components(mask)
            lens = np.array([max(b[0].stop - b[0].start, b[1].stop - b[1].start) for b, sz in zip(boxes, sizes) if sz >= 15])
            for q in (25, 40, 50):
                L = float(np.percentile(lens, q)) if len(lens) else 1.0
                f[f"split_{name}_{src}_q{q}"] = float(np.maximum(1, np.rint(lens / max(L, 1.0))).sum()) if len(lens) else 0.0

    # ---- per-class probability maps (recover symbols that lose the argmax)
    if "iprob" in z.files:
        from app.cubicasa.features import ICON_PROB_CLASSES, ROOM_PROB_CLASSES

        inv_icon = {v: k for k, v in ICON.items()}
        for ci, k in enumerate(ICON_PROB_CLASSES):
            pm = z["iprob"][ci]
            name = inv_icon[k]
            for tau in (0.25, 0.35, 0.5, 0.65):
                _, sizes, _, _ = components(pm >= tau * 255)
                for mn in (10, 20, 40, 80):
                    f[f"ip_{name}_t{int(tau * 100)}_s{mn}"] = float((sizes >= mn).sum())
                f[f"ip_{name}_t{int(tau * 100)}_px"] = float(sizes.sum()) / area * 1e3
        inv_room = {v: k for k, v in ROOM.items()}
        for ci, k in enumerate(ROOM_PROB_CLASSES):
            pm = z["rprob"][ci]
            name = inv_room[k]
            for tau in (0.35, 0.5):
                m = ndimage.binary_opening(pm >= tau * 255, iterations=2)
                _, sizes, _, _ = components(m)
                for r in (5e-4, 2e-3, 8e-3):
                    f[f"rp_{name}_t{int(tau * 100)}_r{r:g}"] = float((sizes >= r * area).sum())

    # ---- junction heatmaps (half res): 0-12 wall junctions, 13-16 opening end points, 17-20 icon corners
    heat = z["heat"].astype(np.float32) / 255
    for ch in range(13, 21):
        h = heat[ch]
        peaks = (h == ndimage.maximum_filter(h, size=5))
        for thr in (0.3, 0.5):
            f[f"hm_{ch}_p{thr}"] = float((peaks & (h >= thr)).sum())
    for thr in (0.3, 0.5):
        f[f"hm_open_p{thr}"] = sum(f[f"hm_{c}_p{thr}"] for c in range(13, 17)) / 2  # ~ openings (2 ends each)
        f[f"hm_icon_p{thr}"] = sum(f[f"hm_{c}_p{thr}"] for c in range(17, 21)) / 4  # ~ icons (4 corners each)

    # ---- rooms (segmentation regions; split by walls, so components ~ rooms)
    for name, k in ROOM.items():
        if name in ("background", "wall", "railing"):
            continue
        m = ndimage.binary_opening(rooms == k, iterations=2)
        _, sizes, mconf, _ = components(m, rconf)
        f[f"rm_{name}_frac"] = float(sizes.sum()) / area
        for r in (5e-4, 2e-3, 8e-3):
            f[f"rm_{name}_r{r:g}"] = float((sizes >= r * area).sum())

    # ---- storeys: big blobs of "building" (any room or wall), merged across small gaps
    fg = rooms != ROOM["background"]
    fg = ndimage.binary_closing(fg, iterations=max(2, int(0.01 * max(H, W))))
    _, sizes, _, boxes = components(fg)
    for r in (0.01, 0.03, 0.06, 0.12):
        f[f"fl_n{r:g}"] = float((sizes >= r * area).sum())
    if len(sizes):
        top = np.sort(sizes)[::-1]
        f["fl_top1"] = top[0] / area
        f["fl_top2_ratio"] = (top[1] / top[0]) if len(top) > 1 else 0.0
        f["fl_top3_ratio"] = (top[2] / top[0]) if len(top) > 2 else 0.0
    else:
        f["fl_top1"] = f["fl_top2_ratio"] = f["fl_top3_ratio"] = 0.0

    # ---- OCR room labels (all / inside the building mask)
    fg_dil = ndimage.binary_dilation(rooms != ROOM["background"], iterations=3)
    counts = {f"ocr_{t}": 0.0 for t in OCR_TYPES}
    counts.update({f"ocr_in_{t}": 0.0 for t in OCR_TYPES})
    f["apt_rooms"] = -1.0
    f["apt_sauna"] = -1.0
    f["ocr_n_tokens"] = float(len(meta["ocr"]))
    from app.cubicasa.token_table import lookup

    for t in OCR_TYPES:
        counts[f"ocr_u_{t}"] = 0.0
        counts[f"ocr_u_in_{t}"] = 0.0
    for tok in meta["ocr"]:
        # union of hand-written dictionary and the table learned on train images
        tu = room_type(tok["text"]) or lookup(tok["text"])
        if tu:
            counts[f"ocr_u_{tu}"] += 1
            x0, y0, x1, y1 = tok["bbox"]
            cx, cy = int(min(W - 1, max(0, (x0 + x1) / 2))), int(min(H - 1, max(0, (y0 + y1) / 2)))
            if fg_dil[cy, cx]:
                counts[f"ocr_u_in_{tu}"] += 1
    for tok in meta["ocr"]:
        t = room_type(tok["text"])
        apt = apartment_type(tok["text"])
        if apt:
            f["apt_rooms"] = float(apt["rooms"])
            f["apt_sauna"] = float(apt["sauna"])
        if not t:
            continue
        counts[f"ocr_{t}"] += 1
        x0, y0, x1, y1 = tok["bbox"]
        cx, cy = int(min(W - 1, max(0, (x0 + x1) / 2))), int(min(H - 1, max(0, (y0 + y1) / 2)))
        if fg_dil[cy, cx]:
            counts[f"ocr_in_{t}"] += 1
    f.update(counts)
    # dimension labels: windows "14x18", "10x5" (height < 20 dm); doors "10x21", "9x23"; door type codes "VO9", "UO10v"
    dims = [re.search(r"(\d{1,2})\s*[xX×*]\s*(\d{1,2})", tok["text"]) for tok in meta["ocr"]]
    f["ocr_win_dims"] = float(sum(1 for m in dims if m and int(m.group(2)) < 20))
    f["ocr_door_dims"] = float(sum(1 for m in dims if m and int(m.group(2)) >= 20))
    f["ocr_door_codes"] = float(sum(1 for tok in meta["ocr"]
                                    if re.match(r"^[VUPKLA]?[O0Q][\s.]?\d{1,2}", tok["text"].upper().strip())))
    # "SAUNA" written anywhere (also inside compounds like "saunaos", "TUPAK.+HUONE+sauna")
    f["ocr_sauna_substr"] = float(sum("SAUNA" in tok["text"].upper() for tok in meta["ocr"]))
    f["ocr_wc_substr"] = float(sum(re.search(r"\bWC\b", tok["text"].upper()) is not None for tok in meta["ocr"]))
    # ---- combined evidence (text and pixels miss different rooms)
    seg = {"bedroom": "rm_bedroom_r0.008", "kitchen": "rm_kitchen_r0.002", "bathroom": "rm_bath_r0.002",
           "balcony": "rm_outdoor_r0.008"}
    for t, sk in seg.items():
        for src in ("ocr_in_", "ocr_", "ocr_u_in_"):
            f[f"max_{src}{t}_seg"] = max(f[f"{src}{t}"], f.get(sk, 0.0))
    f["bed_apt"] = f["apt_rooms"] - 1 if f["apt_rooms"] > 0 else f["max_ocr_in_bedroom_seg"]
    f["sauna_any"] = max(f["ocr_sauna"], f.get("ic_sauna_bench_c7", 0.0) > 0, f["apt_sauna"] > 0)
    f["win_max_ocr_split"] = max(f["ocr_win_dims"], f.get("split_window_am_q40", 0.0))
    f["door_max_ocr_split"] = max(f["ocr_door_codes"], f["ocr_door_dims"], f.get("split_door_am_q40", 0.0))
    return f


@lru_cache(maxsize=8192)
def cached_features(rel: str) -> tuple[tuple[str, float], ...]:
    return tuple(image_features(rel).items())
