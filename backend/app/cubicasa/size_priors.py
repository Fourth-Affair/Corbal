"""Rough plan scale from symbols of known real-world size, and plausibility checks on room sizes.

Not a measurement: a door leaf is 0.7-1.0 m wide, a toilet 0.5-0.8 m long, so the detected symbol boxes bound the
plan's metres-per-pixel to a band. The band gives rough room sizes when no better scale exists (user calibration,
areas written on the drawing) and catches implausible results: a "living room" whose largest possible area is
4 m², a space smaller than a door swing, or a derived scale that disagrees with the symbols by more than 2x.
All defaults are hand-set and conservative (wide ranges): a flag means "check this", never a correction.
"""

import math
import statistics

from app.models import PlanGeometry, PlanGraph, PlanScale, RoomEvidence

# Longest side of the symbol's box in metres (min, max). Doors: the box spans the leaf + swing, so its long side is
# the leaf width. Left out on purpose: windows (widths vary 0.5-3 m), showers (CubiCasa draws the head / drain, not
# the cabin), sauna benches and other furniture.
SYMBOL_SIZES_M = {
    ("door", ""): (0.7, 1.0),
    ("fixture", "toilet"): (0.5, 0.8),
    ("fixture", "sink"): (0.4, 0.7),
    ("fixture", "double_sink"): (0.8, 1.4),
    ("fixture", "bathtub"): (1.4, 1.9),
    ("fixture", "sauna_stove"): (0.35, 0.65),
}
MIN_SYMBOL_CONF = 0.5
MIN_SYMBOLS = 3  # fewer than this: no estimate

# Plausible floor area per room type, m² (min, max). Deliberately wide: only clear outliers are flagged.
ROOM_AREA_M2 = {
    "wc": (0.7, 6), "bathroom": (1.8, 25), "sauna": (1.2, 15), "kitchen": (2, 45), "living": (8, 90),
    "dining": (4, 45), "bedroom": (4.5, 40), "entry": (1, 30), "closet": (0.3, 8), "storage": (0.4, 30),
    "utility": (1.5, 20), "study": (4, 30), "technical": (0.5, 25), "balcony": (1, 40), "terrace": (2, 100),
    "garage": (10, 90), "stairs": (1.5, 25), "basement": (5, 200),
}
MIN_SPACE_M2 = 0.5  # smaller than a door swing: a shaft or a detection artefact, not a room
SCALE_DISAGREE = 2.0  # a calibration / written-area scale this far outside the symbol band gets a warning
POINT_SCALE_TOL = {"calibration": 0.03, "dimensions": 0.03, "written_areas": 0.10}  # relative uncertainty of the other scale sources


def symbol_scale(graph: PlanGraph | None) -> PlanScale | None:
    """Metres-per-pixel band from detected symbols of known size; None with fewer than MIN_SYMBOLS of them.

    Each symbol bounds the scale to [min_m / px, max_m / px]; the band is the median of the lower and of the upper
    bounds (robust to a few odd boxes), the estimate their geometric mean."""
    lows, highs, ids = [], [], []
    for it in (graph.items if graph else []):
        size = SYMBOL_SIZES_M.get((it.category, it.cls))
        if size is None or (it.score is not None and it.score < MIN_SYMBOL_CONF):
            continue
        px = max(it.bbox[2] - it.bbox[0], it.bbox[3] - it.bbox[1])
        if px < 4:
            continue
        lows.append(size[0] / px)
        highs.append(size[1] / px)
        ids.append(it.id)
    if len(ids) < MIN_SYMBOLS:
        return None
    low, high = statistics.median(lows), statistics.median(highs)
    if low >= high:
        return None
    return PlanScale(m_per_px=math.sqrt(low * high), source="symbols", low=low, high=high, items=ids)


def _area_range(area_px: float, scale: PlanScale) -> tuple[float, float]:
    if scale.source == "symbols" and scale.low and scale.high:
        return area_px * scale.low ** 2, area_px * scale.high ** 2
    tol = POINT_SCALE_TOL.get(scale.source, 0.1)
    return area_px * (scale.m_per_px * (1 - tol)) ** 2, area_px * (scale.m_per_px * (1 + tol)) ** 2


def check_sizes(geometry: PlanGeometry) -> None:
    """Rough area ranges and size plausibility, in place. Uses geometry.scale (calibration / written areas) when set,
    else the symbol band. An implausible size never changes a room's type: it adds negative geometry evidence, a
    size_check note, and a strong room drops to weak."""
    sym = geometry.symbol_scale
    scale = geometry.scale or sym
    if scale is None:
        return
    if geometry.scale and sym and sym.low and sym.high:
        m = geometry.scale.m_per_px
        if m < sym.low / SCALE_DISAGREE or m > sym.high * SCALE_DISAGREE:
            geometry.scale.warning = (f"disagrees with door/fixture sizes (they suggest {sym.low * 100:.1f}-"
                                      f"{sym.high * 100:.1f} cm per pixel, this scale is {m * 100:.1f}): check it")
    for room in geometry.rooms:
        lo, hi = _area_range(room.area_px, scale)
        if geometry.scale is None:
            room.area_m2_range = (round(lo, 1), round(hi, 1))
        note = None
        if hi < MIN_SPACE_M2:
            note = f"at most {hi:.1f} m²: smaller than a door swing, likely a shaft or detection artefact"
        elif room.tier != "unknown" and room.type in ROOM_AREA_M2:
            mn, mx = ROOM_AREA_M2[room.type]
            if hi < mn:
                note = f"at most {hi:.1f} m², implausibly small for a {room.type} (usually >= {mn:g} m²)"
            elif lo > mx:
                note = f"at least {lo:.0f} m², implausibly large for a {room.type} (usually <= {mx:g} m²)"
        room.size_check = note
        if note and room.tier != "unknown":
            room.evidence.append(RoomEvidence(source="geometry", type=room.type, weight=-2.0, detail=note))
            if room.tier == "strong":
                room.tier = "weak"
