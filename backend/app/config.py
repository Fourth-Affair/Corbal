"""Paths and settings, overridable via environment variables."""

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# Challenge dataset root (contains data/ and questions.jsonl).
DATASET_DIR = Path(os.environ.get("DATASET_DIR", _REPO / "dataset" / "junction-dataset"))
# Original CubiCasa5K release (extracted Zenodo zip), used only for training/validation labels.
CUBICASA_ORIG_DIR = Path(os.environ.get("CUBICASA_ORIG_DIR", _REPO / "dataset" / "cubicasa5k-original" / "cubicasa5k"))
# Where parsed PlanGraph JSON files are cached.
CACHE_DIR = Path(os.environ.get("CACHE_DIR", _REPO / "dataset" / "cache"))
# Trained models: counting models (cubicasa-fusion.joblib), OCR token table, symbol detector (symbol-det.onnx),
# wall segmenter (wallseg.pt). Everything in here comes from the share bundle or a training run.
MODELS_DIR = Path(os.environ.get("MODELS_DIR", _REPO / "models"))  # tracked in git; see models/MANIFEST.json

# Walls and OCR run at native resolution, so peak memory grows with the pixel count: measured peak RSS of analysis +
# geometry, models included, 2.3 GB at 2.2 MP and 4.9 GB at 12 MP (~1.7 GB + 0.27 GB/MP; the dataset's largest plan
# is 12 MP). Larger uploads are downscaled before analysis (16 MP ~ 6 GB; use ~10 MP on an 8 GB server), and PDF
# pages are rendered at most this large. The viewer draws a graph in its own width/height over the full-size image
# (as it does for PDF points), so citations still line up.
MAX_ANALYSIS_PIXELS = int(os.environ.get("MAX_ANALYSIS_PIXELS", "16000000"))


def pdf_render_scale(width_pt: float, height_pt: float, scale: float = 2.0) -> float:
    """`scale`, lowered so a page of this size renders to at most MAX_ANALYSIS_PIXELS."""
    return min(scale, (MAX_ANALYSIS_PIXELS / max(width_pt * height_pt, 1.0)) ** 0.5)
