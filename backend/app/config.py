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
# Trained detector/segmenter weights.
MODEL_WEIGHTS = Path(os.environ.get("CUBICASA_WEIGHTS", _REPO / "dataset" / "models" / "cubicasa-seg.pt"))
