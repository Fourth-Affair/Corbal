"""Build (or load cached) PlanGraphs for cubicasa images.

source="svg":   ground truth from the original CubiCasa5K model.svg (training / validation / oracle only).
source="model": image-only pipeline (segmentation/detection model + OCR). This is the runtime path.
"""

from pathlib import Path

from app.config import CACHE_DIR, DATASET_DIR
from app.models import PlanGraph


def resolve_dataset_path(rel: str) -> Path:
    """Resolve a dataset-relative path, refusing anything outside the dataset dir."""
    p = (DATASET_DIR / rel).resolve()
    if not p.is_relative_to(DATASET_DIR.resolve()) or not p.is_file():
        raise FileNotFoundError(rel)
    return p


def _cache_path(rel: str, source: str) -> Path:
    sub = source
    if source == "model":  # refitted fusion models must not reuse old predictions
        from app.cubicasa.fit import FUSION_PATH

        if FUSION_PATH.is_file():
            st = FUSION_PATH.stat()
            sub = f"model-{st.st_size:x}-{int(st.st_mtime):x}"
        from app.cubicasa import symdet

        if symdet.available():  # detector boxes are part of the evidence
            sub += "-" + symdet.fingerprint().replace(".", "_")
    return CACHE_DIR / sub / (Path(rel).stem + ".json")


def get_graph(rel: str, source: str = "model", use_cache: bool = True) -> PlanGraph:
    cache = _cache_path(rel, source)
    if use_cache and cache.is_file():
        return PlanGraph.model_validate_json(cache.read_text())

    if source == "svg":
        from app.cubicasa.labels_from_svg import graph_from_svg

        graph = graph_from_svg(rel)
    elif source == "model":
        from app.cubicasa.detect import graph_from_image

        graph = graph_from_image(rel, resolve_dataset_path(rel))
    else:
        raise ValueError(source)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(graph.model_dump_json())
    return graph
