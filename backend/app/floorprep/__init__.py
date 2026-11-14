"""floorprep: simple preprocessing + binary pen-mark flag for floor-plan images.

    from floorprep import load_image, preprocess, detect_penmarks
    img = load_image("plan.png")
    result = preprocess(img)          # result.images -> model input, result.info -> transforms
    has_pen = detect_penmarks(img)    # True / False
"""
from .preprocess import load_image, preprocess, PreprocessResult, DEFAULTS
from .penmarks import detect_penmarks, penmark_score

__all__ = ["load_image", "preprocess", "PreprocessResult", "DEFAULTS", "detect_penmarks", "penmark_score"]
