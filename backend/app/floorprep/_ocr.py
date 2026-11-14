"""Shared, lazily-created RapidOCR engine (used by orientation fix and pen-mark proposals)."""
_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
        # RapidOCR's angle classifier silently turns upside-down text upright, which would hide
        # exactly the 180-degree rotations the orientation fix needs to see.
        _OCR.use_angle_cls = False
    return _OCR
