"""Load every model and heavy import before the first analysis.

A fresh process otherwise pays ~20 s on its first plan (measured on a laptop: ONNX OCR sessions ~10 s, torch op
registration and first-forward setup ~8 s, scikit-learn import ~2 s). The backend runs this in a background thread at
startup (`main.py` lifespan); scripts/profile_analysis.py runs it before timing plans. Each step is optional: missing
weights are skipped, so the app still starts, and the error surfaces on the first real analysis.
"""

from __future__ import annotations

import logging

from PIL import Image

from app.profiling import stage

log = logging.getLogger(__name__)


def warm_up() -> list[str]:
    """Load models and run each network once on a blank image. Returns the steps that failed."""
    blank = Image.new("RGB", (512, 512), "white")
    failed = []

    def step(name, fn):
        try:
            with stage(name):
                fn()
        except Exception as error:  # noqa: BLE001  warm-up must never stop the app
            failed.append(name)
            log.warning("warm-up step %s failed: %s", name, error)

    def imports():
        import skimage.morphology  # noqa: F401
        import shapely.geometry  # noqa: F401

        import app.cubicasa.rooms  # noqa: F401
        import app.cubicasa.room_split  # noqa: F401
        import app.cubicasa.wallvec  # noqa: F401

    def cubinet():
        from app.cubicasa import cubinet

        cubinet.predict(blank, max_side=256)

    def wallseg():
        from app.cubicasa import wallseg

        if wallseg.available():
            wallseg.predict(blank)

    def symbols():
        from app.cubicasa import symdet

        if symdet.available():
            symdet.predict_image(blank)

    def ocr():
        from app.cubicasa.ocr import read_text

        read_text(blank)

    def counting():
        from app.cubicasa.detect import _fusion

        _fusion()

    step("imports", imports)
    step("cubinet", cubinet)
    step("wallseg", wallseg)
    step("symbols", symbols)
    step("ocr", ocr)
    step("counting_models", counting)
    return failed
