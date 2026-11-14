"""The counting features must come from the drawing, never from the file name (dataset names encode the CubiCasa
variant, an upload's name does not)."""

import numpy as np

from app.cubicasa import fuse


class _Maps(dict):
    @property
    def files(self):
        return list(self)


def test_image_features_ignore_the_file_name(monkeypatch):
    rng = np.random.default_rng(0)
    H, W = 64, 80
    z = _Maps(rooms=rng.integers(0, 12, (H, W), dtype=np.uint8), icons=rng.integers(0, 11, (H, W), dtype=np.uint8),
              rooms_conf=rng.integers(0, 256, (H, W), dtype=np.uint8),
              icons_conf=rng.integers(0, 256, (H, W), dtype=np.uint8),
              iprob=rng.integers(0, 256, (len(fuse.ICON_PROB), H, W), dtype=np.uint8),
              rprob=rng.integers(0, 256, (4, H, W), dtype=np.uint8),
              heat=rng.integers(0, 256, (21, H // 2, W // 2), dtype=np.uint8))
    meta = {"width": W, "height": H, "ocr": [{"text": "MH", "bbox": [5, 5, 20, 12], "score": 0.9}]}
    monkeypatch.setattr(fuse, "load", lambda rel: (z, meta))
    a = fuse.image_features("data/cubicasa5k/high_quality_architectural-1.png")
    b = fuse.image_features("uploads/0b1b1c4d-0e9f-424f-aed7-19925edbab0f.png")
    assert a == b
    assert not any(k.startswith("var_") for k in a)
