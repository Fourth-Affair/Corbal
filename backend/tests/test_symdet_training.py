"""Symbol detector training helpers: tiles, rotated labels, tile-merge NMS, weights sidecar."""

import json

from app.cubicasa import symdet
from app.cubicasa.export_yolo import _lines, _rot90, _tile_boxes


def test_tile_starts_cover_the_image():
    assert symdet.tile_starts(800, 1024) == [0]
    starts = symdet.tile_starts(3000, 1024)
    assert starts[0] == 0 and starts[-1] == 3000 - 1024
    assert all(b - a <= 1024 - symdet.TILE_OVERLAP for a, b in zip(starts, starts[1:]))


def test_rot90_matches_pil_rotate_90():
    # PIL ROTATE_90 is counter-clockwise: the top-right corner of a 100-wide image goes to the top-left
    assert _rot90([(0, 90, 0, 100, 10)], 100) == [(0, 0, 0, 10, 10)]


def test_tile_boxes_keep_mostly_visible_boxes():
    boxes = [(1, 1000, 10, 1100, 30), (1, 700, 10, 780, 30)]  # second one is mostly left of the tile at x=768
    assert _tile_boxes(boxes, 768, 0, 1024, 1024) == [(1, 232, 10, 332, 30)]


def test_yolo_lines_are_normalised():
    assert _lines([(2, 0, 0, 50, 100)], 100, 200) == ["2 0.25000 0.25000 0.50000 0.50000"]


def test_nms_merges_tile_fragments_but_keeps_neighbours():
    dets = [{"cls": "window", "conf": 0.9, "bbox": [0, 0, 100, 10]},
            {"cls": "window", "conf": 0.5, "bbox": [60, 0, 100, 10]},  # fragment cut by a tile border
            {"cls": "window", "conf": 0.8, "bbox": [110, 0, 200, 10]},  # adjacent window
            {"cls": "door", "conf": 0.7, "bbox": [60, 0, 100, 10]}]  # other class: never suppressed
    kept = symdet.nms(dets)
    assert [(d["cls"], d["conf"]) for d in kept] == [("window", 0.9), ("window", 0.8), ("door", 0.7)]


def test_weights_meta_reads_sidecar(tmp_path, monkeypatch):
    w = tmp_path / "symbol-det.pt"
    w.write_bytes(b"x")
    monkeypatch.setenv("SYMDET_WEIGHTS", str(w))
    monkeypatch.delenv("SYMDET_IMGSZ", raising=False)
    monkeypatch.delenv("SYMDET_TILE", raising=False)
    assert symdet.weights_meta() == {"imgsz": 1280, "tile": 0}  # no sidecar: the original detector
    w.with_suffix(".json").write_text(json.dumps({"imgsz": 1024, "tile": 1024, "base": "yolo11m.pt"}))
    assert symdet.weights_meta() == {"imgsz": 1024, "tile": 1024}
    monkeypatch.setenv("SYMDET_IMGSZ", "1600")
    assert symdet.weights_meta()["imgsz"] == 1600
