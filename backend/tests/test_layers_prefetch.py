"""layers.prefetch_walls: a wall_classes call during a prefetch waits for it instead of computing the map again."""

import threading
import time

import numpy as np

from app.cubicasa import layers, wallseg


def test_prefetch_then_call_computes_once(monkeypatch, tmp_path):
    if not wallseg.available():
        return  # needs models/wallseg.pt (tracked in git) for the cache key
    calls = []

    def slow_compute(path, model, scale, cache):
        calls.append(threading.current_thread().name)
        time.sleep(0.3)
        return np.ones((4, 4), np.uint8), model

    monkeypatch.setattr(layers, "LAYER_CACHE", tmp_path)
    monkeypatch.setattr(layers, "_compute_wall_classes", slow_compute)
    thread = layers.prefetch_walls("uploads/p.png", tmp_path / "p.png")
    time.sleep(0.05)  # the prefetch has started
    cls, model = layers.wall_classes("uploads/p.png", tmp_path / "p.png")
    thread.join()
    assert calls == ["walls-prefetch"] and cls.shape == (4, 4) and model == "wallseg"
    assert not layers._inflight
