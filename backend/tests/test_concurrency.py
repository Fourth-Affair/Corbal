import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


def test_symbol_detector_predict_is_serialized(monkeypatch):
    """Ultralytics models are not thread-safe: an analysis job and a PDF question may both reach the detector."""
    from app.cubicasa import symdet

    active = peak = 0
    guard = threading.Lock()

    def predict(*_args, **_kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        boxes = SimpleNamespace(xyxy=SimpleNamespace(tolist=list), conf=SimpleNamespace(tolist=list),
                                cls=SimpleNamespace(tolist=list))
        return [SimpleNamespace(names={}, boxes=boxes)]

    monkeypatch.setattr(symdet, "_model", lambda: SimpleNamespace(predict=predict))
    with ThreadPoolExecutor(4) as ex:
        assert list(ex.map(lambda _: symdet._predict(None, 640), range(4))) == [[]] * 4
    assert peak == 1


def test_gemini_client_is_created_once_under_concurrency(monkeypatch):
    pytest.importorskip("google.genai")
    from google import genai

    from app.answering import agent

    created = []

    class SlowClient:
        def __init__(self, **_kwargs):
            time.sleep(0.05)  # widen the race window
            created.append(self)

    monkeypatch.setattr(agent, "_client", None)
    monkeypatch.setattr(genai, "Client", SlowClient)
    with ThreadPoolExecutor(8) as ex:
        clients = list(ex.map(lambda _: agent._get_client(), range(8)))
    assert len(created) == 1 and all(c is created[0] for c in clients)
