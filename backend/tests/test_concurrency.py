import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import concurrency


def _peak_parallelism(run, n_jobs: int = 6) -> int:
    active = peak = 0
    guard = threading.Lock()

    def job():
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with guard:
            active -= 1

    with ThreadPoolExecutor(n_jobs) as ex:
        list(ex.map(lambda _: run(job), range(n_jobs)))
    return peak


def test_analysis_slot_bounds_parallel_work(monkeypatch):
    monkeypatch.setattr(concurrency, "_slots", threading.BoundedSemaphore(2))

    def bounded(job):
        with concurrency.analysis_slot():
            job()

    assert _peak_parallelism(bounded) == 2


def test_analysis_slot_times_out_and_main_answers_503(monkeypatch):
    from app import main

    monkeypatch.setattr(concurrency, "_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(concurrency, "ANALYSIS_QUEUE_TIMEOUT", 0.05)
    holding, release = threading.Event(), threading.Event()

    def hold():
        with concurrency.analysis_slot():
            holding.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    try:
        assert holding.wait(5)
        with pytest.raises(HTTPException) as error:
            main._bounded_analysis(lambda: None)
        assert error.value.status_code == 503
    finally:
        release.set()
        t.join()
    assert main._bounded_analysis(lambda: 42) == 42  # the slot is free again


def test_threadpool_is_resized_at_startup():
    import anyio.to_thread

    from app.main import app

    with TestClient(app) as client:
        size = client.portal.call(lambda: anyio.to_thread.current_default_thread_limiter().total_tokens)
    assert size == concurrency.THREADPOOL_SIZE


def test_symbol_detector_predict_is_serialized(monkeypatch):
    from types import SimpleNamespace

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
