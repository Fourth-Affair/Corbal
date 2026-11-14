"""Background analysis jobs: POST /analyze -> 202, poll GET /analysis -> 200 with the graph; failures; restarts."""

import threading
import time
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.main as main
from app.jobs import AnalysisJobs
from app.models import PlanGraph
from app.profiling import stage


def _graph():
    return PlanGraph(image="uploads/x.png", width=10, height=10, source="model")


def _wait(jobs, plan_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(plan_id)
        if job and job.state in ("done", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_submit_is_idempotent_and_records_stages():
    release = threading.Event()
    calls = []

    def run(plan_id):
        calls.append(plan_id)
        with stage("pipeline"):
            release.wait(5)
        return _graph()

    jobs = AnalysisJobs(run, workers=1)
    pid = uuid4()
    first = jobs.submit(pid)
    assert jobs.submit(pid) is first
    release.set()
    job = _wait(jobs, pid)
    assert job.state == "done" and calls == [pid]
    assert [s["name"] for s in job.profile["stages"]] == ["pipeline"]
    status, body = jobs.response(job)
    assert status == 200 and body["graph"]["width"] == 10


def test_queue_positions_and_failure_resubmit():
    gate = threading.Event()

    def run(plan_id):
        gate.wait(5)
        raise HTTPException(422, "not a plan")

    jobs = AnalysisJobs(run, workers=1)
    a, b, c = uuid4(), uuid4(), uuid4()
    jobs.submit(a)
    time.sleep(0.05)  # a is running
    jb, jc = jobs.submit(b), jobs.submit(c)
    assert jobs.response(jb) == (202, {"state": "queued", "stage": None, "position": 0})
    assert jobs.response(jc)[1]["position"] == 1
    gate.set()
    job = _wait(jobs, c)
    assert (job.state, job.status, job.detail) == ("failed", 422, "not a plan")
    assert jobs.submit(c) is not job  # a failed job is retried on the next submit


def test_endpoints(monkeypatch):
    release = threading.Event()

    def run(plan_id):
        release.wait(5)
        return _graph()

    jobs = AnalysisJobs(run, workers=1)
    monkeypatch.setattr(main, "_jobs", jobs)
    monkeypatch.setattr(main, "_saved_graph", lambda plan_id: None)
    client = TestClient(main.app)
    pid = uuid4()
    assert client.get(f"/api/plans/{pid}/analysis").status_code == 404  # unknown: never submitted
    r = client.post(f"/api/plans/{pid}/analyze")
    assert r.status_code == 202 and r.json()["state"] in ("queued", "running")
    release.set()
    _wait(jobs, pid)
    r = client.get(f"/api/plans/{pid}/analysis")
    assert r.status_code == 200 and r.json()["state"] == "done" and r.json()["graph"]["height"] == 10


def test_saved_graph_is_returned_directly(monkeypatch):
    monkeypatch.setattr(main, "_jobs", AnalysisJobs(lambda plan_id: _graph(), workers=1))
    monkeypatch.setattr(main, "_saved_graph", lambda plan_id: _graph())
    r = TestClient(main.app).post(f"/api/plans/{uuid4()}/analyze")
    assert r.status_code == 200 and r.json()["width"] == 10 and "state" not in r.json()
