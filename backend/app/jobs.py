"""Background analysis jobs: POST /analyze returns at once, a bounded worker pool runs the analysis, the page polls.

One registry per process (uvicorn runs one worker): the queue, its positions and the stage reports live in memory. A
restart forgets running jobs; the page then gets {"state": "unknown"} and submits again. Finished jobs are kept for
JOB_TTL_S so a slow poller still sees the result; the graph itself is persisted by the analysis (Supabase + local
cache), so nothing is lost when a job is evicted.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable
from uuid import UUID

from fastapi import HTTPException

from app.models import PlanGraph
from app.profiling import Profile

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:  # same as app.main: uvicorn does not configure the app's loggers
    log.addHandler(logging.StreamHandler())
log.propagate = False
JOB_TTL_S = 15 * 60
_seq = itertools.count()  # submission order (time.time() ties on Windows' ~15 ms clock)


@dataclass
class Job:
    plan_id: UUID
    seq: int = field(default_factory=lambda: next(_seq))
    state: str = "queued"  # queued | running | done | failed
    stage: str | None = None  # dotted stage name of app.profiling, while running
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    status: int | None = None  # HTTP status of a failure
    detail: str | None = None
    graph: PlanGraph | None = None
    profile: dict | None = None

    def public(self, position: int | None = None) -> dict:
        out = {"state": self.state, "stage": self.stage}
        if self.state == "queued":
            out["position"] = position
        if self.started_at:
            out["elapsed_s"] = round((self.finished_at or time.time()) - self.started_at, 1)
        if self.state == "failed":
            out |= {"status": self.status, "detail": self.detail}
        if self.state == "done":
            out |= {"graph": self.graph.model_dump(mode="json") if self.graph else None, "profile": self.profile}
        return out


class AnalysisJobs:
    def __init__(self, run: Callable[[UUID], PlanGraph], workers: int, ready: threading.Event | None = None,
                 ready_timeout: float = 180.0):
        """run(plan_id) does the analysis (stages recorded into the active profile); `ready` is waited for before
        the first job starts (the model warm-up), at most ready_timeout seconds."""
        self._run_analysis = run
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="analysis")
        self._jobs: dict[UUID, Job] = {}
        self._lock = threading.Lock()
        self._ready = ready
        self._ready_timeout = ready_timeout

    def submit(self, plan_id: UUID) -> Job:
        """The plan's current job, or a new one (also when the last one failed)."""
        with self._lock:
            self._evict()
            job = self._jobs.get(plan_id)
            if job is not None and job.state != "failed":
                return job
            job = self._jobs[plan_id] = Job(plan_id)
        self._pool.submit(self._run, job)
        return job

    def get(self, plan_id: UUID) -> Job | None:
        with self._lock:
            self._evict()
            return self._jobs.get(plan_id)

    def position(self, job: Job) -> int:
        """Jobs queued before this one (0 = next to start)."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.state == "queued" and j.seq < job.seq)

    def response(self, job: Job) -> tuple[int, dict]:
        """(HTTP status, body) for the status endpoints: 202 while queued / running, 200 when finished."""
        position = self.position(job) if job.state == "queued" else None
        return (202 if job.state in ("queued", "running") else 200), job.public(position)

    def _evict(self) -> None:
        now = time.time()
        for pid in [p for p, j in self._jobs.items() if j.finished_at and now - j.finished_at > JOB_TTL_S]:
            del self._jobs[pid]

    def _run(self, job: Job) -> None:
        if self._ready is not None:
            self._ready.wait(self._ready_timeout)
        job.state, job.started_at, job.stage = "running", time.time(), "starting"

        def on_stage(name: str) -> None:
            job.stage = name

        prof = Profile(on_stage=on_stage)
        try:
            with prof:
                job.graph = self._run_analysis(job.plan_id)
            job.state = "done"
        except HTTPException as error:
            job.state, job.status, job.detail = "failed", error.status_code, str(error.detail)
        except Exception as error:  # noqa: BLE001  reported to the page; the traceback goes to the log
            log.exception("plan %s: analysis failed", job.plan_id)
            job.state, job.status, job.detail = "failed", 500, f"Internal error ({type(error).__name__})"
        finally:
            job.finished_at, job.stage = time.time(), None
            job.profile = prof.summary()
            log.info("analysis_profile plan_id=%s state=%s %s", job.plan_id, job.state, prof.to_json())
