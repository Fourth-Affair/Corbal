"""Stage timings for plan analysis.

    with Profile(on_stage=print) as prof:          # activates the profile for this thread / context
        with stage("pipeline"):
            with stage("cubinet"):                 # recorded as "pipeline.cubinet"
                ...
    prof.stages  # [{"name", "start_s", "seconds", "rss_peak_mb"}], in start order

`stage()` is a no-op when no profile is active, so library code (OCR, networks, geometry) marks its own stages without
anything being passed through. A profile belongs to the context that entered it (contextvars): concurrent analyses in
other threads keep their own. Work submitted to another thread records into the profile only if that thread runs
`Profile.attach()` (or uses `contextvars.copy_context()`).
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from contextlib import ExitStack, contextmanager
from typing import Callable

_current: contextvars.ContextVar[Profile | None] = contextvars.ContextVar("analysis_profile", default=None)
_prefix: contextvars.ContextVar[str] = contextvars.ContextVar("analysis_stage_prefix", default="")


class Profile:
    def __init__(self, on_stage: Callable[[str], None] | None = None, sample_rss: bool = True):
        self.stages: list[dict] = []
        self.on_stage = on_stage
        self._sample_rss = sample_rss
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._rss_peak = 0
        self._open: list[dict] = []  # stages whose peak RSS is still being tracked
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._tokens: list = []

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> Profile:
        self._t0 = time.perf_counter()
        self._tokens.append((_current.set(self), _prefix.set("")))
        if self._sample_rss:
            self._start_sampler()
        return self

    def __exit__(self, *exc) -> None:
        cur, pre = self._tokens.pop()
        _current.reset(cur)
        _prefix.reset(pre)
        self._stop.set()
        if self._sampler:
            self._sampler.join(timeout=1)

    @contextmanager
    def attach(self, prefix: str = ""):
        """Record into this profile from another thread (e.g. a worker started by an analysis)."""
        t1, t2 = _current.set(self), _prefix.set(prefix)
        try:
            yield self
        finally:
            _current.reset(t1)
            _prefix.reset(t2)

    def _start_sampler(self) -> None:
        try:
            import psutil
        except ImportError:  # optional: timings still work
            return
        proc = psutil.Process()

        def run():
            while not self._stop.wait(0.05):
                rss = proc.memory_info().rss
                with self._lock:
                    self._rss_peak = max(self._rss_peak, rss)
                    for st in self._open:
                        st["_peak"] = max(st["_peak"], rss)

        self._rss_peak = proc.memory_info().rss
        self._sampler = threading.Thread(target=run, name="profile-rss", daemon=True)
        self._sampler.start()

    # ------------------------------------------------------------------ results
    @property
    def total_s(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def rss_peak_mb(self) -> float | None:
        return round(self._rss_peak / 2**20) if self._rss_peak else None

    def summary(self) -> dict:
        return {"total_s": round(self.total_s, 3), "rss_peak_mb": self.rss_peak_mb, "stages": self.stages}

    def to_json(self) -> str:
        return json.dumps(self.summary(), separators=(",", ":"))


@contextmanager
def stage(name: str):
    """Time a stage of the active profile (nested names are joined with dots). No-op without an active profile."""
    prof = _current.get()
    if prof is None:
        yield
        return
    full = f"{_prefix.get()}.{name}" if _prefix.get() else name
    token = _prefix.set(full)
    rec = {"name": full, "start_s": round(time.perf_counter() - prof._t0, 3), "seconds": None, "_peak": 0}
    with prof._lock:
        prof.stages.append(rec)
        prof._open.append(rec)
    if prof.on_stage:
        try:
            prof.on_stage(full)
        except Exception:  # noqa: BLE001  progress reporting must never break an analysis
            pass
    started = time.perf_counter()
    try:
        yield
    finally:
        _prefix.reset(token)
        with prof._lock:
            rec["seconds"] = round(time.perf_counter() - started, 3)
            prof._open.remove(rec)
            peak = rec.pop("_peak")
            rec["rss_peak_mb"] = round(peak / 2**20) if peak else None


@contextmanager
def steps():
    """Sequential stages without re-indenting a long function: each `step(name)` ends the previous stage.

        with steps() as step:
            step("walls"); ...
            step("rooms"); ...
    """
    stack = ExitStack()

    def step(name: str) -> None:
        stack.close()
        stack.enter_context(stage(name))

    try:
        yield step
    finally:
        stack.close()


def active() -> Profile | None:
    return _current.get()
