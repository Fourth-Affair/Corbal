"""Shared limits for work that must not run freely in parallel across API requests.

Sync FastAPI endpoints run in a thread pool, so several requests execute at once. Three kinds of work need limits:

- CPU-heavy perception (CubiCasa net, OCR, symbol detector, wall segmenter, PDF parsing): one analysis already uses
  every core through torch / onnxruntime thread pools. Measured on a 4-core laptop (8 uncached images): sequential
  37 s, 2-4 threads 34 s, 2 processes 47-67 s. More parallel analyses only add memory and contention, so
  `analysis_slot()` lets MAX_CONCURRENT_ANALYSES (default 1) run at a time and queues the rest. Raise it on machines
  with many cores. Network I/O (Supabase downloads / uploads) stays outside the slot, so it overlaps with another
  plan's computation.
- pdfium is not thread-safe, not even across different documents: every pypdfium2 call holds `pdfium_lock`.
- Ultralytics models keep per-model predictor state: concurrent `predict` calls on one model are guarded in symdet.
"""

import os
import threading
from contextlib import contextmanager

MAX_CONCURRENT_ANALYSES = max(1, int(os.environ.get("MAX_CONCURRENT_ANALYSES", "1")))
# How long a request waits for an analysis slot before 503. Kept under the 300 s response-header timeout of the
# Next.js proxy (undici) so the user sees the backend's message, not a proxy error.
ANALYSIS_QUEUE_TIMEOUT = max(1.0, float(os.environ.get("ANALYSIS_QUEUE_TIMEOUT", "240")))
# Worker threads for sync endpoints (Starlette's default is 40). Requests waiting for an analysis slot hold a thread,
# so a burst of uploads must not take every thread away from questions and other API calls.
THREADPOOL_SIZE = max(8, int(os.environ.get("THREADPOOL_SIZE", "100")))

_slots = threading.BoundedSemaphore(MAX_CONCURRENT_ANALYSES)
pdfium_lock = threading.RLock()


class AnalysisBusy(Exception):
    """No analysis slot became free within ANALYSIS_QUEUE_TIMEOUT."""


@contextmanager
def analysis_slot():
    """Hold one of the MAX_CONCURRENT_ANALYSES slots for CPU-heavy work. Not reentrant: do not nest."""
    if not _slots.acquire(timeout=ANALYSIS_QUEUE_TIMEOUT):
        raise AnalysisBusy(f"no analysis slot free within {ANALYSIS_QUEUE_TIMEOUT:.0f} s")
    try:
        yield
    finally:
        _slots.release()
