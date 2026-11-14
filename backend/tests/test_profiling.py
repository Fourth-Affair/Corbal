"""app.profiling: nested stage names, no-op without a profile, steps(), threads."""

import threading

from app.profiling import Profile, active, stage, steps


def test_stage_is_noop_without_profile():
    assert active() is None
    with stage("anything"):
        pass
    assert active() is None


def test_nested_stages_and_callback():
    seen = []
    with Profile(on_stage=seen.append, sample_rss=False) as prof:
        with stage("pipeline"):
            with stage("ocr"):
                pass
        with steps() as step:
            step("walls")
            with stage("tiles"):
                pass
            step("rooms")
    names = [s["name"] for s in prof.stages]
    assert names == ["pipeline", "pipeline.ocr", "walls", "walls.tiles", "rooms"]
    assert seen == names
    assert all(s["seconds"] is not None and s["seconds"] >= 0 for s in prof.stages)
    assert active() is None


def test_stage_closes_on_error():
    with Profile(sample_rss=False) as prof:
        try:
            with stage("boom"):
                raise ValueError
        except ValueError:
            pass
        with stage("after"):
            pass
    assert [s["name"] for s in prof.stages] == ["boom", "after"]  # prefix was reset after the error


def test_profiles_are_per_thread():
    results = {}

    def work(name):
        with Profile(sample_rss=False) as prof:
            with stage(name):
                pass
        results[name] = [s["name"] for s in prof.stages]

    threads = [threading.Thread(target=work, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {"a": ["a"], "b": ["b"]}
