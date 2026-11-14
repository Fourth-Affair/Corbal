"""Pre-build cached PlanGraphs for all cubicasa images.

  python -m app.cubicasa.build_cache --source svg --workers 8
  python -m app.cubicasa.build_cache --source model --split val
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

from app.cubicasa.graph import get_graph


def _one(args: tuple[str, str, bool]) -> str | None:
    rel, source, force = args
    try:
        get_graph(rel, source, use_cache=not force)
        return None
    except Exception as e:  # report and continue
        return f"{rel}: {type(e).__name__}: {e}"


def main() -> None:
    from app.eval import load_questions, split_of

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["svg", "model"], default="svg")
    ap.add_argument("--split", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = sorted({q["path"] for q in load_questions()})
    if args.split != "all":
        paths = [p for p in paths if split_of(p) == args.split]
    workers = args.workers if args.source == "svg" else 1  # model inference is already multi-threaded
    with ProcessPoolExecutor(workers) as ex:
        errors = [e for e in ex.map(_one, [(p, args.source, args.force) for p in paths], chunksize=16) if e]
    print(f"built {len(paths) - len(errors)}/{len(paths)} graphs ({args.source})")
    for e in errors[:20]:
        print("  ", e)


if __name__ == "__main__":
    main()
