"""Where does an upload's analysis spend its time? Runs the web app's analysis path (image pipeline, then room / wall
geometry) on plans with an empty cache and prints the time and peak memory of every stage (app.profiling).

  uv run python scripts/profile_analysis.py                        # 6 preview plans from 0.3 to 12 MP
  uv run python scripts/profile_analysis.py a.png b.png --threads 4
  uv run python scripts/profile_analysis.py --save-outputs ../dataset/runs/profile/before   # graphs + geometry JSON
  uv run python scripts/profile_analysis.py --diff ../dataset/runs/profile/before ../dataset/runs/profile/after
  uv run python scripts/profile_analysis.py --cprofile out.prof     # + cProfile dump (snakeviz out.prof)

The first plan pays the cold start (model loading, imports); it is reported separately unless --no-warmup. A fresh
temporary CACHE_DIR is used (no cached perception), unless --cache-dir is given.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))  # `app` without PYTHONPATH, like scripts/inspect_plans.py
PREVIEW = REPO / "dataset" / "junction-dataset-preview" / "data" / "cubicasa5k"


def default_images(n: int = 6) -> list[Path]:
    """n preview plans spread over the size range (smallest, quantiles, largest), deterministic."""
    from PIL import Image

    paths = sorted(PREVIEW.glob("*.png"))
    if not paths:
        sys.exit(f"no plans given and none found in {PREVIEW}")
    sized = sorted(paths, key=lambda p: (Image.open(p).size[0] * Image.open(p).size[1], p.name))
    picks = sorted({round(i * (len(sized) - 1) / (n - 1)) for i in range(n)})
    return [sized[i] for i in picks]


def diff(a: Path, b: Path) -> int:
    """Compare two --save-outputs folders: identical files, and for differing graphs what changed."""
    changed = 0
    for fa in sorted(a.glob("*.json")):
        fb = b / fa.name
        if not fb.is_file():
            print(f"{fa.name}: missing in {b}")
            changed += 1
            continue
        if fa.read_bytes() == fb.read_bytes():
            print(f"{fa.name}: identical")
            continue
        changed += 1
        ja, jb = json.loads(fa.read_text(encoding="utf-8")), json.loads(fb.read_text(encoding="utf-8"))
        notes = []
        for key in ("counts",):
            if ja.get(key) != jb.get(key):
                notes.append(f"{key}: " + ", ".join(f"{k} {ja[key].get(k)}->{jb[key].get(k)}"
                                                    for k in sorted(set(ja[key]) | set(jb[key]))
                                                    if ja[key].get(k) != jb[key].get(k)))
        for key in ("items", "rooms", "wall_segments", "openings", "walls"):
            if key in ja and ja[key] != jb.get(key):
                same = sum(x == y for x, y in zip(ja[key], jb[key]))
                notes.append(f"{key}: {len(ja[key])}->{len(jb[key])} ({same} unchanged in place)")
        rest = sorted(k for k in set(ja) | set(jb)
                      if k not in ("counts", "items", "rooms", "wall_segments", "openings", "walls")
                      and ja.get(k) != jb.get(k))
        if rest:
            notes.append("also: " + ", ".join(rest))
        print(f"{fa.name}: DIFFERENT  " + "; ".join(notes))
    print(f"{changed} of {len(list(a.glob('*.json')))} files differ")
    return 1 if changed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--n", type=int, default=6, help="number of default preview plans")
    ap.add_argument("--cache-dir", type=Path, help="CACHE_DIR to use (default: a fresh temporary folder)")
    ap.add_argument("--threads", type=int, help="torch.set_num_threads (e.g. the server's vCPU count)")
    ap.add_argument("--cpus", type=int, help="pin the process to this many cores (emulates a smaller server; also "
                                             "sets --threads)")
    ap.add_argument("--parallel-walls", action="store_true",
                    help="start the wall segmenter while the image pipeline runs (layers.prefetch_walls)")
    ap.add_argument("--no-warmup", action="store_true", help="do not load the models before the first plan")
    ap.add_argument("--save-outputs", type=Path, help="write <plan>.graph.json / <plan>.geometry.json here")
    ap.add_argument("--json", type=Path, help="write all profiles as JSON")
    ap.add_argument("--cprofile", type=Path, help="cProfile dump of the plan runs")
    ap.add_argument("--diff", nargs=2, type=Path, metavar=("A", "B"), help="compare two --save-outputs folders")
    args = ap.parse_args()
    if args.diff:
        return diff(*args.diff)

    images = [p.resolve() for p in args.images] or default_images(args.n)
    tmp = None
    if args.cache_dir is None:
        tmp = tempfile.mkdtemp(prefix="stoa-profile-")
        args.cache_dir = Path(tmp)
    os.environ["CACHE_DIR"] = str(args.cache_dir)  # before any app import: config reads it at import time

    import torch
    from PIL import Image

    from app.config import MAX_ANALYSIS_PIXELS
    from app.cubicasa.detect import graph_from_image
    from app.cubicasa.rooms import geometry_from_image
    from app.profiling import Profile, stage

    if args.cpus:
        import psutil

        psutil.Process().cpu_affinity(list(range(args.cpus)))
        args.threads = args.threads or args.cpus
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch threads {torch.get_num_threads()}, cache {args.cache_dir}", flush=True)

    runs = []
    if not args.no_warmup:
        from app.warmup import warm_up

        with Profile() as prof:
            warm_up()
        print(f"warm-up {prof.total_s:6.1f} s  (model loading and imports, paid once per process)", flush=True)
        runs.append({"plan": "(warm-up)", "megapixels": 0, **prof.summary()})

    profiler = None
    if args.cprofile:
        import cProfile

        profiler = cProfile.Profile()
    for path in images:
        rel = f"uploads/{path.stem}.png"
        dest = args.cache_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(path).convert("RGB")
        if img.width * img.height > MAX_ANALYSIS_PIXELS:  # as the upload path does (main._analyze_plan)
            s = (MAX_ANALYSIS_PIXELS / (img.width * img.height)) ** 0.5
            img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS, reducing_gap=3.0)
        img.save(dest)
        mp = img.width * img.height / 1e6
        if profiler:
            profiler.enable()
        with Profile() as prof:
            if args.parallel_walls:
                from app.cubicasa.layers import prefetch_walls

                prefetch_walls(rel, dest)
            with stage("pipeline"):
                graph = graph_from_image(rel, dest)
            with stage("geometry"):
                geometry = geometry_from_image(rel, dest, graph)
        if profiler:
            profiler.disable()
        runs.append({"plan": path.name, "megapixels": round(mp, 2), **prof.summary()})
        print(f"{path.name:48s} {mp:5.1f} MP  {prof.total_s:6.1f} s  peak {prof.rss_peak_mb} MB", flush=True)
        if args.save_outputs:
            args.save_outputs.mkdir(parents=True, exist_ok=True)
            (args.save_outputs / f"{path.stem}.graph.json").write_text(graph.model_dump_json(indent=1), encoding="utf-8")
            (args.save_outputs / f"{path.stem}.geometry.json").write_text(geometry.model_dump_json(indent=1), encoding="utf-8")

    plans = [r for r in runs if r["plan"] != "(warm-up)"]
    print("\nstage (all plans)                              median s    max s   share of total")
    names = list(dict.fromkeys(st["name"] for r in plans for st in r["stages"]))
    total = sum(r["total_s"] for r in plans)
    for name in names:
        secs = [st["seconds"] for r in plans for st in r["stages"] if st["name"] == name]
        depth = name.count(".")
        label = "  " * depth + name.rsplit(".", 1)[-1]
        print(f"{label:46s} {statistics.median(secs):8.2f} {max(secs):8.2f} {100 * sum(secs) / total:8.1f} %")
    print(f"{'total':46s} {statistics.median(r['total_s'] for r in plans):8.2f} "
          f"{max(r['total_s'] for r in plans):8.2f}   (sum {total:.1f} s over {len(plans)} plans)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "threads": torch.get_num_threads(),
                                         "runs": runs}, indent=1))
    if profiler:
        profiler.dump_stats(args.cprofile)
        print(f"cProfile -> {args.cprofile}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
