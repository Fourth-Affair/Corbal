"""Train the symbol detector or the wall segmenter on a rented Verda GPU VM, download the weights, delete the VM.

  uv run python scripts/verda_train.py list                    # GPU types, prices, availability
  uv run python scripts/verda_train.py run                     # cheapest available >= 24 GB GPU, yolo11s, 80 epochs
  uv run python scripts/verda_train.py run --type 1A100.22V --base rtdetr-l.pt --epochs 60 --batch 8
  uv run python scripts/verda_train.py run --host 1.2.3.4             # VM you created in the dashboard
  uv run python scripts/verda_train.py run --host 1.2.3.4 --task wallseg   # wall segmenter (wallseg prepare first)
  uv run python scripts/verda_train.py attach <instance-id>    # re-follow a run (e.g. after the laptop slept)
  uv run python scripts/verda_train.py delete <instance-id>

Credentials: VERDA_CLIENT_ID / VERDA_CLIENT_SECRET in the environment or the repo-root .env.
The VM downloads CubiCasa5K itself (fast datacenter link); we upload only code, SVG labels and the holdout
questions (the challenge preview, never trained on); for --task wallseg also the label masks (~20 MB).
Results land in dataset/models/: symbol-det.{pt,onnx} or wallseg.{pt,log.csv}.
The VM is deleted when the run ends, fails, times out (--max-hours) or is interrupted, unless --keep.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
DATASET = REPO / "dataset"
REMOTE = "~/stoa"
KNOWN_HOSTS = Path.home() / ".ssh" / "verda_known_hosts"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"


def _default_holdout() -> Path:
    """The challenge preview's questions (= our test images). It moves aside once the full dataset is unpacked."""
    prev = DATASET / "junction-dataset-preview" / "questions.jsonl"
    if prev.is_file():
        return prev
    readme = DATASET / "junction-dataset" / "README.md"
    if readme.is_file() and sum(1 for _ in (DATASET / "junction-dataset" / "data" / "cubicasa5k").glob("*.png")) < 1000:
        return DATASET / "junction-dataset" / "questions.jsonl"  # still the preview in place
    return prev


def _client():
    from verda import VerdaClient

    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    cid, secret = os.environ.get("VERDA_CLIENT_ID"), os.environ.get("VERDA_CLIENT_SECRET")
    if not cid or not secret:
        raise SystemExit("set VERDA_CLIENT_ID and VERDA_CLIENT_SECRET (env or repo-root .env)")
    return VerdaClient(cid, secret)


def _gpu_mem(t) -> float:
    m = t.gpu_memory or {}
    return float(m.get("size_in_gigabytes") or m.get("size") or 0)


def _available(client, spot: bool) -> dict[str, set[str]]:
    """instance type -> locations where it can be deployed now."""
    out: dict[str, set[str]] = {}
    for loc in client.instances.get_availabilities(is_spot=spot):
        for t in loc.get("availabilities", []):
            out.setdefault(t, set()).add(loc.get("location_code", "?"))
    return out


def cmd_list(args) -> None:
    client = _client()
    avail = _available(client, args.spot)
    types = sorted(client.instance_types.get(), key=lambda t: t.spot_price_per_hour if args.spot else t.price_per_hour)
    print(f"{'type':<22}{'gpu':<28}{'GB':>5}{'$/h':>8}  available")
    for t in types:
        if not (t.gpu or {}).get("number_of_gpus"):
            continue
        price = t.spot_price_per_hour if args.spot else t.price_per_hour
        print(f"{t.instance_type:<22}{(t.gpu or {}).get('description', t.model)[:27]:<28}{_gpu_mem(t):>5.0f}"
              f"{price:>8.2f}  {','.join(sorted(avail.get(t.instance_type, []))) or '-'}")


def _pick_type(client, args) -> tuple[str, str]:
    avail = _available(client, args.spot)
    if args.type:
        if args.type not in avail:
            raise SystemExit(f"{args.type} is not available right now (see `list`)")
        return args.type, sorted(avail[args.type])[0]
    single = [t for t in client.instance_types.get()
              if (t.gpu or {}).get("number_of_gpus") == 1 and _gpu_mem(t) >= args.min_gpu_gb and t.instance_type in avail]
    if not single:
        raise SystemExit("no single-GPU type with enough memory is available; pick one with --type (see `list`)")
    best = min(single, key=lambda t: t.spot_price_per_hour if args.spot else t.price_per_hour)
    return best.instance_type, sorted(avail[best.instance_type])[0]


def _pick_image(client, instance_type: str, wanted: str | None) -> str:
    images = [im.image_type for im in client.images.get(instance_type=instance_type)]
    if wanted:
        if wanted not in images:
            raise SystemExit(f"image {wanted} not offered for {instance_type}: {images}")
        return wanted
    cuda = sorted((i for i in images if "ubuntu" in i and "cuda" in i), reverse=True)
    if not cuda:
        raise SystemExit(f"no Ubuntu+CUDA image for {instance_type}: {images}")
    return cuda[0]


def _ssh_key_id(client) -> str:
    pub = SSH_KEY.with_suffix(".pub").read_text().strip()
    for k in client.ssh_keys.get():
        if k.public_key.split()[:2] == pub.split()[:2]:
            return k.id
    return client.ssh_keys.create(f"stoa-{os.uname().nodename}", pub).id


def _ssh_base(ip: str) -> list[str]:
    return ["ssh", "-i", str(SSH_KEY), "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", f"root@{ip}"]


def _ssh(ip: str, cmd: str, check=True, capture=False) -> subprocess.CompletedProcess:
    return subprocess.run(_ssh_base(ip) + [cmd], check=check, text=True, capture_output=capture)


def _rsync(ip: str, src: list[str], dst: str, *extra: str) -> None:
    ssh = f"ssh -i {SSH_KEY} -o UserKnownHostsFile={KNOWN_HOSTS} -o StrictHostKeyChecking=accept-new"
    subprocess.run(["rsync", "-az", "--info=stats1", "-e", ssh, *extra, *src, dst], check=True)


def _wait_running(client, iid: str, timeout_s: float = 1200) -> str:
    from verda.constants import InstanceStatus

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        inst = client.instances.get_by_id(iid)
        if inst.status == InstanceStatus.ERROR:
            raise RuntimeError(f"instance {iid} entered error state")
        if inst.status == InstanceStatus.RUNNING and inst.ip:
            for _ in range(60):  # sshd comes up shortly after "running"
                if _ssh(inst.ip, "true", check=False, capture=True).returncode == 0:
                    return inst.ip
                time.sleep(5)
        print(f"  status: {inst.status}", flush=True)
        time.sleep(15)
    raise TimeoutError(f"instance {iid} not reachable after {timeout_s:.0f}s")


def _delete(client, iid: str) -> None:
    from verda.constants import Actions

    client.instances.action(iid, Actions.DELETE, delete_permanently=True)
    print(f"deleted instance {iid}")


def _follow(ip: str, deadline: float) -> bool:
    """Poll the remote log until DONE/FAILED or the deadline. True on success."""
    last = ""
    while time.time() < deadline:
        r = _ssh(ip, f"cd {REMOTE}; tail -n 2 train.log 2>/dev/null; ls DONE FAILED 2>/dev/null", check=False, capture=True)
        out = r.stdout.strip()
        lines = out.splitlines()
        if "DONE" in lines:
            return True
        if "FAILED" in lines:
            print(_ssh(ip, f"tail -n 40 {REMOTE}/train.log", check=False, capture=True).stdout)
            return False
        tail = lines[-1] if lines else ""
        if tail and tail != last:
            print(f"  {tail[-160:]}", flush=True)
            last = tail
        time.sleep(60)
    print("max-hours reached")
    return False


def _collect(ip: str, task: str = "symdet") -> None:
    models = DATASET / "models"
    models.mkdir(parents=True, exist_ok=True)
    if task == "wallseg":
        _rsync(ip, [f"root@{ip}:{REMOTE}/data/out/wallseg/"], str(models) + "/")
        _rsync(ip, [f"root@{ip}:{REMOTE}/train.log"], str(models / "wallseg.train.log"))
        print(f"weights in {models}: wallseg.pt, wallseg.log.csv")
        return
    _rsync(ip, [f"root@{ip}:{REMOTE}/data/out/"], str(models) + "/", "--exclude=wallseg/")
    _rsync(ip, [f"root@{ip}:{REMOTE}/train.log"], str(models / "symbol-det.train.log"))
    print(f"weights in {models}: symbol-det.pt, symbol-det.onnx, symbol-det.results.csv")


def cmd_run(args) -> None:
    holdout = Path(args.holdout)
    svg = DATASET / "cache" / "svg"
    if not holdout.is_file() or not svg.is_dir():
        raise SystemExit(f"need {holdout} and {svg} (share bundle + challenge preview)")
    if args.host:  # VM created by hand (dashboard): no API calls, never deleted by us
        ip = args.host.split("@")[-1]
        ok = _train_on(ip, args, holdout, svg)
        print(f"done; the VM at {ip} is still running - delete it in the Verda dashboard when finished")
        sys.exit(0 if ok else 1)
    client = _client()
    itype, loc = _pick_type(client, args)
    image = _pick_image(client, itype, args.image)
    print(f"creating {itype} ({'spot' if args.spot else 'on-demand'}) in {loc} with {image}")
    from verda.instances import OSVolume

    inst = client.instances.create(instance_type=itype, image=image, hostname="stoa-symdet",
                                   description="Junction Stoa symbol detector training", ssh_key_ids=[_ssh_key_id(client)],
                                   location=loc, is_spot=args.spot, os_volume=OSVolume(name="stoa-os", size=100),
                                   wait_for_status=None)
    print(f"instance {inst.id} ordered (delete manually with: scripts/verda_train.py delete {inst.id})")
    ok = False
    try:
        ip = _wait_running(client, inst.id)
        ok = _train_on(ip, args, holdout, svg)
    finally:
        if args.keep:
            print(f"--keep: instance {inst.id} is still running and billing")
        else:
            _delete(client, inst.id)
    sys.exit(0 if ok else 1)


def _train_on(ip: str, args, holdout: Path, svg: Path) -> bool:
    """Upload code + labels to a running VM, train there, follow the log, download the results."""
    print(f"up at {ip}; uploading code + labels")
    _ssh(ip, f"mkdir -p {REMOTE}/backend {REMOTE}/data/cache && "
             "(command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync))")
    _rsync(ip, [str(BACKEND) + "/"], f"root@{ip}:{REMOTE}/backend/",
           "--exclude=.venv", "--exclude=__pycache__", "--exclude=.pytest_cache")
    _rsync(ip, [str(svg)], f"root@{ip}:{REMOTE}/data/cache/")
    _rsync(ip, [str(holdout)], f"root@{ip}:{REMOTE}/data/holdout.jsonl")
    if args.task == "wallseg":
        masks = DATASET / "cache" / "wallmasks"
        if not (masks / "index.json").is_file():
            raise SystemExit(f"{masks}/index.json missing: run `python -m app.cubicasa.wallseg prepare` first")
        _rsync(ip, [str(masks)], f"root@{ip}:{REMOTE}/data/cache/")
    env = f"TASK={args.task} BASE={args.base} EPOCHS={args.epochs} IMGSZ={args.imgsz} BATCH={args.batch} DEGREES={args.degrees}"
    # clear markers from an earlier run first, so _follow cannot see a stale DONE
    _ssh(ip, f"cd {REMOTE} && rm -f DONE FAILED && {env} nohup bash backend/scripts/verda_remote.sh > train.log 2>&1 "
             "< /dev/null &")
    print("training started; following the log (if this script stops, training continues on the VM)")
    ok = _follow(ip, time.time() + args.max_hours * 3600)
    if ok:
        _collect(ip, args.task)
    return ok


def cmd_attach(args) -> None:
    client = _client()
    ip = _wait_running(client, args.id)
    ok = _follow(ip, time.time() + args.max_hours * 3600)
    if ok:
        _collect(ip, args.task)
    if not args.keep:
        _delete(client, args.id)
    sys.exit(0 if ok else 1)


def cmd_delete(args) -> None:
    _delete(_client(), args.id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--spot", action="store_true")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("run")
    p.add_argument("--type", help="instance type (default: cheapest available single GPU with >= --min-gpu-gb)")
    p.add_argument("--min-gpu-gb", type=float, default=24)
    p.add_argument("--image", help="OS image (default: newest Ubuntu+CUDA image for the type)")
    p.add_argument("--spot", action="store_true", help="cheaper, but can be reclaimed mid-run")
    p.add_argument("--task", choices=["symdet", "wallseg"], default="symdet")
    p.add_argument("--base", default="yolo11s.pt", help="symdet only")
    p.add_argument("--epochs", type=int, default=None, help="default: 80 (symdet), 40 (wallseg)")
    p.add_argument("--imgsz", type=int, default=1280, help="symdet only")
    p.add_argument("--degrees", type=float, default=0.0, help="symdet only: random rotation augmentation (+/- deg)")
    p.add_argument("--batch", type=int, default=None, help="default: 16 (symdet), 24 (wallseg)")
    p.add_argument("--holdout", default=str(_default_holdout()),
                   help="questions.jsonl whose images must never be trained on")
    p.add_argument("--max-hours", type=float, default=8)
    p.add_argument("--keep", action="store_true", help="don't delete the VM at the end")
    p.add_argument("--host", help="use an existing VM (root@IP or IP) instead of creating one; it is not deleted")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("attach")
    p.add_argument("id")
    p.add_argument("--max-hours", type=float, default=8)
    p.add_argument("--keep", action="store_true")
    p.add_argument("--task", choices=["symdet", "wallseg"], default="symdet")
    p.set_defaults(fn=cmd_attach)
    p = sub.add_parser("delete")
    p.add_argument("id")
    p.set_defaults(fn=cmd_delete)
    args = ap.parse_args()
    if args.cmd == "run":
        wall = args.task == "wallseg"
        args.epochs = args.epochs or (40 if wall else 80)
        args.batch = args.batch or (24 if wall else 16)
    args.fn(args)


if __name__ == "__main__":
    main()
