"""Visual inspection sheets: everything the pipeline knows about a plan, side by side, to spot the usual problems.

  uv run python scripts/inspect_plans.py                                   # 12 preview plans, spread over the set
  uv run python scripts/inspect_plans.py --limit 40 --random --seed 3
  uv run python scripts/inspect_plans.py data/cubicasa5k/high_quality_architectural-7731.png other.png
  uv run python scripts/inspect_plans.py --uploads 5                       # the 5 newest uploaded plans

Per plan one PNG with four panels, and a header with counts and flagged issues:
  1. rooms      room polygons filled by type (strong solid, weak dashed, unknown grey), borders, "id type area tier";
                implausible sizes / conflicts outlined red
  2. walls      wall map (solid dark, openings blue), wall centrelines (exterior red, interior blue), openings joining
                rooms, dimension lines used for the scale (orange)
  3. detections every graph item: YOLO doors / windows / fixtures with scores, CubiCasa net blobs, OCR text
                (room labels teal, other text grey)
  4. truth      CubiCasa SVG ground truth (rooms, doors, windows, fixtures) when the plan has one
Plus index.html (DATASET or uploads; plans sorted by number of issues) in --out (default dataset/runs/inspect/).
Counts compare the calibrated counting models (graph.counts) with the ground truth's counting rules.
"""

import argparse
import html
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from app.config import CACHE_DIR, DATASET_DIR  # noqa: E402

ROOM_COLOURS = {
    "bedroom": "#5b8def", "living": "#e0a43b", "kitchen": "#e0613b", "bathroom": "#3bb3c9", "wc": "#3b8fc9",
    "sauna": "#c9733b", "balcony": "#6dbb5f", "terrace": "#6dbb5f", "entry": "#9b8bd6", "closet": "#b07fc0",
    "storage": "#a0a0a0", "utility": "#5fb3a0", "dining": "#d6b13b", "study": "#7f9bc0", "technical": "#8a8a8a",
    "garage": "#8a8a8a", "stairs": "#b0906a", "basement": "#8a8a8a", "other": "#b5a58a", "unknown": "#7a8a80",
}
ITEM_COLOURS = {"door": "#dc1e1e", "window": "#1e88e5", "toilet": "#2e7d4f", "sink": "#00897b", "double_sink": "#00897b",
                "shower": "#8e24aa", "bathtub": "#8e24aa", "sauna_stove": "#ef6c00", "sauna_bench": "#ef6c00",
                "room": "#9e9e9e", "floor": "#bdbdbd"}
PANEL_W = 1000  # px per panel in the sheet
BIG_ROOM = 0.35  # of the image: probably a leak through a missing wall


def rgb(hex_: str, a: int = 255) -> tuple[int, int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a


def ascii_safe(s: str) -> str:
    """The built-in bitmap font has no ² or bullets."""
    return s.replace("²", "2").replace("•", "-").replace("…", "...").replace("×", "x")


def font(size: float):
    try:
        return ImageFont.load_default(size=max(9, round(size)))
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class Canvas:
    """The plan faded to the background, with an RGBA layer for translucent fills."""

    def __init__(self, img: Image.Image, fade: float = 0.55):
        base = img.convert("RGB")
        self.base = Image.blend(base, Image.new("RGB", base.size, "white"), fade).convert("RGBA")
        self.fill = Image.new("RGBA", base.size, (0, 0, 0, 0))
        self.top = Image.new("RGBA", base.size, (0, 0, 0, 0))
        self.f = ImageDraw.Draw(self.fill)
        self.d = ImageDraw.Draw(self.top)
        W, H = base.size
        self.lw = max(1, round(max(W, H) / 700))
        self.font = font(max(W, H) / 85)
        self.small = font(max(W, H) / 120)

    def text(self, xy, s, colour="#1f2a23", small=False):
        s = ascii_safe(s)
        f = self.small if small else self.font
        self.d.text(xy, s, fill=rgb(colour), font=f, stroke_width=max(1, self.lw), stroke_fill=(255, 255, 255, 230))

    def dashed(self, pts, colour, width, dash=8):
        import math

        pts = list(pts) + [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / dash))
            for k in range(0, n, 2):
                a, b = k / n, min(1, (k + 1) / n)
                self.d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                            fill=rgb(colour), width=width)

    def image(self) -> Image.Image:
        return Image.alpha_composite(Image.alpha_composite(self.base, self.fill), self.top).convert("RGB")


def panel_rooms(img, geo) -> Image.Image:
    c = Canvas(img)
    for rm in geo.rooms:
        col = ROOM_COLOURS.get(rm.type if rm.tier != "unknown" else "unknown", "#7a8a80")
        alpha = {"strong": 95, "weak": 45, "unknown": 0}[rm.tier]
        if len(rm.polygon) >= 3:
            c.f.polygon(rm.polygon, fill=rgb(col, alpha))
            bad = rm.size_check or rm.conflicts
            if rm.tier == "strong":
                c.d.polygon(rm.polygon, outline=rgb("#c62828" if bad else col), width=c.lw * (3 if bad else 2))
            else:
                c.dashed(rm.polygon, "#c62828" if bad else col, c.lw * 2)
    for rm in geo.rooms:
        x0, y0, x1, y1 = rm.bbox
        area = rm.labeled_area_m2 or rm.area_m2
        a = f" {area:.1f}m²" if area else (f" ~{rm.area_m2_range[0]:.0f}-{rm.area_m2_range[1]:.0f}m²" if rm.area_m2_range else "")
        tier = {"strong": "", "weak": "?", "unknown": "??"}[rm.tier]
        what = "+".join(rm.purposes) if len(rm.purposes) > 1 else (rm.type if rm.tier != "unknown" else "")
        c.text(((x0 + x1) / 2 - 20, (y0 + y1) / 2 - 8), f"{rm.id.replace('space-', '#')} {what}{tier}{a}")
    return c.image()


def panel_walls(img, geo) -> Image.Image:
    import numpy as np

    from app.cubicasa import layers

    c = Canvas(img, fade=0.8)
    try:
        cls, _ = layers.wall_classes(geo.image, Path(img.filename))
        a = np.zeros((*cls.shape, 4), np.uint8)
        a[cls == 1] = (38, 44, 58, 150)
        a[cls == 2] = (30, 136, 229, 170)
        c.fill = Image.fromarray(a, "RGBA")
        c.f = ImageDraw.Draw(c.fill)
    except Exception:  # noqa: BLE001  rooms from a PDF / no wall map
        pass
    for rm in geo.rooms:  # room borders
        if len(rm.polygon) >= 3:
            c.d.polygon(rm.polygon, outline=rgb("#6a1b9a", 160), width=c.lw)
    for sg in geo.wall_segments:
        c.d.line([tuple(sg.p0), tuple(sg.p1)], fill=rgb("#c0392b" if sg.exterior else "#2e6fb5"),
                 width=c.lw * (3 if sg.exterior else 2))
    for op in geo.openings:
        c.d.rectangle(op.bbox, outline=rgb("#00acc1"), width=c.lw)
        if op.symbol:
            c.text((op.bbox[0], op.bbox[3]), op.symbol[0].upper(), "#00838f", small=True)
    for p0, p1 in geo.room_cuts:  # cuts between two room labels in one space
        c.d.line([tuple(p0), tuple(p1)], fill=rgb("#e91e63"), width=c.lw * 3)
    if geo.scale and geo.scale.dimensions:
        for dm in geo.scale.dimensions:
            c.d.line([tuple(dm.p0), tuple(dm.p1)], fill=rgb("#d35400"), width=c.lw * 3)
            for p in (dm.p0, dm.p1):
                r = c.lw * 3
                c.d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=rgb("#d35400"))
            c.text(dm.bbox[:2], dm.text, "#d35400", small=True)
    return c.image()


def panel_detections(img, graph, tokens) -> Image.Image:
    c = Canvas(img)
    for t in tokens:  # OCR first (underneath)
        col = "#00897b" if t.room_type else "#9e9e9e"
        c.d.rectangle(t.bbox, outline=rgb(col, 200), width=1)
        c.text((t.bbox[0], t.bbox[3]), t.text[:14], col, small=True)
    for it in graph.items:
        if it.raw == "ocr" or it.category in ("floor",):
            continue
        key = it.category if it.category in ("door", "window", "room") else it.cls
        col = ITEM_COLOURS.get(key, "#43a047")
        net = it.raw != "detector" and not it.user_verified
        if it.category == "room":  # CubiCasa net room blobs: faint
            c.d.rectangle(it.bbox, outline=rgb(col, 110), width=1)
            continue
        if net:
            c.dashed([(it.bbox[0], it.bbox[1]), (it.bbox[2], it.bbox[1]), (it.bbox[2], it.bbox[3]), (it.bbox[0], it.bbox[3])],
                     col, c.lw, dash=4)
        else:
            c.d.rectangle(it.bbox, outline=rgb(col), width=c.lw * 2)
        name = (it.cls or it.category)[:8] + (f" {it.score:.2f}" if it.score is not None else "")
        c.text((it.bbox[0], it.bbox[1] - c.small.size - 2), name, col, small=True)
    return c.image()


def panel_truth(img, gt) -> Image.Image:
    c = Canvas(img)
    for it in gt.items:
        if it.category == "room" and it.polygon and len(it.polygon) >= 3:
            col = ROOM_COLOURS.get(it.cls, "#b5a58a")
            c.f.polygon(it.polygon, fill=rgb(col, 70))
            c.d.polygon(it.polygon, outline=rgb(col), width=c.lw * 2)
    for it in gt.items:
        if it.category == "room":
            x0, y0, x1, y1 = it.bbox
            c.text(((x0 + x1) / 2 - 20, (y0 + y1) / 2 - 8), f"{it.cls} {it.text or ''}"[:22])
        elif it.category in ("door", "window", "fixture"):
            key = it.category if it.category != "fixture" else it.cls
            if it.category == "fixture" and it.cls == "other":
                c.d.rectangle(it.bbox, outline=rgb("#bdbdbd"), width=1)
                continue
            c.d.rectangle(it.bbox, outline=rgb(ITEM_COLOURS.get(key, "#43a047")), width=c.lw * 2)
    return c.image()


def issues_and_counts(graph, geo, tokens, gt) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Flagged problems (most useful first) and (what, model, truth) count rows."""
    from app.cubicasa.answer import count

    issues: list[str] = []
    rows: list[tuple[str, str, str]] = []
    W, H = graph.width, graph.height
    for t, v in sorted(graph.counts.items()):
        g = count(gt, t).answer if gt else None
        rows.append((t, str(v), "" if g is None else str(g)))
        if g is not None and g != v:
            issues.append(f"count {t}: model {v}, truth {g}")
    raw = {}
    for it in graph.items:
        if it.raw == "detector":
            raw[it.cls or it.category] = raw.get(it.cls or it.category, 0) + 1
    rows.append(("YOLO boxes", ", ".join(f"{k} {v}" for k, v in sorted(raw.items())) or "none", ""))
    if geo is None:
        issues.append("no room geometry (wall model missing?)")
        return issues, rows
    tiers = {t: sum(r.tier == t for r in geo.rooms) for t in ("strong", "weak", "unknown")}
    rows.append(("rooms strong/weak/unknown", f"{tiers['strong']}/{tiers['weak']}/{tiers['unknown']}",
                 f"{sum(it.category == 'room' for it in gt.items)} spaces" if gt else ""))
    sc = geo.scale
    rows.append(("scale", f"{sc.source} {sc.m_per_px * 1000:.2f} mm/px" if sc else
                 ("symbols only (rough)" if geo.symbol_scale else "none"), ""))
    if not geo.rooms:
        issues.append("no rooms found")
    if sc is None:
        issues.append("no scale (no dimension lines / written areas read)")
    elif sc.warning:
        issues.append(f"scale: {sc.warning}")
    for rm in geo.rooms:
        if rm.area_px > BIG_ROOM * W * H:
            issues.append(f"{rm.id} covers {rm.area_px / (W * H):.0%} of the image (leak through a wall?)")
        if rm.conflicts:
            issues.append(f"{rm.id} {rm.type}: conflicting evidence {', '.join(rm.conflicts)}")
        if rm.size_check:
            issues.append(f"{rm.id} {rm.type}: {rm.size_check}")
        if rm.tier == "unknown" and any(lb.text.strip() for lb in rm.labels):
            issues.append(f"{rm.id} unknown though text inside: {', '.join(lb.text for lb in rm.labels)[:40]}")
    in_rooms = {tuple(lb.bbox) for rm in geo.rooms for lb in rm.labels}
    lost = [t.text for t in tokens if t.room_type and tuple(t.bbox) not in in_rooms]
    if lost:
        issues.append(f"room labels outside every room: {', '.join(lost[:6])}")
    if not any(t.room_type for t in tokens):
        issues.append("OCR read no room labels")
    if tiers["unknown"]:
        issues.append(f"{tiers['unknown']} spaces of unknown purpose")
    return issues, rows


def sheet(name: str, panels: list[tuple[str, Image.Image]], issues: list[str], rows) -> Image.Image:
    k = min(1.0, PANEL_W / panels[0][1].width)
    pw, ph = round(panels[0][1].width * k), round(panels[0][1].height * k)
    cols = 2
    nrows = (len(panels) + 1) // 2
    title, body, mono = font(22), font(15), font(14)
    head = 40 + 20 * max(len(rows), min(len(issues), 14)) + 10
    out = Image.new("RGB", (cols * pw + 10, head + nrows * (ph + 26)), "white")
    d = ImageDraw.Draw(out)
    d.text((10, 8), name, fill=(20, 20, 20), font=title)
    for i, (what, model, truth) in enumerate(rows):
        differs = truth and model.isdigit() and truth != model
        d.text((10, 40 + 20 * i), f"{what:<26} {model}" + (f"   truth {truth}" if truth else ""),
               fill=(160, 30, 30) if differs else (40, 40, 40), font=mono)
    for i, s in enumerate(issues[:14]):
        d.text((pw + 10, 40 + 20 * i), "- " + ascii_safe(s)[:110], fill=(160, 30, 30), font=body)
    if len(issues) > 14:
        d.text((pw + 10, 40 + 20 * 14), f"... {len(issues) - 14} more", fill=(160, 30, 30), font=body)
    for n, (label, im) in enumerate(panels):
        x, y = (n % cols) * pw + 5, head + (n // cols) * (ph + 26)
        d.text((x, y + 2), label, fill=(60, 60, 60), font=body)
        out.paste(im.resize((pw, ph), Image.LANCZOS), (x, y + 22))
    return out


def inspect(rel: str, path: Path):
    from app.cubicasa import layers
    from app.cubicasa.detect import graph_from_image
    from app.cubicasa.graph import get_graph
    from app.cubicasa.rooms import geometry_from_image
    from app.main import with_metrics

    img = Image.open(path)
    img.filename = str(path)
    graph = get_graph(rel, "model") if rel.startswith("data/") else graph_from_image(rel, path)
    tokens = layers.text(rel, path).tokens
    try:
        geo = with_metrics(geometry_from_image(rel, path, graph), None)
    except FileNotFoundError:
        geo = None
    gt = None
    svg = CACHE_DIR / "svg" / f"{Path(rel).stem}.json"
    if svg.is_file():
        from app.models import PlanGraph

        gt = PlanGraph.model_validate_json(svg.read_text(encoding="utf-8"))
    panels = []
    if geo is not None:
        panels += [("1 rooms (fill = type; dashed = weak; grey = unknown; red = size / conflict)", panel_rooms(img, geo)),
                   ("2 walls, centrelines (red exterior), openings (cyan), room cuts (pink), scale lines (orange)",
                    panel_walls(img, geo))]
    panels.append(("3 detections: YOLO solid, net dashed, OCR thin (teal = room label)", panel_detections(img, graph, tokens)))
    panels.append(("4 ground truth (SVG)", panel_truth(img, gt)) if gt else ("4 original", img.convert("RGB")))
    issues, rows = issues_and_counts(graph, geo, tokens, gt)
    return sheet(rel, panels, issues, rows), issues


def pick(args) -> list[tuple[str, Path]]:
    if args.uploads:
        ups = sorted((CACHE_DIR / "uploads").glob("*.png"), key=lambda p: -p.stat().st_mtime)[: args.uploads]
        return [(f"uploads/{p.name}", p) for p in ups]
    if args.plans:
        out = []
        for s in args.plans:
            p = Path(s)
            if p.is_file() and not s.startswith("data/"):
                out.append((f"inspect/{p.name}", p.resolve()))
            else:
                out.append((s, DATASET_DIR / s))
        return out
    imgs = sorted((DATASET_DIR / "data" / "cubicasa5k").glob("*.png"))
    if args.random:
        random.Random(args.seed).shuffle(imgs)
        imgs = imgs[: args.limit]
    else:
        imgs = imgs[:: max(1, len(imgs) // args.limit)][: args.limit]
    return [(f"data/cubicasa5k/{p.name}", p) for p in imgs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="*", help="dataset-relative (data/cubicasa5k/x.png) or image file paths")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uploads", type=int, default=0, help="inspect the N newest uploaded plans instead")
    ap.add_argument("--out", type=Path, default=CACHE_DIR.parent / "runs" / "inspect")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for rel, path in pick(a):
        t = time.time()
        try:
            im, issues = inspect(rel, path)
        except Exception as e:  # noqa: BLE001  keep going, report it
            print(f"FAILED {rel}: {e!r}", file=sys.stderr)
            results.append((rel, None, [f"failed: {e!r}"]))
            continue
        png = a.out / f"{Path(rel).stem}.png"
        im.save(png, optimize=True)
        results.append((rel, png.name, issues))
        print(f"{rel}: {len(issues)} issues, {time.time() - t:.1f} s -> {png}", file=sys.stderr)
    results.sort(key=lambda r: -len(r[2]))
    rows = "\n".join(f'<tr><td><a href="#{html.escape(Path(r).stem)}">{html.escape(r)}</a></td><td>{len(i)}</td>'
                     f'<td>{html.escape("; ".join(i[:3]))}</td></tr>' for r, _, i in results)
    blocks = "\n".join(f'<h2 id="{html.escape(Path(r).stem)}">{html.escape(r)}</h2>'
                       + (f'<a href="{png}"><img loading="lazy" src="{png}"></a>' if png else "<p>failed</p>")
                       for r, png, _ in results)
    (a.out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Plan inspection</title>
<style>body{{font:14px system-ui;margin:16px}}img{{max-width:100%;border:1px solid #ddd}}td{{padding:2px 10px;vertical-align:top}}
tr:nth-child(even){{background:#f4f4f4}}</style><h1>Plan inspection ({len(results)} plans, most issues first)</h1>
<table><tr><th>plan</th><th>issues</th><th>first issues</th></tr>{rows}</table>{blocks}""", encoding="utf-8")
    print(f"\n{a.out / 'index.html'}")


if __name__ == "__main__":
    main()
