"""Fit per-template count models on cached features and evaluate.

  python -m app.cubicasa.fit                 # fit on train, select on val, report val (and test with --test)
  python -m app.cubicasa.fit --test --final  # refit chosen models on train+val, report test, save

Candidates per template:
  rule  : best single evidence feature + integer offset (fully transparent)
  hgb_c : gradient-boosted classifier over the count
  hgb_p : gradient-boosted Poisson regressor, rounded
"""

import argparse
import json
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np

from app.config import CACHE_DIR, MODELS_DIR
from app.cubicasa.features import feature_paths
from app.router import route

warnings.filterwarnings("ignore")
TABLE = CACHE_DIR / "features_table.jsonl"
FUSION_PATH = MODELS_DIR / "cubicasa-fusion.joblib"
FEATURE_VERSION = 10
TEMPLATES = ["windows", "doors", "floors", "wc_seats", "sinks", "showers_baths", "saunas", "kitchens",
             "bedrooms", "bathrooms", "wc_rooms", "balconies"]


def _feat(rel: str):
    from app.cubicasa.fuse import image_features

    try:
        return rel, image_features(rel)
    except Exception as e:  # noqa: BLE001
        return rel, {"__error__": str(e)}


def build_table(paths: list[str]) -> dict[str, dict]:
    table: dict[str, dict] = {}
    if TABLE.exists():
        for line in TABLE.read_text().splitlines():
            r = json.loads(line)
            if r.get("v") == FEATURE_VERSION:
                table[r["path"]] = r["f"]
    from app.cubicasa.features import _npz_ok

    todo = [p for p in paths if p not in table and feature_paths(p)[1].exists() and _npz_ok(feature_paths(p)[0])]
    if todo:
        with ProcessPoolExecutor(4) as ex, TABLE.open("a") as out:
            for rel, f in ex.map(_feat, todo, chunksize=8):
                if "__error__" in f:
                    print("feature error", rel, f["__error__"])
                    continue
                table[rel] = f
                out.write(json.dumps({"path": rel, "v": FEATURE_VERSION, "f": f}) + "\n")
    _add_det_features(table)
    return table


def _add_det_features(table: dict[str, dict]) -> None:
    """Merge symbol-detector features (cached separately, see app.cubicasa.symdet) into every row.

    All-or-nothing: rows without cached detections would read as zero counts and poison the fit."""
    from app.cubicasa import symdet

    if not symdet.available():
        return
    dets = {p: symdet.cached(p) for p in table}
    missing = [p for p, d in dets.items() if d is None]
    if missing:
        print(f"symbol detector: {len(missing)}/{len(table)} images lack cached detections "
              f"(run: python -m app.cubicasa.symdet); fitting WITHOUT det_* features")
        return
    for p, d in dets.items():
        table[p] = {**table[p], **symdet.det_features(d)}
    print(f"symbol detector: added det_* features for {len(table)} images ({symdet.fingerprint()})")


def dataset(table, qs, paths, template):
    keys = None
    X, y, P = [], [], []
    for p in paths:
        if p in table and (p, template) in qs:
            f = table[p]
            keys = keys or sorted(f)
            X.append([f.get(k, 0.0) for k in keys])
            y.append(qs[(p, template)])
            P.append(p)
    return np.array(X, dtype=np.float64), np.array(y), keys, P


class RuleModel:
    """count = round(scale * feature + offset), choosing feature/scale/offset with best exact match.

    The scale corrects detectors that miss a roughly constant fraction of symbols (e.g. merged windows)."""

    SCALES = np.round(np.arange(0.8, 1.41, 0.05), 2)
    OFFSETS = np.arange(-2.0, 2.01, 0.25)

    def fit(self, X, y, keys):
        best = (-1.0, 0, 1.0, 0.0)
        for j in range(X.shape[1]):
            x = X[:, j]
            if np.all(x == x[0]):
                continue
            # coarse: scale 1, integer offsets; then refine scale/offset on the best few features
            for off in range(-3, 4):
                acc = np.mean(np.rint(x) + off == y)
                if acc > best[0]:
                    best = (acc, j, 1.0, float(off))
        cands = {best[1]}
        accs = [(np.max([np.mean(np.rint(X[:, j]) + o == y) for o in range(-3, 4)]), j) for j in range(X.shape[1])]
        cands |= {j for _, j in sorted(accs, reverse=True)[:15]}
        for j in cands:
            x = X[:, j]
            for a in self.SCALES:
                for b in self.OFFSETS:
                    acc = np.mean(np.maximum(0, np.rint(a * x + b)) == y)
                    if acc > best[0] + 1e-9:
                        best = (acc, j, float(a), float(b))
        _, self.j, self.a, self.b = best
        self.feature = keys[self.j]
        return self

    def predict(self, X):
        return np.maximum(0, np.rint(self.a * X[:, self.j] + self.b)).astype(int)


class RuleResidModel:
    """Best rule feature as the base count, plus a small boosted classifier over the integer correction."""

    def fit(self, X, y, keys):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.rule = RuleModel().fit(X, y, keys)
        resid = np.clip(y - self.rule.predict(X), -2, 2)
        self.const = None
        if len(np.unique(resid)) < 2:
            self.const = int(resid[0])
            return self
        self.m = HistGradientBoostingClassifier(**{**HGB_KW, "early_stopping": False, "max_iter": 60,
                                                    "l2_regularization": 3.0, "min_samples_leaf": 15})
        self.m.fit(X, resid)
        return self

    def predict(self, X):
        corr = self.const if self.const is not None else self.m.predict(X).astype(int)
        return np.maximum(0, self.rule.predict(X) + corr)


def make_candidates():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    # import via the package path so pickled models load outside `python -m app.cubicasa.fit`
    from app.cubicasa.fit import RuleModel, RuleResidModel

    return {
        "rule": lambda: RuleModel(),
        "resid": lambda: RuleResidModel(),
        "hgb_c": lambda: HistGradientBoostingClassifier(**{**HGB_KW, "early_stopping": False, "max_iter": 80}),
        "hgb_p": lambda: HistGradientBoostingRegressor(loss="poisson", **HGB_KW),
    }


HGB_KW = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=10,
              max_features=0.3, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=0)


def fit_predict(name, factory, Xtr, ytr, keys, Xte):
    m = factory()
    if name in ("rule", "resid"):
        m.fit(Xtr, ytr, keys)
        return m, m.predict(Xte)
    if name == "hgb_c":
        cap = min(int(np.percentile(ytr, 99.5)), 6)  # many-count templates are left to hgb_p / rule
        m.fit(Xtr, np.minimum(ytr, cap))
        return m, m.predict(Xte).astype(int)
    m.fit(Xtr, ytr)
    return m, np.maximum(0, np.rint(m.predict(Xte))).astype(int)


def main() -> None:
    from app.eval import load_questions, split_of

    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="also report on the test split")
    ap.add_argument("--final", action="store_true", help="refit chosen models and save them")
    ap.add_argument("--fit-on", choices=["val", "train", "train+val"], default="val",
                    help="train images are biased (the CubiCasa net was trained on them); default fits on val only")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    qs = {}
    for q in load_questions():
        qs.setdefault((q["path"], route(q["question"])), q["answer"])
    paths = sorted({p for p, _ in qs})
    by_split = defaultdict(list)
    for p in paths:
        by_split[split_of(p)].append(p)
    table = build_table(paths)
    tr = [p for p in by_split["train"] if p in table]
    va = [p for p in by_split["val"] if p in table]
    te = [p for p in by_split["test"] if p in table]
    print(f"features: train={len(tr)} val={len(va)} test={len(te)}  fit-on={args.fit_on}")

    cands = make_candidates()
    saved = {}
    tot_v, hit_v, tot_t, hit_t = Counter(), Counter(), Counter(), Counter()
    print(f"{'template':<14}{'n_val':>6}" + "".join(f"{c:>8}" for c in cands) + "   chosen")
    for t in TEMPLATES:
        Xva, yva, keys, _ = dataset(table, qs, va, t)
        Xtr, ytr, _, _ = dataset(table, qs, tr, t)
        if len(yva) < 10:
            continue
        accs = {}
        # many-valued counts: the capped classifier / residual model never win there; skip to save time
        names = ["rule", "hgb_p"] if t in ("windows", "doors") else ["rule", "hgb_c", "hgb_p"]
        for name in names:
            fac = cands[name]
            # K-fold over val; train images are optionally added to every fold's training set
            pred = np.zeros_like(yva)
            folds = np.arange(len(yva)) % args.folds
            for k in range(args.folds):
                trn, tst = folds != k, folds == k
                Xf, yf = Xva[trn], yva[trn]
                if args.fit_on == "train+val" and len(ytr):
                    Xf, yf = np.vstack([Xtr, Xf]), np.concatenate([ytr, yf])
                elif args.fit_on == "train" and len(ytr):
                    Xf, yf = Xtr, ytr
                _, pred[tst] = fit_predict(name, fac, Xf, yf, keys, Xva[tst])
            accs[name] = float(np.mean(pred == yva))
        best = max(accs, key=accs.get)
        tot_v[t], hit_v[t] = len(yva), round(accs[best] * len(yva))
        print(f"{t:<14}{len(yva):>6}" + "".join(f"{accs[c]:>8.1%}" if c in accs else f"{'-':>8}" for c in cands)
              + f"   {best}", flush=True)
        if args.test or args.final:
            Xfit, yfit = Xva, yva
            if args.fit_on == "train+val" and len(ytr):
                Xfit, yfit = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])
            elif args.fit_on == "train" and len(ytr):
                Xfit, yfit = Xtr, ytr
            m, _ = fit_predict(best, cands[best], Xfit, yfit, keys, Xva[:1])
            Xte, yte, _, _ = dataset(table, qs, te, t)
            if len(yte):
                pt = _predict(m, best, Xte)
                tot_t[t], hit_t[t] = len(yte), int(np.sum(pt == yte))
            saved[t] = {"model": m, "kind": best, "keys": keys}
    print(f"VAL (CV) overall exact: {sum(hit_v.values()) / max(1, sum(tot_v.values())):.1%} over {sum(tot_v.values())} questions")
    if tot_t:
        print("TEST per template: " + ", ".join(f"{t} {hit_t[t] / tot_t[t]:.1%}" for t in TEMPLATES if tot_t[t]))
        print(f"TEST overall exact: {sum(hit_t.values()) / sum(tot_t.values()):.1%} over {sum(tot_t.values())} questions")
    if args.final:
        FUSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        from app.cubicasa import symdet

        uses_det = any(k.startswith("det_") for spec in saved.values() for k in spec["keys"])
        joblib.dump({"version": FEATURE_VERSION, "symdet": symdet.fingerprint() if uses_det else None,
                     "templates": saved}, FUSION_PATH)
        print("saved", FUSION_PATH)


def _predict(m, kind, X):
    if kind in ("rule", "resid"):
        return m.predict(X)
    if kind == "hgb_c":
        return m.predict(X).astype(int)
    return np.maximum(0, np.rint(m.predict(X))).astype(int)


if __name__ == "__main__":
    main()
