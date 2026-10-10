"""Rung 3a: rank ImageNet backbones with frozen features + logistic regression
under the same video-grouped folds.

For each backbone x geometry (native 86x128, x2 = 172x256): extract pooled
penultimate features once per image (no augmentation, ImageNet
normalisation), then per outer fold fit StandardScaler + LogisticRegression.
C is chosen by an INNER video-grouped CV on the four training folds
(grid 0.01/0.1/1/10, by macro-F1), so the test fold never influences it.
Run with and without class_weight="balanced".

Feature extraction and lbfgs are deterministic, so the "seeds" axis is
degenerate here (std = 0); reported as a single run.

    python scripts/linear_probe.py --arch resnet18 --geometry native
    python scripts/linear_probe.py --summary
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolid.data import CLASSES, index_dataset  # noqa: E402
from toolid.folds import fold_of  # noqa: E402
from toolid.metrics import cv_report, fmt  # noqa: E402
from toolid.models import feature_extractor  # noqa: E402
from toolid.preprocess import load_rgb, prepare, stack_uint8  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0)
OUT = ROOT / "results" / "cv"


@torch.no_grad()
def extract(arch, geometry, samples):
    cache = ROOT / "runs" / "features" / f"{arch}_{geometry}.npy"
    if cache.exists():
        return np.load(cache)
    m = feature_extractor(arch)
    x = stack_uint8([load_rgb(s.path) for s in samples])
    feats = np.concatenate([m(prepare(x[i:i + 64], {"geometry": geometry})).numpy() for i in range(0, len(x), 64)])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)
    return feats


def clf(C, balanced):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000,
                                                              class_weight="balanced" if balanced else None))


def probe(X, y, folds, balanced):
    pred, chosen = np.empty_like(y), []
    for f in np.unique(folds):
        tr = folds != f
        scores = {}
        for C in C_GRID:  # inner grouped CV over the remaining folds
            p = np.empty(tr.sum(), dtype=int)
            for g in np.unique(folds[tr]):
                itr, ite = folds[tr] != g, folds[tr] == g
                p[ite] = clf(C, balanced).fit(X[tr][itr], y[tr][itr]).predict(X[tr][ite])
            scores[C] = f1_score(y[tr], p, average="macro")
        C = max(scores, key=scores.get)
        chosen.append(C)
        pred[folds == f] = clf(C, balanced).fit(X[tr], y[tr]).predict(X[folds == f])
    return pred, chosen


def run(arch, geometry):
    samples = index_dataset()
    y = np.array([s.label for s in samples])
    folds = np.array([fold_of(s) for s in samples])
    t0 = time.time()
    X = extract(arch, geometry, samples)
    print(f"{arch} {geometry}: features {X.shape} in {time.time() - t0:.0f}s")
    for balanced in (False, True):
        name = f"r3a_probe_{arch}_{geometry}{'_balanced' if balanced else ''}"
        pred, chosen = probe(X, y, folds, balanced)
        r = cv_report(y, pred, folds)
        print(fmt(r, name), " C per fold:", chosen)
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / f"{name}.json", "w") as fh:
            json.dump({"experiment": name, "arch": arch, "geometry": geometry, "class_weight_balanced": balanced,
                       "C_per_fold": chosen, "summary": {"macro_f1_mean": r["macro_f1"], "macro_f1_std": 0.0,
                                                         "per_class_f1_mean": r["per_class_f1"]},
                       "report": r, "note": "deterministic: single run"}, fh, indent=1)


def summary():
    rows = []
    for p in sorted(OUT.glob("r3a_probe_*.json")):
        d = json.load(open(p))
        pc = d["summary"]["per_class_f1_mean"]
        rows.append((d["summary"]["macro_f1_mean"], d["experiment"], pc))
    for mf, name, pc in sorted(rows, reverse=True):
        print(f"{name:45s} macroF1 {mf:.3f} | " + " ".join(f"{c} {pc[c]:.3f}" for c in CLASSES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch")
    ap.add_argument("--geometry", default="native")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run(a.arch, a.geometry)
