"""Is colour a class signal or a per-video nuisance?

For every image: mean and std of R,G,B and chromaticity (channel share of
the mean, i.e. colour independent of brightness). Reports:
  * eta^2: share of each statistic's variance explained by video vs by class
  * per-video and per-class mean colour
  * macro-F1 of a class-balanced logistic regression on these 9 numbers
    under the video-grouped folds (how much class signal global colour holds)
  * 10-way video-id accuracy from the same numbers (how strong the video
    "signature" is)
Usage: python scripts/colour_analysis.py
"""
import collections
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolid.data import CLASSES, index_dataset  # noqa: E402
from toolid.folds import N_FOLDS, VIDEO_TO_FOLD  # noqa: E402

NAMES = ["meanR", "meanG", "meanB", "stdR", "stdG", "stdB", "chromR", "chromG", "chromB"]


def eta2(x, g):
    tot = ((x - x.mean()) ** 2).sum()
    return sum(((x[g == k].mean() - x.mean()) ** 2) * (g == k).sum() for k in np.unique(g)) / tot


def main():
    S = index_dataset()
    X, V, Y = [], [], []
    for s in S:
        a = np.asarray(Image.open(s.path).convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255
        m, sd = a.mean(0), a.std(0)
        X.append(np.r_[m, sd, m / m.sum()])
        V.append(s.video)
        Y.append(s.label)
    X, V, Y = np.array(X), np.array(V), np.array(Y)

    print("variance explained (eta^2)   by video   by class")
    for i, n in enumerate(NAMES):
        print(f"  {n:7s}                      {eta2(X[:, i], V):.2f}       {eta2(X[:, i], Y):.2f}")
    print("\nper-video mean RGB")
    for v in sorted(set(V)):
        k = V == v
        print(f"  {v}: n={k.sum():3d} mean={np.round(X[k, :3].mean(0), 2)}")
    print("\nper-class mean RGB / std RGB")
    for c, name in enumerate(CLASSES):
        k = Y == c
        print(f"  {name:8s} mean={np.round(X[k, :3].mean(0), 2)} std={np.round(X[k, 3:6].mean(0), 2)}")

    folds = np.array([VIDEO_TO_FOLD[v] for v in V])
    pred = np.zeros_like(Y)
    for f in range(N_FOLDS):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[folds != f], Y[folds != f])
        pred[folds == f] = clf.predict(X[folds == f])
    print(f"\nclass from 9 colour stats (video-grouped CV): macro-F1 {f1_score(Y, pred, average='macro'):.3f} "
          f"per-class {dict(zip(CLASSES, np.round(f1_score(Y, pred, average=None), 2)))}")
    pv = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, V,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0))
    chance = max(collections.Counter(V).values()) / len(V)
    print(f"video id from the same stats (random 5-fold): accuracy {(pv == V).mean():.3f} (majority {chance:.3f})")


if __name__ == "__main__":
    main()
