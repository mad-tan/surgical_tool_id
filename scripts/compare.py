"""Paired comparison of two CV experiments (same folds, same seeds).

    python scripts/compare.py CHILD PARENT [--boot 2000]

Reports, for macro-F1 averaged over seeds:
  * the paired per-seed differences (child - parent, seed by seed)
  * a 95% video-bootstrap interval of the difference: resample the 10
    videos with replacement, recompute both experiments' macro-F1 on the
    resampled images, take the difference. This measures how much the
    verdict depends on WHICH videos we happen to have, which is the main
    uncertainty for a held-out set of unseen videos.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(exp):
    z = np.load(ROOT / "results" / "cv" / f"{exp}_oof.npz")
    return z["logits"], z["labels"], z["keys"]


def macro(y, pred):
    return f1_score(y, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)


def compare(child, parent, n_boot=2000, seed=0):
    lc, y, keys = load(child)
    lp, y2, keys2 = load(parent)
    assert (keys == keys2).all() and (y == y2).all() and lc.shape[0] == lp.shape[0]
    pc, pp = lc.argmax(2), lp.argmax(2)
    per_seed = [macro(y, pc[i]) - macro(y, pp[i]) for i in range(len(pc))]
    videos = np.array([k.split("/")[-1].split("_")[0] for k in keys])
    uv = np.unique(videos)
    idx_of = {v: np.where(videos == v)[0] for v in uv}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[v] for v in rng.choice(uv, len(uv), replace=True)])
        diffs.append(np.mean([macro(y[idx], pc[i][idx]) - macro(y[idx], pp[i][idx]) for i in range(len(pc))]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(per_seed)), per_seed, float(lo), float(hi), float(np.mean(np.array(diffs) > 0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("child")
    ap.add_argument("parent")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    m, ps, lo, hi, frac = compare(a.child, a.parent, a.boot)
    print(f"{a.child} - {a.parent}: mean {m:+.3f}  per-seed {[round(x, 3) for x in ps]}  "
          f"video-bootstrap 95% [{lo:+.3f}, {hi:+.3f}]  P(diff>0) {frac:.2f}")
