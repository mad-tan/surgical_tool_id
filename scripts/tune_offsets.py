"""Rung 2c: nested per-class logit offsets on a finished CV experiment's
out-of-fold logits (results/cv/<exp>_oof.npz). Writes
results/cv/<exp>+offsets.json in the same format as scripts/cv.py.

    python scripts/tune_offsets.py --exp r1b_smallcnn_fixed_aug
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolid.data import CLASSES  # noqa: E402
from toolid.metrics import cv_report, fmt  # noqa: E402
from toolid.offsets import nested_offsets_predictions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    d = np.load(ROOT / "results" / "cv" / f"{args.exp}_oof.npz")
    y, folds = d["labels"], d["folds"]
    runs, offsets = {}, {}
    for seed, lg in zip(d["seeds"].tolist(), d["logits"]):
        pred, used = nested_offsets_predictions(y, lg, folds)
        runs[str(seed)] = cv_report(y, pred, folds)
        offsets[str(seed)] = used
        print(fmt(runs[str(seed)], f"{args.exp}+offsets seed {seed}"))
    mf = [r["macro_f1"] for r in runs.values()]
    summary = {"macro_f1_mean": float(np.mean(mf)), "macro_f1_std": float(np.std(mf)),
               "per_class_f1_mean": {c: float(np.mean([r["per_class_f1"][c] for r in runs.values()])) for c in CLASSES},
               "per_class_f1_std": {c: float(np.std([r["per_class_f1"][c] for r in runs.values()])) for c in CLASSES},
               "accuracy_mean": float(np.mean([r["accuracy"] for r in runs.values()]))}
    print(f"{args.exp}+offsets: macro-F1 {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f} | "
          + " ".join(f"{c} {summary['per_class_f1_mean'][c]:.3f}" for c in CLASSES))
    with open(ROOT / "results" / "cv" / f"{args.exp}+offsets.json", "w") as fh:
        json.dump({"experiment": f"{args.exp}+offsets", "base": args.exp, "summary": summary, "runs": runs,
                   "offsets_per_seed_per_fold": offsets, "classes": CLASSES}, fh, indent=1)


if __name__ == "__main__":
    main()
