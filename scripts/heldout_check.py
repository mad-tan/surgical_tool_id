"""End-to-end held-out check of the SHIPPED path (train.py -> checkpoint -> predict.py).

1. Train with the shipped recipe while leaving one CV fold's videos out:
       python train.py --exp r3g_mbv3_ft_colour_affine --seeds 0 --exclude-fold 0 \\
           --out runs/heldout/fold0_seed0.pt
2. Run this script: it copies that fold's images into a temporary folder laid
   out like the graders' data (class subfolders, original filenames), runs
   `python predict.py --data-dir ... --out ...` as a subprocess exactly as the
   graders will, scores the CSV against the folder labels, and compares with
   the CV out-of-fold number for the same fold and seed.

       python scripts/heldout_check.py --fold 0 --seed 0 --checkpoint runs/heldout/fold0_seed0.pt

Rows are matched to images by position (predict.py lists class folders and
files in sorted order), because in a filename-keyed CSV two crops of the same
frame that sit in different class folders cannot be told apart.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolid.data import CLASSES, index_dataset  # noqa: E402
from toolid.folds import split  # noqa: E402
from toolid.metrics import fmt, report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--exp", default="r3g_mbv3_ft_colour_affine")
    args = ap.parse_args()

    _, test = split(index_dataset(), args.fold)
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "heldout"
        for s in test:  # graders' layout: <dir>/<class>/<filename>; split folders merged
            d = data / CLASSES[s.label]
            d.mkdir(parents=True, exist_ok=True)
            assert not (d / s.path.name).exists()
            shutil.copy2(s.path, d / s.path.name)
        out = Path(tmp) / "preds.csv"
        proc = subprocess.run([sys.executable, "predict.py", "--data-dir", str(data), "--out", str(out),
                               "--checkpoint", str(args.checkpoint.resolve())],
                              cwd=ROOT, capture_output=True, text=True)
        print(proc.stdout.strip(), proc.stderr.strip(), sep="\n")
        if proc.returncode != 0:
            sys.exit(f"predict.py failed with exit {proc.returncode}")
        rows = list(csv.reader(open(out)))
    assert rows[0] == ["filename", "predicted_class"]
    rows = rows[1:]
    # same order predict.py uses: sorted class folders, sorted files within each
    order = sorted(test, key=lambda s: (CLASSES[s.label], s.path.name))
    assert len(rows) == len(order), f"{len(rows)} rows for {len(order)} images"
    assert [r[0] for r in rows] == [s.path.name for s in order]
    y = [s.label for s in order]
    pred = [CLASSES.index(r[1]) for r in rows]
    r = report(y, pred)
    print(fmt(r, f"predict.py, fold {args.fold} held out"))

    z = np.load(ROOT / "results" / "cv" / f"{args.exp}_oof.npz")
    seeds = z["seeds"].tolist()
    m = z["folds"] == args.fold
    cv = report(z["labels"][m], z["logits"][seeds.index(args.seed)][m].argmax(1))
    print(fmt(cv, f"CV OOF, fold {args.fold}, seed {args.seed} (GPU)"))
    res = {"fold": args.fold, "seed": args.seed, "n": len(y), "predict_py": r, "cv_oof_same_fold_seed": cv,
           "checkpoint": str(args.checkpoint)}
    outp = ROOT / "results" / f"heldout_check_fold{args.fold}_seed{args.seed}.json"
    json.dump(res, open(outp, "w"), indent=1)
    print("wrote", outp)


if __name__ == "__main__":
    main()
