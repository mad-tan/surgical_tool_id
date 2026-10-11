"""Step 4: the honest "before" number for the shipped pipeline.

Part A (--part shipped): score checkpoints/model_best.pt under the video-grouped
folds, using predict.py's own preprocessing code. This model was trained on
ALL 1402 images (train + validation pooled via labels.csv), so every number
here is in-sample: it measures fit, not generalisation. Reported twice:
with predict.py's shipped index->name mapping, and with the mapping the
checkpoint was actually trained with (alphabetical).

Part B (--part retrain): retrain the same SmallCNN with its original recipe
(legacy/cnn_baseline_v2.py main(): grayscale 128x128 /255, batch size 2,
Adam 1e-3, MSE between softmax and one-hot, <=30 epochs, early stopping on
TRAINING loss with patience 3, keep the lowest-train-loss weights) inside
each fold, and predict the held-out videos. Label source:
  --labels csv     labels.csv looked up by filename, exactly as the original
                   (4% flipped labels; the 26 duplicate names resolve to one image)
  --labels folder  class-folder labels keyed by path (isolates the label-noise cost)

Usage:
  python scripts/baseline_legacy.py --part shipped
  python scripts/baseline_legacy.py --part retrain --labels csv --seeds 0 1 2
  (optional --budget SECS --shard i --n-shards k: run k shards in parallel,
   pause after SECS and resume on re-run; per-epoch state in runs/baseline_legacy/)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolid.data import CLASSES, DATA_ROOT, index_dataset  # noqa: E402
from toolid.folds import N_FOLDS, fold_of, split  # noqa: E402
from toolid.metrics import cv_report, fmt  # noqa: E402
from toolid.models import SmallCNN  # noqa: E402  (same architecture/state-dict keys as legacy/cnn_baseline_v2.py)

# The original code, labels.csv and model_best.pt were removed from the working
# tree in the cleanup; they are read from the commit that imported the zip
# (needs full git history: a shallow clone will not have it).
ORIGINAL_COMMIT = "e2f05e1200c017ad27719259bc9f9332dbdcd0e6"


def original_file(relpath):
    out = ROOT / "runs" / "original" / relpath
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(["git", "show", f"{ORIGINAL_COMMIT}:{relpath}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        out.write_bytes(blob)
    return out


# The original predict.py wrote SHIPPED_NAMES[argmax] (data_utils.py:15 at the
# original commit); the checkpoint's output index i actually means CLASSES[i].
SHIPPED_NAMES = ["grasper", "hook", "clipper", "scissor"]
SHIPPED_TO_TRUE_IDX = np.array([CLASSES.index(n) for n in SHIPPED_NAMES])
RESULTS = ROOT / "results"
torch.set_num_threads(os.cpu_count() or 4)


def legacy_tensor(path):
    """Exactly the legacy/predict.py preprocessing (grayscale, 128x128, /255,
    replicated to 3 channels), vectorised with numpy instead of getdata()."""
    img = Image.open(path).convert("L").resize((128, 128))
    t = torch.from_numpy(np.asarray(img, dtype=np.float32)).view(1, 128, 128) / 255.0
    return t.repeat(3, 1, 1)


def preload(samples):
    return torch.stack([legacy_tensor(s.path) for s in samples])


def predict_idx(model, x, bs=256):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)]).numpy()


def both_reports(samples, pred_idx):
    """pred_idx are model output indices (model index i == CLASSES[i])."""
    y = [s.label for s in samples]
    folds = [fold_of(s) for s in samples]
    correct = cv_report(y, pred_idx, folds)
    # what predict.py would have written: name SHIPPED_NAMES[i] -> true index of that name
    as_shipped = cv_report(y, SHIPPED_TO_TRUE_IDX[pred_idx], folds)
    return as_shipped, correct


def part_shipped(samples):
    # legacy_tensor() was asserted identical to the original predict.py
    # InferenceDataset preprocessing when this script was first run (commit 91b7d66).
    model = SmallCNN(4)
    model.load_state_dict(torch.load(original_file("checkpoints/model_best.pt"), map_location="cpu",
                                     weights_only=True))
    pred = predict_idx(model, preload(samples))
    np.savez_compressed(RESULTS / "baseline_shipped_checkpoint_oof.npz", pred=pred[None],
                        labels=np.array([s.label for s in samples]), folds=np.array([fold_of(s) for s in samples]),
                        keys=np.array([s.key for s in samples]), seeds=np.array([0]))
    as_shipped, correct = both_reports(samples, pred)
    print(fmt(as_shipped, "A shipped ckpt, shipped mapping (IN-SAMPLE)"))
    print(fmt(correct, "A shipped ckpt, correct mapping (IN-SAMPLE)"))
    return {"as_shipped_mapping": as_shipped, "correct_mapping": correct,
            "note": "in-sample: checkpoint was trained on all 1402 images"}


def csv_labels():
    """Reproduce the legacy label lookup: labels.csv rows, filename -> last path
    found while walking train/ then validation/ (os.listdir order)."""
    with open(original_file("labels.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    locate = {}
    for sp in ("train", "validation"):
        for cls in os.listdir(DATA_ROOT / sp):
            d = DATA_ROOT / sp / cls
            if d.is_dir():
                for f in os.listdir(d):
                    locate[f] = d / f
    return [(locate[r["filename"]], CLASSES.index(r["label"])) for r in rows]


def _job_path(label_src, seed, fold):
    return ROOT / "runs" / "baseline_legacy" / f"{label_src}_s{seed}_f{fold}.pt"


def train_legacy_resumable(x, y, seed, state_path, deadline, max_epochs=30, patience=3):
    """Original recipe, checkpointed after every epoch (model, optimiser,
    early-stopping counters and RNG state), so a run that is interrupted and
    resumed is bit-identical to an uninterrupted one. Returns the model with
    the lowest-training-loss weights once finished, or None if the deadline hit."""
    torch.manual_seed(seed)  # before building the model: seeds the weight init
    model = SmallCNN(4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if state_path.exists():
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng"])
    else:
        st = {"epoch": 0, "best": float("inf"), "best_state": None, "left": patience,
              "done": False, "secs_per_epoch": None}
    crit = nn.MSELoss()
    onehot = torch.eye(4)[y]
    while not st["done"]:
        spe = st["secs_per_epoch"] or 0
        if time.time() + spe * 1.2 > deadline:
            return None
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 2):  # batch_size=2, shuffle=True
            idx = perm[i:i + 2]
            opt.zero_grad()
            loss = crit(torch.softmax(model(x[idx]), dim=1), onehot[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        ep_loss = total / len(x)
        st["epoch"] += 1
        if ep_loss < st["best"] - 1e-4:  # early stopping on TRAINING loss, as in the original
            st["best"], st["left"] = ep_loss, patience
            st["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            st["left"] -= 1
        st["done"] = st["left"] == 0 or st["epoch"] >= max_epochs
        st["secs_per_epoch"] = time.time() - t0
        st.update(model=model.state_dict(), opt=opt.state_dict(), rng=torch.get_rng_state())
        tmp = state_path.with_suffix(".tmp")
        torch.save(st, tmp)
        tmp.replace(state_path)
    model.load_state_dict(st["best_state"])
    return model


def fold_data(samples, f, label_src, pool):
    train, test = split(samples, f)
    if label_src == "csv":
        test_videos = {s.video for s in test}
        rows = [(p, l) for p, l in pool if p.name.split("_")[0] not in test_videos]
        xtr = torch.stack([legacy_tensor(p) for p, _ in rows])
        ytr = torch.tensor([l for _, l in rows])
    else:
        xtr, ytr = preload(train), torch.tensor([s.label for s in train])
    return xtr, ytr, test


def part_retrain(samples, label_src, seeds, budget, shard, n_shards):
    """Runs unfinished (seed, fold) jobs of this shard until the time budget is
    used, then exits; re-run to continue. Aggregates when all jobs are done."""
    torch.set_num_threads(1)
    deadline = time.time() + budget
    pool = csv_labels() if label_src == "csv" else None
    jobs = [(s, f) for s in seeds for f in range(N_FOLDS)]
    for j, (seed, f) in enumerate(jobs):
        if j % n_shards != shard:
            continue
        path = _job_path(label_src, seed, f)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and "pred" in torch.load(path, weights_only=False):
            continue
        xtr, ytr, test = fold_data(samples, f, label_src, pool)
        model = train_legacy_resumable(xtr, ytr, seed, path, deadline)
        if model is None:
            print(f"  seed {seed} fold {f}: paused", flush=True)
            return None
        st = torch.load(path, weights_only=False)
        st["pred"] = {s.key: int(k) for s, k in zip(test, predict_idx(model, preload(test)))}
        st["n_train"] = len(xtr)
        torch.save(st, path)
        print(f"  seed {seed} fold {f}: n_train={len(xtr)} epochs={st['epoch']} done", flush=True)
    if not all(_job_path(label_src, s, f).exists() and "pred" in torch.load(_job_path(label_src, s, f), weights_only=False)
               for s, f in jobs):
        print("shard finished; other jobs still pending")
        return None
    runs, preds = {}, []
    for seed in seeds:
        merged, epochs = {}, []
        for f in range(N_FOLDS):
            st = torch.load(_job_path(label_src, seed, f), weights_only=False)
            merged.update(st["pred"])
            epochs.append(st["epoch"])
        pred = np.array([merged[s.key] for s in samples])
        preds.append(pred)
        as_shipped, correct = both_reports(samples, pred)
        print(fmt(as_shipped, f"B retrain[{label_src}] s{seed} shipped mapping"))
        print(fmt(correct, f"B retrain[{label_src}] s{seed} correct mapping"))
        runs[seed] = {"as_shipped_mapping": as_shipped, "correct_mapping": correct, "epochs": epochs}
    summary = {}
    for m in ("as_shipped_mapping", "correct_mapping"):
        v = [r[m]["macro_f1"] for r in runs.values()]
        pc = {c: float(np.mean([r[m]["per_class_f1"][c] for r in runs.values()])) for c in CLASSES}
        summary[m] = {"macro_f1_mean": float(np.mean(v)), "macro_f1_std": float(np.std(v)),
                      "per_class_f1_mean": pc}
        print(f"B retrain[{label_src}] {m}: macroF1 {np.mean(v):.3f} ± {np.std(v):.3f} over seeds {seeds}")
    # raw outputs (model output index per image and seed) for scripts/make_results_table.py
    np.savez_compressed(RESULTS / f"baseline_legacy_retrain_{label_src}_oof.npz", pred=np.stack(preds),
                        labels=np.array([s.label for s in samples]), folds=np.array([fold_of(s) for s in samples]),
                        keys=np.array([s.key for s in samples]), seeds=np.array(seeds))
    return {"runs": {str(k): v for k, v in runs.items()}, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["shipped", "retrain"], required=True)
    ap.add_argument("--labels", choices=["csv", "folder"], default="csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget", type=float, default=1e9, help="seconds before pausing (resumable)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    samples = index_dataset()
    RESULTS.mkdir(exist_ok=True)
    if args.part == "shipped":
        out, name = part_shipped(samples), "baseline_shipped_checkpoint.json"
    else:
        out, name = part_retrain(samples, args.labels, args.seeds, args.budget, args.shard, args.n_shards), \
            f"baseline_legacy_retrain_{args.labels}.json"
        if out is None:
            return
    out["torch"] = torch.__version__
    with open(RESULTS / name, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", RESULTS / name)


if __name__ == "__main__":
    main()
