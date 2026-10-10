"""Video-grouped CV for a named experiment (toolid/experiments.py).

    python scripts/cv.py --exp r1a_smallcnn_fixed               # run everything
    python scripts/cv.py --exp NAME --budget 150 --shard 0 --n-shards 4
                                         # time-sliced, resumable, parallel shards

Per (seed, fold) job: train on the other folds' videos with the fixed
schedule, predict the held-out videos, store their logits in
runs/cv/<exp>/s<seed>_f<fold>.logits.pt (the large per-epoch training state
is deleted once the job is finished). When every job is done, writes
  results/cv/<exp>.json      per-seed pooled OOF reports + mean/std summary
  results/cv/<exp>_oof.npz   OOF logits (seeds x N x 4), labels, folds, keys
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolid import experiments  # noqa: E402
from toolid.data import CLASSES, index_dataset  # noqa: E402
from toolid.folds import N_FOLDS, fold_of, split  # noqa: E402
from toolid.metrics import cv_report, fmt  # noqa: E402
from toolid.preprocess import load_rgb, stack_uint8  # noqa: E402
from toolid.trainer import predict_logits, train  # noqa: E402


def load_all(samples):
    return stack_uint8([load_rgb(s.path) for s in samples])


def job_dir(exp):
    return ROOT / "runs" / "cv" / exp


def summarise(exp, samples, seeds, cfg):
    keys = [s.key for s in samples]
    y = np.array([s.label for s in samples])
    folds = np.array([fold_of(s) for s in samples])
    all_logits, runs = [], {}
    for seed in seeds:
        lg = np.zeros((len(samples), 4), np.float32)
        pos = {k: i for i, k in enumerate(keys)}
        for f in range(N_FOLDS):
            d = torch.load(job_dir(exp) / f"s{seed}_f{f}.logits.pt", weights_only=False)
            for k, v in zip(d["keys"], d["logits"]):
                lg[pos[k]] = v
        all_logits.append(lg)
        r = cv_report(y, lg.argmax(1), folds)
        runs[str(seed)] = r
        print(fmt(r, f"{exp} seed {seed}"))
    mf = [r["macro_f1"] for r in runs.values()]
    summary = {"macro_f1_mean": float(np.mean(mf)), "macro_f1_std": float(np.std(mf)),
               "per_class_f1_mean": {c: float(np.mean([r["per_class_f1"][c] for r in runs.values()])) for c in CLASSES},
               "per_class_f1_std": {c: float(np.std([r["per_class_f1"][c] for r in runs.values()])) for c in CLASSES},
               "accuracy_mean": float(np.mean([r["accuracy"] for r in runs.values()]))}
    # ensemble of the seeds' OOF probabilities (what averaging 3 models would give)
    probs = np.mean([torch.softmax(torch.from_numpy(l), 1).numpy() for l in all_logits], 0)
    summary["seed_ensemble"] = cv_report(y, probs.argmax(1), folds)
    print(f"{exp}: macro-F1 {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f} | "
          + " ".join(f"{c} {summary['per_class_f1_mean'][c]:.3f}±{summary['per_class_f1_std'][c]:.3f}" for c in CLASSES)
          + f" | 3-seed ensemble {summary['seed_ensemble']['macro_f1']:.3f}")
    out = ROOT / "results" / "cv"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{exp}.json", "w") as fh:
        json.dump({"experiment": exp, "config": cfg, "seeds": seeds, "summary": summary, "runs": runs,
                   "torch": torch.__version__}, fh, indent=1)
    np.savez_compressed(out / f"{exp}_oof.npz", logits=np.stack(all_logits), labels=y, folds=folds,
                        keys=np.array(keys), seeds=np.array(seeds))
    print("wrote", out / f"{exp}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget", type=float, default=None, help="seconds; pause (resumable) when exceeded")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--claim", action="store_true",
                    help="claim jobs via lock files instead of static sharding (locks cleared by run_slices.sh)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    deadline = time.time() + args.budget if args.budget is not None else None
    cfg = experiments.get(args.exp)
    samples = index_dataset()
    jd = job_dir(args.exp)
    jd.mkdir(parents=True, exist_ok=True)
    jobs = [(s, f) for s in args.seeds for f in range(N_FOLDS)]
    done = lambda s, f: (jd / f"s{s}_f{f}.logits.pt").exists()  # noqa: E731
    x_all = None
    for j, (seed, f) in enumerate(jobs):
        if done(seed, f):
            continue
        if args.claim:  # dynamic: take any job no other running process holds
            try:
                os.close(os.open(jd / f"s{seed}_f{f}.lock", os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
        elif j % args.n_shards != args.shard:
            continue
        if deadline is not None and deadline - time.time() < 45:  # no time to start a job safely
            print(f"[{args.exp}] seed {seed} fold {f}: paused", flush=True)
            return
        if x_all is None:
            x_all = load_all(samples)
            pos = {s.key: i for i, s in enumerate(samples)}
        train_s, test_s = split(samples, f)
        tr = torch.tensor([pos[s.key] for s in train_s])
        te = torch.tensor([pos[s.key] for s in test_s])
        y_tr = torch.tensor([s.label for s in train_s])
        state = jd / f"s{seed}_f{f}.state.pt"
        t0 = time.time()
        model = train(cfg, x_all[tr], y_tr, seed, device=args.device, state_path=state, deadline=deadline)
        if model is None:
            print(f"[{args.exp}] seed {seed} fold {f}: paused", flush=True)
            return
        lg = predict_logits(model, x_all[te], cfg["pre"], device=args.device)
        torch.save({"keys": [s.key for s in test_s], "logits": lg}, jd / f"s{seed}_f{f}.logits.pt")
        state.unlink(missing_ok=True)
        print(f"[{args.exp}] seed {seed} fold {f}: done ({time.time() - t0:.0f}s this slice)", flush=True)
    if all(done(s, f) for s, f in jobs):
        summarise(args.exp, samples, args.seeds, cfg)
    else:
        print(f"[{args.exp}] {sum(done(s, f) for s, f in jobs)}/{len(jobs)} jobs done")


if __name__ == "__main__":
    main()
