"""Train the final model on ALL labelled images (train/ + validation/ folders)
with a named experiment config, and write a self-contained checkpoint for
predict.py.

    python train.py --exp r1b_smallcnn_fixed_aug --seeds 0 1 2 --out checkpoints/model.pt [--tta]

Each seed becomes one ensemble member (predictions = mean softmax). The
schedule is the experiment's fixed schedule, the same one evaluated under
video-grouped CV (scripts/cv.py), so the CV numbers describe this recipe.
Resumable: --budget SECS pauses after SECS and a re-run continues; seeds can
be trained in parallel processes (one --seeds value each) and packed by a
final call with all seeds.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from toolid import checkpoint, experiments  # noqa: E402
from toolid.data import CLASSES, index_dataset  # noqa: E402
from toolid.folds import split  # noqa: E402
from toolid.preprocess import load_rgb, stack_uint8  # noqa: E402
from toolid.trainer import train  # noqa: E402


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "model.pt")
    ap.add_argument("--tta", action="store_true", help="store flip-TTA flag in the checkpoint")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-pack", action="store_true", help="train only, do not write the checkpoint")
    ap.add_argument("--exclude-fold", type=int, default=None,
                    help="leave out this CV fold's videos (for the end-to-end held-out check)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    deadline = time.time() + args.budget if args.budget is not None else None
    cfg = experiments.get(args.exp)
    tag = args.exp if args.exclude_fold is None else f"{args.exp}_without_fold{args.exclude_fold}"
    work = ROOT / "runs" / "final" / tag
    work.mkdir(parents=True, exist_ok=True)

    samples = index_dataset()
    if args.exclude_fold is not None:
        samples, _ = split(samples, args.exclude_fold)
    x = y = None
    for seed in args.seeds:
        done = work / f"s{seed}.weights.pt"
        if done.exists():
            continue
        if x is None:
            x = stack_uint8([load_rgb(s.path) for s in samples])
            y = torch.tensor([s.label for s in samples])
            print(f"training {args.exp} on {len(y)} images: {torch.bincount(y).tolist()} per class {CLASSES}")
        model = train(cfg, x, y, seed, device=args.device, state_path=work / f"s{seed}.state.pt",
                      deadline=deadline, log=lambda m: print(f"  seed {seed} {m}", flush=True))
        if model is None:
            print(f"seed {seed}: paused (re-run to continue)")
            return
        torch.save(model.state_dict(), done)
        (work / f"s{seed}.state.pt").unlink(missing_ok=True)

    if args.no_pack:
        return
    members = [torch.load(work / f"s{s}.weights.pt", map_location="cpu", weights_only=True) for s in args.seeds]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.save(args.out, cfg["arch"], CLASSES, cfg["pre"], members, tta_hflip=args.tta,
                    info={"experiment": args.exp, "config": cfg, "seeds": args.seeds, "n_train": len(samples),
                          "excluded_fold": args.exclude_fold,
                          "git_commit": git_commit(), "torch": torch.__version__})
    print(f"wrote {args.out} ({len(members)} member(s), {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
