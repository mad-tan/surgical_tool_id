"""Predict surgical tool classes for a folder of 128x86 PNG crops.

    python predict.py --data-dir <DIR> --out <CSV>

<DIR> may contain class subfolders (like data/cholec-tinytools/validation)
and/or images directly. Hidden files (e.g. .DS_Store) are ignored; the
extension check is case-insensitive. Output: CSV with header
`filename,predicted_class`, one row per image (basename only, as the graders
expect), predicted_class in {clipper, grasper, hook, scissor}.

Runs on CPU, never downloads anything: the checkpoint (default
checkpoints/model.pt) carries the architecture, class names, preprocessing
and all weights. If any image cannot be read, the script lists them and exits
with status 2 instead of guessing a class from a blank image.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from toolid import checkpoint  # noqa: E402
from toolid.preprocess import UnreadableImage, load_rgb, prepare, stack_uint8  # noqa: E402

DEFAULT_CKPT = ROOT / "checkpoints" / "model.pt"
IMAGE_EXTS = {".png"}


def cpu_threads() -> int:
    """Threads for CPU inference: the cores this process may actually use
    (respects container/cgroup CPU affinity), capped at 8 (more gives no speed-up
    for these small models), or OMP_NUM_THREADS if the caller set it."""
    if os.environ.get("OMP_NUM_THREADS", "").isdigit():
        return max(1, int(os.environ["OMP_NUM_THREADS"]))
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        n = os.cpu_count() or 1
    return max(1, min(n, 8))


def find_images(data_dir: Path):
    """Images directly in data_dir plus images one level down (class folders)."""
    def images_in(d):
        return sorted(p for p in d.iterdir()
                      if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS)
    files = images_in(data_dir)
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        files += images_in(d)
    return files


@torch.no_grad()
def predict_probs(models, x_uint8, pre, tta_hflip, batch_size=256):
    out = []
    for i in range(0, len(x_uint8), batch_size):
        xb = prepare(x_uint8[i:i + batch_size], pre)
        p = 0
        for m in models:
            p = p + m(xb).softmax(1)
            if tta_hflip:
                p = p + m(xb.flip(3)).softmax(1)
        out.append(p / (len(models) * (2 if tta_hflip else 1)))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(cpu_threads())

    if not args.data_dir.is_dir():
        sys.exit(f"error: --data-dir {args.data_dir} is not a directory")
    files = find_images(args.data_dir)
    if not files:
        sys.exit(f"error: no {'/'.join(sorted(IMAGE_EXTS))} images found in {args.data_dir} or its subfolders")

    arrays, bad = [], []
    for p in files:
        try:
            arrays.append(load_rgb(p))
        except UnreadableImage as e:
            bad.append(str(e))
    if bad:
        print(f"error: {len(bad)} unreadable image(s); refusing to guess:", *bad, sep="\n  ", file=sys.stderr)
        sys.exit(2)

    ck, models = checkpoint.load(args.checkpoint)  # CPU; architectures built with weights=None, no download
    probs = predict_probs(models, stack_uint8(arrays), ck["pre"], ck["tta_hflip"])
    names = [ck["classes"][i] for i in probs.argmax(1).tolist()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_class"])
        w.writerows((p.name, n) for p, n in zip(files, names))

    dup = [n for n, c in Counter(p.name for p in files).items() if c > 1]
    if dup:
        print(f"note: {len(dup)} filename(s) occur in more than one subfolder (e.g. {dup[0]}); "
              f"each gets its own row", file=sys.stderr)
    print(f"wrote {len(files)} predictions to {args.out} "
          f"({len(models)} model(s){' + flip TTA' if ck['tta_hflip'] else ''}, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
