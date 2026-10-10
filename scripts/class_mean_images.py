"""Per-class mean image (docs/figures/class_means.png) and left/right
brightness asymmetry. Used to explain why horizontal-flip augmentation hurt:
hook/clipper/scissor shafts enter from the right, graspers from both sides.
Usage: python scripts/class_mean_images.py"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolid.data import CLASSES, index_dataset  # noqa: E402
from toolid.preprocess import load_rgb, stack_uint8  # noqa: E402


def main():
    S = index_dataset()
    x = stack_uint8([load_rgb(s.path) for s in S]).float() / 255
    y = torch.tensor([s.label for s in S])
    tiles = []
    for c, name in enumerate(CLASSES):
        xc = x[y == c]
        tiles.append(xc.mean(0))
        third = x.shape[3] // 3
        print(f"{name:8s} n={len(xc):3d}  mean brightness left third {xc[..., :third].mean():.3f}  "
              f"right third {xc[..., -third:].mean():.3f}")
    img = torch.cat([F.interpolate(t[None], scale_factor=3, mode="nearest")[0] for t in tiles], 2)
    img = (img - img.min()) / (img.max() - img.min())
    out = ROOT / "docs" / "figures" / "class_means.png"
    Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(out)
    print("wrote", out, "(left to right:", ", ".join(CLASSES) + "; contrast stretched)")


if __name__ == "__main__":
    main()
