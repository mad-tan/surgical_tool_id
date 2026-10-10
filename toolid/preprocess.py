"""Image loading and deterministic preprocessing, shared by training and predict.py.

Pipeline (all on float tensors in [0,1], batch-first):
    uint8 RGB 86x128  ->  [train only: augment]  ->  colour normalisation
    ->  geometry  ->  ImageNet mean/std

geometry:
    "native"  keep 86x128 (no distortion)
    "pad128"  pad top/bottom with black to 128x128 (no distortion; lets the
              legacy SmallCNN keep its 128x128 architecture)
    "x2"      bilinear upsample to 172x256 (more spatial resolution for
              pretrained backbones whose last feature map is otherwise 3x4)
colour:
    "none"    as is
    "wb"      gray-world white balance: scale each channel so the three
              channel means are equal, keeping the image's mean brightness.
              Removes per-video colour casts; keeps brightness/contrast,
              which carry class signal (see scripts/colour_analysis.py).
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

NATIVE_H, NATIVE_W = 86, 128
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class UnreadableImage(RuntimeError):
    pass


def load_rgb(path) -> np.ndarray:
    """uint8 HxWx3 at the native 86x128. Raises UnreadableImage instead of
    silently substituting anything."""
    try:
        with Image.open(path) as im:
            im.load()
            im = im.convert("RGB")
            if im.size != (NATIVE_W, NATIVE_H):
                im = im.resize((NATIVE_W, NATIVE_H), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8).copy()
    except Exception as e:  # noqa: BLE001 - re-raised with the path attached
        raise UnreadableImage(f"{path}: {type(e).__name__}: {e}") from e


def stack_uint8(arrays) -> torch.Tensor:
    """list of HxWx3 uint8 -> Bx3xHxW uint8 tensor."""
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def white_balance(x: torch.Tensor) -> torch.Tensor:
    m = x.mean(dim=(2, 3), keepdim=True).clamp_min(1e-3)  # B,3,1,1
    gain = m.mean(dim=1, keepdim=True) / m
    return (x * gain).clamp(0, 1)


def geometry(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "native":
        return x
    if mode == "pad128":
        pad = NATIVE_W - x.shape[2]
        return F.pad(x, (0, 0, pad // 2, pad - pad // 2))
    if mode == "x2":
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    raise ValueError(f"unknown geometry {mode!r}")


def prepare(x_uint8: torch.Tensor, pre: dict, augment=None) -> torch.Tensor:
    """pre = {"geometry": ..., "colour": ...}; augment: callable on [0,1] floats or None."""
    x = x_uint8.float() / 255.0
    if augment is not None:
        x = augment(x)
    if pre.get("colour", "none") == "wb":
        x = white_balance(x)
    x = geometry(x, pre["geometry"])
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
