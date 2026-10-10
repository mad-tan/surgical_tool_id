"""Batched, per-sample random augmentation on float [0,1] tensors (B,3,H,W).
Uses only the global torch RNG, so a run resumed with the saved RNG state
is bit-identical to an uninterrupted one.

Config keys (all optional, default off):
    hflip: bool                      horizontal flip, p=0.5
    affine: {"scale": s, "translate": t, "rotate": deg}
                                     scale in [1-s, 1+s], shift up to t of
                                     width/height, rotation up to +-deg
    colour: {"gain": g, "hue": deg, "saturation": s, "brightness": b, "contrast": c}
        gain        per-channel multiplicative gain exp(U(-g, g)): simulates
                    white-balance / colour-cast differences between videos
        hue         rotation of the chroma plane (YIQ) by U(-deg, deg)
        saturation  blend with grayscale, factor U(1-s, 1+s)
        brightness  multiply by U(1-b, 1+b)
        contrast    blend with the image mean, factor U(1-c, 1+c)
"""
import math

import torch
import torch.nn.functional as F

_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def _u(n, lo, hi, device):
    return torch.rand(n, device=device) * (hi - lo) + lo


def _bcast(v):
    return v.view(-1, 1, 1, 1)


class Augment:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    def __call__(self, x):
        c, B, dev = self.cfg, x.shape[0], x.device
        if c.get("hflip"):
            flip = torch.rand(B, device=dev) < 0.5
            x = torch.where(_bcast(flip), x.flip(3), x)
        if c.get("affine"):
            a = c["affine"]
            s = _u(B, 1 - a.get("scale", 0), 1 + a.get("scale", 0), dev)
            r = _u(B, -a.get("rotate", 0), a.get("rotate", 0), dev) * math.pi / 180
            tx = _u(B, -a.get("translate", 0), a.get("translate", 0), dev) * 2
            ty = _u(B, -a.get("translate", 0), a.get("translate", 0), dev) * 2
            cos, sin = torch.cos(r) / s, torch.sin(r) / s
            theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if c.get("colour"):
            k = c["colour"]
            if k.get("gain"):
                g = k["gain"]
                x = x * torch.exp(_u(B * 3, -g, g, dev)).view(B, 3, 1, 1)
            if k.get("hue"):
                ang = _u(B, -k["hue"], k["hue"], dev) * math.pi / 180
                cos, sin = torch.cos(ang), torch.sin(ang)
                rot = torch.zeros(B, 3, 3, device=dev)
                rot[:, 0, 0] = 1
                rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = cos, -sin, sin, cos
                m = _YIQ2RGB.to(dev) @ rot @ _RGB2YIQ.to(dev)  # B,3,3
                x = torch.einsum("bij,bjhw->bihw", m, x)
            if k.get("saturation"):
                gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
                f = _bcast(_u(B, 1 - k["saturation"], 1 + k["saturation"], dev))
                x = gray + f * (x - gray)
            if k.get("brightness"):
                x = x * _bcast(_u(B, 1 - k["brightness"], 1 + k["brightness"], dev))
            if k.get("contrast"):
                mean = x.mean(dim=(1, 2, 3), keepdim=True)
                x = mean + _bcast(_u(B, 1 - k["contrast"], 1 + k["contrast"], dev)) * (x - mean)
            x = x.clamp(0, 1)
        return x
