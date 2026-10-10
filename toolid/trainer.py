"""Training loop shared by CV (scripts/cv.py) and the final fit (train.py).

Deterministic and resumable: batches are drawn from a preloaded uint8 tensor
with the global torch RNG (no DataLoader workers), and the full state
(model, optimiser, epoch, RNG) is saved after every epoch, so a run that is
paused and resumed gives exactly the same weights as an uninterrupted one.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .augment import Augment
from .models import build
from .preprocess import prepare


def class_counts(y, k=4):
    return torch.bincount(y, minlength=k).float()


def lr_at(step, steps_per_epoch, cfg):
    total = cfg["epochs"] * steps_per_epoch
    warm = cfg["warmup_epochs"] * steps_per_epoch
    if step < warm:
        return cfg["lr"] * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * t))


def epoch_indices(y, cfg):
    n = len(y)
    if cfg["sampler"] == "shuffle":
        return torch.randperm(n)
    if cfg["sampler"] == "balanced":
        w = (1.0 / class_counts(y))[y]
        return torch.multinomial(w, n, replacement=True)
    raise ValueError(cfg["sampler"])


def make_model(cfg, device):
    return build(cfg["arch"], 4, pretrained=cfg["pretrained"]).to(device)


def train(cfg, x, y, seed, device="cpu", state_path=None, deadline=None, log=None):
    """x: uint8 (N,3,86,128) tensor on CPU, y: long (N,). Returns the trained
    model, or None if `deadline` (time.time() value) was reached first; in that
    case the state is in `state_path` and calling again resumes."""
    torch.manual_seed(seed)
    model = make_model(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    st = {"epoch": 0, "secs_per_epoch": 0.0}
    if state_path is not None and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng"])
    weight = None
    if cfg["loss"]["class_weights"]:
        cnt = class_counts(y)
        weight = (cnt.sum() / (len(cnt) * cnt)).to(device)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg["loss"]["label_smoothing"])
    aug = Augment(cfg["augment"]) if cfg["augment"] else None
    bs = cfg["batch_size"]
    steps = math.ceil(len(y) / bs)
    while st["epoch"] < cfg["epochs"]:
        if deadline is not None and time.time() + 1.15 * st["secs_per_epoch"] > deadline:
            return None
        t0 = time.time()
        model.train()
        idx = epoch_indices(y, cfg)
        tot = 0.0
        for b in range(steps):
            bi = idx[b * bs:(b + 1) * bs]
            if len(bi) < 2:  # BatchNorm needs >1 sample
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(st["epoch"] * steps + b, steps, cfg)
            # move the uint8 batch first so augmentation runs on the GPU when there is one
            xb = prepare(x[bi].to(device, non_blocking=True), cfg["pre"], aug)
            loss = crit(model(xb), y[bi].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        st["epoch"] += 1
        st["secs_per_epoch"] = time.time() - t0
        if log:
            log(f"epoch {st['epoch']}/{cfg['epochs']} loss {tot / len(y):.4f} {st['secs_per_epoch']:.1f}s")
        if state_path is not None:
            st.update(model=model.state_dict(), opt=opt.state_dict(), rng=torch.get_rng_state())
            tmp = state_path.with_suffix(".tmp")
            torch.save(st, tmp)
            tmp.replace(state_path)
    return model


@torch.no_grad()
def predict_logits(model, x, cfg_pre, device="cpu", bs=128, tta_hflip=False):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = prepare(x[i:i + bs].to(device), cfg_pre)
        lg = model(xb)
        if tta_hflip:
            lg = (lg.softmax(1) + model(xb.flip(3)).softmax(1)).log()
        out.append(lg.float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 4), np.float32)
