"""Self-describing checkpoint format shared by train.py and predict.py.

A checkpoint is a plain dict (loadable with torch.load(weights_only=True)):
    format      "toolid-v1"
    arch        model architecture name (toolid.models.build)
    classes     list of class names; index i of the model output == classes[i]
    pre         preprocessing dict (toolid.preprocess.prepare)
    tta_hflip   bool, average predictions over the image and its mirror
    members     list of state_dicts; predictions are the mean of their softmax
    info        provenance (experiment name, config, seeds, git commit, ...)
"""
import json

import torch

from .models import build

FORMAT = "toolid-v1"


def save(path, arch, classes, pre, members, tta_hflip=False, info=None):
    members = [{k: v.detach().cpu() for k, v in m.items()} for m in members]
    # metadata must be plain JSON types so torch.load(weights_only=True) accepts it
    # (e.g. torch.__version__ is a TorchVersion object, which it rejects)
    plain = lambda o: json.loads(json.dumps(o, default=str))  # noqa: E731
    torch.save({"format": FORMAT, "arch": str(arch), "classes": [str(c) for c in classes], "pre": plain(dict(pre)),
                "tta_hflip": bool(tta_hflip), "members": members, "info": plain(info or {})}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} checkpoint")
    models = []
    for sd in ck["members"]:
        m = build(ck["arch"], len(ck["classes"]), pretrained=False)
        m.load_state_dict(sd)
        models.append(m.to(device).eval())
    return ck, models
