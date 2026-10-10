"""Named experiment configs. Each one is evaluated with scripts/cv.py
(5 video-grouped folds x 3 seeds) and logged in docs/REVIEW.md.

Schedules are fixed in advance (no early stopping, no epoch picking on the
test fold): AdamW, 1 warm-up epoch, cosine decay to 0 over `epochs`.
"""
import copy

BASE = {
    "arch": "smallcnn",
    "pretrained": False,
    "pre": {"geometry": "pad128", "colour": "none"},
    "augment": {},
    "epochs": 30,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "warmup_epochs": 1,
    "loss": {"class_weights": False, "label_smoothing": 0.0},
    "sampler": "shuffle",          # or "balanced" (class-balanced, with replacement)
}


def _cfg(**over):
    c = copy.deepcopy(BASE)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(c.get(k), dict):
            c[k].update(v)
        else:
            c[k] = v
    return c


EXPERIMENTS = {
    # Rung 1: the legacy SmallCNN with the bugs fixed and nothing else:
    # RGB instead of grayscale, black padding to 128x128 instead of stretching,
    # cross-entropy on logits instead of MSE on softmax, batch 32 instead of 2,
    # fixed 30-epoch cosine schedule instead of early stopping on train loss,
    # folder labels keyed by path. No augmentation.
    "r1a_smallcnn_fixed": _cfg(),
    # Rung 1b: + light geometric augmentation (flip, small scale/shift/rotate).
    "r1b_smallcnn_fixed_aug": _cfg(augment={"hflip": True,
                                            "affine": {"scale": 0.1, "translate": 0.05, "rotate": 10}}),
}


def get(name):
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}")
    return copy.deepcopy(EXPERIMENTS[name])
