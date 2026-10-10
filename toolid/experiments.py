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
    # Rung 1b came out WORSE than 1a (0.581 vs 0.689). Class-mean images show
    # hook/clipper/scissor shafts entering from the right and graspers from both
    # sides, so horizontal flip may destroy a real cue. Split 1b to find out:
    "r1c_smallcnn_affine": _cfg(augment={"affine": {"scale": 0.1, "translate": 0.05, "rotate": 10}}),
    "r1d_smallcnn_hflip": _cfg(augment={"hflip": True}),
    # Rung 2: class imbalance (scissor ~10%, clipper ~19%), on top of the best
    # rung-1 recipe so far (1a: no augmentation). (A first version stacked on 1b
    # was stopped unfinished once 1b turned out worse.)
    # 2a: cross-entropy weighted by inverse class frequency.
    "r2a_smallcnn_classweights": _cfg(loss={"class_weights": True}),
    # 2b: class-balanced sampling (each batch ~equal classes, with replacement).
    "r2b_smallcnn_balanced": _cfg(sampler="balanced"),
    # 2c is post-hoc (scripts/tune_offsets.py), no training.
}

# ---- Rung 3b+: fine-tuning ImageNet backbones (run on a GPU; see notebooks/colab_ladder.ipynb).
# Choices fixed from earlier rungs: native 86x128 input (no stretch), NO flips
# (rung 1d), no post-hoc offsets (2c). Fixed schedule: 20 epochs, AdamW 3e-4,
# wd 0.05, 1 warm-up epoch, cosine. Backbones from rung 3a: MobileNetV3-L and
# EfficientNet-B0 (ResNet-18 features were clearly worse).
COLOUR = {"gain": 0.15, "hue": 8, "saturation": 0.2, "brightness": 0.1, "contrast": 0.1}
AFFINE = {"scale": 0.1, "translate": 0.05, "rotate": 10}
_ARCH = {"mbv3": "mobilenet_v3_large", "effb0": "efficientnet_b0"}


def _ft(arch, **over):
    base = dict(arch=_ARCH[arch], pretrained=True, pre={"geometry": "native", "colour": "none"},
                epochs=20, lr=3e-4, weight_decay=0.05)
    base.update(over)
    return _cfg(**base)


for _a in _ARCH:
    EXPERIMENTS.update({
        # 3b: plain fine-tune, no augmentation
        f"r3b_{_a}_ft": _ft(_a),
        # 3c: + colour augmentation (strong per-channel gain/hue: per-video cast is a nuisance;
        #     mild brightness/contrast: they carry class signal) - scripts/colour_analysis.py
        f"r3c_{_a}_ft_colour": _ft(_a, augment={"colour": COLOUR}),
        # 3d: gray-world white-balance normalisation instead of colour augmentation
        f"r3d_{_a}_ft_wb": _ft(_a, pre={"geometry": "native", "colour": "wb"}),
        # 3e: colour aug + 2x upsampled input (172x256)
        f"r3e_{_a}_ft_colour_x2": _ft(_a, augment={"colour": COLOUR}, pre={"geometry": "x2", "colour": "none"}),
        # 3f: colour aug + inverse-frequency class weights (rung 2a was a tie on SmallCNN)
        f"r3f_{_a}_ft_colour_cw": _ft(_a, augment={"colour": COLOUR}, loss={"class_weights": True}),
        # 3g: colour aug + mild affine (affine hurt the position-specific SmallCNN, 1c;
        #     a globally pooled backbone may react differently)
        f"r3g_{_a}_ft_colour_affine": _ft(_a, augment={"colour": COLOUR, "affine": AFFINE}),
    })

# Groups of experiments run together in one Colab session.
STAGES = {
    "3b": [f"r3{k}_{a}_{suffix}" for a in _ARCH for k, suffix in
           [("b", "ft"), ("c", "ft_colour"), ("d", "ft_wb"), ("e", "ft_colour_x2"),
            ("f", "ft_colour_cw"), ("g", "ft_colour_affine")]],
}


def get(name):
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}")
    return copy.deepcopy(EXPERIMENTS[name])
