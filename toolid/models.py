"""Model registry. Architectures are built WITHOUT downloading anything;
ImageNet weights are loaded from weights/ only when training asks for them
(see README.md). A trained checkpoint is self-contained."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_FILES = {
    "resnet18": ROOT / "weights" / "resnet18-f37072fd.pth",
    "efficientnet_b0": ROOT / "weights" / "efficientnet_b0_rwightman-7f5810bc.pth",
    "mobilenet_v3_large": ROOT / "weights" / "mobilenet_v3_large-5c1a4163.pth",
}


class SmallCNN(nn.Module):
    """Same architecture as legacy/cnn_baseline_v2.SmallCNN (expects 128x128)."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        return self.classifier(self.dropout(self.features(x).flatten(1)))


def build(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if arch == "smallcnn":
        if pretrained:
            raise ValueError("smallcnn has no pretrained weights")
        return SmallCNN(num_classes)
    if arch not in WEIGHT_FILES:
        raise ValueError(f"unknown arch {arch!r}")
    m = getattr(models, arch)(weights=None)
    if pretrained:
        path = WEIGHT_FILES[arch]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - see README.md for the download command")
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    head = _head_parent(m, arch)
    parent, attr = head
    old = getattr(parent, attr)
    setattr(parent, attr, nn.Linear(old.in_features, num_classes))
    return m


def _head_parent(m, arch):
    if arch == "resnet18":
        return m, "fc"
    if arch == "efficientnet_b0":
        return m.classifier, "1"
    if arch == "mobilenet_v3_large":
        return m.classifier, "3"
    raise ValueError(arch)


def feature_extractor(arch: str) -> nn.Module:
    """ImageNet-pretrained backbone returning pooled penultimate features
    (resnet18: 512, efficientnet_b0: 1280, mobilenet_v3_large: 960)."""
    m = build(arch, 1000 if arch != "smallcnn" else 4, pretrained=False)
    m.load_state_dict(torch.load(WEIGHT_FILES[arch], map_location="cpu", weights_only=True))
    if arch == "resnet18":
        m.fc = nn.Identity()
    elif arch == "efficientnet_b0":
        m.classifier = nn.Identity()
    elif arch == "mobilenet_v3_large":
        m.classifier = nn.Identity()
    return m.eval()
