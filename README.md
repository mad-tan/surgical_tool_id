# Surgical tool classification (cholec-tinytools)

Classifies 128×86 laparoscopic crops into `clipper`, `grasper`, `hook`, `scissor`.
Work in progress; see `docs/REVIEW.md` for the review of the original code,
the evaluation protocol and every experiment.

## Pretrained ImageNet weights (training only)
`predict.py` never needs these: the shipped checkpoint contains all of its
weights. Training a pretrained backbone needs them in `weights/` (git-ignored):

```bash
mkdir -p weights
curl -fL -o weights/resnet18-f37072fd.pth                   https://download.pytorch.org/models/resnet18-f37072fd.pth
curl -fL -o weights/efficientnet_b0_rwightman-7f5810bc.pth  https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
curl -fL -o weights/mobilenet_v3_large-5c1a4163.pth         https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth
shasum -a 256 weights/*.pth
```

Expected SHA-256:
```
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec  weights/resnet18-f37072fd.pth
7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934  weights/efficientnet_b0_rwightman-7f5810bc.pth
5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997  weights/mobilenet_v3_large-5c1a4163.pth
```
