# Surgical tool classification (cholec-tinytools)

Classifies 128×86 laparoscopic image crops into `clipper`, `grasper`, `hook`
or `scissor`. This is a rewrite of the provided `surgical_tool_id` codebase:
the review of what was wrong is in [`docs/REVIEW.md`](docs/REVIEW.md).

| Evaluated on videos the model never saw (5-fold, video-grouped) | macro-F1 |
|---|---|
| Original submission as shipped | 0.11 |
| Original model with the class-name order fixed | 0.51 |
| **This repo: 3 × MobileNetV3-L ensemble (`checkpoints/model.pt`)** | **0.858** |

The estimate rests on 10 videos and on picking the best of 12 recipes, so
treat it as "a little under 0.86, give or take a few points"; see the limits
section of the review.

## Run predictions (what the graders run)
Python 3.10-3.13, CPU only, no internet needed at prediction time.

```bash
pip install -r requirements.txt          # on Linux, for a small CPU-only torch, first:
                                          # pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python predict.py --data-dir <DIR> --out <CSV>
```

- `<DIR>`: images in class subfolders (like `data/cholec-tinytools/validation`)
  or directly in the folder; `.png` in any letter case; hidden files ignored.
- `<CSV>`: header `filename,predicted_class`, exactly one row per image.
  Filenames that occur in two subfolders get one row each (the two rows
  cannot be told apart by name; see the review).
- An unreadable image stops the run with exit code 2 and its path, instead
  of a guessed label. Images of another size are resized to 128×86.
- Speed: about 13 s for 1400 images on 4 CPU threads, ~1.3 GB RAM. Threads =
  CPUs available to the process (max 8), or set `OMP_NUM_THREADS`.
- The checkpoint is self-describing (architecture, class order,
  preprocessing, weights) and the network is built with `weights=None`, so
  nothing is downloaded.

## Test
```bash
python check_submission.py      # graders' preflight: runs predict.py on data/.../validation, checks the CSV format
pip install pytest && python -m pytest -q    # 14 tests, ~15 s on CPU
```
The tests cover predict.py's input handling (class folders, flat folder,
duplicate names, `.PNG`, hidden files, corrupt and odd-sized images), the
checkpoint (class order, no download, no test-time flip), the folds (no video
leakage), preprocessing and bit-identical resumed training.

## Reproduce

**ImageNet weights** (training only; `weights/` is git-ignored):
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

**Evaluation and training**
```bash
python scripts/describe_folds.py                              # the 5 video-grouped folds, with leak checks
python scripts/cv.py --exp r3g_mbv3_ft_colour_affine          # 5 folds x 3 seeds -> results/cv/ (GPU: ~10 min on a T4)
python scripts/compare.py r3g_mbv3_ft_colour_affine r3c_mbv3_ft_colour   # paired + video-bootstrap comparison
python scripts/ensemble_oof.py r3g_mbv3_ft_colour_affine r3g_effb0_ft_colour_affine
python scripts/make_results_table.py                          # rebuild docs/results_table.md from stored outputs
python train.py --exp r3g_mbv3_ft_colour_affine --seeds 0 1 2 --out checkpoints/model.pt   # ~15 min on 4 CPU cores
python scripts/baseline_legacy.py --part shipped              # the 'before' numbers (needs full git history)
```
All recipes are named in `toolid/experiments.py`. `notebooks/colab_ladder.ipynb`
runs a group of them on a Colab GPU (set `REPO_URL` and `STAGE`, Run all) and
downloads a results zip. The provided data (`data/cholec-tinytools/`) is used
as-is: train and validation folders are pooled and labelled by folder name.

## Layout
```
predict.py, train.py        entry points (inference / final training)
check_submission.py, contract.py   graders' preflight, unchanged
checkpoints/model.pt        shipped model (3 x MobileNetV3-L, float16 storage, 25.7 MB)
toolid/                     library: data, folds, preprocessing, augmentation, models, trainer, checkpoint, metrics
scripts/                    cross-validation, comparisons, ensembles, baseline, data analyses, results table
results/                    raw out-of-fold outputs (*_oof.npz) and summaries (*.json) of every experiment
tests/                      pytest suite
notebooks/colab_ladder.ipynb   GPU runner
docs/REVIEW.md              written review: what was wrong, what we changed, what we found, limits
docs/EXPERIMENTS.md         detailed log of the audit and every experiment
docs/results_table.md       full results table (generated)
docs/WALKTHROUGH.md         plain-language tour of the code and likely reviewer questions
docs/figures/               data figures (class examples, same-name pairs, class-mean images)
```

## How this was built
Developed with an AI coding agent (Claude). The complete conversation is
submitted alongside this repository; the commit history follows it step by
step, one commit per change or experiment.
