# Code review: what was wrong and what we changed

Running log, polished at the end. Each entry: **where**, **what**, **why it
matters for held-out macro-F1**, **evidence**, **fix** (filled in when made).
Line numbers refer to the original commit (`Import original ...`).

Ranking is by expected impact on the score the graders compute:
`python predict.py --data-dir <DIR> --out <CSV>` → macro-F1 on unseen images.

## Definitely bugs

### 1. predict.py maps class indices to the wrong names
- **Where:** `predict.py:19-22` takes names from `data_utils.py:15`
  (`grasper, hook, clipper, scissor`), but the checkpoint was trained with
  `legacy/cnn_baseline_v2.py:21` (`sorted(...)` → `clipper, grasper, hook, scissor`).
- **Why:** every prediction except scissor is relabelled as another class.
- **Evidence:** shipped `predict.py` on `validation/`: macro-F1 0.21, acc 0.11;
  same predictions with the correct index→name order: macro-F1 0.79 (in-sample,
  see #6). Preflight still passes; it only checks CSV shape.
- **Fix:** _pending_ (save class names inside the checkpoint; predict reads them).

### 2. predict.py ships the legacy CNN, not what train.py trains
- **Where:** `predict.py:20,23` loads `SmallCNN` + `checkpoints/model_best.pt`;
  `train.py:129` writes `resnet18_final.pt`, which nothing loads.
- **Why:** improvements to train.py could never reach the submission.
- **Fix:** _pending_.

### 3. train.py destroys its inputs (train and val)
- **Where:** `train.py:41-50` never divides by 255 (inputs reach ~1136 after
  "normalisation"); `train.py:27` `ColorJitter` on a 0-255 float tensor clamps
  to [0,1] → an image collapses to 3-4 distinct values; `train.py:88` also
  augments the validation set.
- **Evidence:** sample min/max/unique values printed before and after.
- **Fix:** _pending_.

### 4. Cross-entropy applied to softmax probabilities
- **Where:** `train.py:108,119`.
- **Why:** double softmax: loss floor 0.744 on a perfect prediction, gradient
  4x smaller at init; the model can never become confident.
- **Fix:** _pending_.

### 5. Backbone frozen despite "full finetune"; BN stats drift; no eval mode
- **Where:** `train.py:72-75` (2,052 of 11.2M params trainable); validation
  loop `train.py:115` never calls `model.eval()`/`no_grad`. BatchNorm running
  stats still update in train mode with frozen weights (verified).
- **Fix:** _pending_.

### 6. No honest validation anywhere
- **Where:** `train.py:61,83` pools train+validation, random image-level split;
  8 of 10 videos are in both the given train/ and validation/ folders;
  `legacy/cnn_baseline_v2.py:32` trains on everything, so validation/ is
  in-sample for the shipped model.
- **Why:** consecutive frames of one video are near-duplicates, so a random
  split measures memorisation of a video, not tool recognition. No number in
  the repo predicted held-out performance, so model choices were blind.
- **Fix:** _pending_ (video-grouped CV, step 3).

### 7. Shipped model trained on corrupted labels
- **Where:** `notebooks/EDA_final.ipynb` (`n_flip` cell) deliberately flips 4%
  of labels into `labels.csv` (56 files disagree with their folder);
  filename-keyed lookup (`legacy/cnn_baseline_v2.py:32-44`, `data_utils.py:44`)
  collides on the 26 filenames that exist in two class folders.
- **Fix:** _pending_ (labels = folder name, samples keyed by full path).

### 8. "Best" legacy checkpoint chosen by training loss
- **Where:** `legacy/cnn_baseline_v2.py:107`.

### 9. evaluate_model.py reports misleading numbers
- **Where:** per-class names in wrong order (`helpers/image_helpers.py:38`);
  preprocessing (cv2 BGR, colour, ImageNet norm) differs from how the CNN was
  trained (grayscale /255): reports 0.66 where matched preprocessing gives 0.82.

### 10. exp02 EfficientNet result is meaningless
- **Where:** `experiments/exp02_efficientnet/model_def.py:8` `weights=None`
  (not pretrained); `run_experiment.py:58` Adam lr=0.1, wd=0.1, re-created every
  epoch; `:48` `shuffle=False` on class-sorted ImageFolder (33/36 batches are a
  single class); `:74` `val_loader = train_loader`. results.json = 0.103
  (chance) while the docstring claims 0.94.

## Suspicious, to test under video-grouped CV
1. No class-imbalance handling (scissor ≈10%); macro-F1 weighs it equally.
2. 128×86 images squashed to 128×128 / 224×224 (aspect distortion).
3. Grayscale input discards colour.
4. Small from-scratch CNN vs pretrained backbone (weights must be bundled:
   grading is offline).
5. Legacy recipe: MSE on softmax, batch size 2 with BatchNorm.
6. exp02 augmentations (hue 0.15, RandomResizedCrop scale 0.3, NEAREST).
7. stats.json normalisation computed over train+val; not ImageNet stats.
8. Robustness at grading: `.png` case-sensitive filter, silent zero-image
   fallback, `.DS_Store` read as a class by `os.listdir`, predict.py import
   depends on labels.csv, 26 duplicate filenames collapse in a filename-keyed CSV.

## Data observations (step 2, `scripts/make_data_figures.py`)
- Every image is a 128×86 crop centred on one instrument tip, cut from a
  full laparoscopic frame (the black circular endoscope border often shows in
  a corner). Figures: `docs/figures/class_grid.png`, `docs/figures/same_name_pairs.png`.
- Classes are visually distinct for a human: hook = white insulated cone on a
  black shaft; grasper = long thin fenestrated jaws; clipper and scissor =
  short metallic jaws on a banded silver shaft. Clipper vs scissor is the
  hard pair; some scissor tips are also pale and cone-shaped like a hook.
- Strong per-video appearance differences (white balance, blue/orange casts,
  tissue colour), which is why splitting by video matters.
- Consecutive frames 25 apart (1 s) are near-identical (e.g. 9152_15026 …
  9152_15276), confirming that an image-level split leaks.
- The 26 same-name pairs are **not duplicates or label errors**: they are two
  different crops from the same video frame, one per instrument visible in
  that frame (grasper + hook, grasper + clipper, grasper + scissor). Both labels
  are correct. Consequences: (a) any code that keys samples by filename
  silently loses or mislabels one of each pair; (b) in a filename-keyed CSV the
  two predictions for such a frame cannot be told apart; (c) some crops show
  part of the second tool, so a little label ambiguity is inherent.

## Evaluation protocol (step 3, `toolid/`)
Video-grouped 5-fold CV over train+validation pooled (1402 images, 10
videos), labels = class folder, samples keyed by relative path. Fold
assignment and its rationale: `toolid/folds.py`; check with
`python scripts/describe_folds.py`. Headline metric: macro-F1 on the pooled
out-of-fold predictions; per-fold macro-F1 shown for spread. Every model
variant is run with ≥3 seeds and reported as mean ± std.
Reference point: uniform random guessing scores ≈0.24 macro-F1 on this class mix.

## Baseline: the shipped pipeline (step 4, `scripts/baseline_legacy.py`)
| Setting | macro-F1 | clipper | grasper | hook | scissor |
|---|---|---|---|---|---|
| A. shipped `model_best.pt`, shipped predict.py mapping, **in-sample** | 0.152 | 0.06 | 0.06 | 0.04 | 0.45 |
| A. shipped `model_best.pt`, correct mapping, **in-sample** | 0.740 | 0.74 | 0.87 | 0.90 | 0.45 |
| B. original recipe retrained per fold (labels.csv), shipped mapping | **0.111 ± 0.004** | 0.14 | 0.12 | 0.06 | 0.12 |
| B. original recipe retrained per fold (labels.csv), correct mapping | **0.514 ± 0.063** | 0.42 | 0.77 | 0.75 | 0.12 |
| B. same, folder labels instead of labels.csv, correct mapping | 0.534 ± 0.034 | 0.40 | 0.78 | 0.75 | 0.20 |

(B rows: pooled out-of-fold, mean over seeds 0,1,2; per-class F1 are seed means.)

What this says:
- **Honest "before" = 0.11 macro-F1** for the submission as shipped (worse
  than random guessing, because of the name mapping), and **0.51** for the
  same model with the mapping fixed. A is in-sample (the checkpoint saw every
  image), so it only bounds fit; the 0.74 → 0.51 gap is what the model had
  memorised about the videos it was trained on.
- **Scissor is the bottleneck**: F1 0.12-0.20; the model almost never predicts
  scissor (28 of 429 true scissors over 3 seeds, labels.csv run). Clipper is
  the second weakest (often called grasper).
- **Seed noise is large** for this recipe (0.43-0.56 across seeds): early
  stopping on a batch-size-2 training loss stops anywhere between 5 and 30
  epochs. A change has to beat the baseline by clearly more than ~0.05 to
  count.
- Removing the 4% label noise (+0.02) is within seed noise for this model; it
  is still the correct thing to do (the graders' labels are the folders).
