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
- **Fix (done):** checkpoints are self-describing (`toolid/checkpoint.py`): they
  store the class names in output-index order, the architecture and the
  preprocessing; `predict.py` reads all three from the checkpoint, so the
  mapping can no longer drift between training and inference.

### 2. predict.py ships the legacy CNN, not what train.py trains
- **Where:** `predict.py:20,23` loads `SmallCNN` + `checkpoints/model_best.pt`;
  `train.py:129` writes `resnet18_final.pt`, which nothing loads.
- **Why:** improvements to train.py could never reach the submission.
- **Fix (done):** `train.py` now trains the final model with a named,
  CV-evaluated experiment config and writes `checkpoints/model.pt`, which is
  exactly what `predict.py` loads by default. No dependency on legacy/,
  data_utils.py or labels.csv.

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
   **Fixed in the new predict.py:** case-insensitive extension, hidden files and
   folders ignored, flat folder or class subfolders, unreadable images listed and
   exit status 2 (no silent black-image guess), no labels.csv/legacy imports,
   one row per image even when names repeat (with a stderr note; the duplicate
   rows are unavoidable given a filename-keyed CSV). Images not 128×86 are
   resized to 128×86 rather than rejected.

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

## Colour: signal or nuisance? (`scripts/colour_analysis.py`)
Per-image colour statistics, share of variance explained by video vs class (eta²):
- **Chromaticity (hue / white balance, brightness removed) is a video
  signature, not a class cue**: blue share 0.47 by video vs 0.03 by class,
  red share 0.35 vs 0.08. Video 7697 is bluish, 7695 and 7697 are darker; the
  same 9 colour numbers identify the video 63% of the time (majority 31%).
- **Brightness and contrast do carry class signal**: per-channel mean/std
  0.18-0.33 by class (hook = white insulated cone → brighter, higher contrast).
  The 9 numbers alone reach macro-F1 0.41 under video-grouped CV (hook 0.75).
- Implications for the ladder: hue / white-balance (per-channel gain)
  augmentation should be strong because it removes a shortcut that does not
  transfer to new videos; brightness/contrast jitter should be mild; per-image
  standardisation of mean/std would delete real hook evidence, so test
  white-balance-only normalisation (equalise channel ratios, keep luminance)
  rather than full per-image normalisation.

## Experiment log (video-grouped 5-fold CV, pooled OOF macro-F1, mean ± std over seeds 0,1,2)
Bar to beat = previous best + more than the seed spread. Every row is kept,
including ones that did not help. Full reports: `results/cv/<name>.json`.

| # | Experiment | macro-F1 | clipper F1 | scissor F1 | Verdict |
|---|---|---|---|---|---|
| 0 | Legacy recipe retrained in folds, folder labels (`baseline_legacy_retrain_folder`) | 0.534 ± 0.034 | 0.40 | 0.20 | baseline |
| 1a | `r1a_smallcnn_fixed`: same SmallCNN, bugs fixed only: RGB, black-pad to 128×128 (no stretch), CE on logits, batch 32, AdamW 1e-3 wd 1e-2, fixed 30-epoch cosine, no augmentation | **0.689 ± 0.008** | 0.65 ± 0.03 | 0.42 ± 0.01 | **+0.155, keep.** Seed spread also falls 4× (0.034 → 0.008): the fixed schedule removes the early-stopping lottery. |
| 1b | `r1b_smallcnn_fixed_aug`: 1a + horizontal flip + affine (scale ±10%, shift ±5%, rotate ±10°) | 0.581 ± 0.021 | 0.55 ± 0.02 | 0.29 ± 0.03 | **−0.108, reject.** Augmentation verified visually (tools intact, mild transforms), so not a bug. Hypothesis: flip destroys a real cue. Class-mean images (`docs/figures/class_means.png`, `scripts/class_mean_images.py`) show hook/clipper/scissor shafts entering from the right (left third brighter than right third: hook 0.43 vs 0.30) and graspers from both sides (X pattern; grasper is the only class darker on the left). Mirroring a right-hand tool makes it look like a grasper; grasper F1 fell 0.84 → 0.70. Split into 1c/1d to confirm. (Run on the cloud workspace: torch 2.14 x86; others on torch 2.5.1 arm64.) |
| 1c | `r1c_smallcnn_affine`: 1a + affine only (scale ±10%, shift ±5%, rotate ±10°), no flip | 0.654 ± 0.030 | 0.60 ± 0.06 | 0.39 ± 0.04 | **−0.035, reject.** Geometric jitter alone also hurts this model and makes it seed-sensitive. Plausible reasons: the crops are centred on the tool tip with a consistent framing, and SmallCNN ends in a position-specific 4096→4 linear layer (no global pooling), so moving content around costs more than it regularises within 30 epochs. |
| 1d | `r1d_smallcnn_hflip`: 1a + horizontal flip only | 0.621 ± 0.008 | 0.61 ± 0.01 | 0.31 ± 0.03 | **−0.068, reject. Confirms the orientation hypothesis:** flip alone accounts for most of 1b's loss, and the class that drops most is grasper (0.84 → 0.76), the class whose shafts come from both sides. Mirror augmentation (and mirror test-time augmentation) is ruled out for every later model. |
| 2a | `r2a_smallcnn_classweights`: 1a + cross-entropy weighted by inverse class frequency | 0.695 ± 0.010 | 0.66 ± 0.01 | 0.43 ± 0.02 | **+0.006, tie (inside the seed spread).** Rare classes move the right way (clipper +0.015, scissor +0.018) without hurting the others, but the gain has not earned a place on its own. Re-tested on the pretrained model, where it is free. |
| 2b | `r2b_smallcnn_balanced`: 1a + class-balanced sampling (with replacement) | 0.683 ± 0.017 | 0.64 ± 0.01 | 0.43 ± 0.04 | **−0.006, tie, not adopted.** Scissor +0.015 but clipper −0.014 and a wider seed spread (oversampling 143 scissors repeats the same few videos' frames). Seeds 0-1 ran on the cloud workspace (torch 2.14 x86), seed 2 on the VM (torch 2.5.1 arm64). |
| 2c | `r1a_smallcnn_fixed+offsets`: per-class logit offsets maximising macro-F1, tuned NESTED (on the other 4 folds' OOF predictions, applied to the 5th), `scripts/tune_offsets.py` | 0.664 ± 0.015 | 0.63 | 0.38 | **−0.025, reject.** The best offsets do not transfer across videos: the tuned scissor offset ranges from −2.5 to +1.0 between folds, i.e. the "right" class prior depends on which videos you look at. The model is also over-confident (median top probability 0.99, half of OOF predictions above 0.99 yet only 92% correct there), so small offsets change nothing and large ones overcorrect. |

### Rung 3a: frozen ImageNet features + logistic regression (`scripts/linear_probe.py`)
Penultimate features extracted once (no augmentation), StandardScaler +
LogisticRegression, C chosen by inner video-grouped CV on the training folds
(grid 0.01-10). Deterministic, so one run per setting (no seed spread);
differences below ~0.02 should be read as ties.

| Backbone | input | class_weight | macro-F1 | clipper | scissor |
|---|---|---|---|---|---|
| MobileNetV3-L | native 86×128 | balanced | **0.740** | 0.76 | 0.48 |
| MobileNetV3-L | native | none | 0.732 | 0.76 | 0.46 |
| EfficientNet-B0 | native | balanced | 0.730 | 0.74 | 0.49 |
| EfficientNet-B0 | native | none | 0.726 | 0.74 | 0.48 |
| EfficientNet-B0 | x2 172×256 | none | 0.720 | 0.75 | 0.42 |
| EfficientNet-B0 | x2 | balanced | 0.716 | 0.75 | 0.40 |
| MobileNetV3-L | x2 | none | 0.715 | 0.75 | 0.40 |
| MobileNetV3-L | x2 | balanced | 0.711 | 0.74 | 0.40 |
| ResNet-18 | x2 | balanced | 0.657 | 0.65 | 0.36 |
| ResNet-18 | x2 | none | 0.645 | 0.64 | 0.34 |
| ResNet-18 | native | balanced | 0.639 | 0.60 | 0.34 |
| ResNet-18 | native | none | 0.620 | 0.57 | 0.28 |

Reading: (1) frozen MobileNetV3-L / EfficientNet-B0 features with a linear
classifier already beat the best from-scratch CNN (0.689) by ~0.05;
(2) ResNet-18 features are clearly worse (-0.08), so it is not the backbone
to fine-tune first; (3) upsampling 2x does not help frozen features (-0.01
to -0.03; ResNet-18 is the exception, +0.02), so native vs x2 is re-tested
only when fine-tuning; (4) class-balanced weighting is a small, consistent
plus at native resolution (+0.004 to +0.019), mostly on scissor.
(First pass used C grid 0.01-1; EfficientNet picked the edge C=1 in every
fold, so the grid was widened to include 10 and all twelve were re-run.)

## Inference check (after rung 1)
Interim checkpoint = rung-1a recipe trained on all 1402 images (seed 0),
0.3 MB. `python check_submission.py` → PREFLIGHT PASSED. CPU (4-core arm64
VM): 277 images 2.0 s, 1125 images 4.6 s wall including start-up. Tested:
flat folder, `.PNG`, `.DS_Store`/hidden folder, a corrupt PNG (exit 2 with the
path), duplicate names across subfolders. A first preflight run failed
because checkpoint metadata held a `TorchVersion` object that
`torch.load(weights_only=True)` rejects; metadata is now forced to plain JSON
types. Expected held-out macro-F1 of this recipe: ~0.69 (CV).

### Rung 1-2 conclusion
Fixing the bugs is what mattered: 0.534 → 0.689. For this from-scratch
SmallCNN, every augmentation hurt (flip most, because tool orientation is a
real class cue), and none of the three imbalance remedies beat the seed
spread (class weights +0.006, balanced sampler −0.006, nested logit offsets
−0.025). Carried forward: no flips anywhere (training or test time), no
post-hoc offsets; class weights re-tested on the fine-tuned backbones.

## Rungs 3b-3g: fine-tuning ImageNet backbones (Colab T4, torch 2.11, commit 6cb2fa6)
Run with `notebooks/colab_ladder.ipynb`, stage `3b`; results zip checked
before use (sha256 df8b1aea…): all 12 configs × 5 folds × 3 seeds present,
stored configs identical to `toolid/experiments.py`, keys/labels/folds
identical to the local index, no unfilled or duplicated seeds, every
macro-F1 recomputed from the raw logits, finish times evenly spaced (no
disconnect/resume). Fixed for all: native 86×128, no flips, AdamW 3e-4,
wd 0.05, 20-epoch cosine.

Verdicts use `scripts/compare.py`: paired per-seed differences against the
parent config plus a 95% **video-bootstrap** interval (resample the 10
videos), because which videos end up in the held-out set is a bigger source
of noise than the seed. "Tie" = interval clearly includes 0.

| # | Experiment | macro-F1 | clipper F1 | scissor F1 | Δ vs parent (per-seed; video-bootstrap 95%) | Verdict |
|---|---|---|---|---|---|---|
| 3b | `r3b_mbv3_ft`: MobileNetV3-L fine-tuned, no augmentation | 0.796 ± 0.006 | 0.82 ± 0.00 | 0.54 ± 0.03 | vs 1a SmallCNN: +0.107 (+0.091/+0.111/+0.120; [+0.056, +0.186]) | **Keep.** Clear win over every from-scratch model, and +0.056 over the frozen-feature probe of the same backbone (0.740), so fine-tuning is earning its cost. |
| 3b | `r3b_effb0_ft`: EfficientNet-B0 fine-tuned, no augmentation | 0.808 ± 0.017 | 0.84 ± 0.03 | 0.55 ± 0.03 | vs 1a: +0.119 ([+0.081, +0.198]); vs MobileNetV3-L 3b: +0.012 (−0.002/+0.026/+0.011; [−0.002, +0.042]) | **Keep as the second candidate.** Same picture as MobileNetV3-L; the two backbones are tied within noise (EfficientNet is 3× more seed-sensitive here). |
| 3c | `r3c_mbv3_ft_colour`: MobileNetV3-L + colour aug (channel gain ±0.15 log, hue ±8°, saturation ±0.2, brightness/contrast ±0.1) | 0.816 ± 0.012 | 0.85 ± 0.02 | 0.57 ± 0.04 | vs 3b: +0.020 (+0.011/+0.022/+0.028; [−0.000, +0.039]) | **Keep.** Positive for all three seeds, interval just touching 0. Matches the colour analysis: per-video colour cast is a nuisance worth randomising. |
| 3c | `r3c_effb0_ft_colour`: EfficientNet-B0 + colour aug | 0.815 ± 0.014 | 0.86 ± 0.02 | 0.57 ± 0.03 | vs 3b: +0.007 (+0.009/−0.007/+0.019; [−0.006, +0.015]) | **Tie;** kept as the base for 3e-3g so both backbones are compared on the same footing. |
| 3d | `r3d_mbv3_ft_wb`: MobileNetV3-L + gray-world white-balance normalisation (no aug) | 0.784 ± 0.014 | 0.84 ± 0.01 | 0.49 ± 0.03 | vs 3b: −0.012 (+0.016/−0.021/−0.029; [−0.029, +0.004]) | **Reject.** Normalising the colour cast away is worse than randomising it (3c). |
| 3d | `r3d_effb0_ft_wb`: EfficientNet-B0 + white-balance normalisation | 0.783 ± 0.028 | 0.81 ± 0.04 | 0.52 ± 0.05 | vs 3b: −0.025 (−0.044/−0.022/−0.008; [−0.064, −0.011]) | **Reject, clearly.** Interval excludes 0. Gray-world also shifts the brightness/tissue-colour balance the classes partly rely on, and it changes the input statistics the ImageNet weights expect. |
| 3e | `r3e_mbv3_ft_colour_x2`: MobileNetV3-L + colour aug + 2× upsampled input (172×256) | 0.828 ± 0.013 | 0.85 ± 0.02 | 0.60 ± 0.02 | vs 3c: +0.012 (+0.040/−0.013/+0.009; [−0.005, +0.046]) | **Tie at ~4× the compute;** not adopted. Scissor improves (+0.04) but not reliably across seeds. |
| 3e | `r3e_effb0_ft_colour_x2`: EfficientNet-B0 + colour aug + 2× input | 0.828 ± 0.023 | 0.87 ± 0.04 | 0.60 ± 0.04 | vs 3c: +0.013 (+0.005/+0.036/−0.001; [−0.009, +0.045]) | **Tie at ~4× the compute;** not adopted. |
