# Review of the original code and what we changed

**Bottom line.** Under an honest evaluation (models never tested on a video
they were trained on), the submission as shipped scores **macro-F1 0.11**,
worse than guessing at random (0.24). The pipeline in this repo scores
**0.858** under the same evaluation (3-model ensemble; single model
0.838 ± 0.006). Most of that gain comes from fixing bugs; the rest from a
pretrained backbone, augmentation chosen from the data, and ensembling.

All numbers: video-grouped 5-fold cross-validation over the 1402 labelled
images (10 videos), pooled out-of-fold macro-F1, 3 training seeds.
Full table: [`results_table.md`](results_table.md) (regenerated from stored
raw outputs by `scripts/make_results_table.py`). Detailed evidence for every
claim, with line numbers in the original code: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## The problems that mattered most, in order of impact on macro-F1

**1. `predict.py` mapped class indices to the wrong names.** The checkpoint was
trained with classes in alphabetical order (`clipper, grasper, hook, scissor`);
`predict.py` named its outputs from a hard-coded list in a different order
(`grasper, hook, clipper, scissor`). Three of the four classes were always
reported as another class: on the validation folder this alone takes
macro-F1 from 0.79 to 0.21. The graders' preflight check still passed,
because it only checks the CSV's shape. *Fix:* the checkpoint now stores the
class names in output order along with the architecture and preprocessing,
and `predict.py` reads all three from it (`toolid/checkpoint.py`).

**2. The validation measured memory, not recognition.** The images are frames
from 10 surgery videos, one second apart, and neighbouring frames are nearly
identical. Eight of the ten videos have frames in both the train and the
validation folder, `train.py` pooled both folders and split them randomly by
image, and the shipped model had been trained on the validation folder
itself. So no number in the repo said anything about unseen videos (the
validation folder's 0.79 drops to 0.51 for the same recipe tested on held-out
videos). *Fix:* video-grouped cross-validation (`toolid/folds.py`), every
decision taken on out-of-fold predictions, 3 seeds, and paired
comparisons with a bootstrap over videos (`scripts/compare.py`).

**3. The training code could not train a good model.**
`train.py` never divided pixels by 255, and its colour jitter, applied to
0-255 values, clipped every image to 3-4 distinct values (on the validation
set too); it applied cross-entropy to probabilities instead of logits (a loss
that cannot go below 0.74); it froze the whole backbone while its comment said
"full fine-tune", and validated in training mode. None of this mattered to
the submission anyway, because `predict.py` loaded a different, older model
that `train.py` never produced: a small CNN trained in grayscale on images
squashed to a square, with mean-squared error on probabilities, batch size 2,
and "early stopping" on training loss. The EfficientNet experiment that
claimed 0.94 was not pretrained, used learning rate 0.1, never shuffled its
class-sorted data, and "validated" on its training set (its own results file
says 0.10). *Fix:* one training path (`toolid/trainer.py`, `train.py`) that
produces exactly what `predict.py` loads. Fixing the bugs alone, with the same
small CNN, moves macro-F1 from 0.534 to 0.689 and cuts the seed spread 4×.

**4. Label noise and filename collisions.** `labels.csv`, which the shipped
model learned from, had 4% of its labels deliberately flipped by the EDA
notebook ("to avoid over-confidence"). Its rows were matched to images by
filename, but 26 filenames occur in two class folders (two crops of the same
frame, one per instrument), so 26 rows loaded the wrong image. *Fix:* labels
come from the class folders, which are what the graders use, and every image
is identified by its path. Measured effect on the old model: +0.02, inside
the seed noise; it is still the correct thing to do.

## What we found beyond the bugs

- **Mirroring destroys a real cue.** In this data the hook, clipper and
  scissor shafts enter from the right, while graspers come from both sides
  (`docs/figures/class_means.png`). Horizontal-flip augmentation alone cost
  0.068 macro-F1, most of it on grasper. No flips anywhere, including at test
  time.
- **Hue identifies the video, brightness identifies the tool.** Colour cast
  is explained by the video, not the class (`scripts/colour_analysis.py`), so
  randomising hue and channel gains during training helps (+0.020 on
  MobileNetV3-L); normalising the cast away (gray-world white balance) hurts
  (−0.012 and −0.025).
- **Fine-tuning a pretrained backbone is the big step after the fixes**
  (+0.107 over the bug-fixed small CNN). MobileNetV3-L and EfficientNet-B0
  tie; ResNet-18 features were clearly worse. Mild affine jitter (±10% scale,
  ±5% shift, ±10° rotation) helped the fine-tuned MobileNetV3-L (+0.021) even
  though it hurt the small CNN, whose last layer is position-specific.
- **Averaging three seeds is a reliable +0.020** (95% bootstrap interval
  +0.016 to +0.026), and it makes the result depend less on one training run.

**Shipped:** 3 × MobileNetV3-L, ImageNet-pretrained, fine-tuned 20 epochs at
the native 86×128 with colour + mild affine augmentation, trained on all 1402
images; 25.7 MB; about 13 s for 1400 images on 4 CPU threads.

## What did not help (all kept in the log)
Inverse-frequency class weights (a tie on each of the three models tried), a
class-balanced sampler (tie), per-class decision offsets tuned for macro-F1 (−0.025: the best offsets
differ from video to video), 2× upsampled input (tie at 4× the compute),
white-balance normalisation (worse), adding a second architecture to the
ensemble (+0.005, within noise), ResNet-18 (worse features).

## Duplicate filenames in the output CSV
26 filenames appear in two class folders of the provided data because they
are two crops of the same frame, each centred on a different instrument. The
required output is keyed by filename only, so the two rows for such a pair
cannot be told apart by name. `predict.py` writes exactly one row per image
file (both rows, in a deterministic order: class folders sorted, then
filenames) and notes the repeats on stderr. A scorer that maps filename →
label can keep only one of the two rows and has two possible true labels for
that name, so at most one of the pair can count as correct whatever we
predict; this affects 52 of 1402 images in the provided data. We did not add
a path column or rename files, because that would break the required
two-column format.

## Limits of the evidence
- **10 videos.** One video is 31% of the data and holds 47% of the scissors,
  so fold-level scores swing a lot (the shipped recipe ranges 0.66-0.74 on
  that fold alone). Intervals from resampling videos are wide; the top five
  fine-tuning configurations are statistically tied, and the shipped one was
  chosen on tie-breakers (stability, scissor F1, cost).
- **Best-of-12 selection.** The winning configuration was picked on the same
  cross-validation data it is scored on, which flatters it slightly: expect a
  little under 0.86 on new videos.
- **Assumption about the held-out set.** We assumed it comes from videos we
  have not seen. If it comes from these same 10 videos, scores will be higher
  than our estimate; if its videos look very different (other hospital,
  camera or tools), lower. It must be the same kind of 128×86 crop centred on
  one instrument; other sizes are resized, which we have not evaluated.
- **One end-to-end check.** The full train → checkpoint → `predict.py` path
  was checked on one held-out fold (0.738 vs 0.66-0.70 in CV for that fold, a
  difference explained by 12 clipper images), not on all five.
