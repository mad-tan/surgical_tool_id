# Walkthrough: how this repo works and how to defend it

Written for someone who did not write the code. Read [`REVIEW.md`](REVIEW.md)
first for what was wrong with the original; this file explains what is here now,
why, and the questions you are likely to be asked.

## The task in one paragraph
Each image is a 128×86 crop from a laparoscopic (keyhole surgery) video,
centred on the tip of one instrument. We must say which of four instruments
it is: `clipper`, `grasper`, `hook` or `scissor`. The graders run
`python predict.py --data-dir <DIR> --out <CSV>` on images we have never seen
and score **macro-F1**: the F1 score of each class, averaged with equal
weight. Scissor is only 10% of the data but counts for 25% of the score, so
rare classes matter as much as common ones.

## What happens when `predict.py` runs
1. **Find images**: every `.png` (any case) directly in `<DIR>` and one level
   down (class folders), skipping hidden files such as `.DS_Store`.
2. **Load each image** as RGB at 128×86 (`toolid/preprocess.load_rgb`). If any
   file is unreadable, list them all and exit with status 2: we refuse to
   guess rather than silently predicting from a blank image.
3. **Load `checkpoints/model.pt`**. It holds the architecture name, the class
   names in the model's output order, the preprocessing settings and three
   sets of weights. The architecture is built empty (`weights=None`, nothing
   downloaded) and our weights are loaded into it.
4. **Preprocess** exactly as in training, minus augmentation: scale to [0,1],
   keep the native 86×128 size, normalise with ImageNet mean/std.
5. **Predict**: each of the three models gives class probabilities; we
   average them and take the most likely class.
6. **Write the CSV**: header `filename,predicted_class`, one row per image.

## Tour of every file

**Top level**
- `predict.py`: the graders' entry point (described above). CPU only; picks
  a sensible thread count (the cores available to the process, capped at 8,
  or `OMP_NUM_THREADS`).
- `train.py`: trains the final model on all labelled images with a named
  recipe, one ensemble member per seed, and writes the checkpoint. Can pause
  and resume (`--budget`), and can leave out one fold (`--exclude-fold`) for
  the end-to-end check.
- `check_submission.py`, `contract.py`: the graders' own preflight check,
  unchanged. It runs `predict.py` on the validation folder and checks the
  CSV's format (not its correctness).
- `requirements.txt`: tested version ranges; `README.md`: how to install,
  run, test and reproduce.
- `checkpoints/model.pt`: the shipped model (3 × MobileNetV3-L, 25.7 MB).
- `data/cholec-tinytools/{train,validation}/<class>/*.png`: the provided
  data, unchanged. We use both folders together, with the folder name as label.

**`toolid/`: the library everything else uses**
- `data.py`: lists all labelled images. Each image is identified by its path
  (not its filename, which can repeat) and carries its label (from the folder),
  video ID and frame number (from the filename `<video>_<frame>.png`).
- `folds.py`: the 5 cross-validation folds. Each fold is a set of whole
  videos, so a model is never tested on a video it trained on. The file
  explains how the 10 videos were assigned.
- `preprocess.py`: image loading and the deterministic preprocessing shared
  by training and prediction.
- `augment.py`: random training-time changes to images (colour, small
  shifts/zoom/rotation; flips exist in the code but are not used by the
  shipped recipe).
- `models.py`: builds the networks (a small CNN and three ImageNet
  architectures). Pretrained weights are read from `weights/` only when
  training asks for them.
- `trainer.py`: the training loop. Fixed schedule (warm-up, then cosine
  decay over a fixed number of epochs), state saved every epoch so a run can
  be paused and resumed with bit-identical results.
- `experiments.py`: every recipe we tried, by name (e.g.
  `r3g_mbv3_ft_colour_affine` = rung 3g, MobileNetV3-L, fine-tuned, colour +
  affine augmentation), and the groups run together on Colab.
- `checkpoint.py`: saves/loads the self-describing checkpoint (stored in
  float16 to halve the size, computed in float32).
- `metrics.py`: macro-F1, per-class F1 and confusion matrices.
- `offsets.py`: the per-class decision offsets we tested and rejected.

**`scripts/`: evaluation and analysis**
- `cv.py`: runs one recipe through the 5 folds × 3 seeds and stores the
  out-of-fold outputs in `results/cv/`. `run_slices.sh` and `cv_status.py`
  run it in short slices (needed only in the sandbox this was built in, which
  limited each command to 3 minutes).
- `describe_folds.py`: prints the folds and checks them (no leaks).
- `baseline_legacy.py`: reproduces the "before" numbers by retraining the
  original small CNN recipe inside the folds; reads the original files from
  git history.
- `linear_probe.py`: ranks backbones by frozen features + logistic regression.
- `compare.py`: is recipe A better than recipe B? Paired by seed, with a
  bootstrap over videos.
- `ensemble_oof.py`: does averaging models help?
- `tune_offsets.py`: the rejected per-class offsets experiment.
- `heldout_check.py`: the end-to-end check through `predict.py`.
- `make_results_table.py`: rebuilds `docs/results_table.md` from the stored
  raw outputs and checks every number.
- `make_data_figures.py`, `class_mean_images.py`, `colour_analysis.py`: the
  data inspection behind the figures and the colour/flip decisions.

**Other folders**
- `results/`: raw out-of-fold outputs (`*_oof.npz`) and summaries (`*.json`)
  for every experiment.
- `docs/`: `REVIEW.md` (the written review), `EXPERIMENTS.md` (detailed log),
  `results_table.md`, this file, and `figures/`.
- `notebooks/colab_ladder.ipynb`: runs a group of experiments on a Colab GPU
  and downloads the results.
- `tests/`: 14 automated checks (`python -m pytest`), about 15 s on a CPU.
- Not in git: `weights/` (ImageNet weights, downloaded per README; only for
  training) and `runs/` (temporary training state).

## Key decisions in one line each
- Evaluate on **unseen videos**, never unseen frames of seen videos.
- **Macro-F1 pooled over all out-of-fold predictions**, 3 seeds, paired
  comparisons; a change is kept only if it beats the noise.
- **Labels from folders, images keyed by path**; `labels.csv` (4% flipped) dropped.
- **No stretching** (native 86×128) and **no flips** (left/right is a real cue).
- **Colour randomised, not normalised** (hue identifies the video, brightness the tool).
- **MobileNetV3-L fine-tuned**, colour + mild affine augmentation, 20 epochs.
- **Average 3 seeds** (+0.02, reliable).
- **Self-describing checkpoint** so the class order can never drift again.

## Questions a reviewer is likely to ask

**1. Why not just use the provided train/validation folders?**
Eight of the ten videos have frames in both, and frames one second apart are
nearly identical, so the validation folder mostly tests whether the model
remembers a video. For a held-out set from new videos that is badly
optimistic: the old model's 0.79 on the validation folder is 0.51 on unseen
videos.

**2. How were the folds made? Why not scikit-learn's GroupKFold or leave-one-video-out?**
The videos are very unequal (9 to 440 images). sklearn's stratified group
split produced folds of up to 848 images or folds missing a class. We
assigned videos by hand so each fold has all four classes and similar sizes,
and attached the three tiny videos (9-22 images) to larger folds. One video
alone holds 31% of the images, so perfectly balanced folds are impossible.
Leave-one-video-out would give folds with one or two classes, where per-fold
macro-F1 is undefined.

**3. What does "pooled out-of-fold macro-F1" mean?**
Every image is predicted once, by the model trained on the folds that do not
contain its video. We then compute macro-F1 over all 1402 predictions
together. That is more stable than averaging five per-fold scores, because
some folds contain only 7 scissors.

**4. Flipping is standard augmentation. Why is it off?**
Because in this data it changes the answer. Hook, clipper and scissor shafts
come in from the right; graspers come from both sides
(`docs/figures/class_means.png`). A mirrored hook looks like something only a
grasper does. We measured it: flips alone cost 0.068 macro-F1, mostly on
grasper. For the same reason there is no flip at test time.

**5. Why colour augmentation but not colour normalisation?**
The colour cast (hue, white balance) tells you which video an image is from,
almost nothing about the tool, so the model should learn to ignore it;
randomising it during training does that (+0.020). Brightness and contrast do
carry tool information (the hook's white tip). Gray-world white balance
removes the cast but also distorts that information, and it made results worse.

**6. Why MobileNetV3-L?**
With frozen features it ranked first (tied with EfficientNet-B0; ResNet-18
was 0.08 worse). Fine-tuned, the two tie within noise; MobileNetV3-L's best
recipe had the highest mean, the smallest seed spread, the best scissor F1,
and it is about half the compute. A bigger model is not obviously better with
1400 images from 10 videos.

**7. Why not resize to 224×224 like ImageNet models expect?**
Stretching 128×86 to a square distorts the tool shapes. We tested 2×
upsampling without distortion (172×256): a tie, at 4× the compute.

**8. Why an ensemble of exactly three seeds?**
Averaging three seeds improved every one of the 12 recipes, by +0.018 to
+0.034, with a bootstrap interval that excludes zero. More seeds would likely
add a little; three keeps the checkpoint at 26 MB and prediction at about 10 s.
Adding EfficientNet-B0 models gave +0.005 (noise) for twice the size.

**9. What does 0.858 mean, and what doesn't it mean?**
It is our estimate of macro-F1 on new videos of the same kind. It is slightly
optimistic (we picked the best of 12 recipes on the same data), and it rests
on only 10 videos, so the true number could easily be several points either
way. If the held-out images come from the same 10 videos, expect higher; from
a different hospital or camera, lower.

**10. Scissor is only 10% of the data. Why didn't class weighting help?**
We tried inverse-frequency loss weights (three times), a balanced sampler, and
per-class decision offsets. None beat the noise. The scissor errors come from
scissor looking like the other tools in particular videos, not from the model
ignoring a rare class. Reweighting mostly trades clipper errors for scissor
errors. The offsets that maximised macro-F1 differed from video to video.

**11. How is the checkpoint loaded? Could the class order break again?**
`torch.load(..., weights_only=True)` reads a plain dictionary: architecture
name, class names in output order, preprocessing, weights. The network is
built with `weights=None` (no download) and our weights loaded into it. The
class names come from the same file as the weights, so they cannot disagree.
A test checks the order and that loading never calls a download function.

**12. What happens with two images that share a filename?**
In the provided data, 26 names appear in two class folders (two crops of one
frame, different tools). `predict.py` writes one row for each image, so both
appear. A CSV keyed only by filename cannot tell those two rows apart, so a
filename-keyed scorer can credit at most one of the pair. That is a property
of the required format; we did not change the format to work around it.

**13. What breaks if the held-out images are different?**
- Different size or grayscale: resized to 128×86 / converted to RGB
  (handled, tested, but accuracy on such images is untested).
- Flat folder or class subfolders: both handled; deeper nesting is not read
  (the graders' format is one level).
- Corrupt file: the script stops with exit code 2 and names the file. That is
  deliberate but means one bad file gives no CSV; easy to change if the
  graders prefer a guess.
- Upper-case `.PNG`: handled. Non-PNG files: ignored.
- Images from a new hospital or camera, or a fifth instrument: no code
  failure, but accuracy will drop, and a fifth tool will be forced into one
  of the four classes.

**14. How were the hyperparameters chosen?**
Fixed in advance, not tuned: AdamW, learning rate 3e-4, weight decay 0.05,
20 epochs, cosine schedule, batch 32. No early stopping, so no test data
leaks into the choice of epoch. Tuning them with nested cross-validation is
one of the obvious next steps.

**15. Can we trust the Colab results and the reproducibility?**
The zip was checked before use: all 12 × 5 × 3 runs present, configs
identical to the repo's, every metric recomputed from the raw outputs, the
commit hash recorded. Training is seeded; on CPU a paused and resumed run is
bit-identical (tested). GPU runs are not bit-identical to CPU runs, which is
why we report spreads over seeds rather than single numbers.
`scripts/make_results_table.py` rebuilds the main table from stored outputs.

**16. Did you use the video ID or frame number from the filename?**
No. It would be a shortcut (tools follow the stages of surgery), and a
single-frame classifier should not depend on it.

**17. What would you do with more time?**
Get more videos (the main limit); nested tuning of learning rate and epochs;
look at the remaining clipper/scissor confusions image by image; test more
seeds and an EfficientNet + MobileNet ensemble on a larger set; calibrate
probabilities if someone needs confidence scores.

## Numbers to remember
| | macro-F1 |
|---|---|
| Random guessing | 0.24 |
| As shipped, honest evaluation | 0.11 |
| Same model, class order fixed | 0.51 |
| Bugs fixed, same small CNN | 0.69 |
| Fine-tuned MobileNetV3-L + colour + affine (one model) | 0.838 ± 0.006 |
| **Shipped 3-seed ensemble** | **0.858** |
