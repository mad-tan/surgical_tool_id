"""Video-grouped 5-fold cross-validation over train+validation combined.

Why grouped by video: frames 1 s apart from one video are near-identical
and share lighting / white balance / instrument wear, so any split that puts
a video on both sides measures memorisation. The held-out set is assumed to
come from unseen videos.

Why this hand-made assignment rather than sklearn's StratifiedGroupKFold:
with only 10 videos of very unequal size (9 to 440 images), StratifiedGroupKFold
produced folds of 848/1402 images or folds missing a class for most seeds.
Rules used:
  * the 7 large videos (70-440 images) are spread so fold sizes are as even
    as the 440-image video 7201 allows, and every held-out fold contains all
    four classes (so per-fold macro-F1 is defined);
  * the 3 tiny videos (7301: 12, 8610: 22, 8611: 9 images; one or two classes
    each, mostly clipper/scissor) are never a fold of their own (per-fold
    F1 would be meaningless); they are attached to folds, still grouped, so
    no tiny video ever straddles train and test;
  * scissor (the rarest class) is spread over 4 of the 5 folds; 7201 holds
    67 of the 143 scissors, which is unavoidable and makes fold 0 hardest.

Headline metric = macro-F1 over the pooled out-of-fold predictions (each of
the 1402 images predicted once by a model that never saw its video). Per-fold
macro-F1 is reported alongside as a spread estimate.
"""
FOLD_VIDEOS = (
    ("7201",),
    ("9152", "8611"),
    ("7013", "7301"),
    ("7654", "8610"),
    ("7695", "7697", "7414"),
)
N_FOLDS = len(FOLD_VIDEOS)
VIDEO_TO_FOLD = {v: i for i, vids in enumerate(FOLD_VIDEOS) for v in vids}


def fold_of(sample):
    return VIDEO_TO_FOLD[sample.video]


def split(samples, fold):
    """(train_samples, test_samples) for one fold."""
    unknown = {s.video for s in samples} - set(VIDEO_TO_FOLD)
    if unknown:
        raise ValueError(f"videos not assigned to a fold: {sorted(unknown)}")
    train = [s for s in samples if VIDEO_TO_FOLD[s.video] != fold]
    test = [s for s in samples if VIDEO_TO_FOLD[s.video] == fold]
    return train, test
