"""Print the video-grouped CV folds and verify their invariants.
Usage: python scripts/describe_folds.py"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolid.data import CLASSES, index_dataset  # noqa: E402
from toolid.folds import N_FOLDS, split  # noqa: E402


def main():
    samples = index_dataset()
    print(f"{len(samples)} samples, {len({s.video for s in samples})} videos")
    seen = set()
    for f in range(N_FOLDS):
        train, test = split(samples, f)
        assert not ({s.video for s in train} & {s.video for s in test}), "video leak"
        c = collections.Counter(CLASSES[s.label] for s in test)
        assert all(c[k] > 0 for k in CLASSES), f"fold {f} missing a class"
        seen |= {s.key for s in test}
        print(f"fold {f}: videos {sorted({s.video for s in test})}  n_test={len(test):4d}  "
              f"n_train={len(train):4d}  test " + " ".join(f"{k}={c[k]}" for k in CLASSES))
    assert seen == {s.key for s in samples}, "every sample must be tested exactly once"
    print("OK: no video in both train and test of any fold; every image tested once; all classes in every fold")


if __name__ == "__main__":
    main()
