"""Dataset indexing and the video-grouped folds."""
from collections import Counter

from toolid.data import CLASSES, index_dataset
from toolid.folds import N_FOLDS, split


def test_index_is_path_keyed_and_complete():
    s = index_dataset()
    assert len(s) == 1402
    assert len({x.key for x in s}) == 1402
    names = Counter(x.path.name for x in s)
    assert sum(1 for c in names.values() if c > 1) == 26  # two crops of one frame, different tools
    assert Counter(CLASSES[x.label] for x in s) == {"hook": 505, "grasper": 492, "clipper": 262, "scissor": 143}


def test_folds_have_no_video_leak_and_cover_everything_once():
    s = index_dataset()
    seen = Counter()
    for f in range(N_FOLDS):
        tr, te = split(s, f)
        assert not ({x.video for x in tr} & {x.video for x in te})
        assert {x.label for x in te} == {0, 1, 2, 3}
        seen.update(x.key for x in te)
    assert len(seen) == 1402 and set(seen.values()) == {1}
