"""Dataset indexing.

Every sample is identified by its path relative to the data root
(e.g. "train/hook/7013_14951.png"), never by bare filename: 26 filenames
exist in two class folders because they are two different crops (one per
instrument) from the same video frame. Labels come from the class folder,
which is the ground truth the graders use. labels.csv is NOT used (4% of its
labels were deliberately flipped in notebooks/EDA_final.ipynb).
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "cholec-tinytools"
# Fixed, alphabetical. Saved with every checkpoint so predict.py never has to
# guess the index -> name mapping again.
CLASSES = ("clipper", "grasper", "hook", "scissor")
IMAGE_EXTS = {".png"}


@dataclass(frozen=True)
class Sample:
    key: str      # path relative to DATA_ROOT, unique
    path: Path
    label: int    # index into CLASSES
    video: str    # filename prefix before "_"
    frame: int    # filename number after "_"
    folder: str   # original split folder: "train" or "validation"


def parse_name(name: str):
    stem = Path(name).stem
    video, frame = stem.split("_")
    return video, int(frame)


def index_dataset(data_root: Path = DATA_ROOT, splits=("train", "validation")):
    """All labelled images under data_root/<split>/<class>/*.png, pooled.
    Hidden entries (e.g. macOS .DS_Store) and unknown folders are ignored;
    an unexpected class folder raises instead of silently becoming a class."""
    data_root = Path(data_root)
    samples = []
    for split in splits:
        for cls_dir in sorted((data_root / split).iterdir()):
            if cls_dir.name.startswith(".") or not cls_dir.is_dir():
                continue
            if cls_dir.name not in CLASSES:
                raise ValueError(f"unexpected class folder {cls_dir}")
            for p in sorted(cls_dir.iterdir()):
                if p.name.startswith(".") or p.suffix.lower() not in IMAGE_EXTS:
                    continue
                video, frame = parse_name(p.name)
                samples.append(Sample(
                    key=str(p.relative_to(data_root)), path=p,
                    label=CLASSES.index(cls_dir.name), video=video, frame=frame, folder=split,
                ))
    keys = [s.key for s in samples]
    assert len(keys) == len(set(keys))
    return samples
