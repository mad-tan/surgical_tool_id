import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "cholec-tinytools"


def run_predict(data_dir, out, *extra):
    return subprocess.run([sys.executable, str(ROOT / "predict.py"), "--data-dir", str(data_dir), "--out", str(out),
                           *extra], capture_output=True, text=True, cwd=ROOT)


def read_csv(path):
    import csv
    with open(path, newline="") as f:
        return list(csv.reader(f))


@pytest.fixture
def real_images():
    """A few real crops, one per class."""
    return {c: sorted((DATA / "validation" / c).glob("*.png"))[:2] for c in ("clipper", "grasper", "hook", "scissor")}
