"""Progress of a time-sliced CV run: python scripts/cv_status.py EXP"""
import sys
from pathlib import Path

import torch

d = Path(__file__).resolve().parents[1] / "runs" / "cv" / sys.argv[1]
done = sorted(p.name.split(".")[0] for p in d.glob("*.logits.pt"))
part = {p.name.split(".")[0]: torch.load(p, weights_only=False)["epoch"] for p in d.glob("*.state.pt")}
print(f"{sys.argv[1]}: {len(done)}/15 done; in progress: " + ", ".join(f"{k}@{v}" for k, v in sorted(part.items())))
