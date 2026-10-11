"""Does averaging models help? Evaluated on existing out-of-fold logits, no training.

Pre-declared candidates (chosen before looking, to avoid picking the best of
many combinations on the same OOF data):
  A  one model                         mean over the 3 seeds of single-model macro-F1
  B  3-seed ensemble of one config     mean softmax of seeds 0,1,2 (what train.py ships)
  C  2-architecture ensemble           B of config 1 + B of config 2 (6 members)
Each image's members were all trained without that image's video, exactly as
in CV. 95% video-bootstrap intervals for B-A and C-B.

    python scripts/ensemble_oof.py r3g_mbv3_ft_colour_affine r3g_effb0_ft_colour_affine
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolid.data import CLASSES  # noqa: E402


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main(a, b, n_boot=2000):
    za, zb = (np.load(ROOT / "results" / "cv" / f"{e}_oof.npz") for e in (a, b))
    assert (za["keys"] == zb["keys"]).all()
    y, keys = za["labels"], za["keys"]
    pa, pb = softmax(za["logits"]), softmax(zb["logits"])  # seeds x N x 4
    f = lambda yy, p: f1_score(yy, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)  # noqa: E731
    pcf = lambda yy, p: f1_score(yy, p, average=None, labels=[0, 1, 2, 3], zero_division=0)  # noqa: E731
    single = lambda idx: np.mean([f(y[idx], pa[i][idx].argmax(1)) for i in range(len(pa))])  # noqa: E731
    ensB = lambda idx: f(y[idx], pa.mean(0)[idx].argmax(1))  # noqa: E731
    ensC = lambda idx: f(y[idx], np.concatenate([pa, pb]).mean(0)[idx].argmax(1))  # noqa: E731
    allidx = np.arange(len(y))
    res = {"A_single_mean": single(allidx), "B_seed_ensemble": ensB(allidx), "C_two_arch_ensemble": ensC(allidx)}
    res["per_class_B"] = dict(zip(CLASSES, pcf(y, pa.mean(0).argmax(1)).round(3).tolist()))
    res["per_class_C"] = dict(zip(CLASSES, pcf(y, np.concatenate([pa, pb]).mean(0).argmax(1)).round(3).tolist()))
    videos = np.array([k.split("/")[-1].split("_")[0] for k in keys])
    uv = np.unique(videos)
    idx_of = {v: np.where(videos == v)[0] for v in uv}
    rng = np.random.default_rng(0)
    d_ba, d_cb = [], []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[v] for v in rng.choice(uv, len(uv), replace=True)])
        b_ = ensB(idx)
        d_ba.append(b_ - single(idx))
        d_cb.append(ensC(idx) - b_)
    res["B_minus_A_95"] = np.percentile(d_ba, [2.5, 97.5]).round(3).tolist()
    res["C_minus_B_95"] = np.percentile(d_cb, [2.5, 97.5]).round(3).tolist()
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    out = ROOT / "results" / "cv" / f"ensemble_{a}+{b}.json"
    json.dump({"a": a, "b": b, **res}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
