"""Regenerate the main results table from the stored raw outputs.

Reads only results/**/*_oof.npz (per-image out-of-fold logits or predicted
class indices, per seed), recomputes every metric from scratch, checks each
number against the summary JSON written when the experiment ran (fails
loudly on any disagreement), and writes docs/results_table.md.

    python scripts/make_results_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolid.data import CLASSES  # noqa: E402
from toolid.offsets import nested_offsets_predictions  # noqa: E402

R = ROOT / "results"
SHIPPED_TO_TRUE = np.array([CLASSES.index(n) for n in ["grasper", "hook", "clipper", "scissor"]])

# (section, label, source npz, how to turn it into predictions, summary json to check against)
ROWS = [
    ("Baseline (legacy SmallCNN recipe retrained inside the video-grouped folds)", None, None, None, None),
    ("", "original recipe, labels.csv, **as shipped** (wrong class-name order)", "baseline_legacy_retrain_csv", "pred_shipped",
     ("baseline_legacy_retrain_csv.json", "as_shipped_mapping")),
    ("", "original recipe, labels.csv, class order fixed", "baseline_legacy_retrain_csv", "pred",
     ("baseline_legacy_retrain_csv.json", "correct_mapping")),
    ("", "original recipe, folder labels, class order fixed", "baseline_legacy_retrain_folder", "pred",
     ("baseline_legacy_retrain_folder.json", "correct_mapping")),
    ("Rungs 1-2: SmallCNN from scratch", None, None, None, None),
    ("", "1a bugs fixed (RGB, pad, CE, batch 32, fixed 30-ep cosine)", "cv/r1a_smallcnn_fixed", "logits", "cv/r1a_smallcnn_fixed.json"),
    ("", "1b + flip + affine", "cv/r1b_smallcnn_fixed_aug", "logits", "cv/r1b_smallcnn_fixed_aug.json"),
    ("", "1c + affine only", "cv/r1c_smallcnn_affine", "logits", "cv/r1c_smallcnn_affine.json"),
    ("", "1d + flip only", "cv/r1d_smallcnn_hflip", "logits", "cv/r1d_smallcnn_hflip.json"),
    ("", "2a 1a + class weights", "cv/r2a_smallcnn_classweights", "logits", "cv/r2a_smallcnn_classweights.json"),
    ("", "2b 1a + balanced sampler", "cv/r2b_smallcnn_balanced", "logits", "cv/r2b_smallcnn_balanced.json"),
    ("", "2c 1a + nested per-class logit offsets", "cv/r1a_smallcnn_fixed", "offsets", "cv/r1a_smallcnn_fixed+offsets.json"),
    ("Rung 3a: frozen ImageNet features + logistic regression (deterministic, 1 run)", None, None, None, None),
] + [("", f"{arch} {geo}{' balanced' if bal else ''}", f"cv/r3a_probe_{arch}_{geo}{'_balanced' if bal else ''}", "pred",
      f"cv/r3a_probe_{arch}_{geo}{'_balanced' if bal else ''}.json")
     for arch in ("mobilenet_v3_large", "efficientnet_b0", "resnet18") for geo in ("native", "x2") for bal in (False, True)] + [
    ("Rungs 3b-3g: fine-tuned (Colab T4), native 86x128 unless noted, no flips", None, None, None, None),
] + [("", f"{tag} {arch_label} {desc}", f"cv/r3{tag}_{arch}_{suf}", "logits", f"cv/r3{tag}_{arch}_{suf}.json")
     for arch, arch_label in (("mbv3", "MobileNetV3-L"), ("effb0", "EfficientNet-B0"))
     for tag, suf, desc in (("b", "ft", "plain"), ("c", "ft_colour", "+ colour aug"), ("d", "ft_wb", "+ white-balance norm"),
                            ("e", "ft_colour_x2", "+ colour, 2x input"), ("f", "ft_colour_cw", "+ colour, class weights"),
                            ("g", "ft_colour_affine", "+ colour + affine"))] + [
    ("Ensembles (mean softmax of the seeds' out-of-fold predictions)", None, None, None, None),
    ("", "**SHIPPED: 3-seed MobileNetV3-L 3g**", "cv/r3g_mbv3_ft_colour_affine", "ensemble",
     ("ens", "cv/ensemble_r3g_mbv3_ft_colour_affine+r3g_effb0_ft_colour_affine.json", "B_seed_ensemble", "per_class_B")),
    ("", "3-seed MobileNetV3-L 3g + 3-seed EfficientNet-B0 3g", ("cv/r3g_mbv3_ft_colour_affine", "cv/r3g_effb0_ft_colour_affine"),
     "ensemble2", ("ens", "cv/ensemble_r3g_mbv3_ft_colour_affine+r3g_effb0_ft_colour_affine.json", "C_two_arch_ensemble", "per_class_C")),
]


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def per_class(y, p):
    return f1_score(y, p, labels=[0, 1, 2, 3], average=None, zero_division=0)


def predictions(src, how):
    if how == "ensemble2":
        za, zb = (np.load(R / f"{s}_oof.npz") for s in src)
        return za["labels"], [np.concatenate([softmax(za["logits"]), softmax(zb["logits"])]).mean(0).argmax(1)]
    z = np.load(R / f"{src}_oof.npz")
    y = z["labels"]
    if how == "logits":
        return y, [lg.argmax(1) for lg in z["logits"]]
    if how == "pred":
        return y, list(z["pred"])
    if how == "pred_shipped":
        return y, [SHIPPED_TO_TRUE[p] for p in z["pred"]]
    if how == "offsets":
        return y, [nested_offsets_predictions(y, lg, z["folds"])[0] for lg in z["logits"]]
    if how == "ensemble":
        return y, [softmax(z["logits"]).mean(0).argmax(1)]
    raise ValueError(how)


def stored(check):
    if check is None:
        return None
    if check[0] == "ens":  # scripts/ensemble_oof.py output (per-class rounded to 3 decimals there)
        d = json.load(open(R / check[1]))
        return d[check[2]], d[check[3]], 5e-4
    if isinstance(check, tuple):
        d = json.load(open(R / check[0]))["summary"][check[1]]
    else:
        d = json.load(open(R / check))["summary"]
    return d["macro_f1_mean"], d["per_class_f1_mean"], 1e-9


def main():
    lines = ["| Experiment | macro-F1 (mean ± std over seeds) | clipper | grasper | hook | scissor |",
             "|---|---|---|---|---|---|"]
    for section, label, src, how, check in ROWS:
        if label is None:
            lines.append(f"| **{section}** | | | | | |")
            continue
        y, preds = predictions(src, how)
        pcs = np.array([per_class(y, p) for p in preds])
        mfs = pcs.mean(1)
        ref = stored(check)
        if ref is not None:
            tol = ref[2]
            assert abs(ref[0] - mfs.mean()) < 1e-9, (label, ref[0], mfs.mean())
            for i, c in enumerate(CLASSES):
                assert abs(ref[1][c] - pcs[:, i].mean()) < tol, (label, c)
        sd = f" ± {mfs.std():.3f}" if len(mfs) > 1 else ""
        lines.append(f"| {label} | {mfs.mean():.3f}{sd} | " + " | ".join(f"{v:.2f}" for v in pcs.mean(0)) + " |")
    md = ("<!-- generated by scripts/make_results_table.py from results/**/*_oof.npz; do not edit by hand -->\n"
          "Video-grouped 5-fold CV over all 1402 images (10 videos), pooled out-of-fold macro-F1; "
          "per-class columns are F1 averaged over seeds.\n\n" + "\n".join(lines) + "\n")
    (ROOT / "docs" / "results_table.md").write_text(md)
    print(md)
    print("all recomputed numbers match the stored summaries; wrote docs/results_table.md")


if __name__ == "__main__":
    main()
