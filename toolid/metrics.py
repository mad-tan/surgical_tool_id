"""Evaluation metrics. Macro-F1 is what the graders score."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .data import CLASSES


def report(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "macro_f1": float(np.mean(per_class)),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "per_class_f1": {c: float(f) for c, f in zip(CLASSES, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),  # rows=true, cols=pred
    }


def cv_report(oof_true, oof_pred, fold_ids):
    """Pooled out-of-fold report + per-fold macro-F1."""
    oof_true, oof_pred, fold_ids = map(np.asarray, (oof_true, oof_pred, fold_ids))
    pooled = report(oof_true, oof_pred)
    per_fold = [report(oof_true[fold_ids == f], oof_pred[fold_ids == f])["macro_f1"]
                for f in sorted(set(fold_ids.tolist()))]
    pooled["per_fold_macro_f1"] = per_fold
    pooled["fold_mean"] = float(np.mean(per_fold))
    pooled["fold_std"] = float(np.std(per_fold))
    return pooled


def fmt(r, name=""):
    pc = "  ".join(f"{c} {f:.3f}" for c, f in r["per_class_f1"].items())
    s = f"{name:<38s} macroF1 {r['macro_f1']:.3f}  acc {r['accuracy']:.3f} | {pc}"
    if "per_fold_macro_f1" in r:
        s += "  | folds " + " ".join(f"{x:.2f}" for x in r["per_fold_macro_f1"])
    return s
