"""Per-class logit offsets tuned for macro-F1 (rung 2c).

A classifier trained on imbalanced data under-predicts rare classes; adding
a constant b_c to class c's log-probability moves the decision boundaries
without retraining. b is found by coordinate search maximising macro-F1 on
held-out predictions. To keep the estimate honest it is NESTED: for each
test fold, b is tuned on the out-of-fold predictions of the other four folds
only, then applied to the test fold.
"""
import numpy as np
from sklearn.metrics import f1_score

GRID = np.round(np.arange(-3.0, 3.01, 0.1), 2)


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def macro_f1(y, lp, b):
    return f1_score(y, (lp + b).argmax(1), average="macro", labels=list(range(lp.shape[1])), zero_division=0)


def fit_offsets(y, logits, rounds=3):
    lp = log_softmax(logits)
    b = np.zeros(lp.shape[1])
    best = macro_f1(y, lp, b)
    for _ in range(rounds):
        improved = False
        for c in range(1, lp.shape[1]):  # class 0 fixed at 0 (only differences matter)
            for v in GRID:
                trial = b.copy()
                trial[c] = v
                s = macro_f1(y, lp, trial)
                if s > best + 1e-9:
                    best, b, improved = s, trial, True
        if not improved:
            break
    return b


def nested_offsets_predictions(y, logits, folds):
    """Returns (predictions with nested-tuned offsets, list of per-fold offsets)."""
    lp = log_softmax(logits)
    pred = np.empty(len(y), dtype=int)
    used = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        b = fit_offsets(y[tr], logits[tr])
        pred[te] = (lp[te] + b).argmax(1)
        used.append(b.tolist())
    return pred, used
