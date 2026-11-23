"""
evaluation.py
=============
Multi-label metrics and the homology-aware cross-validation used both
inside the wrapper (for fitness) and for the final reported results.

Key metric: label-averaged Matthews Correlation Coefficient (MCC), which is
robust to the heavy class imbalance typical of localization data.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (matthews_corrcoef, f1_score,
                             jaccard_score, accuracy_score)
from sklearn.multiclass import OneVsRestClassifier

from config import Config


# ----------------------------------------------------------------------
# Classifiers
# ----------------------------------------------------------------------
class ThresholdTunedOvR:
    """One-vs-rest multi-label classifier with PER-LABEL decision thresholds.

    Why this matters: the localization data is heavily imbalanced (e.g. ~300
    peroxisome proteins vs ~9900 cytoplasm). A fixed 0.5 cut-off badly hurts
    rare classes. DeepLoc 2.0 tunes a threshold per label by maximising MCC on
    the training data; we do the same here. This typically gives a large jump
    in macro-averaged MCC/F1 without changing the representation at all.
    """

    def __init__(self, base_factory, n_jobs=-1):
        self.base_factory = base_factory
        self.n_jobs = n_jobs
        self.models_ = []
        self.thresholds_ = None

    def fit(self, X, Y):
        n_labels = Y.shape[1]
        self.models_ = []
        self.thresholds_ = np.full(n_labels, 0.5)
        for j in range(n_labels):
            y = Y[:, j]
            clf = self.base_factory()
            if len(np.unique(y)) < 2:      # degenerate label
                self.models_.append(None)
                continue
            clf.fit(X, y)
            self.models_.append(clf)
            # tune the threshold on the TRAINING data (never the test set)
            p = self._proba_one(clf, X)
            best_t, best_mcc = 0.5, -2.0
            for t in np.linspace(0.05, 0.95, 19):
                pred = (p >= t).astype(int)
                if len(np.unique(pred)) < 2 and len(np.unique(y)) < 2:
                    continue
                m = matthews_corrcoef(y, pred)
                if m > best_mcc:
                    best_mcc, best_t = m, t
            self.thresholds_[j] = best_t
        return self

    @staticmethod
    def _proba_one(clf, X):
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)[:, 1]
        d = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))

    def predict(self, X):
        out = np.zeros((X.shape[0], len(self.models_)), dtype=int)
        for j, clf in enumerate(self.models_):
            if clf is None:
                continue
            p = self._proba_one(clf, X)
            out[:, j] = (p >= self.thresholds_[j]).astype(int)
        return out


def make_classifier(kind: str, C: float, max_iter: int, n_jobs: int):
    """
    Classifiers.

    SPEED: the wrapper evaluates thousands of subsets, so its inner model must
    be fast -> "ridge". The FINAL reported model can be much stronger:
      "mlp"    : small neural network + per-label threshold tuning
                 (closest to DeepLoc 2.0's MLP prediction heads)
      "logreg_t": logistic regression + per-label threshold tuning
      "logreg" / "ridge": plain, no threshold tuning
    """
    if kind == "mlp":
        return ThresholdTunedOvR(
            lambda: MLPClassifier(hidden_layer_sizes=(256, 64),
                                  activation="relu", alpha=1e-4,
                                  learning_rate_init=1e-3,
                                  max_iter=max_iter, early_stopping=True,
                                  n_iter_no_change=8, random_state=0),
            n_jobs=n_jobs)
    if kind == "logreg_t":
        return ThresholdTunedOvR(
            lambda: LogisticRegression(C=C, max_iter=max_iter,
                                       class_weight="balanced",
                                       solver="lbfgs"),
            n_jobs=n_jobs)
    if kind == "logreg":
        base = LogisticRegression(C=C, max_iter=max_iter,
                                  class_weight="balanced", solver="lbfgs")
    elif kind == "ridge":
        base = RidgeClassifier(alpha=1.0 / max(C, 1e-6),
                               class_weight="balanced")
    else:
        raise ValueError(f"Unknown classifier: {kind}")
    return OneVsRestClassifier(base, n_jobs=n_jobs)


def label_avg_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean MCC across labels (skips degenerate single-value labels)."""
    mccs = []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        if len(np.unique(yt)) < 2 and len(np.unique(yp)) < 2:
            continue
        mccs.append(matthews_corrcoef(yt, yp))
    return float(np.mean(mccs)) if mccs else 0.0


def full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mcc_macro": label_avg_mcc(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "jaccard": jaccard_score(y_true, y_pred, average="samples",
                                 zero_division=0),
        "subset_acc": accuracy_score(y_true, y_pred),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      labels: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for j, lab in enumerate(labels):
        yt, yp = y_true[:, j], y_pred[:, j]
        mcc = (matthews_corrcoef(yt, yp)
               if len(np.unique(yt)) > 1 else float("nan"))
        out[lab] = {
            "mcc": mcc,
            "f1": f1_score(yt, yp, zero_division=0),
            "support": int(yt.sum()),
        }
    return out


# ----------------------------------------------------------------------
# Cross-validation over homology partitions
# ----------------------------------------------------------------------
def cv_mcc(X: np.ndarray, Y: np.ndarray, partitions: np.ndarray,
           inner_parts: List[int], cfg: Config) -> float:
    """
    Leave-one-partition-out CV restricted to `inner_parts`. Returns the mean
    label-averaged MCC. This is the fitness signal for the wrapper.
    """
    folds = list(inner_parts)
    k = getattr(cfg, 'wrapper_folds', 0)
    if k and 0 < k < len(folds):
        folds = folds[:k]
    scores = []
    for held in folds:
        tr = np.isin(partitions, [p for p in inner_parts if p != held])
        va = partitions == held
        if tr.sum() == 0 or va.sum() == 0:
            continue
        clf = make_classifier(cfg.wrapper_clf, cfg.wrapper_C,
                              cfg.wrapper_max_iter, cfg.n_jobs)
        clf.fit(X[tr], Y[tr])
        pred = clf.predict(X[va])
        scores.append(label_avg_mcc(Y[va], np.asarray(pred)))
    return float(np.mean(scores)) if scores else 0.0
