"""
baselines.py
============
Baselines that the SS-SA wrapper is compared against, all evaluated under
the SAME classifier and the SAME homology-partitioned test split:

  - full            : use the entire embedding (no selection)
  - pca             : PCA to k components (matched dimensionality)
  - variance        : keep the top-k highest-variance dimensions
  - mutual_info     : keep the top-k dimensions by mutual information

Each returns a transform function mapping X -> X_reduced so the downstream
evaluation is identical across methods.
"""

from __future__ import annotations
from typing import Callable, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


def full_baseline() -> Tuple[str, Callable[[np.ndarray], np.ndarray]]:
    return "full", (lambda X: X)


def pca_baseline(X_train: np.ndarray, k: int, seed: int = 0):
    pca = PCA(n_components=min(k, X_train.shape[1]), random_state=seed)
    pca.fit(X_train)
    return f"pca_{k}", (lambda X: pca.transform(X))


def variance_baseline(X_train: np.ndarray, k: int):
    var = X_train.var(axis=0)
    idx = np.argsort(var)[::-1][:k]
    return f"variance_{k}", (lambda X: X[:, idx])


def mutual_info_baseline(X_train: np.ndarray, Y_train: np.ndarray,
                         k: int, seed: int = 0):
    # aggregate MI across labels (sum), then keep the top-k dimensions
    mi = np.zeros(X_train.shape[1])
    for j in range(Y_train.shape[1]):
        if len(np.unique(Y_train[:, j])) < 2:
            continue
        mi += mutual_info_classif(X_train, Y_train[:, j],
                                  random_state=seed)
    idx = np.argsort(mi)[::-1][:k]
    return f"mutual_info_{k}", (lambda X: X[:, idx])
