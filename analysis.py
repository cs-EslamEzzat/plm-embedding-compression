"""
analysis.py
===========
Exploratory data analysis (EDA) and all publication-quality figures.

Figures produced (saved to cfg.figures_dir as 300-dpi PNG + PDF):
  EDA
    01_class_distribution      : proteins per localization
    02_label_cardinality       : how many labels per protein
    03_sequence_length         : sequence-length distribution
    04_partition_balance       : class balance across the 5 partitions
    05_embedding_pca_variance  : explained variance of the embedding
    06_embedding_projection    : 2-D map of proteins coloured by location
  RESULTS
    07_convergence             : SS-SA best fitness over generations
    08_size_over_generations   : selected-dimension count over generations
    09_dims_vs_performance     : the key efficiency curve
    10_per_class_comparison    : full vs selected, per localization
    11_method_comparison       : SS-SA vs baselines (bar chart)
"""

from __future__ import annotations
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import Config

sns.set_theme(style="whitegrid", context="talk")


def _save(fig, cfg: Config, name: str):
    os.makedirs(cfg.figures_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(cfg.figures_dir, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------
def eda_class_distribution(Y: np.ndarray, labels: List[str], cfg: Config):
    counts = Y.sum(axis=0)
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=[labels[i] for i in order], y=counts[order], ax=ax,
                color="#4C72B0")
    ax.set_ylabel("Number of proteins")
    ax.set_xlabel("")
    ax.set_title("Class distribution across localizations")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    _save(fig, cfg, "01_class_distribution")


def eda_label_cardinality(Y: np.ndarray, cfg: Config):
    card = Y.sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(card, bins=range(0, card.max() + 2), discrete=True, ax=ax,
                 color="#55A868")
    ax.set_xlabel("Labels per protein")
    ax.set_ylabel("Count")
    ax.set_title(f"Label cardinality (mean = {card.mean():.2f})")
    _save(fig, cfg, "02_label_cardinality")


def eda_sequence_length(df: pd.DataFrame, cfg: Config):
    lengths = df["Sequence"].str.len()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(lengths, bins=50, ax=ax, color="#C44E52")
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Count")
    ax.set_title(f"Sequence-length distribution (median = {lengths.median():.0f})")
    _save(fig, cfg, "03_sequence_length")


def eda_partition_balance(Y: np.ndarray, partitions: np.ndarray,
                          labels: List[str], cfg: Config):
    parts = np.unique(partitions)
    mat = np.zeros((len(labels), len(parts)))
    for pi, p in enumerate(parts):
        mat[:, pi] = Y[partitions == p].sum(axis=0)
    mat = mat / mat.sum(axis=0, keepdims=True)  # column-normalise
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"P{p}" for p in parts],
                yticklabels=labels, ax=ax)
    ax.set_title("Class proportion within each homology partition")
    _save(fig, cfg, "04_partition_balance")


def eda_embedding_variance(X: np.ndarray, cfg: Config):
    from sklearn.decomposition import PCA
    k = min(100, X.shape[1])
    pca = PCA(n_components=k, random_state=cfg.seed).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(range(1, k + 1), cum, marker="o", ms=3, color="#8172B3")
    ax.axhline(0.95, ls="--", color="grey", lw=1)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Embedding intrinsic dimensionality")
    _save(fig, cfg, "05_embedding_pca_variance")


def eda_embedding_projection(X: np.ndarray, Y: np.ndarray,
                             labels: List[str], cfg: Config):
    """2-D map of proteins, coloured by their (first) localization label."""
    try:
        import umap
        reducer = umap.UMAP(random_state=cfg.seed)
        method = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=cfg.seed,
                       init="pca", perplexity=30)
        method = "t-SNE"
    # subsample for speed if large
    n = X.shape[0]
    idx = (np.random.default_rng(cfg.seed).choice(n, 2000, replace=False)
           if n > 2000 else np.arange(n))
    Z = reducer.fit_transform(X[idx])
    dom = np.array([labels[i] if Y[r].sum() else "None"
                    for r, i in zip(idx, Y[idx].argmax(axis=1))])
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=dom, s=18, ax=ax,
                    palette="tab10", linewidth=0, alpha=0.8)
    ax.set_title(f"{method} projection of PLM embeddings")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    _save(fig, cfg, "06_embedding_projection")


# ----------------------------------------------------------------------
# Results figures
# ----------------------------------------------------------------------
def fig_convergence(history, cfg: Config):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(range(1, len(history.best_fitness) + 1),
            history.best_fitness, marker="o", ms=4, color="#4C72B0")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (MCC - lambda*size)")
    ax.set_title("SS-SA convergence")
    _save(fig, cfg, "07_convergence")


def fig_size_over_generations(history, D: int, cfg: Config):
    fig, ax = plt.subplots(figsize=(9, 6))
    frac = [s / D for s in history.best_size]
    ax.plot(range(1, len(frac) + 1), frac, marker="s", ms=4, color="#55A868")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Selected dimensions (fraction of D)")
    ax.set_title("Representation size over the search")
    _save(fig, cfg, "08_size_over_generations")


def fig_dims_vs_performance(curve: pd.DataFrame, cfg: Config):
    """curve columns: dims, mcc_macro, method"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=curve, x="dims", y="mcc_macro", hue="method",
                 marker="o", ax=ax)
    ax.set_xlabel("Number of retained dimensions")
    ax.set_ylabel("Test MCC (macro)")
    ax.set_title("Efficiency: performance vs representation size")
    _save(fig, cfg, "09_dims_vs_performance")


def fig_per_class_comparison(full_pc: Dict, sel_pc: Dict,
                             labels: List[str], cfg: Config):
    rows = []
    for lab in labels:
        rows.append({"loc": lab, "metric": "MCC",
                     "Full": full_pc[lab]["mcc"], "SS-SA": sel_pc[lab]["mcc"]})
    df = pd.DataFrame(rows).melt(id_vars=["loc", "metric"],
                                 var_name="method", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="loc", y="value", hue="method", ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("MCC")
    ax.set_title("Per-localization MCC: full embedding vs SS-SA subset")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    _save(fig, cfg, "10_per_class_comparison")


def fig_method_comparison(summary: pd.DataFrame, cfg: Config):
    """summary columns: method, mcc_macro, f1_macro, dims"""
    fig, ax = plt.subplots(figsize=(11, 6))
    melted = summary.melt(id_vars=["method", "dims"],
                          value_vars=["mcc_macro", "f1_macro"],
                          var_name="metric", value_name="value")
    sns.barplot(data=melted, x="method", y="value", hue="metric", ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("Score (test partition)")
    ax.set_title("Method comparison on the held-out partition")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, cfg, "11_method_comparison")
