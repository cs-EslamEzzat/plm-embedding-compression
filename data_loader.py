"""
data_loader.py
==============
Loads the DeepLoc SwissProt localization CSV, builds the multi-label
target matrix, and exposes the homology partitions.

The DeepLoc release stores the data slightly differently across versions
(sometimes one-hot label columns, sometimes a single semicolon-separated
"label" column). This loader detects both and normalises them into:

    df        : the raw dataframe (with ACC, Sequence, Partition)
    Y         : (n_samples, n_labels) binary numpy array
    labels    : ordered list of label names matching Y columns
    partitions: (n_samples,) int array of partition ids
"""

from __future__ import annotations
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import Config, LOCALIZATIONS


def _find_col(df: pd.DataFrame, wanted: str) -> str:
    """Case-insensitive, punctuation-insensitive column matching."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    target = norm(wanted)
    for c in df.columns:
        if norm(c) == target:
            return c
    raise KeyError(
        f"Could not find a column matching '{wanted}'. "
        f"Available columns: {list(df.columns)}"
    )


def _build_label_matrix(df: pd.DataFrame, cfg: Config
                        ) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (Y, labels). Handles two storage formats:
      (a) one binary column per localization, or
      (b) a single column holding semicolon/comma-separated location names.
    """
    # Format (a): try to find one-hot columns
    found = []
    for lab in LOCALIZATIONS:
        try:
            found.append(_find_col(df, lab))
        except KeyError:
            found = []
            break
    if found:
        Y = df[found].to_numpy()
        Y = (Y > 0).astype(int)
        return Y, list(LOCALIZATIONS)

    # Format (b): a single multi-label text column
    candidates = [c for c in df.columns
                  if re.search(r"local|label|target|class", c, re.I)]
    if not candidates:
        raise KeyError(
            "No one-hot label columns and no obvious multi-label text "
            "column found. Please inspect the CSV header."
        )
    col = candidates[0]
    split = df[col].astype(str).apply(
        lambda s: [t.strip() for t in re.split(r"[;,|]", s) if t.strip()]
    )
    labels = sorted({t for row in split for t in row})
    index = {lab: i for i, lab in enumerate(labels)}
    Y = np.zeros((len(df), len(labels)), dtype=int)
    for r, row in enumerate(split):
        for t in row:
            Y[r, index[t]] = 1
    return Y, labels


def load_dataset(cfg: Config):
    """Load the DeepLoc CSV into a tidy structure."""
    df = pd.read_csv(cfg.data_csv)

    acc_col = _find_col(df, cfg.acc_col)
    seq_col = _find_col(df, cfg.seq_col)
    part_col = _find_col(df, cfg.partition_col)

    df = df.rename(columns={acc_col: "ACC",
                            seq_col: "Sequence",
                            part_col: "Partition"})
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip()
    df["Partition"] = df["Partition"].astype(int)

    Y, labels = _build_label_matrix(df, cfg)
    partitions = df["Partition"].to_numpy()

    if cfg.verbose:
        print(f"[data] loaded {len(df)} proteins, "
              f"{len(labels)} labels, "
              f"{len(np.unique(partitions))} partitions")
    return df.reset_index(drop=True), Y, labels, partitions


def make_synthetic_dataset(n: int = 400, D_seqlen: int = 60,
                           n_labels: int = 10, n_parts: int = 5,
                           seed: int = 0):
    """
    Tiny fake dataset for the --smoke test. NOT real data; only checks that
    the whole pipeline runs end to end without errors.
    """
    rng = np.random.default_rng(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = ["".join(rng.choice(list(aa), size=rng.integers(30, D_seqlen)))
            for _ in range(n)]
    df = pd.DataFrame({
        "ACC": [f"SYN_{i:04d}" for i in range(n)],
        "Sequence": seqs,
        "Partition": rng.integers(0, n_parts, size=n),
    })
    # random multi-label targets with at least one label each
    Y = (rng.random((n, n_labels)) < 0.2).astype(int)
    for r in range(n):
        if Y[r].sum() == 0:
            Y[r, rng.integers(0, n_labels)] = 1
    labels = LOCALIZATIONS[:n_labels]
    return df, Y, labels, df["Partition"].to_numpy()


def split_by_partition(partitions: np.ndarray, test_partition: int,
                       inner_parts: List[int]):
    """Boolean masks for the outer test split and the training split."""
    test_mask = partitions == test_partition
    # never allow the held-out partition into training, whatever the caller
    # passed in inner_parts
    safe_inner = [p for p in inner_parts if p != test_partition]
    train_mask = np.isin(partitions, safe_inner)
    return train_mask, test_mask
