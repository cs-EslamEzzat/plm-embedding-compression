"""
config.py
=========
Central configuration for the pipeline of
"How Much Compression Can Protein Language Model Embeddings Tolerate?
 A Scatter Search-Simulated Annealing Study".

The defaults below are the settings used for the reported results.

All tunable knobs live here so experiments are reproducible and
nothing important is hidden inside the code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# The ten subcellular compartments used by the DeepLoc SwissProt dataset.
LOCALIZATIONS: List[str] = [
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell_membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic_reticulum",
    "Lysosome_Vacuole",
    "Golgi_apparatus",
    "Peroxisome",
]


@dataclass
class Config:
    # ----- Paths -----
    # DeepLoc 2.0 SwissProt set (identical to multisub_5_partitions_unique.csv)
    data_csv: str = "Swissprot_Train_Validation_dataset.csv"
    cache_dir: str = "cache"        # embeddings cached here (.npy)
    figures_dir: str = "figures"    # all plots saved here
    results_dir: str = "results"    # metrics / selected masks saved here

    # ----- Data schema (auto-detected if names differ; see data_loader) -----
    acc_col: str = "ACC"
    seq_col: str = "Sequence"
    partition_col: str = "Partition"
    label_cols: List[str] = field(default_factory=lambda: list(LOCALIZATIONS))

    # ----- Embeddings -----
    embedder: str = "esm2"          # "esm2" | "prott5" | "synthetic"
    esm2_model: str = "esm2_t33_650M_UR50D"   # 1280-d
    prott5_model: str = "Rostlab/prot_t5_xl_half_uniref50-enc"  # 1024-d
    embed_batch_size: int = 8
    max_seq_len: int = 1022         # truncate very long sequences

    # ----- Cross-validation (homology-aware via the Partition column) -----
    test_partition: int = 0         # outer held-out partition (0..4)
    inner_cv_partitions: List[int] = field(default_factory=lambda: [1, 2, 3, 4])

    # ----- Downstream classifier used INSIDE the wrapper (must be fast) -----
    wrapper_clf: str = "ridge"     # "logreg" | "ridge"
    wrapper_C: float = 1.0
    wrapper_max_iter: int = 100
    wrapper_folds: int = 2  # folds used inside the wrapper (speed); 0 = all

    # ----- Final classifier used for REPORTED results (can be stronger) -----
    final_clf: str = "logreg_t"
    final_C: float = 1.0
    final_max_iter: int = 1000

    # ----- Fitness: f(mask) = MCC_cv(mask) - lambda * (|mask| / D) -----
    # Weight of the size penalty in  f = MCC - lambda * (size/D).
    # NOTE: 0.05 is far too weak. The maximum possible penalty is lambda itself
    # (when every dimension is used), i.e. 0.05, while accuracy differences
    # between subsets are typically 0.05-0.15 MCC. The optimizer therefore
    # ignores size and returns huge subsets -- which defeats this paper's whole
    # claim of COMPACT representations. 0.30 makes size a real competing force
    # while still letting accuracy dominate.
    parsimony_lambda: float = 0.30

    # ----- Scatter Search parameters -----
    # Minimum subset size: stops the search returning degenerate 1-2 dim models.
    min_dims: int = 30

    ss_pool_size: int = 60          # initial diverse population
    ss_refset_quality: int = 8      # best-by-fitness members
    ss_refset_diverse: int = 8      # most-diverse members
    ss_generations: int = 15        # outer SS iterations (15 in the paper)
    init_density: float = 0.5       # expected fraction of bits ON at init

    # ----- Simulated Annealing (local improvement) parameters -----
    sa_steps: int = 40              # moves per local search
    sa_T0: float = 0.10             # initial temperature
    sa_cooling: float = 0.92        # geometric cooling factor
    sa_flip_k: int = 3              # bits flipped per neighbour move

    # ----- Misc -----
    seed: int = 42
    # Parallel workers for the SEARCH itself (each candidate's SA local search
    # is independent). This is the real bottleneck: the classifier's n_jobs
    # only parallelises across ~10 labels, while a generation evaluates
    # hundreds of candidates. Set to your CPU core count.
    # NOTE: keep n_jobs=1 when search_workers>1, otherwise the two levels of
    # parallelism fight over cores and everything gets SLOWER.
    # Reproducibility: with search_workers > 1 every local search gets its own
    # seeded random stream (independent of the number of workers), so results
    # are identical for any value > 1. The paper used --workers 12.
    search_workers: int = 8

    n_jobs: int = -1
    verbose: bool = True

    def p(self, *parts: str) -> str:
        """Join a path relative to the project root."""
        return str(Path(*parts))
