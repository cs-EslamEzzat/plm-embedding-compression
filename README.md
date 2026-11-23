# How Much Compression Can Protein Language Model Embeddings Tolerate?

Code and results for the article

> **How Much Compression Can Protein Language Model Embeddings Tolerate? A Scatter Search–Simulated Annealing Study**
> Eslam E. M. Attia, Department of Computer Science, Minia University, Egypt.

The study measures how many dimensions of a frozen ESM-2 650M protein embedding
(1,280 dimensions, mean-pooled) multi-label subcellular localization needs. It
uses the five homology partitions of the DeepLoc 2.0 SwissProt benchmark
(28,303 proteins, ten compartments) and the independent DeepLoc 2.0 Human
Protein Atlas (HPA) test set (1,717 proteins). The compared reductions are:

- random subsets;
- variance and mutual-information filters;
- an L1-regularized (Lasso) embedded selector;
- a hybrid Scatter Search–Simulated Annealing (SS–SA) wrapper;
- PCA at the same size.

Every reduction is scored with the same linear classifier, whose
regularization is re-tuned for every representation.

## Main results (five folds, regularization tuned per size)

| Representation | Dimensions | Macro MCC | % of full |
|---|---|---|---|
| Full embedding | 1,280 | 0.571 ± 0.018 | 100 |
| PCA | 188.6 | 0.543 ± 0.017 | 95.2 |
| L1 embedded (Lasso) | 188.6 | 0.510 ± 0.011 | 89.4 |
| SS–SA wrapper | 188.6 ± 14.1 | 0.508 ± 0.012 | 89.0 |
| Mutual information | 188.6 | 0.499 ± 0.006 | 87.4 |
| Random subsets | 188.6 | 0.497 ± 0.010 | 87.1 |
| Variance filter | 188.6 | 0.496 ± 0.013 | 87.0 |

- **Accuracy retained by random subsets:**
  - 15% of the dimensions keep 87% of full-embedding MCC.
  - 512 dimensions keep 96%.
  - 896 dimensions keep 99%.
  - No subset reaches the full embedding on average.
- **Fixed regularization is misleading.** With regularization fixed at C = 1,
  subsets appear to match or beat the full embedding. That apparent result is
  an artefact.
- **Multivariate selection gains little.** The SS–SA wrapper and the L1
  selector perform alike (p = 0.63) and beat random subsets by about 0.01 MCC.
  The univariate filters do not beat random subsets.
- **External test set.** On the HPA set, the full embedding and PCA lead, and
  the selected subsets perform like random ones.

## Repository layout

```
config.py            all settings (defaults = settings used in the paper)
data_loader.py       DeepLoc CSV -> label matrix and homology partitions
embeddings.py        ESM-2 / ProtT5 mean-pooled embeddings, cached in cache/
optimizer.py         SS-SA wrapper (Scatter Search + Simulated Annealing)
evaluation.py        classifiers, per-label thresholds, metrics, homology-aware CV
baselines.py         full / PCA / variance / mutual-information baselines
analysis.py          exploratory and search figures written by main.py
main.py              one fold: embeddings, SS-SA search, evaluation
run_all_folds.py     main.py on all five folds, then aggregate_folds.py
aggregate_folds.py   five-fold summary and paired tests of the main.py runs
rerun_baselines.py   earlier fixed-C re-evaluation with random subsets (runs/V2_*)
gap_fixes.py         final evaluation: L1 selector, C tuned per size, HPA test
                     set, consensus core  -> runs/V4/  (all reported numbers)
paper/make_figures.py      Figures 2-11 of the paper from runs/V4
paper/paper_statistics.py  all statistics of the paper from runs/V4
runs/                saved results (see below)
```

Saved results in `runs/`:

| Path | Content |
|---|---|
| `runs/fold_N/results/selected_mask.npy` | SS–SA subset of fold N (boolean mask over 1,280 dims) |
| `runs/fold_N/results/*.csv, run_summary.json` | output of `main.py` for fold N |
| `runs/fold_N/results_v2/`, `runs/V2_*.csv` | output of `rerun_baselines.py` |
| `runs/FINAL_*.csv` | output of `aggregate_folds.py` (preliminary evaluation, C fixed at 1) |
| `runs/V4/` | output of `gap_fixes.py`: every reported number (both C regimes, curves, HPA, consensus, L1 and MI selections, chosen C) |

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Install a CUDA build of PyTorch if a GPU is available. A GPU is used only
once, to compute the embeddings; everything else runs on CPU.

## Data

Both files come from the DeepLoc 2.0 page,
<https://services.healthtech.dtu.dk/services/DeepLoc-2.0/>. Download them into
the repository root:

```bash
curl -O https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv
curl -O https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/hpa_testset.csv
```

| File | MD5 |
|---|---|
| `Swissprot_Train_Validation_dataset.csv` | `96ef595bab0a71e7a21049f21b2ce2e8` |
| `hpa_testset.csv` | `8956fe1343e9a6b01520ab974f94e94e` |

The SwissProt file is identical to `multisub_5_partitions_unique.csv`. Either
name works: pass `--data-csv <file>` or edit `data_csv` in `config.py`.

## Usage

**1. Self-test (synthetic data, about a minute, no download needed)**

```bash
python main.py --smoke
```

**2. SS–SA search and first evaluation, all five folds**

```bash
python run_all_folds.py --generations 15 --final-clf logreg_t --workers 12
```

- The first run computes the ESM-2 embeddings and caches them in `cache/`.
- The search is resumable: rerun the same command after an interruption.
- Any `--workers` value above 1 gives identical results, because every local
  search has its own seeded random stream.

**3. Final evaluation (L1 selector, C tuned per size, HPA test set)**

```bash
python gap_fixes.py --workers 12 --out runs/V4_rerun
```

The script reads the subsets from `runs/fold_N/results/`. The shipped
`runs/V4/` is the output used in the paper.

**4. Statistics and figures of the paper**

```bash
python paper/paper_statistics.py > paper/statistics_report.txt
python paper/make_figures.py
```

Both read `runs/V4` by default. Pass another folder as the first argument to
use a rerun.

- Figures 2–11 are written to `paper/figures/`.
- Figures 1, 12 and 13 are drawn by `main.py`. Their files in `figures/` are:
  - Figure 1: `01_class_distribution` and `02_label_cardinality`;
  - Figure 12: `07_convergence` and `08_size_over_generations`;
  - Figure 13: `05_embedding_pca_variance`.

## Settings used in the paper

| Component | Setting |
|---|---|
| Embedding | ESM-2 `esm2_t33_650M_UR50D`, last layer, mean over residues, max 1,022 residues |
| SS–SA fitness | ridge proxy, MCC on 2 inner partitions − 0.30 × (size / 1,280) |
| SS–SA search | pool 60, reference set 8 + 8, 15 generations |
| Local search | simulated annealing, 40 steps, T0 0.10, cooling 0.92 |
| Size limit | minimum 30 dimensions; seed 42 |
| Final classifier | one-vs-rest logistic regression, class-balanced, per-label MCC thresholds |
| Regularization | C tuned on {1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 1} by inner CV on the training partitions, then refit |
| L1 selector | liblinear L1 one-vs-rest, C found by bisection until the union of non-zero coefficients reaches the SS–SA size |
| Hardware | embeddings on an NVIDIA RTX 4050 (6 GB); search and evaluation on a 16-thread CPU with 12 workers |

The search took 253–435 minutes per fold (mean 317). The final evaluation took
53–57 minutes per fold.

## Citation

See `CITATION.cff`. The article reference will be added on publication.

## License

MIT. See `LICENSE`. The DeepLoc 2.0 datasets are not redistributed here and
remain under the terms of their providers.
