"""
main.py
=======
End-to-end pipeline for Paper 1.

Stages:
  1. Load data (real DeepLoc CSV, or synthetic for --smoke)
  2. Generate / load cached PLM embeddings
  3. Exploratory data analysis figures
  4. Run the SS-SA wrapper on the training partitions
  5. Evaluate the selected subset on the held-out test partition
  6. Run baselines (full / PCA / variance / mutual-info)
  7. Build the dims-vs-performance efficiency curve
  8. Save all metrics, the selected mask, and every figure

Usage:
  python main.py --smoke                 # fast synthetic end-to-end test
  python main.py                         # full run (needs data + embeddings)
  python main.py --embedder prott5       # switch embedder
  python main.py --test-partition 2      # choose the held-out fold
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import Config, LOCALIZATIONS
import data_loader as dl
import embeddings as emb
import evaluation as ev
import baselines as bl
import analysis as viz
from optimizer import SSSAOptimizer


def parse_args() -> Config:
    cfg = Config()
    ap = argparse.ArgumentParser(description="Paper 1 pipeline")
    ap.add_argument("--smoke", action="store_true",
                    help="run on tiny synthetic data to test the pipeline")
    ap.add_argument("--data-csv", default=cfg.data_csv)
    ap.add_argument("--embedder", default=cfg.embedder,
                    choices=["esm2", "prott5", "synthetic"])
    ap.add_argument("--test-partition", type=int, default=cfg.test_partition)
    ap.add_argument("--lambda", dest="plam", type=float,
                    default=cfg.parsimony_lambda)
    ap.add_argument("--generations", type=int, default=cfg.ss_generations)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel CPU workers for the search "
                         "(set to your core count, e.g. 16)")
    ap.add_argument("--final-clf", default=None,
                    choices=["logreg", "logreg_t", "mlp"],
                    help="final reported classifier (default logreg_t)")
    ap.add_argument("--min-dims", type=int, default=None,
                    help="minimum dimensions per solution (default 30)")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any saved checkpoint and start the search over")
    ap.add_argument("--sample", type=int, default=0,
                    help="use only N proteins (quick test); 0 = all")
    ap.add_argument("--fast", action="store_true",
                    help="lighter search: results in minutes (testing)")
    a = ap.parse_args()

    cfg.data_csv = a.data_csv
    cfg.embedder = "synthetic" if a.smoke else a.embedder
    cfg.test_partition = a.test_partition
    cfg.parsimony_lambda = a.plam
    cfg.ss_generations = a.generations
    cfg.seed = a.seed
    if a.smoke:
        # shrink the search so the smoke test finishes in seconds
        cfg.ss_pool_size = 10
        cfg.ss_refset_quality = 4
        cfg.ss_refset_diverse = 4
        cfg.ss_generations = 3
        cfg.sa_steps = 4
        cfg.wrapper_max_iter = 80
        cfg.inner_cv_partitions = [1, 2, 3, 4]
    cfg._smoke = a.smoke  # type: ignore[attr-defined]
    cfg.sample = getattr(a, "sample", 0)  # type: ignore[attr-defined]
    if getattr(a, "fast", False):
        cfg.ss_pool_size = 20
        cfg.ss_generations = 8
        cfg.sa_steps = 10
        cfg.wrapper_folds = 1
        cfg.wrapper_max_iter = 50
        print("[mode] FAST MODE: lighter search (testing). Drop --fast for final run.")
    cfg.resume = not getattr(a, "restart", False)  # type: ignore[attr-defined]
    if getattr(a, "min_dims", None) is not None:
        cfg.min_dims = a.min_dims
    if getattr(a, "final_clf", None):
        cfg.final_clf = a.final_clf
    if getattr(a, "workers", None) is not None:
        cfg.search_workers = a.workers
        # avoid nested parallelism (workers x classifier threads = slowdown)
        cfg.n_jobs = 1 if a.workers > 1 else -1
        print(f"[mode] parallel search: {a.workers} workers")
    return cfg


def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # 1. data ----------------------------------------------------------
    if getattr(cfg, "_smoke", False):
        print("[mode] SMOKE TEST on synthetic data (results are meaningless)")
        df, Y, labels, parts = dl.make_synthetic_dataset(
            n=150, n_labels=6, seed=cfg.seed)
    else:
        df, Y, labels, parts = dl.load_dataset(cfg)
        _n = getattr(cfg, "sample", 0)
        if _n and 0 < _n < len(df):
            _rng = np.random.default_rng(cfg.seed)
            _idx = np.sort(_rng.choice(len(df), _n, replace=False))
            df = df.iloc[_idx].reset_index(drop=True); Y = Y[_idx]; parts = parts[_idx]
            print(f"[mode] SAMPLE MODE: {_n} proteins (testing only)")

    # 2. embeddings ----------------------------------------------------
    X = emb.get_embeddings(df["Sequence"].tolist(), cfg)
    X = X.astype(np.float32)
    D = X.shape[1]
    print(f"[emb] embedding matrix: {X.shape}")

    # standardise features (fit on training partitions only, below)
    # CRITICAL: inner CV partitions must be EVERY partition except the held-out
    # one. A fixed config list silently leaks the test partition into training
    # for every fold except partition 0.
    all_parts = sorted(int(p) for p in np.unique(parts))
    cfg.inner_cv_partitions = [p for p in all_parts
                               if p != cfg.test_partition]
    print(f"[split] test partition = {cfg.test_partition}   "
          f"inner CV partitions = {cfg.inner_cv_partitions}")

    train_mask, test_mask = dl.split_by_partition(
        parts, cfg.test_partition, cfg.inner_cv_partitions)
    assert not np.any(train_mask & test_mask), \
        "LEAK: training rows overlap the test partition"
    mu = X[train_mask].mean(0)
    sd = X[train_mask].std(0) + 1e-8
    Xz = (X - mu) / sd

    # 3. EDA -----------------------------------------------------------
    print("[viz] exploratory data analysis ...")
    viz.eda_class_distribution(Y, labels, cfg)
    viz.eda_label_cardinality(Y, cfg)
    viz.eda_sequence_length(df, cfg)
    viz.eda_partition_balance(Y, parts, labels, cfg)
    viz.eda_embedding_variance(Xz, cfg)
    try:
        viz.eda_embedding_projection(Xz, Y, labels, cfg)
    except Exception as e:
        print(f"[viz] projection skipped ({e})")

    # 4. SS-SA wrapper on the training partitions ----------------------
    print("[opt] running SS-SA wrapper ...")
    Xtr, Ytr = Xz[train_mask], Y[train_mask]
    parts_tr = parts[train_mask]

    def fitness(mask: np.ndarray) -> float:
        cols = np.flatnonzero(mask)
        mcc = ev.cv_mcc(Xtr[:, cols], Ytr, parts_tr,
                        cfg.inner_cv_partitions, cfg)
        return mcc - cfg.parsimony_lambda * (mask.sum() / D)

    opt = SSSAOptimizer(D, fitness, cfg)
    best_mask, best_f = opt.run()
    sel = np.flatnonzero(best_mask)
    print(f"[opt] selected {len(sel)}/{D} dims  best_fitness={best_f:.4f}")

    viz.fig_convergence(opt.history, cfg)
    viz.fig_size_over_generations(opt.history, D, cfg)

    # 5/6. evaluate selected subset + baselines on the test partition --
    def evaluate(transform, name):
        Xt = transform(Xz)
        clf = ev.make_classifier(cfg.final_clf, cfg.final_C,
                                 cfg.final_max_iter, cfg.n_jobs)
        clf.fit(Xt[train_mask], Y[train_mask])
        pred = np.asarray(clf.predict(Xt[test_mask]))
        m = ev.full_metrics(Y[test_mask], pred)
        m.update({"method": name, "dims": Xt.shape[1]})
        pc = ev.per_class_metrics(Y[test_mask], pred, labels)
        return m, pc, pred

    print("[eval] evaluating methods on the held-out partition ...")
    results = []

    sssa_metrics, sssa_pc, _ = evaluate(lambda Z: Z[:, sel], "SS-SA")
    results.append(sssa_metrics)

    full_name, full_t = bl.full_baseline()
    full_metrics_, full_pc, _ = evaluate(full_t, full_name)
    results.append(full_metrics_)

    k = len(sel)   # match SS-SA dimensionality for fair comparison
    for name, t in [
        bl.pca_baseline(Xz[train_mask], k, cfg.seed),
        bl.variance_baseline(Xz[train_mask], k),
        bl.mutual_info_baseline(Xz[train_mask], Y[train_mask], k, cfg.seed),
    ]:
        m, _, _ = evaluate(t, name)
        results.append(m)

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(cfg.results_dir, "method_comparison.csv"),
                   index=False)
    print("\n=== Method comparison (held-out partition) ===")
    print(summary[["method", "dims", "mcc_macro", "f1_macro",
                   "f1_micro", "jaccard"]].to_string(index=False))

    viz.fig_per_class_comparison(full_pc, sssa_pc, labels, cfg)
    viz.fig_method_comparison(summary, cfg)

    # 7. dims-vs-performance efficiency curve --------------------------
    print("[eval] building dims-vs-performance curve ...")
    curve_rows = []
    var = Xz[train_mask].var(0)
    order = np.argsort(var)[::-1]
    grid = sorted(set([max(1, int(D * f))
                       for f in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)]))
    for kk in grid:
        idx = order[:kk]
        m, _, _ = evaluate(lambda Z, i=idx: Z[:, i], f"top-var-{kk}")
        curve_rows.append({"dims": kk, "mcc_macro": m["mcc_macro"],
                           "method": "variance"})
    curve_rows.append({"dims": len(sel),
                       "mcc_macro": sssa_metrics["mcc_macro"],
                       "method": "SS-SA"})
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(os.path.join(cfg.results_dir, "dims_vs_performance.csv"),
                 index=False)
    viz.fig_dims_vs_performance(curve, cfg)

    # 8. save selected mask + run config -------------------------------
    np.save(os.path.join(cfg.results_dir, "selected_mask.npy"), best_mask)
    with open(os.path.join(cfg.results_dir, "run_summary.json"), "w") as f:
        json.dump({
            "embedder": cfg.embedder,
            "D": int(D),
            "selected_dims": int(len(sel)),
            "compression_ratio": round(len(sel) / D, 4),
            "best_fitness": round(float(best_f), 4),
            "test_partition": cfg.test_partition,
            "evaluations": opt.history.evaluations,
            "sssa_test_mcc": round(float(sssa_metrics["mcc_macro"]), 4),
            "full_test_mcc": round(float(full_metrics_["mcc_macro"]), 4),
        }, f, indent=2)

    opt.clear_ckpt()   # finished successfully -> checkpoint no longer needed
    print(f"\n[done] figures -> {cfg.figures_dir}/   "
          f"results -> {cfg.results_dir}/")


if __name__ == "__main__":
    main()
