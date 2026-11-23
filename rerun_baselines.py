"""
rerun_baselines.py
==================
Re-evaluates the baselines for all five folds WITHOUT re-running the SS-SA
search. It reuses the masks already saved in runs/fold_N/results/.

Fixes / additions
-----------------
1. Variance filter computed on the ORIGINAL embeddings. The old code ranked
   dimensions by variance AFTER standardisation, where every dimension has
   variance ~1, so the "variance filter" was effectively arbitrary.
2. Explicit random-subset baseline (several seeds, same size as SS-SA).
3. Per-compartment MCC for SS-SA and the full embedding on ALL five folds
   (the old per-class analysis existed for one fold only).
4. Dims-vs-performance curve rebuilt with the corrected variance filter and
   the random baseline.
5. Reproducibility check: SS-SA, full, PCA and mutual-information are
   recomputed and compared with the numbers already reported.

Run from the project folder (next to main.py and the runs/ folder):
    python rerun_baselines.py --smoke          # quick self-test, seconds
    python rerun_baselines.py --workers 8      # real run, all five folds

It is resumable: finished folds are skipped on re-run.

Outputs
-------
runs/fold_N/results_v2/method_comparison_v2.csv
runs/fold_N/results_v2/per_class_v2.csv
runs/fold_N/results_v2/dims_vs_performance_v2.csv
runs/V2_summary.csv          five-fold mean +/- SD for every method
runs/V2_stats.csv            paired t-tests, SS-SA vs every baseline
runs/V2_per_class.csv        per-compartment MCC, five-fold mean
runs/V2_curve.csv            dims-vs-performance, every fold
runs/V2_repro_check.csv      recomputed vs originally reported values
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# worker: one fit + evaluation (runs in a separate process)
# ----------------------------------------------------------------------
def run_task(Xz, Y, train_mask, test_mask, cols, Xalt, clf_args, labels,
             want_pc, tag):
    import evaluation as ev
    Xt = Xalt if Xalt is not None else Xz[:, cols]
    clf = ev.make_classifier(*clf_args)
    clf.fit(Xt[train_mask], Y[train_mask])
    pred = np.asarray(clf.predict(Xt[test_mask]))
    m = ev.full_metrics(Y[test_mask], pred)
    pc = ev.per_class_metrics(Y[test_mask], pred, labels) if want_pc else None
    return tag, m, pc


# ----------------------------------------------------------------------
def load_fold_data(cfg, fold, smoke):
    import data_loader as dl
    import embeddings as emb
    if smoke:
        df, Y, labels, parts = dl.make_synthetic_dataset(
            n=150, n_labels=6, seed=cfg.seed)
        cfg.embedder = "synthetic"
    else:
        df, Y, labels, parts = dl.load_dataset(cfg)
    X = emb.get_embeddings(df["Sequence"].tolist(), cfg).astype(np.float32)

    all_parts = sorted(int(p) for p in np.unique(parts))
    inner = [p for p in all_parts if p != fold]
    train_mask, test_mask = dl.split_by_partition(parts, fold, inner)
    assert not np.any(train_mask & test_mask), "LEAK"
    mu = X[train_mask].mean(0)
    sd = X[train_mask].std(0) + 1e-8
    Xz = (X - mu) / sd
    return X, Xz, Y, labels, train_mask, test_mask


def process_fold(fold, a, cfg):
    from joblib import Parallel, delayed
    import baselines as bl

    out_dir = os.path.join("runs", f"fold_{fold}", "results_v2")
    done_flag = os.path.join(out_dir, "dims_vs_performance_v2.csv")
    if os.path.exists(done_flag) and not a.smoke:
        print(f"[fold {fold}] already done -> skipping")
        return
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    X, Xz, Y, labels, tr, te = load_fold_data(cfg, fold, a.smoke)
    D = X.shape[1]

    mask_path = os.path.join("runs", f"fold_{fold}", "results",
                             "selected_mask.npy")
    if a.smoke:
        rng = np.random.default_rng(fold)
        mask = np.zeros(D, dtype=bool)
        mask[rng.choice(D, 30, replace=False)] = True
    else:
        if not os.path.exists(mask_path):
            sys.exit(f"ERROR: {mask_path} not found")
        mask = np.load(mask_path).astype(bool)
    sel = np.flatnonzero(mask)
    k = len(sel)
    print(f"[fold {fold}] D={D}  SS-SA retained k={k}")

    clf_args = (cfg.final_clf, cfg.final_C, cfg.final_max_iter, 1)

    # ---- selections computed once, in the main process --------------
    var_true = X[tr].astype(np.float64).var(0)          # FIXED: raw scale
    order_var = np.argsort(var_true)[::-1]

    _, pca_t = bl.pca_baseline(Xz[tr], k, cfg.seed)      # as before
    X_pca = pca_t(Xz).astype(np.float32)
    print(f"[fold {fold}] computing mutual information (slow step) ...")
    _, mi_t = bl.mutual_info_baseline(Xz[tr], Y[tr], k, cfg.seed)
    probe = np.arange(D, dtype=np.float32)[None, :]
    mi_cols = mi_t(probe)[0].astype(int)                 # recover indices

    tasks = []
    tasks.append((sel, None, True, ("SS-SA", k, -1)))
    tasks.append((np.arange(D), None, True, ("full", D, -1)))
    tasks.append((None, X_pca, False, ("pca", k, -1)))
    tasks.append((mi_cols, None, False, ("mutual_info", k, -1)))
    tasks.append((order_var[:k], None, False, ("variance", k, -1)))
    for s in range(a.random_seeds):
        r = np.random.default_rng(1000 + s).choice(D, k, replace=False)
        tasks.append((r, None, False, ("random", k, s)))

    grid = sorted(set(max(1, int(D * f))
                      for f in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7)))
    for kk in grid:
        tasks.append((order_var[:kk], None, False, ("curve_variance", kk, -1)))
        for s in range(a.curve_seeds):
            r = np.random.default_rng(2000 + 97 * kk + s).choice(
                D, kk, replace=False)
            tasks.append((r, None, False, ("curve_random", kk, s)))

    print(f"[fold {fold}] {len(tasks)} evaluations on {a.workers} workers ...")
    res = Parallel(n_jobs=a.workers, backend="loky", verbose=5)(
        delayed(run_task)(Xz, Y, tr, te, c, xa, clf_args, labels, pc, tag)
        for c, xa, pc, tag in tasks)

    rows, pcs, curve = [], [], []
    full_mcc = None
    for (name, dims, seed), m, pc in res:
        row = {"fold": fold, "method": name, "dims": dims, "seed": seed, **m}
        if name.startswith("curve_"):
            curve.append({"fold": fold, "method": name.replace("curve_", ""),
                          "dims": dims, "seed": seed,
                          "mcc_macro": m["mcc_macro"]})
        else:
            rows.append(row)
        if name == "full":
            full_mcc = m["mcc_macro"]
        if pc is not None:
            for lab, v in pc.items():
                pcs.append({"fold": fold, "method": name, "label": lab, **v})
    # full-size point of the curve = full embedding (identical by definition)
    curve.append({"fold": fold, "method": "full", "dims": D, "seed": -1,
                  "mcc_macro": full_mcc})
    ss = [r for r in rows if r["method"] == "SS-SA"][0]
    curve.append({"fold": fold, "method": "SS-SA", "dims": k, "seed": -1,
                  "mcc_macro": ss["mcc_macro"]})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "method_comparison_v2.csv"),
                              index=False)
    pd.DataFrame(pcs).to_csv(os.path.join(out_dir, "per_class_v2.csv"),
                             index=False)
    pd.DataFrame(curve).to_csv(done_flag, index=False)
    print(f"[fold {fold}] done in {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------------
def aggregate(folds):
    from scipy import stats as st

    mc, pc, cv = [], [], []
    for f in folds:
        d = os.path.join("runs", f"fold_{f}", "results_v2")
        if not os.path.exists(os.path.join(d, "dims_vs_performance_v2.csv")):
            print(f"[agg] fold {f} missing -> skipped")
            continue
        mc.append(pd.read_csv(os.path.join(d, "method_comparison_v2.csv")))
        pc.append(pd.read_csv(os.path.join(d, "per_class_v2.csv")))
        cv.append(pd.read_csv(os.path.join(d, "dims_vs_performance_v2.csv")))
    if not mc:
        print("[agg] nothing to aggregate")
        return
    mc, pc, cv = (pd.concat(x, ignore_index=True) for x in (mc, pc, cv))
    nf = mc["fold"].nunique()

    # random baseline: average over seeds within each fold first
    metrics = ["mcc_macro", "f1_macro", "f1_micro", "jaccard"]
    per_fold = (mc.groupby(["fold", "method"])[metrics + ["dims"]]
                .mean().reset_index())
    summ = per_fold.groupby("method")[metrics + ["dims"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.sort_values("mcc_macro_mean", ascending=False).round(4)
    summ.to_csv(os.path.join("runs", "V2_summary.csv"))

    piv = per_fold.pivot(index="fold", columns="method", values="mcc_macro")
    stats = []
    for b in [c for c in piv.columns if c != "SS-SA"]:
        diff = piv["SS-SA"] - piv[b]
        if nf >= 2:
            t = st.ttest_rel(piv["SS-SA"], piv[b])
            tval, pval = float(t.statistic), float(t.pvalue)
        else:
            tval = pval = float("nan")
        stats.append({"comparison": f"SS-SA vs {b}",
                      "mean_diff": round(diff.mean(), 4),
                      "t": round(tval, 3), "p": round(pval, 4),
                      "wins": int((diff > 0).sum()), "n_folds": nf})
    stats = pd.DataFrame(stats)
    stats.to_csv(os.path.join("runs", "V2_stats.csv"), index=False)

    pcm = (pc.groupby(["method", "label"])[["mcc", "support"]]
           .agg(["mean", "std"]).round(4))
    pcm.columns = [f"{a}_{b}" for a, b in pcm.columns]
    pcm.to_csv(os.path.join("runs", "V2_per_class.csv"))

    cvm = (cv.groupby(["fold", "method", "dims"])["mcc_macro"]
           .mean().reset_index().round(4))
    cvm.to_csv(os.path.join("runs", "V2_curve.csv"), index=False)

    # reproducibility check against originally reported numbers
    rep = []
    for f in sorted(mc["fold"].unique()):
        p = os.path.join("runs", f"fold_{f}", "results", "method_comparison.csv")
        if not os.path.exists(p):
            continue
        old = pd.read_csv(p)
        old["m"] = old["method"].str.replace(r"_\d+$", "", regex=True)
        for name in ["SS-SA", "full", "pca", "mutual_info"]:
            o = old[old["m"] == name]
            n = per_fold[(per_fold.fold == f) & (per_fold.method == name)]
            if len(o) and len(n):
                rep.append({"fold": f, "method": name,
                            "reported": round(float(o.mcc_macro.iloc[0]), 4),
                            "recomputed": round(float(n.mcc_macro.iloc[0]), 4)})
    rep = pd.DataFrame(rep)
    if len(rep):
        rep["abs_diff"] = (rep.reported - rep.recomputed).abs().round(4)
        rep.to_csv(os.path.join("runs", "V2_repro_check.csv"), index=False)

    print("\n" + "=" * 66)
    print(f"FIVE-FOLD SUMMARY (n = {nf} folds)")
    print("=" * 66)
    print(summ[["dims_mean", "mcc_macro_mean", "mcc_macro_std"]].to_string())
    print("\nPAIRED TESTS (SS-SA vs baseline)")
    print(stats.to_string(index=False))
    if len(rep):
        print(f"\nREPRODUCIBILITY: max |reported - recomputed| = "
              f"{rep.abs_diff.max():.4f}")
    print("\nSaved: runs/V2_summary.csv, V2_stats.csv, V2_per_class.csv, "
          "V2_curve.csv, V2_repro_check.csv")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--random-seeds", type=int, default=10)
    ap.add_argument("--curve-seeds", type=int, default=3)
    ap.add_argument("--data-csv", default=None,
                    help="DeepLoc 2.0 SwissProt CSV (default: config.py)")
    ap.add_argument("--smoke", action="store_true",
                    help="self-test on synthetic data (writes to runs_smoke/)")
    a = ap.parse_args()

    if not os.path.exists("main.py"):
        sys.exit("ERROR: run this from the project folder (main.py not found)")

    from config import Config
    cfg = Config()
    cfg.verbose = False
    if a.data_csv:
        cfg.data_csv = os.path.abspath(a.data_csv)

    if a.smoke:
        os.makedirs("runs_smoke", exist_ok=True)
        os.chdir("runs_smoke")
        sys.path.insert(0, os.path.abspath(".."))
        a.folds, a.random_seeds, a.curve_seeds = [0, 1], 3, 1
        a.workers = min(a.workers, 2)
        print("[smoke] synthetic self-test; results are meaningless")

    for f in a.folds:
        process_fold(f, a, cfg)
    aggregate(a.folds)


if __name__ == "__main__":
    main()
