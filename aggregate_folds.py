"""
aggregate_folds.py
==================
Collects runs/fold_N/results/method_comparison.csv (written by main.py for
each held-out partition), reports the five-fold mean and SD of every method,
and runs paired t-tests of the SS-SA subset against every baseline.

Usage (from the repository root):
    python aggregate_folds.py

Outputs:
    runs/FINAL_summary.csv     five-fold mean and SD per method
    runs/FINAL_per_fold.csv    all per-fold rows
    runs/FINAL_stats.csv       paired tests, SS-SA vs every baseline
"""

import os
import glob
import numpy as np
import pandas as pd

RUNS = "runs"


def find_fold_files():
    """Locate every fold's method_comparison.csv, whatever the path style."""
    hits = {}
    # try several layouts so Windows/Linux path quirks don't matter
    patterns = [
        os.path.join(RUNS, "fold_*", "results", "method_comparison.csv"),
        os.path.join(RUNS, "fold_*", "method_comparison.csv"),
        os.path.join(RUNS, "**", "method_comparison.csv"),
    ]
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            # pull the fold number out of the path
            parts = p.replace("\\", "/").split("/")
            fold = None
            for seg in parts:
                if seg.startswith("fold_"):
                    try:
                        fold = int(seg.split("_")[1])
                    except (IndexError, ValueError):
                        pass
            if fold is not None and fold not in hits:
                hits[fold] = p
    return dict(sorted(hits.items()))


def main():
    if not os.path.isdir(RUNS):
        print(f"ERROR: '{RUNS}/' not found. Run this from the project folder.")
        return

    files = find_fold_files()
    if not files:
        print(f"ERROR: no method_comparison.csv found under {RUNS}/")
        print("Contents:")
        for p in glob.glob(os.path.join(RUNS, "*")):
            print("   ", p)
        return

    print(f"found {len(files)} folds: {sorted(files)}\n")

    rows = []
    for fold, path in files.items():
        d = pd.read_csv(path)
        d["fold"] = fold
        rows.append(d)
    allr = pd.concat(rows, ignore_index=True)
    allr.to_csv(os.path.join(RUNS, "FINAL_per_fold.csv"), index=False)

    # normalise the method names (pca_202 -> pca, etc.)
    allr["method_clean"] = (allr["method"]
                            .str.replace(r"_\d+$", "", regex=True))

    metrics = [c for c in ["mcc_macro", "f1_macro", "f1_micro", "jaccard",
                           "dims"] if c in allr.columns]

    # ---------------- mean +/- SD ----------------
    print("=" * 62)
    print("FIVE-FOLD MEAN +/- SD   ")
    print("=" * 62)
    summary = []
    order = (allr.groupby("method_clean")["mcc_macro"].mean()
             .sort_values(ascending=False).index)
    for m in order:
        sub = allr[allr["method_clean"] == m]
        row = {"method": m, "n_folds": len(sub)}
        line = f"{m:14s}"
        for k in metrics:
            mu, sd = sub[k].mean(), sub[k].std(ddof=1)
            row[f"{k}_mean"], row[f"{k}_sd"] = round(mu, 4), round(sd, 4)
            if k == "mcc_macro":
                line += f"  MCC {mu:.4f} +/- {sd:.4f}"
        summary.append(row)
        print(line)
    pd.DataFrame(summary).to_csv(
        os.path.join(RUNS, "FINAL_summary.csv"), index=False)

    # ---------------- paired tests vs SS-SA ----------------
    print("\n" + "=" * 62)
    print("PAIRED COMPARISON vs SS-SA   (df = n_folds - 1)")
    print("=" * 62)
    piv = allr.pivot_table(index="fold", columns="method_clean",
                           values="mcc_macro")
    if "SS-SA" not in piv.columns:
        print("SS-SA column not found; skipping tests.")
        return

    stats = []
    for m in piv.columns:
        if m == "SS-SA":
            continue
        d = (piv["SS-SA"] - piv[m]).dropna()
        n = len(d)
        if n < 2:
            continue
        se = d.std(ddof=1) / np.sqrt(n)
        t = d.mean() / se if se > 0 else np.nan
        wins = int((d > 0).sum())
        # two-sided p from the t distribution, without scipy
        try:
            from scipy import stats as st
            p = 2 * (1 - st.t.cdf(abs(t), df=n - 1))
            ptxt = f"{p:.4f}"
        except Exception:
            crit = {4: 2.776, 3: 3.182, 2: 4.303}.get(n - 1, 2.0)
            ptxt = "<0.05" if abs(t) > crit else ">=0.05"
            p = np.nan
        verdict = ("SS-SA higher" if d.mean() > 0 else "SS-SA lower")
        print(f"  vs {m:14s} mean diff {d.mean():+.4f}   "
              f"t={t:+6.2f}   p={ptxt:>8s}   wins {wins}/{n}   {verdict}")
        stats.append({"comparison": f"SS-SA vs {m}",
                      "mean_diff": round(d.mean(), 4),
                      "t": round(float(t), 3), "p": p,
                      "wins": wins, "n_folds": n})
    pd.DataFrame(stats).to_csv(
        os.path.join(RUNS, "FINAL_stats.csv"), index=False)

    # ---------------- rank consistency ----------------
    print("\n" + "=" * 62)
    print("RANK PER FOLD  (consistency is itself a result)")
    print("=" * 62)
    for f in sorted(piv.index):
        r = piv.loc[f].sort_values(ascending=False)
        print(f"  fold {f}: " + " > ".join(r.index))

    print(f"\nSaved to {RUNS}/FINAL_summary.csv, FINAL_per_fold.csv, "
          f"FINAL_stats.csv")


if __name__ == "__main__":
    main()
