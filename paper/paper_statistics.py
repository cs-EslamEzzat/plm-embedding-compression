"""
paper/paper_statistics.py
=========================
Recomputes the statistics reported in the paper from the saved results in
runs/V4 (written by gap_fixes.py) and prints them as one plain-text report.

Covers: five-fold means and retention (both regularization regimes), paired
t-tests with 95% CI, d_z and Holm adjustment, saturation points of the
size-performance curves, the fixed-C artefact, per-compartment effects,
external (HPA) results, the consensus core, and subset stability
(overlap, Jaccard index and permutation tests).

Usage (from the repository root):
    python paper/paper_statistics.py                 # reads runs/V4
    python paper/paper_statistics.py runs/V4_rerun   # any gap_fixes output
    python paper/paper_statistics.py > paper/statistics_report.txt
"""
import glob
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

V = sys.argv[1] if len(sys.argv) > 1 else os.path.join("runs", "V4")
MET = ["mcc_macro", "f1_macro", "f1_micro", "jaccard"]
pd.set_option("display.width", 160)


def load(name):
    files = sorted(glob.glob(os.path.join(V, "fold_*", name)))
    if not files:
        sys.exit(f"ERROR: no {name} under {V}/fold_*/")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired(a, b):
    """Paired t-test over folds: mean diff, 95% CI, t, p, d_z, wins."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    t = stats.ttest_rel(a, b)
    return dict(diff=d.mean(), ci_lo=d.mean() - h, ci_hi=d.mean() + h,
                t=t.statistic, p=t.pvalue, d_z=d.mean() / sd,
                wins=int((d > 0).sum()))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out, run = np.empty(len(p)), 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        out[j] = min(1.0, run)
    return out


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


tr = load("test_results.csv")
pc = load("test_per_class.csv")
main = tr[~tr.method.str.startswith("curve_")]
pf = main.groupby(["regime", "fold", "method"])[MET + ["dims"]].mean().reset_index()
cv = tr[tr.method.str.startswith("curve_")].copy()
cv["method"] = cv.method.str.replace("curve_", "")
cvm = cv.groupby(["regime", "fold", "method", "dims"]).mcc_macro.mean().reset_index()
full = pf[pf.method == "full"].set_index(["regime", "fold"]).mcc_macro
cvm["ret"] = cvm.apply(lambda r: 100 * r.mcc_macro / full[(r.regime, r.fold)], axis=1)
COMPS = ["random", "variance", "mutual_info", "L1_embedded", "full", "pca"]

# ----------------------------------------------------------------------
section("1. Five-fold means (cross-validation) and retention")
ssa_k = pf[(pf.regime == "tuned") & (pf.method == "SS-SA")].dims
print(f"SS-SA subset sizes per fold: {ssa_k.astype(int).tolist()}  "
      f"mean {ssa_k.mean():.1f} +/- {ssa_k.std(ddof=1):.1f}")
for reg in ["tuned", "fixed"]:
    piv = pf[pf.regime == reg].pivot(index="fold", columns="method", values="mcc_macro")
    s = pf[pf.regime == reg].groupby("method")[MET].agg(["mean", "std"]).round(4)
    print(f"\n[{reg}]\n{s}")
    print("MCC retained (% of full):",
          (100 * piv.mean() / piv["full"].mean()).round(1).to_dict())

# ----------------------------------------------------------------------
section("2. Paired comparisons, SS-SA minus each strategy (Holm over six)")
for reg in ["tuned", "fixed"]:
    piv = pf[pf.regime == reg].pivot(index="fold", columns="method", values="mcc_macro")
    R = pd.DataFrame([dict(comparison=f"SS-SA - {b}", **paired(piv["SS-SA"], piv[b]))
                      for b in COMPS])
    R["p_holm"] = holm(R.p)
    print(f"\n[{reg}]\n{R.round(4).to_string(index=False)}")
    for a_, b_ in [("L1_embedded", "random"), ("L1_embedded", "variance"),
                   ("L1_embedded", "mutual_info"), ("pca", "full"),
                   ("pca", "L1_embedded")]:
        r = paired(piv[a_], piv[b_])
        print(f"  {a_} - {b_}: " + ", ".join(f"{k}={v:.4f}" for k, v in r.items()))

# ----------------------------------------------------------------------
section("3. Size-performance curves, saturation and ceiling tests")
print(cvm.groupby(["regime", "method", "dims"])[["mcc_macro", "ret"]]
      .agg(["mean", "std"]).round(3))
for reg in ["tuned", "fixed"]:
    for meth in ["random", "variance"]:
        out = {0.90: [], 0.95: [], 0.99: []}
        for f in sorted(cvm.fold.unique()):
            c = cvm[(cvm.regime == reg) & (cvm.method == meth) & (cvm.fold == f)].sort_values("dims")
            d = np.r_[c.dims, 1280]
            r = np.r_[c.ret, 100.0] / 100
            for thr in out:
                j = int(np.where(r >= thr)[0][0])
                if j == 0:
                    out[thr].append(d[0])
                else:   # log-linear interpolation between neighbouring sizes
                    fr = (thr - r[j - 1]) / (r[j] - r[j - 1])
                    out[thr].append(np.exp(np.log(d[j - 1]) + fr * (np.log(d[j]) - np.log(d[j - 1]))))
        print(f"saturation [{reg}, {meth}]: " + ", ".join(
            f"{int(t * 100)}% at {np.mean(v):.0f} +/- {np.std(v, ddof=1):.0f} dims"
            for t, v in out.items()))
for reg in ["tuned", "fixed"]:
    for meth in ["random", "variance"]:
        for dd in [256, 512, 896]:
            x = cvm[(cvm.regime == reg) & (cvm.method == meth) & (cvm.dims == dd)].set_index("fold").mcc_macro
            r = paired(x, full[reg])
            print(f"ceiling [{reg}] {meth} {dd} - full: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in r.items()))

# ----------------------------------------------------------------------
section("4. Fixed-regularization artefact (tuned minus fixed)")
for m in ["full", "pca", "L1_embedded", "SS-SA", "mutual_info", "random", "variance"]:
    a_ = pf[(pf.regime == "tuned") & (pf.method == m)].set_index("fold").mcc_macro
    b_ = pf[(pf.regime == "fixed") & (pf.method == m)].set_index("fold").mcc_macro
    r = paired(a_, b_)
    print(f"{m:12s} " + ", ".join(f"{k}={v:.4f}" for k, v in r.items()))
ct = pd.read_csv(os.path.join(V, "V4_C_tuning_all.csv"))
print("\nchosen C (counts):")
print(ct[~ct.method.str.startswith("curve")].groupby(["method", "best_C"]).size().unstack(fill_value=0))

# ----------------------------------------------------------------------
section("5. Per-compartment effects (tuned)")
t = pc[pc.regime == "tuned"].groupby(["method", "label", "fold"]).agg(
    mcc=("mcc", "mean"), sup=("support", "mean")).reset_index()
w = t.groupby(["label", "method"]).mcc.mean().unstack()
sup = t[t.method == "full"].groupby("label").sup.mean()
tab = pd.DataFrame({"support_per_fold": sup, "full": w["full"], "SS-SA": w["SS-SA"],
                    "d_SS-SA": w["SS-SA"] - w["full"], "d_L1": w["L1_embedded"] - w["full"]})
print(tab.sort_values("support_per_fold", ascending=False).round(4))
for col in ["d_SS-SA", "d_L1"]:
    rho = stats.spearmanr(tab.support_per_fold, tab[col])
    print(f"Spearman(support, {col}) = {rho.statistic:.3f}, p = {rho.pvalue:.3f}")

# ----------------------------------------------------------------------
if glob.glob(os.path.join(V, "fold_*", "hpa_results.csv")):
    section("6. External HPA test set")
    hr = load("hpa_results.csv")
    hpc = load("hpa_per_class.csv")
    hpf = hr.groupby(["regime", "fold", "method"])[MET].mean().reset_index()
    for reg in ["tuned", "fixed"]:
        piv = hpf[hpf.regime == reg].pivot(index="fold", columns="method", values="mcc_macro")
        print(f"\n[{reg}]\n{hpf[hpf.regime == reg].groupby('method')[MET].agg(['mean', 'std']).round(4)}")
        R = pd.DataFrame([dict(comparison=f"SS-SA - {b}", **paired(piv["SS-SA"], piv[b]))
                          for b in COMPS])
        R["p_holm"] = holm(R.p)
        print(R.round(4).to_string(index=False))
    print("\nper-compartment HPA MCC (tuned):")
    print(hpc[hpc.regime == "tuned"].groupby(["label", "method"]).mcc.mean().unstack().round(3))
    cp = os.path.join(V, "consensus", "consensus_results.csv")
    if os.path.exists(cp):
        c = pd.read_csv(cp)
        print("\nconsensus core on HPA:")
        print(c.groupby(["regime", "method"]).mcc_macro.agg(["mean", "std", "count"]).round(4))

# ----------------------------------------------------------------------
section("7. Stability of the SS-SA subsets across folds")
M = np.array([np.load(f).astype(bool) for f in
              sorted(glob.glob(os.path.join(V, "fold_*", "ssa_mask.npy")))])
D = M.shape[1]
ks = M.sum(1)
cnt = M.sum(0)
dist = np.array([1.0])
for p in ks / D:                                   # Poisson-binomial
    dist = np.convolve(dist, [1 - p, p])
print("folds selecting a dimension (0..5): observed", np.bincount(cnt, minlength=6).tolist(),
      " expected", (dist * D).round(1).tolist())
inter, jac, exp_int = [], [], []
for i, j in combinations(range(len(M)), 2):
    inter.append(int((M[i] & M[j]).sum()))
    exp_int.append(ks[i] * ks[j] / D)
    jac.append((M[i] & M[j]).sum() / (M[i] | M[j]).sum())
print(f"pairwise intersections {min(inter)}-{max(inter)} "
      f"(expected {min(exp_int):.0f}-{max(exp_int):.0f}); "
      f"mean Jaccard {np.mean(jac):.3f}; distinct dims {(cnt >= 1).sum()}; "
      f">=3 folds {(cnt >= 3).sum()}; 4 folds {(cnt == 4).sum()}; 5 folds {(cnt == 5).sum()}")
rng = np.random.default_rng(42)          # same seed as Figure 10
N = 20000
ge3, mj, distinct = np.empty(N), np.empty(N), np.empty(N)
for n in range(N):
    R = np.zeros_like(M)
    for i, k in enumerate(ks):
        R[i, rng.choice(D, k, replace=False)] = True
    c = R.sum(0)
    ge3[n], distinct[n] = (c >= 3).sum(), (c >= 1).sum()
    mj[n] = np.mean([(R[i] & R[j]).sum() / (R[i] | R[j]).sum()
                     for i, j in combinations(range(len(M)), 2)])
print(f"null (20,000 random draws of the same sizes): mean Jaccard {mj.mean():.4f} "
      f"(95% range {np.percentile(mj, 2.5):.3f}-{np.percentile(mj, 97.5):.3f}), "
      f"P(null >= observed) = {(mj >= np.mean(jac)).mean():.4f}; "
      f">=3-fold dims {ge3.mean():.1f}, P(null >= {(cnt >= 3).sum()}) = "
      f"{(ge3 >= (cnt >= 3).sum()).mean():.4f}; distinct dims {distinct.mean():.0f}")
