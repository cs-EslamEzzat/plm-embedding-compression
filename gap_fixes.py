"""
gap_fixes.py
============
Paper 1 -- closes the three gaps raised in review, WITHOUT re-running the
SS-SA search and WITHOUT re-computing the SwissProt embeddings.

  Gap 1  Embedded selection.  An L1-regularised (Lasso) one-vs-rest logistic
         regression is tuned, inside the training partitions of every fold,
         until it keeps the same number of dimensions as the SS-SA subset of
         that fold. Its subset is then scored with the same final classifier
         as every other method.

  Gap 2  Regularisation re-tuned for every size.  For every method and every
         subset size, the inverse regularisation strength C of the final
         classifier is chosen by inner cross-validation on the TRAINING
         partitions only (the held-out partition is never touched). Results
         are reported side by side for the original fixed C (= 1.0) and the
         tuned C, including the size-performance curve.

  Gap 3  External test set.  The independent Human Protein Atlas test set
         released with DeepLoc 2.0 (hpa_testset.csv; >30% identity to the
         SwissProt set removed by its authors) is embedded once with ESM-2
         and every fold's model is evaluated on it. A consensus subset (the
         dimensions selected in >= 3 of the 5 folds) is also tested on it.

Inputs:
    cache/emb_esm2_*.npy                     SwissProt embeddings (main.py)
    runs/fold_N/results/selected_mask.npy    SS-SA subsets (main.py)

New computation:
    HPA embeddings (1,717 proteins, GPU, run once and cached)
    mutual-information ranking (recomputed with the same seed as before, cached)
    L1 subsets, C tuning, external evaluation (CPU)

HOW TO RUN (from the repository root):
    python gap_fixes.py --smoke                          # self-test
    python gap_fixes.py --workers 12 --out runs/V4_rerun # full run, resumable

The results reported in the paper are in runs/V4 (produced with the default
settings below). Folds that already contain fold_done.json are skipped, so
use a new --out folder to recompute them.

Put hpa_testset.csv (DeepLoc 2.0 download page) in the project folder first,
or pass --hpa PATH.  Use --no-hpa to skip gap 3.

C grid: 1e-4 ... 1 (8 values). External labels with fewer than 20 positives
(lysosome/vacuole, peroxisome) are excluded from the external summary.
--reuse can point to an earlier output folder whose mutual-information and
L1 selections are copied instead of recomputed.

All outputs go to the --out folder and are also packed into <out>_results.zip.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # avoid thread oversubscription

import argparse
import json
import re
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FIXED_C = 1.0                                # value used in all earlier runs


# ======================================================================
# Worker (runs in a separate process): tune C, then evaluate
# ======================================================================
def tune_task(tag, Xz, Y, parts, tr, te, cols, Xalt, Xh, Xh_alt, Yh, hpa_idx,
              Cs, max_iter, val_parts, want_pc, labels, hpa_labels):
    import warnings as _w
    _w.filterwarnings("ignore")
    import evaluation as ev

    Xt = Xalt if Xalt is not None else Xz[:, cols]
    Xht = None
    if Xh is not None:
        Xht = Xh_alt if Xh_alt is not None else Xh[:, cols]

    def fit(C, mask):
        clf = ev.make_classifier("logreg_t", C, max_iter, 1)
        clf.fit(Xt[mask], Y[mask])
        return clf

    # ---- inner CV on the TRAINING partitions only --------------------
    cv_scores = {}
    for C in Cs:
        s = []
        for v in val_parts:
            trv = tr & (parts != v)
            va = parts == v
            clf = fit(C, trv)
            s.append(ev.label_avg_mcc(Y[va], np.asarray(clf.predict(Xt[va]))))
        cv_scores[C] = float(np.mean(s))
    best = max(cv_scores.values())
    # ties -> the C closest to the original value (log scale)
    best_C = min([c for c, v in cv_scores.items() if v >= best - 1e-12],
                 key=lambda c: abs(np.log10(c) - np.log10(FIXED_C)))

    out = {"tag": tag, "cv_scores": cv_scores, "best_C": best_C, "eval": {}}
    regimes = [("fixed", FIXED_C), ("tuned", best_C)]
    cache = {}
    for regime, C in regimes:
        if C not in cache:
            clf = fit(C, tr)
            pred = np.asarray(clf.predict(Xt[te]))
            r = {"metrics": ev.full_metrics(Y[te], pred), "pc": None,
                 "hpa": None, "hpa_pc": None}
            if want_pc:
                r["pc"] = ev.per_class_metrics(Y[te], pred, labels)
            if Xht is not None:
                ph = np.asarray(clf.predict(Xht))[:, hpa_idx]
                r["hpa"] = ev.full_metrics(Yh, ph)
                r["hpa_pc"] = ev.per_class_metrics(Yh, ph, hpa_labels)
            cache[C] = r
        out["eval"][regime] = dict(cache[C], C=C)
    return out


# ======================================================================
# Data
# ======================================================================
def load_swissprot(cfg, smoke):
    import data_loader as dl
    import embeddings as emb
    if smoke:
        df, Y, labels, parts = dl.make_synthetic_dataset(
            n=150, n_labels=6, seed=cfg.seed)
        cfg.embedder = "synthetic"
    else:
        df, Y, labels, parts = dl.load_dataset(cfg)
    X = emb.get_embeddings(df["Sequence"].tolist(), cfg).astype(np.float32)
    return df, X, Y, list(labels), parts


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def read_hpa_csv(path, labels):
    """Robust reader: detects sequence, id and label columns."""
    raw = pd.read_csv(path)
    cols = list(raw.columns)
    # sequence column
    seq_col = None
    for c in cols:
        if _norm(c) in ("sequence", "seq", "fasta", "proteinsequence"):
            seq_col = c
            break
    if seq_col is None:
        for c in cols:
            v = raw[c].astype(str)
            if v.str.len().mean() > 50 and v.str.fullmatch(r"[A-Za-z]+").mean() > .95:
                seq_col = c
                break
    if seq_col is None:
        sys.exit(f"ERROR: no sequence column found in {path}. Columns: {cols}")
    id_col = None
    for c in cols:
        if _norm(c) in ("acc", "sid", "entry", "id", "uniprot", "accession",
                        "uniprotid", "protein"):
            id_col = c
            break
    # label columns: one-hot first
    lab_map = {}
    for lab in labels:
        for c in cols:
            if _norm(c) == _norm(lab):
                lab_map[lab] = c
                break
    Yh_cols, hpa_labels = [], []
    if len(lab_map) >= 2:
        for lab in labels:
            if lab in lab_map:
                y = (pd.to_numeric(raw[lab_map[lab]], errors="coerce")
                     .fillna(0).to_numpy() > 0).astype(int)
                Yh_cols.append(y)
                hpa_labels.append(lab)
    else:                                   # one text column with names
        txt = None
        for c in cols:
            if c != seq_col and re.search(r"loc|label|target|class|compartment", c, re.I):
                txt = c
                break
        if txt is None:
            sys.exit(f"ERROR: no label columns found in {path}. Columns: {cols}")
        toks = raw[txt].astype(str).apply(
            lambda s: {_norm(t) for t in re.split(r"[;,|]", s) if t.strip()})
        for lab in labels:
            y = toks.apply(lambda t: int(_norm(lab) in t)).to_numpy()
            Yh_cols.append(y)
            hpa_labels.append(lab)
    Yh = np.vstack(Yh_cols).T
    keep = Yh.sum(0) > 0                      # labels that actually occur
    Yh = Yh[:, keep]
    hpa_labels = [l for l, k in zip(hpa_labels, keep) if k]
    seqs = raw[seq_col].astype(str).str.upper().str.strip().tolist()
    ids = (raw[id_col].astype(str).tolist() if id_col
           else [f"HPA_{i}" for i in range(len(raw))])
    info = {"file": path, "columns": cols, "sequence_column": seq_col,
            "id_column": id_col, "n_rows": len(raw)}
    return ids, seqs, Yh, hpa_labels, info


def load_hpa(a, cfg, df, labels):
    """Returns dict with embeddings and labels, or None."""
    import embeddings as emb
    import data_loader as dl
    if a.smoke:
        dfh, Yh_all, _, _ = dl.make_synthetic_dataset(n=60, n_labels=6, seed=99)
        Xh = emb.get_embeddings(dfh["Sequence"].tolist(), cfg).astype(np.float32)
        hpa_labels = labels[:4]
        Yh = Yh_all[:, :4]
        keep = Yh.sum(1) > 0
        return {"X": Xh[keep], "Y": Yh[keep], "labels": hpa_labels,
                "idx": [labels.index(l) for l in hpa_labels],
                "info": {"smoke": True, "n_used": int(keep.sum())}}
    if a.no_hpa:
        return None
    if not os.path.exists(a.hpa):
        sys.exit(f"ERROR: {a.hpa} not found. Download hpa_testset.csv from the "
                 f"DeepLoc 2.0 data page and put it in the project folder, "
                 f"or run with --no-hpa.")
    ids, seqs, Yh, hpa_labels, info = read_hpa_csv(a.hpa, labels)
    # remove exact duplicates of training sequences / accessions (safety)
    train_seqs = set(df["Sequence"].tolist())
    train_ids = set(df["ACC"].astype(str).tolist())
    dup = np.array([(s in train_seqs) or (i in train_ids)
                    for s, i in zip(seqs, ids)])
    nolab = Yh.sum(1) == 0
    keep = ~dup & ~nolab
    pos = Yh[keep].sum(0)
    good = pos >= a.hpa_min_pos
    dropped = {l: int(n) for l, n, g in zip(hpa_labels, pos, good) if not g}
    Yh = Yh[:, good]
    hpa_labels = [l for l, g in zip(hpa_labels, good) if g]
    # proteins whose only labels were dropped stay in as negatives for the
    # remaining compartments (as in the DeepLoc 2.0 evaluation on 1,717
    # proteins); keeping them also reuses the cached HPA embeddings
    info["labels_dropped_too_few_positives"] = dropped
    info.update({"n_exact_overlap_removed": int(dup.sum()),
                 "n_without_label_removed": int((nolab & ~dup).sum()),
                 "n_used": int(keep.sum()), "labels": hpa_labels,
                 "positives": {l: int(Yh[keep, j].sum())
                               for j, l in enumerate(hpa_labels)}})
    print(f"[hpa] {info['n_rows']} rows | sequence column '{info['sequence_column']}'"
          f" | id column '{info['id_column']}'")
    print(f"[hpa] labels found: {hpa_labels}")
    print(f"[hpa] removed {info['n_exact_overlap_removed']} exact overlaps with "
          f"training data, {info['n_without_label_removed']} without labels -> "
          f"{info['n_used']} proteins used")
    print("[hpa] computing / loading ESM-2 embeddings (GPU if available) ...")
    Xh = emb.get_embeddings([s for s, k in zip(seqs, keep) if k],
                            cfg).astype(np.float32)
    return {"X": Xh, "Y": Yh[keep], "labels": hpa_labels,
            "idx": [labels.index(l) for l in hpa_labels], "info": info}


# ======================================================================
# Selections computed once per fold (cached)
# ======================================================================
def mi_order(Xtr, Ytr, seed, workers, path):
    if os.path.exists(path):
        return np.load(path)
    from joblib import Parallel, delayed
    from sklearn.feature_selection import mutual_info_classif
    labs = [j for j in range(Ytr.shape[1]) if len(np.unique(Ytr[:, j])) > 1]
    res = Parallel(n_jobs=workers, backend="loky")(
        delayed(mutual_info_classif)(Xtr, Ytr[:, j], random_state=seed)
        for j in labs)
    mi = np.sum(res, axis=0)
    order = np.argsort(mi)[::-1]             # same rule as baselines.py
    np.save(path, order)
    return order


def _l1_fit_one(Xtr, y, C, seed):
    import warnings as _w
    _w.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y)) < 2:
        return np.zeros(Xtr.shape[1])
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C,
                           class_weight="balanced", max_iter=2000,
                           tol=1e-4, random_state=seed)
    m.fit(Xtr, y)
    return m.coef_.ravel()


def l1_select(Xtr, Ytr, k, workers, seed, path_cols, path_info):
    """Embedded selection: L1-logistic OvR, C searched (log-bisection) until
    the union of non-zero coefficients across labels reaches k; ties trimmed by
    the summed |coefficient|. Uses the training partitions only."""
    if os.path.exists(path_cols):
        return np.load(path_cols), json.load(open(path_info))
    from joblib import Parallel, delayed

    def coefs(C):
        r = Parallel(n_jobs=workers, backend="loky")(
            delayed(_l1_fit_one)(Xtr, Ytr[:, j], C, seed)
            for j in range(Ytr.shape[1]))
        return np.vstack(r)

    lo, hi = -5.0, 1.0                       # log10 C
    W_hi = coefs(10 ** hi)
    log = [{"C": 10 ** hi, "union": int((np.abs(W_hi) > 0).any(0).sum())}]
    for _ in range(14):
        mid = (lo + hi) / 2
        W = coefs(10 ** mid)
        u = int((np.abs(W) > 0).any(0).sum())
        log.append({"C": 10 ** mid, "union": u})
        if u >= k:
            hi, W_hi = mid, W
            if u <= int(np.ceil(1.1 * k)):
                break
        else:
            lo = mid
    score = np.abs(W_hi).sum(0)
    cols = np.sort(np.argsort(score)[::-1][:k])
    info = {"k_target": int(k), "C_selected": float(10 ** hi),
            "union_at_C": int((np.abs(W_hi) > 0).any(0).sum()),
            "search_log": log}
    np.save(path_cols, cols)
    json.dump(info, open(path_info, "w"), indent=1)
    return cols, info


# ======================================================================
# One fold
# ======================================================================
def process_fold(fold, a, cfg, data, hpa, t_start, folds_left):
    from joblib import Parallel, delayed
    import baselines as bl
    import data_loader as dl

    df, X, Y, labels, parts = data
    out = os.path.join(a.out, f"fold_{fold}")
    done = os.path.join(out, "fold_done.json")
    if os.path.exists(done) and not a.smoke:
        print(f"[fold {fold}] already done -> skipping")
        return
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    all_parts = sorted(int(p) for p in np.unique(parts))
    inner = [p for p in all_parts if p != fold]
    tr, te = dl.split_by_partition(parts, fold, inner)
    assert not np.any(tr & te), "LEAK: train and test overlap"
    val_parts = inner[:a.val_folds]
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-8
    Xz = (X - mu) / sd
    D = X.shape[1]
    Xh = Yh = None
    if hpa is not None:
        Xh = (hpa["X"] - mu) / sd
        Yh = hpa["Y"]

    # ---- SS-SA subset ---------------------------------------------
    if a.smoke:
        rng = np.random.default_rng(fold)
        mask = np.zeros(D, bool)
        mask[rng.choice(D, 30, replace=False)] = True
    else:
        mp = os.path.join("runs", f"fold_{fold}", "results", "selected_mask.npy")
        if not os.path.exists(mp):
            sys.exit(f"ERROR: {mp} not found")
        mask = np.load(mp).astype(bool)
        np.save(os.path.join(out, "ssa_mask.npy"), mask)
    sel = np.flatnonzero(mask)
    k = len(sel)
    print(f"\n[fold {fold}] D={D}  SS-SA k={k}  val partitions for tuning: {val_parts}")

    # ---- reference selections (same rules as rerun_baselines.py) ---
    order_var = np.argsort(X[tr].astype(np.float64).var(0))[::-1]
    _, pca_t = bl.pca_baseline(Xz[tr], k, cfg.seed)
    X_pca = pca_t(Xz).astype(np.float32)
    Xh_pca = pca_t(Xh).astype(np.float32) if Xh is not None else None
    print(f"[fold {fold}] mutual information (cached after first run) ...")
    if a.reuse and not a.smoke:
        src = os.path.join(a.reuse, f"fold_{fold}")
        for fn in ("mi_order.npy", "l1_cols.npy", "l1_info.json"):
            if os.path.exists(os.path.join(src, fn)) and not os.path.exists(
                    os.path.join(out, fn)):
                shutil.copy(os.path.join(src, fn), os.path.join(out, fn))
                print(f"[fold {fold}] reused {fn} from {src}")
    order_mi = mi_order(Xz[tr], Y[tr], cfg.seed, a.workers,
                        os.path.join(out, "mi_order.npy"))
    print(f"[fold {fold}] L1 embedded selection, target k={k} ...")
    t1 = time.time()
    l1_cols, l1_info = l1_select(Xz[tr], Y[tr], k, a.workers, cfg.seed,
                                 os.path.join(out, "l1_cols.npy"),
                                 os.path.join(out, "l1_info.json"))
    print(f"[fold {fold}] L1: C={l1_info['C_selected']:.4g}, union "
          f"{l1_info['union_at_C']} -> kept {len(l1_cols)} "
          f"({(time.time() - t1) / 60:.1f} min)")

    # ---- task list ---------------------------------------------------
    tasks = []                       # (tag, cols, Xalt, Xh_alt, want_pc)
    tasks.append((("full", D, -1), np.arange(D), None, None, True))
    tasks.append((("SS-SA", k, -1), sel, None, None, True))
    tasks.append((("L1_embedded", k, -1), l1_cols, None, None, True))
    tasks.append((("pca", k, -1), None, X_pca, Xh_pca, False))
    tasks.append((("mutual_info", k, -1), order_mi[:k], None, None, False))
    tasks.append((("variance", k, -1), order_var[:k], None, None, False))
    for s in range(a.random_seeds):
        r = np.random.default_rng(1000 + s).choice(D, k, replace=False)
        tasks.append((("random", k, s), r, None, None, False))
    grid = sorted(set(max(1, int(D * f))
                      for f in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7)))
    for kk in grid:
        tasks.append((("curve_variance", kk, -1), order_var[:kk], None, None, False))
        for s in range(a.curve_seeds):
            r = np.random.default_rng(2000 + 97 * kk + s).choice(D, kk, replace=False)
            tasks.append((("curve_random", kk, s), r, None, None, False))
    tasks.sort(key=lambda t: -t[0][1])        # big subsets first (load balance)

    hpa_idx = hpa["idx"] if hpa else None
    hpa_labels = hpa["labels"] if hpa else None
    print(f"[fold {fold}] {len(tasks)} subsets x {len(a.Cs)} C values x "
          f"{len(val_parts)} inner folds, on {a.workers} workers ...")
    res = Parallel(n_jobs=a.workers, backend="loky", verbose=5)(
        delayed(tune_task)(tag, Xz, Y, parts, tr, te, cols, xa,
                           None if tag[0].startswith("curve_") else Xh,
                           xha, Yh, hpa_idx, a.Cs, cfg.final_max_iter,
                           val_parts, pc, labels, hpa_labels)
        for tag, cols, xa, xha, pc in tasks)

    # ---- save --------------------------------------------------------
    rows, pcs, hrows, hpcs, crows = [], [], [], [], []
    for r in res:
        name, dims, seed = r["tag"]
        crows.append({"fold": fold, "method": name, "dims": dims, "seed": seed,
                      "best_C": r["best_C"],
                      **{f"cv_C={c:g}": v for c, v in r["cv_scores"].items()}})
        for regime, e in r["eval"].items():
            base = {"fold": fold, "method": name, "dims": dims, "seed": seed,
                    "regime": regime, "C": e["C"]}
            rows.append({**base, **e["metrics"]})
            if e["pc"]:
                for lab, v in e["pc"].items():
                    pcs.append({**base, "label": lab, **v})
            if e["hpa"]:
                hrows.append({**base, **e["hpa"]})
                for lab, v in e["hpa_pc"].items():
                    hpcs.append({**base, "label": lab, **v})
    pd.DataFrame(rows).to_csv(os.path.join(out, "test_results.csv"), index=False)
    pd.DataFrame(pcs).to_csv(os.path.join(out, "test_per_class.csv"), index=False)
    pd.DataFrame(crows).to_csv(os.path.join(out, "C_tuning.csv"), index=False)
    if hrows:
        pd.DataFrame(hrows).to_csv(os.path.join(out, "hpa_results.csv"), index=False)
        pd.DataFrame(hpcs).to_csv(os.path.join(out, "hpa_per_class.csv"), index=False)
    mins = (time.time() - t0) / 60
    json.dump({"fold": fold, "k": int(k), "minutes": round(mins, 1),
               "val_parts": val_parts, "Cs": a.Cs}, open(done, "w"))
    print(f"[fold {fold}] done in {mins:.1f} min")
    if folds_left:
        print(f"[time] about {mins * folds_left:.0f} more minutes for the "
              f"remaining {folds_left} fold(s), judging by this one")


# ======================================================================
# Consensus subset on the external set (trained on all five partitions)
# ======================================================================
def consensus_on_hpa(a, cfg, data, hpa, folds):
    from joblib import Parallel, delayed
    out = os.path.join(a.out, "consensus")
    done = os.path.join(out, "consensus_results.csv")
    if os.path.exists(done) and not a.smoke:
        return
    masks = []
    for f in folds:
        p = os.path.join(a.out, f"fold_{f}", "ssa_mask.npy")
        if a.smoke:
            D = data[1].shape[1]
            rng = np.random.default_rng(f)
            m = np.zeros(D, bool)
            m[rng.choice(D, 30, replace=False)] = True
            masks.append(m)
        elif os.path.exists(p):
            masks.append(np.load(p).astype(bool))
    if len(masks) < 5 and not a.smoke:
        print("[consensus] needs all five folds -> skipped")
        return
    os.makedirs(out, exist_ok=True)
    df, X, Y, labels, parts = data
    cnt = np.sum(masks, axis=0)
    core = np.flatnonzero(cnt >= 3)
    if len(core) == 0:
        print("[consensus] no dimension in >= 3 folds -> skipped")
        return
    np.save(os.path.join(out, "consensus_dims.npy"), core)
    tr = np.ones(len(Y), bool)
    te = np.zeros(len(Y), bool)
    te[:1] = True                                  # unused placeholder
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xz = (X - mu) / sd
    Xh = (hpa["X"] - mu) / sd
    D = X.shape[1]
    allp = sorted(int(p) for p in np.unique(parts))
    val_parts = allp[:a.val_folds]
    tasks = [(("consensus_core", len(core), -1), core),
             (("full", D, -1), np.arange(D))]
    for s in range(10):
        tasks.append((("random_same_size", len(core), s),
                      np.random.default_rng(3000 + s).choice(D, len(core), replace=False)))
    print(f"[consensus] {len(core)} dims in >= 3 folds; testing on HPA ...")
    res = Parallel(n_jobs=a.workers, backend="loky", verbose=5)(
        delayed(tune_task)(tag, Xz, Y, parts, tr, te, cols, None, Xh, None,
                           hpa["Y"], hpa["idx"], a.Cs, cfg.final_max_iter,
                           val_parts, False, labels, hpa["labels"])
        for tag, cols in tasks)
    rows = []
    for r in res:
        name, dims, seed = r["tag"]
        for regime, e in r["eval"].items():
            rows.append({"method": name, "dims": dims, "seed": seed,
                         "regime": regime, "C": e["C"], **e["hpa"]})
    pd.DataFrame(rows).to_csv(done, index=False)


# ======================================================================
# Statistics and report
# ======================================================================
def paired(a, b):
    from scipy import stats as st
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if n < 2:
        return dict(mean_diff=float(d.mean()) if n else np.nan, ci_lo=np.nan,
                    ci_hi=np.nan, t=np.nan, p=np.nan, d_z=np.nan,
                    wins=int((d > 0).sum()), n=n)
    t = st.ttest_rel(a, b)
    se = d.std(ddof=1) / np.sqrt(n)
    h = st.t.ppf(0.975, n - 1) * se
    return dict(mean_diff=float(d.mean()), ci_lo=float(d.mean() - h),
                ci_hi=float(d.mean() + h), t=float(t.statistic),
                p=float(t.pvalue),
                d_z=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan,
                wins=int((d > 0).sum()), n=n)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def compare(per_fold, ref="SS-SA"):
    piv = per_fold.pivot(index="fold", columns="method", values="mcc_macro")
    rows = []
    for b in [c for c in piv.columns if c != ref]:
        ok = piv[[ref, b]].dropna()
        rows.append({"comparison": f"{ref} - {b}", **paired(ok[ref], ok[b])})
    s = pd.DataFrame(rows)
    if len(s):
        s["p_holm"] = holm(s["p"].fillna(1.0))
    return s


def aggregate(a, folds, hpa):
    R = []
    lines = []

    def say(x=""):
        print(x)
        lines.append(str(x))

    def load(name):
        fs = [os.path.join(a.out, f"fold_{f}", name) for f in folds]
        fs = [p for p in fs if os.path.exists(p)]
        return pd.concat([pd.read_csv(p) for p in fs], ignore_index=True) if fs else None

    tr = load("test_results.csv")
    if tr is None:
        say("nothing to aggregate")
        return
    nf = tr["fold"].nunique()
    met = ["mcc_macro", "f1_macro", "f1_micro", "jaccard"]
    say("=" * 72)
    say(f"PAPER 1 -- GAP FIXES  ({nf} folds; C grid {a.Cs}; "
        f"inner folds {a.val_folds})")
    say("=" * 72)

    matched = tr[~tr.method.str.startswith("curve_")]
    pf = (matched.groupby(["regime", "fold", "method"])[met + ["dims"]]
          .mean().reset_index())
    summ = pf.groupby(["regime", "method"])[met + ["dims"]].agg(["mean", "std"])
    summ.columns = [f"{x}_{y}" for x, y in summ.columns]
    summ = summ.round(4).reset_index()
    summ.to_csv(os.path.join(a.out, "V4_summary.csv"), index=False)
    for reg in ("fixed", "tuned"):
        s = summ[summ.regime == reg].sort_values("mcc_macro_mean", ascending=False)
        say(f"\n[{reg.upper()} C]  five-fold mean MCC at matched size")
        say(s[["method", "dims_mean", "mcc_macro_mean", "mcc_macro_std",
               "f1_macro_mean", "f1_micro_mean", "jaccard_mean"]]
            .to_string(index=False))

    st_all = []
    for reg in ("fixed", "tuned"):
        s = compare(pf[pf.regime == reg])
        s.insert(0, "regime", reg)
        st_all.append(s)
    st_all = pd.concat(st_all, ignore_index=True).round(5)
    st_all.to_csv(os.path.join(a.out, "V4_stats.csv"), index=False)
    say("\nPAIRED TESTS, SS-SA minus each method (Holm within each regime)")
    say(st_all[["regime", "comparison", "mean_diff", "ci_lo", "ci_hi", "t", "p",
                "p_holm", "d_z", "wins"]].to_string(index=False))

    # gap 2: tuned vs fixed for the same method
    g2 = []
    for m in pf.method.unique():
        f_ = pf[(pf.method == m) & (pf.regime == "fixed")].set_index("fold").mcc_macro
        t_ = pf[(pf.method == m) & (pf.regime == "tuned")].set_index("fold").mcc_macro
        ok = pd.concat([t_, f_], axis=1, keys=["t", "f"]).dropna()
        g2.append({"method": m, **paired(ok["t"], ok["f"])})
    g2 = pd.DataFrame(g2).round(5)
    g2.to_csv(os.path.join(a.out, "V4_tuned_minus_fixed.csv"), index=False)
    say("\nEFFECT OF RE-TUNING C (tuned minus fixed, same method)")
    say(g2[["method", "mean_diff", "ci_lo", "ci_hi", "p", "wins"]].to_string(index=False))

    ct = load("C_tuning.csv")
    ct.to_csv(os.path.join(a.out, "V4_C_tuning_all.csv"), index=False)
    lo_, hi_ = min(a.Cs), max(a.Cs)
    edge = ct[(ct.best_C == lo_) | (ct.best_C == hi_)]
    say(f"\nGRID-EDGE CHECK: {len(edge)} of {len(ct)} tuned configurations chose "
        f"C at the edge of the grid ({lo_:g} or {hi_:g})")
    if len(edge):
        say(edge.groupby(["method", "best_C"]).size().to_string())
    say("\nCHOSEN C (count over folds and seeds)")
    say(pd.crosstab(ct.method + "@" + ct.dims.astype(str), ct.best_C).to_string())

    # curve, both regimes
    cv = tr[tr.method.str.startswith("curve_")].copy()
    cv["method"] = cv.method.str.replace("curve_", "")
    full = pf[pf.method == "full"][["regime", "fold", "mcc_macro"]]
    cvm = (cv.groupby(["regime", "fold", "method", "dims"]).mcc_macro.mean()
           .reset_index())
    cvm.to_csv(os.path.join(a.out, "V4_curve.csv"), index=False)
    say("\nSIZE-PERFORMANCE CURVE (five-fold mean MCC)")
    say(cvm.groupby(["regime", "method", "dims"]).mcc_macro.mean().round(4)
        .unstack("regime").to_string())
    ceil = []
    for reg in ("fixed", "tuned"):
        fr = full[full.regime == reg].set_index("fold").mcc_macro
        for m in ("random", "variance"):
            for d in sorted(cvm.dims.unique()):
                x = cvm[(cvm.regime == reg) & (cvm.method == m) & (cvm.dims == d)
                        ].set_index("fold").mcc_macro
                ok = pd.concat([x, fr], axis=1, keys=["x", "f"]).dropna()
                if len(ok):
                    ceil.append({"regime": reg, "method": m, "dims": d,
                                 **paired(ok["x"], ok["f"])})
    ceil = pd.DataFrame(ceil).round(5)
    ceil.to_csv(os.path.join(a.out, "V4_curve_vs_full.csv"), index=False)
    say("\nIS THE FULL EMBEDDING A CEILING?  (subset minus full, per regime)")
    say(ceil[ceil.dims >= ceil.dims.max() // 3][
        ["regime", "method", "dims", "mean_diff", "ci_lo", "ci_hi", "p", "wins"]]
        .to_string(index=False))

    pc = load("test_per_class.csv")
    pcm = pc.groupby(["regime", "method", "label"])[["mcc", "support"]].agg(["mean", "std"])
    pcm.columns = [f"{x}_{y}" for x, y in pcm.columns]
    pcm.round(4).to_csv(os.path.join(a.out, "V4_per_class.csv"))

    # L1 info
    l1 = []
    for f in folds:
        p = os.path.join(a.out, f"fold_{f}", "l1_info.json")
        if os.path.exists(p):
            j = json.load(open(p))
            l1.append({"fold": f, "k": j["k_target"], "C": j["C_selected"],
                       "union": j["union_at_C"]})
    if l1:
        say("\nL1 EMBEDDED SELECTION PER FOLD")
        say(pd.DataFrame(l1).to_string(index=False))
        ov = []
        for f in folds:
            pm = os.path.join(a.out, f"fold_{f}", "ssa_mask.npy")
            pl = os.path.join(a.out, f"fold_{f}", "l1_cols.npy")
            if os.path.exists(pm) and os.path.exists(pl):
                s = set(np.flatnonzero(np.load(pm)))
                l_ = set(np.load(pl).tolist())
                ov.append({"fold": f, "ssa": len(s), "l1": len(l_),
                           "shared": len(s & l_)})
        if ov:
            say("overlap SS-SA vs L1 subsets:")
            say(pd.DataFrame(ov).to_string(index=False))

    # gap 3: external set
    hr = load("hpa_results.csv")
    if hr is not None:
        say("\n" + "=" * 72)
        if hpa:
            say(f"EXTERNAL TEST SET (HPA): {hpa['info'].get('n_used')} proteins, "
                f"labels {hpa['labels']}")
            say(f"positives: {hpa['info'].get('positives')}")
        hpf = (hr.groupby(["regime", "fold", "method"])[met + ["dims"]]
               .mean().reset_index())
        hs = hpf.groupby(["regime", "method"])[met + ["dims"]].agg(["mean", "std"])
        hs.columns = [f"{x}_{y}" for x, y in hs.columns]
        hs = hs.round(4).reset_index()
        hs.to_csv(os.path.join(a.out, "V4_hpa_summary.csv"), index=False)
        for reg in ("fixed", "tuned"):
            s = hs[hs.regime == reg].sort_values("mcc_macro_mean", ascending=False)
            say(f"\n[{reg.upper()} C]  HPA, mean over the five fold models")
            say(s[["method", "dims_mean", "mcc_macro_mean", "mcc_macro_std",
                   "f1_macro_mean", "f1_micro_mean", "jaccard_mean"]]
                .to_string(index=False))
        hst = []
        for reg in ("fixed", "tuned"):
            s = compare(hpf[hpf.regime == reg])
            s.insert(0, "regime", reg)
            hst.append(s)
        hst = pd.concat(hst, ignore_index=True).round(5)
        hst.to_csv(os.path.join(a.out, "V4_hpa_stats.csv"), index=False)
        say("\nHPA PAIRED TESTS (paired by fold model), SS-SA minus each method")
        say(hst[["regime", "comparison", "mean_diff", "ci_lo", "ci_hi", "p",
                 "p_holm", "wins"]].to_string(index=False))
        hpc = load("hpa_per_class.csv")
        hpcm = hpc.groupby(["regime", "method", "label"])[["mcc", "support"]].agg(["mean", "std"])
        hpcm.columns = [f"{x}_{y}" for x, y in hpcm.columns]
        hpcm.round(4).to_csv(os.path.join(a.out, "V4_hpa_per_class.csv"))
        say("\nHPA per-compartment MCC (tuned C):")
        say(hpc[hpc.regime == "tuned"].groupby(["label", "method"]).mcc.mean()
            .round(3).unstack("method").to_string())
    cr = os.path.join(a.out, "consensus", "consensus_results.csv")
    if os.path.exists(cr):
        c = pd.read_csv(cr)
        say("\nCONSENSUS SUBSET (dims in >= 3 folds), trained on all partitions, "
            "tested on HPA")
        say(c.groupby(["regime", "method", "dims"]).mcc_macro.agg(["mean", "std"])
            .round(4).to_string())

    # reproducibility vs earlier V2 results (fixed C)
    rep = []
    for f in folds:
        p = os.path.join("runs", f"fold_{f}", "results_v2", "method_comparison_v2.csv")
        if not os.path.exists(p):
            continue
        old = pd.read_csv(p).groupby("method").mcc_macro.mean()
        new = pf[(pf.regime == "fixed") & (pf.fold == f)].set_index("method").mcc_macro
        for m in ("SS-SA", "full", "pca", "mutual_info", "variance"):
            if m in old.index and m in new.index:
                rep.append({"fold": f, "method": m, "v2": round(old[m], 4),
                            "v3_fixed": round(new[m], 4),
                            "abs_diff": round(abs(old[m] - new[m]), 4)})
    if rep:
        rep = pd.DataFrame(rep)
        rep.to_csv(os.path.join(a.out, "V4_repro_check.csv"), index=False)
        say(f"\nREPRODUCIBILITY vs V2 (fixed C): max |diff| = {rep.abs_diff.max():.4f}")

    with open(os.path.join(a.out, "V4_REPORT.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def pack(a):
    base = a.out.rstrip("/\\")
    shutil.make_archive(base + "_results", "zip", root_dir=os.path.dirname(base) or ".",
                        base_dir=os.path.basename(base))
    print(f"\n[done] results packed into {base}_results.zip")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--Cs", type=float, nargs="+",
                    default=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0])
    ap.add_argument("--out", default=os.path.join("runs", "V4"))
    ap.add_argument("--reuse", default="",
                    help="optional earlier output folder whose MI/L1 "
                         "selections are reused (e.g. runs/V4)")
    ap.add_argument("--hpa-min-pos", type=int, default=20)
    ap.add_argument("--val-folds", type=int, default=2,
                    help="inner validation partitions used to tune C")
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--curve-seeds", type=int, default=3)
    ap.add_argument("--hpa", default="hpa_testset.csv")
    ap.add_argument("--data-csv", default=None,
                    help="DeepLoc 2.0 SwissProt CSV (default: config.py)")
    ap.add_argument("--no-hpa", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if not os.path.exists("main.py"):
        sys.exit("ERROR: run this from the project folder (main.py not found)")
    if FIXED_C not in a.Cs:
        a.Cs = sorted(a.Cs + [FIXED_C])

    from config import Config
    cfg = Config()
    cfg.verbose = True
    if a.data_csv:
        cfg.data_csv = os.path.abspath(a.data_csv)
    if a.smoke:
        os.makedirs("runs_smoke", exist_ok=True)
        os.chdir("runs_smoke")
        sys.path.insert(0, os.path.abspath(".."))
        a.folds, a.random_seeds, a.curve_seeds = [0, 1, 2, 3, 4], 2, 1
        a.workers = min(a.workers, 2)
        a.Cs = [0.1, 1.0]
        a.out = os.path.join("runs", "V4")
        print("[smoke] synthetic self-test; numbers are meaningless")
    os.makedirs(a.out, exist_ok=True)

    t_start = time.time()
    data = load_swissprot(cfg, a.smoke)
    hpa = load_hpa(a, cfg, data[0], data[3])
    if hpa is not None:
        json.dump(hpa["info"], open(os.path.join(a.out, "hpa_info.json"), "w"),
                  indent=1)
    for i, f in enumerate(a.folds):
        process_fold(f, a, cfg, data, hpa, t_start, len(a.folds) - i - 1)
    if hpa is not None:
        consensus_on_hpa(a, cfg, data, hpa, a.folds)
    aggregate(a, a.folds, hpa)
    pack(a)
    print(f"total time {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
