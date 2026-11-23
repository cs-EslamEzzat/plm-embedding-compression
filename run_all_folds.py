"""
run_all_folds.py
================
Runs main.py once per homology partition (each partition is held out in
turn), archives every fold's outputs in its own folder, and then aggregates
the five folds with aggregate_folds.py.

Usage (from the repository root):
    python run_all_folds.py --generations 15 --final-clf logreg_t --workers 12
    python run_all_folds.py --folds 0 1 2            # selected folds only
    python run_all_folds.py --resume-runs            # skip finished folds
    python run_all_folds.py --sample 2000 --fast     # quick test only

Outputs:
    runs/fold_0/ ... runs/fold_4/    results/ and figures/ of every fold
    runs/FINAL_summary.csv           five-fold mean and SD per method
    runs/FINAL_per_fold.csv          all per-fold rows
    runs/FINAL_stats.csv             paired tests, SS-SA vs every baseline
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time


RUNS_DIR = "runs"


def parse_args():
    p = argparse.ArgumentParser(description="Run all folds and aggregate")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="test partitions to run (default: 0 1 2 3 4)")
    p.add_argument("--embedder", default="esm2")
    p.add_argument("--sample", type=int, default=0,
                   help="use only N proteins (testing); 0 = all")
    p.add_argument("--fast", action="store_true",
                   help="lighter search (testing only, not for reported results)")
    p.add_argument("--generations", type=int, default=None,
                   help="search generations per fold (paper: 15)")
    p.add_argument("--min-dims", type=int, default=None,
                   help="minimum dimensions per solution (default 30)")
    p.add_argument("--final-clf", default=None,
                   choices=["logreg", "logreg_t", "mlp"],
                   help="final classifier; must be the same for every fold")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel CPU workers for the search (paper: 12)")
    p.add_argument("--resume-runs", action="store_true",
                   help="skip folds that already produced results")
    p.add_argument("--python", default=sys.executable,
                   help="python executable to use")
    return p.parse_args()


def run_one_fold(fold: int, a) -> bool:
    """Run main.py for a single held-out partition and archive its outputs."""
    out_dir = os.path.join(RUNS_DIR, f"fold_{fold}")
    done_marker = os.path.join(out_dir, "results", "method_comparison.csv")

    if a.resume_runs and os.path.exists(done_marker):
        print(f"[fold {fold}] already done -> skipping")
        return True

    cmd = [a.python, "main.py",
           "--embedder", a.embedder,
           "--test-partition", str(fold),
           "--restart"]                    # fresh search for every fold
    if a.sample:
        cmd += ["--sample", str(a.sample)]
    if a.fast:
        cmd += ["--fast"]
    if a.generations is not None:
        cmd += ["--generations", str(a.generations)]
    if a.min_dims is not None:
        cmd += ["--min-dims", str(a.min_dims)]
    if a.final_clf is not None:
        cmd += ["--final-clf", a.final_clf]
    if a.workers is not None:
        cmd += ["--workers", str(a.workers)]

    print("\n" + "=" * 70)
    print(f"[fold {fold}] running: {' '.join(cmd)}")
    print("=" * 70)
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        print(f"[fold {fold}] FAILED (exit {r.returncode})")
        return False
    print(f"[fold {fold}] finished in {dt:.1f} min")

    # archive this fold's outputs (main.py writes to results/ and figures/)
    os.makedirs(out_dir, exist_ok=True)
    for sub in ("results", "figures"):
        dst = os.path.join(out_dir, sub)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if os.path.exists(sub):
            shutil.copytree(sub, dst)
    print(f"[fold {fold}] archived -> {out_dir}/")
    return True


def main():
    a = parse_args()
    if not os.path.exists("main.py"):
        print("ERROR: run this from the repository root (main.py not found).")
        sys.exit(1)

    if a.fast or a.sample:
        print("\n*** WARNING: --fast / --sample are for testing only. ***\n")

    os.makedirs(RUNS_DIR, exist_ok=True)
    ok = [f for f in a.folds if run_one_fold(f, a)]
    if len(ok) < 5:
        print(f"\n[warn] {len(ok)} of 5 folds available; "
              "the reported results need all five.")

    import aggregate_folds
    aggregate_folds.main()


if __name__ == "__main__":
    main()
