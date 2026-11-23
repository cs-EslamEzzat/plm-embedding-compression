"""
optimizer.py
============
The core contribution of Paper 1: a hybrid Scatter Search (SS) +
Simulated Annealing (SA) WRAPPER that selects a compact subset of
protein-language-model embedding dimensions for multi-label localization.

A candidate solution is a binary mask m in {0,1}^D over the D embedding
dimensions. Fitness rewards predictive quality and penalises size:

    f(m) = MCC_cv(X[:, m], Y) - lambda * (|m| / D)

Scatter Search drives global exploration (diverse pool -> reference set ->
structured combination); Simulated Annealing performs local refinement of
each combined solution via Metropolis-accepted bit flips. Fitness values are
memoised so repeated masks are never re-evaluated (a large speed-up).
"""

from __future__ import annotations
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from config import Config


FitnessFn = Callable[[np.ndarray], float]


@dataclass
class OptHistory:
    best_fitness: List[float] = field(default_factory=list)
    best_size: List[int] = field(default_factory=list)
    evaluations: int = 0


class SSSAOptimizer:
    def __init__(self, D: int, fitness_fn: FitnessFn, cfg: Config):
        self.D = D
        self.raw_fitness = fitness_fn
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._cache: Dict[bytes, float] = {}
        self.history = OptHistory()

    # ---------- fitness with memoisation ----------
    def fitness(self, mask: np.ndarray) -> float:
        if mask.sum() == 0:                       # never allow empty subset
            return -1.0
        key = np.packbits(mask).tobytes()
        if key in self._cache:
            return self._cache[key]
        val = self.raw_fitness(mask)
        self._cache[key] = val
        # Counts evaluations made in this process only; evaluations performed
        # inside parallel workers are not added back to this counter.
        self.history.evaluations += 1
        return val

    # ---------- minimum-size enforcement ----------
    def _min_dims(self) -> int:
        k = int(getattr(self.cfg, "min_dims", 1) or 1)
        return max(1, min(k, self.D))

    def _repair(self, mask: np.ndarray) -> np.ndarray:
        """Guarantee at least cfg.min_dims active dimensions."""
        need = self._min_dims() - int(mask.sum())
        if need > 0:
            off = np.flatnonzero(mask == 0)
            if off.size:
                mask[self.rng.choice(off, size=min(need, off.size),
                                     replace=False)] = 1
        return mask

    # ---------- Scatter Search: diversification ----------
    def _random_mask(self, density: float) -> np.ndarray:
        m = (self.rng.random(self.D) < density).astype(np.uint8)
        if m.sum() == 0:
            m[self.rng.integers(self.D)] = 1
        return self._repair(m)

    def _initial_pool(self) -> List[np.ndarray]:
        """
        Seed across the WHOLE size range (log-spaced, ~1% -> 90%).
        Uniform(0.2, 0.8) can never reach small subsets: with only a few bit
        flips per SA move, a 256-dim start cannot shrink to 50 dims. Since this
        paper's claim is COMPACT representations, small subsets must be
        reachable from the start.
        """
        n = self.cfg.ss_pool_size
        lo = max(getattr(self.cfg, "min_dims", 1) / self.D, 0.01)
        dens = np.logspace(np.log10(lo), np.log10(0.90), num=n)
        self.rng.shuffle(dens)
        return [self._random_mask(float(d)) for d in dens]

    # ---------- diversity (Hamming) ----------
    @staticmethod
    def _hamming(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.count_nonzero(a != b))

    def _build_refset(self, pool: List[np.ndarray]
                      ) -> List[Tuple[np.ndarray, float]]:
        scored = sorted(((m, self.fitness(m)) for m in pool),
                        key=lambda t: t[1], reverse=True)
        ref = scored[: self.cfg.ss_refset_quality]          # best by quality
        chosen = [m for m, _ in ref]
        # add the most diverse remaining members
        rest = [m for m, _ in scored[self.cfg.ss_refset_quality:]]
        for _ in range(self.cfg.ss_refset_diverse):
            if not rest:
                break
            div = [min(self._hamming(c, ch) for ch in chosen) for c in rest]
            idx = int(np.argmax(div))
            m = rest.pop(idx)
            ref.append((m, self.fitness(m)))
            chosen.append(m)
        return ref

    # ---------- Scatter Search: subset combination ----------
    def _combine(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Structured recombination: keep agreed-on bits, randomly resolve the
        bits where the two parents disagree (a uniform-style crossover).
        """
        agree = a == b
        child = np.where(agree, a, self.rng.integers(0, 2, self.D)).astype(np.uint8)
        if child.sum() == 0:
            child[self.rng.integers(self.D)] = 1
        return self._repair(child)

    # ---------- Simulated Annealing: local improvement ----------
    def _sa_improve(self, mask: np.ndarray, seed: int | None = None
                    ) -> Tuple[np.ndarray, float]:
        """Local SA refinement.

        `seed` gives this call its OWN random stream. Without it, parallel
        workers each receive a copy of the same RNG state and explore the
        exact same moves, destroying search diversity.
        """
        rng = self.rng if seed is None else np.random.default_rng(seed)
        current = mask.copy()
        cur_f = self.fitness(current)
        best, best_f = current.copy(), cur_f
        T = self.cfg.sa_T0
        for _ in range(self.cfg.sa_steps):
            nb = current.copy()
            # Mixed neighbourhood: mostly small local flips, plus occasional
            # BULK add/remove moves so the search can travel along the size
            # axis instead of being stuck near its starting size.
            u = rng.random()
            if u < 0.25:                      # bulk REMOVE (shrink)
                on = np.flatnonzero(nb)
                if on.size > 1:
                    k = max(1, int(rng.integers(1, max(2, on.size // 2))))
                    nb[rng.choice(on, size=min(k, on.size - 1),
                                       replace=False)] = 0
            elif u < 0.45:                    # bulk ADD (grow)
                off = np.flatnonzero(nb == 0)
                if off.size:
                    k = max(1, int(rng.integers(1, max(2, off.size // 8))))
                    nb[rng.choice(off, size=min(k, off.size),
                                       replace=False)] = 1
            else:                             # fine local flips
                flips = rng.integers(0, self.D, size=self.cfg.sa_flip_k)
                nb[flips] ^= 1
            nb = self._repair(nb)
            if nb.sum() == 0:
                continue
            nb_f = self.fitness(nb)
            delta = nb_f - cur_f
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                current, cur_f = nb, nb_f
                if cur_f > best_f:
                    best, best_f = current.copy(), cur_f
            T *= self.cfg.sa_cooling
        return best, best_f

    # ---------- checkpointing ----------
    def _ckpt_path(self) -> str:
        d = getattr(self.cfg, "cache_dir", "cache")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"ss_ckpt_D{self.D}_seed{self.cfg.seed}.pkl")

    def _save_ckpt(self, gen, refset, best_mask, best_f):
        """Save progress so an interrupted run can resume from this generation."""
        try:
            with open(self._ckpt_path(), "wb") as f:
                pickle.dump({
                    "gen": gen,
                    "refset": refset,
                    "best_mask": best_mask,
                    "best_f": best_f,
                    "cache": self._cache,
                    "history": self.history,
                }, f)
        except Exception as e:
            print(f"[ckpt] warning: could not save checkpoint ({e})")

    def _load_ckpt(self):
        p = self._ckpt_path()
        if not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            self._cache = d.get("cache", {})
            self.history = d.get("history", self.history)
            print(f"[ckpt] resuming from generation {d['gen']}/"
                  f"{self.cfg.ss_generations}  (best={d['best_f']:.4f})")
            return d
        except Exception as e:
            print(f"[ckpt] warning: could not load checkpoint ({e})")
            return None

    def clear_ckpt(self):
        try:
            os.remove(self._ckpt_path())
        except OSError:
            pass

    # ---------- main loop ----------
    def run(self) -> Tuple[np.ndarray, float]:
        start_gen = 0
        ck = self._load_ckpt() if getattr(self.cfg, "resume", True) else None
        if ck is not None:
            refset = ck["refset"]
            best_mask, best_f = ck["best_mask"], ck["best_f"]
            start_gen = ck["gen"]
        else:
            pool = self._initial_pool()
            refset = self._build_refset(pool)
            best_mask, best_f = max(refset, key=lambda t: t[1])

        for gen in range(start_gen, self.cfg.ss_generations):
            new_solutions: List[Tuple[np.ndarray, float]] = []
            members = [m for m, _ in refset]

            # Build all children first, then run their SA local searches.
            # Each local search is INDEPENDENT, so they can run in parallel
            # across CPU cores -- this is the real bottleneck of the search
            # (the classifier inside only parallelises over ~10 labels).
            children = [self._combine(members[i], members[j])
                        for i in range(len(members))
                        for j in range(i + 1, len(members))]

            workers = int(getattr(self.cfg, "search_workers", 1) or 1)
            if workers > 1 and len(children) > 1:
                from joblib import Parallel, delayed
                base = int(self.rng.integers(0, 2**31 - 1))
                results = Parallel(n_jobs=workers, backend="loky",
                                   batch_size=1)(
                    delayed(self._sa_improve)(c, base + k)
                    for k, c in enumerate(children))
                new_solutions.extend(results)
            else:
                for c in children:
                    new_solutions.append(self._sa_improve(c))

            # update reference set: quality + diversity from the union
            union = refset + new_solutions
            # de-duplicate by mask
            seen, uniq = set(), []
            for m, f in union:
                k = m.tobytes()
                if k not in seen:
                    seen.add(k)
                    uniq.append((m, f))
            uniq.sort(key=lambda t: t[1], reverse=True)
            quality = uniq[: self.cfg.ss_refset_quality]
            chosen = [m for m, _ in quality]
            rest = [t for t in uniq[self.cfg.ss_refset_quality:]]
            diverse = []
            for _ in range(self.cfg.ss_refset_diverse):
                if not rest:
                    break
                divs = [min(self._hamming(c, ch) for ch in chosen)
                        for c, _ in rest]
                idx = int(np.argmax(divs))
                diverse.append(rest.pop(idx))
                chosen.append(diverse[-1][0])
            refset = quality + diverse

            gen_best_mask, gen_best_f = max(refset, key=lambda t: t[1])
            if gen_best_f > best_f:
                best_mask, best_f = gen_best_mask.copy(), gen_best_f

            self.history.best_fitness.append(best_f)
            self.history.best_size.append(int(best_mask.sum()))
            if self.cfg.verbose:
                print(f"[SS-SA] gen {gen+1:02d}/{self.cfg.ss_generations} "
                      f"best_f={best_f:.4f} size={int(best_mask.sum())} "
                      f"evals={self.history.evaluations}")
            self._save_ckpt(gen + 1, refset, best_mask, best_f)

        return best_mask.astype(int), best_f
