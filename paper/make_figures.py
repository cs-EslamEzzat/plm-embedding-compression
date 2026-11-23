"""
paper/make_figures.py
=====================
Rebuilds Figures 2-11 of the paper from the saved results in runs/V4
(written by gap_fixes.py). Figures 1, 12 and 13 come from main.py
(figures/01_class_distribution + 02_label_cardinality, 07_convergence +
08_size_over_generations, and 05_embedding_pca_variance).

Usage (from the repository root):
    python paper/make_figures.py                # reads runs/V4
    python paper/make_figures.py runs/V4_rerun  # any other gap_fixes output

Outputs: paper/figures/figN.pdf and figN.png (600 dpi, grayscale).
"""
import os, sys
import glob, json, itertools
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats

V = (sys.argv[1] if len(sys.argv) > 1 else os.path.join('runs', 'V4')).rstrip('/\\') + os.sep
O = os.path.join('paper', 'figures') + os.sep
os.makedirs(O, exist_ok=True)
NUM = {'fig_frontier': 2, 'fig_retention_folds': 3, 'fig_artifact': 4, 'fig_Cprofile': 5,
       'fig_forest': 6, 'fig_metrics': 7, 'fig_perfold': 8, 'fig_external': 9,
       'fig_stability': 10, 'fig_percompartment': 11}
K, D1, D2, G, L = '#000000', '#3a3a3a', '#6e6e6e', '#a0a0a0', '#d9d9d9'
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
    'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.6,
    'axes.edgecolor': D2, 'axes.labelcolor': K, 'xtick.color': D1, 'ytick.color': D1, 'text.color': K,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#ececec',
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'lines.linewidth': 1.4, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'legend.frameon': False, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'hatch.color': D1, 'hatch.linewidth': 0.6})
def save(f, n):
    base = O + f'fig{NUM[n]}'
    f.savefig(base + '.pdf'); f.savefig(base + '.png', dpi=600); plt.close(f)

def Lc(n): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(V + f'fold_*/{n}'))], ignore_index=True)
tr, pc, hr, hpc = Lc('test_results.csv'), Lc('test_per_class.csv'), Lc('hpa_results.csv'), Lc('hpa_per_class.csv')
met = ['mcc_macro', 'f1_macro', 'f1_micro', 'jaccard']
mt = tr[~tr.method.str.startswith('curve_')]
pf = mt.groupby(['regime', 'fold', 'method'])[met + ['dims']].mean().reset_index()
cv = tr[tr.method.str.startswith('curve_')].copy(); cv['method'] = cv.method.str.replace('curve_', '')
cvm = cv.groupby(['regime', 'fold', 'method', 'dims']).mcc_macro.mean().reset_index()
full = pf[pf.method == 'full'].set_index(['regime', 'fold']).mcc_macro
tc = stats.t.ppf(.975, 4)
LB = {'full': 'Full embedding (1,280)', 'pca': 'PCA', 'L1_embedded': 'L1 embedded (Lasso)',
      'SS-SA': 'SS–SA wrapper', 'mutual_info': 'Mutual information', 'random': 'Random subset',
      'variance': 'Variance filter'}
MK = {'full': 'o', 'pca': '^', 'L1_embedded': 'X', 'SS-SA': 'D', 'mutual_info': 'v', 'random': 'o', 'variance': 's'}
FC = {'full': K, 'pca': 'white', 'L1_embedded': D2, 'SS-SA': K, 'mutual_info': 'white', 'random': G, 'variance': 'white'}
order = ['full', 'pca', 'L1_embedded', 'SS-SA', 'mutual_info', 'random', 'variance']
def piv(reg, src=pf): return src[src.regime == reg].pivot(index='fold', columns='method', values='mcc_macro')
def logx(ax):
    ax.set_xscale('log', base=2); ax.set_xticks([25, 64, 128, 256, 512, 1280])
    ax.set_xticklabels(['25', '64', '128', '256', '512', '1280'])

# ---------------- Fig: frontier (tuned)
P = piv('tuned'); kmean = pf[pf.method == 'SS-SA'].dims.mean()
f, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={'width_ratios': [1.35, 1]})
for m, ls in [('random', '-'), ('variance', '--')]:
    g = cvm[(cvm.regime == 'tuned') & (cvm.method == m)].groupby('dims').mcc_macro
    mu, sd = g.mean(), g.std()
    a.fill_between(mu.index, mu - sd, mu + sd, color=L, alpha=.55, lw=0)
    a.plot(mu.index, mu, ls=ls, color=D1, marker=MK[m], ms=4.5, mfc=FC[m], mec=D1, label=LB[m] + ' (mean ± SD)')
fm = P['full'].mean(); a.axhline(fm, color=K, ls=':', lw=1.2)
a.text(24, fm + .006, f'Full embedding, 1,280 dims (MCC {fm:.3f})', fontsize=7.4)
for m in ['pca', 'L1_embedded', 'SS-SA']:
    a.errorbar(kmean, P[m].mean(), yerr=P[m].std(), fmt=MK[m], ms=6.5, color=K, mfc=FC[m], mec=K,
               capsize=2, lw=1, label=LB[m] + ' (189 dims)', zorder=5)
logx(a); a.set_ylim(.26, .60); a.set_xlabel('Retained dimensions (log scale)'); a.set_ylabel('Macro MCC (test partition)')
a.set_title('a  Accuracy versus size (C tuned per size)'); a.legend(loc='lower right', fontsize=7)
for i, m in enumerate(order):
    mu = P[m].mean(); ci = tc * P[m].std() / np.sqrt(5)
    b.errorbar(mu, i, xerr=ci, fmt=MK[m], ms=6.5, color=K, mfc=FC[m], mec=K, capsize=2.5, lw=1.1, zorder=4)
    b.scatter(P[m], np.full(5, i) + .24, s=7, color=D2, lw=0, zorder=3)
    b.text(mu + ci + .003, i, f'{mu:.3f}', va='center', fontsize=7.4, color=D1)
b.set_yticks(range(len(order))); b.set_yticklabels([LB[m] for m in order]); b.invert_yaxis()
b.set_xlim(.47, .615); b.set_xlabel('Macro MCC, mean and 95% CI (5 folds)'); b.set_title('b  At ~189 dimensions'); b.grid(axis='y', visible=False)
f.tight_layout(); save(f, 'fig_frontier')

# ---------------- Fig: regularisation artefact
f, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
for reg, ls, mf, lab in [('fixed', '--', 'white', 'C fixed at 1.0'), ('tuned', '-', K, 'C tuned per size')]:
    g = cvm[(cvm.regime == reg) & (cvm.method == 'random')].groupby('dims').mcc_macro.mean()
    a.plot(g.index, g, ls=ls, color=K, marker='o', ms=4.5, mfc=mf, mec=K, label='Random subsets, ' + lab)
    a.axhline(full[reg].mean(), color=D2, ls=ls, lw=1.1)
    if reg == 'tuned':
        a.text(24, full[reg].mean() + .006, f'Full embedding, {lab}: {full[reg].mean():.3f}', fontsize=7.1, color=D1, va='bottom')
    else:
        a.text(24, full[reg].mean() + .004, f'Full, {lab}: {full[reg].mean():.3f}', fontsize=7.1, color=D1, va='bottom')
logx(a); a.set_ylim(.26, .62); a.set_xlabel('Retained dimensions (log scale)'); a.set_ylabel('Macro MCC (test partition)')
a.set_title('a  Absolute performance'); a.legend(loc='lower right', bbox_to_anchor=(1, .02))
cvm['ret'] = cvm.apply(lambda r: 100 * r.mcc_macro / full[(r.regime, r.fold)], axis=1)
for reg, ls, mf, lab in [('fixed', '--', 'white', 'C fixed at 1.0'), ('tuned', '-', K, 'C tuned per size')]:
    g = cvm[(cvm.regime == reg) & (cvm.method == 'random') & (cvm.dims >= 256)].groupby('dims').ret
    b.errorbar(g.mean().index, g.mean(), yerr=g.std(), ls=ls, color=K, marker='o', ms=4.5, mfc=mf, mec=K, capsize=2, label=lab)
b.axhline(100, color=D2, lw=1); b.text(215, 100.5, '100% = full embedding (same regime)', fontsize=7.1, color=D1)
logx(b); b.set_xlim(220, 1100); b.set_ylim(85, 105); b.set_xticks([256, 512, 896, 1280]); b.set_xticklabels(['256', '512', '896', '1280'])
b.set_xlabel('Retained dimensions (log scale)'); b.set_ylabel('MCC retained (% of full embedding)')
b.set_title('b  Retention, 256–1,280 dims'); b.legend(loc='lower right')
f.tight_layout(); save(f, 'fig_artifact')

# ---------------- Fig: forest (tuned)
def pd_(a_, b_):
    d = a_ - b_; se = d.std(ddof=1) / np.sqrt(5); t = stats.ttest_rel(a_, b_)
    return d.mean(), d.mean() - tc * se, d.mean() + tc * se, t.pvalue, d.mean() / d.std(ddof=1)
rows = []
for bm in ['random', 'variance', 'mutual_info', 'L1_embedded', 'full', 'pca']:
    rows.append(('SS–SA  −  ' + LB[bm].replace(' (1,280)', ''),) + pd_(P['SS-SA'], P[bm]))
ps = np.array([r[4] for r in rows]); o = np.argsort(ps); run = 0; holm = np.empty(6)
for i, j in enumerate(o): run = max(run, (6 - i) * ps[j]); holm[j] = min(1, run)
rows = [r[:4] + (h,) + r[5:] for r, h in zip(rows, holm)]
for dd in [512, 896]:
    x = cvm[(cvm.regime == 'tuned') & (cvm.method == 'random') & (cvm.dims == dd)].set_index('fold').mcc_macro
    rows.append((f'Random {dd} dims  −  Full',) + pd_(x, full['tuned']))
f, ax = plt.subplots(figsize=(5.2, 3.3)); ys = [0, 1, 2, 3, 4, 5, 6.6, 7.6]
for y, r in zip(ys, rows):
    lab, d, lo, hi, p, dz = r
    ax.plot([lo, hi], [y, y], color=K, lw=1.5); ax.plot(d, y, 'o', ms=6, color=K, mfc=K if d > 0 else 'white', mec=K, zorder=4)
    pt = f'p = {p:.3f}' if p >= .001 else 'p < 0.001'
    ax.text(.032, y, f'{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {dz:+.2f}  {pt}', va='center', fontsize=7, family='DejaVu Sans Mono', color=D1)
ax.axvline(0, color=K, lw=.9); ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.axhline(5.8, color=L, lw=1.2); ax.grid(axis='y', visible=False); ax.set_xlim(-.085, .03)
ax.text(-.083, -.8, 'SS–SA versus each strategy (Holm-adjusted p)', fontsize=7.4, weight='bold', color=D1)
ax.text(-.083, 6.0, 'Is the full embedding a ceiling? (unadjusted p)', fontsize=7.4, weight='bold', color=D1)
ax.text(.032, -.8, 'Δ MCC   [95% CI]       d_z   p', fontsize=7, weight='bold', family='DejaVu Sans Mono', color=D1)
ax.set_xlabel('Paired difference in macro MCC (C tuned; 5 folds)')
save(f, 'fig_forest')

# ---------------- Fig: stability
masks = [np.load(V + f'fold_{i}/ssa_mask.npy').astype(bool) for i in range(5)]
k = [int(m.sum()) for m in masks]; Dn = 1280; cnt = np.sum(masks, 0)
pk = np.array(k) / Dn; dist = np.array([1.])
for p in pk: dist = np.convolve(dist, [1 - p, p])
exp = dist * Dn; obs = np.bincount(cnt, minlength=6)
rng = np.random.default_rng(42); sim3 = []
for _ in range(20000):
    c = np.zeros(Dn, int)
    for kk in k: c[rng.choice(Dn, kk, replace=False)] += 1
    sim3.append((c >= 3).sum())
sim3 = np.array(sim3); o3 = int((cnt >= 3).sum()); pperm = (sim3 >= o3).mean()
f, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9)); x = np.arange(6)
a.bar(x - .19, obs, width=.36, color=D1, label='Observed (SS–SA subsets)')
a.bar(x + .19, exp, width=.36, color='white', edgecolor=D1, hatch='////', lw=.8, label='Expected under random selection')
a.set_yscale('log'); a.set_ylim(.05, 3e4)
for xi, ob in zip(x, obs): a.text(xi - .19, max(ob, .07) * 1.25, str(ob), ha='center', fontsize=7.2, color=D1)
a.set_xticks(x); a.set_xlabel('Number of folds (of 5) selecting a dimension'); a.set_ylabel('Dimensions (log scale)')
a.set_title('a  Re-selection frequency'); a.legend(loc='upper right', fontsize=7.2); a.grid(axis='x', visible=False)
bins = np.arange(sim3.min() - .5, sim3.max() + 1.5); b.hist(sim3, bins=bins, color=G, lw=0)
b.axvline(o3, color=K, lw=2); b.text(o3 - .8, b.get_ylim()[1] * .6, f'Observed\n{o3} dims\np = {pperm:.3f}', ha='right', fontsize=7.6)
b.text(sim3.min() - 1, b.get_ylim()[1] * .95, f'Random selection\n(same subset sizes)\nmean = {sim3.mean():.1f}', va='top', fontsize=7.4, color=D1)
b.set_xlabel('Dimensions selected in ≥3 of 5 folds'); b.set_ylabel('Count in 20,000 random draws'); b.set_title('b  Consensus beyond chance'); b.grid(axis='x', visible=False)
f.tight_layout(); save(f, 'fig_stability')
json.dump({'obs': obs.tolist(), 'exp': exp.round(2).tolist(), 'o3': o3, 'sim3_mean': float(sim3.mean()), 'p': float(pperm)}, open(O + 'stability.json', 'w'))

# ---------------- Fig: per compartment (tuned)
w = pc[pc.regime == 'tuned'].groupby(['label', 'method']).mcc.mean().unstack()
sup = pc[(pc.regime == 'tuned') & (pc.method == 'full')].groupby('label').support.mean()
w['dS'] = w['SS-SA'] - w['full']; w['dL'] = w['L1_embedded'] - w['full']; w['n'] = sup; w = w.sort_values('n')
f, ax = plt.subplots(figsize=(7.2, 3.1)); y = np.arange(len(w))
ax.barh(y + .19, w.dS, height=.36, color=D1, label='SS–SA subset − full')
ax.barh(y - .19, w.dL, height=.36, color='white', edgecolor=D1, hatch='////', lw=.7, label='L1 subset − full')
for yi, d in zip(y, w.dS): ax.text(d - .002, yi + .19, f'{d:+.3f}', va='center', ha='right', fontsize=6.8, color=D1)
names = [l.replace('_', ' ').replace('Lysosome Vacuole', 'Lysosome / Vacuole') for l in w.index]
ax.set_yticks(y); ax.set_yticklabels([f'{n}  (n ≈ {s:.0f})' for n, s in zip(names, w.n)])
ax.axvline(0, color=K, lw=.9); ax.set_xlim(-.135, .01); ax.grid(axis='y', visible=False)
ax.set_xlabel('Δ MCC per compartment versus full embedding (C tuned; mean of 5 folds)'); ax.legend(loc='upper left')
save(f, 'fig_percompartment')

# ---------------- Fig: external HPA
H = piv('tuned', hr.groupby(['regime', 'fold', 'method'])[met].mean().reset_index())
dl = {'Cytoplasm': (.29, .36), 'Nucleus': (.41, .44), 'Cell_membrane': (.34, .36), 'Mitochondrion': (.60, .56),
      'Endoplasmic_reticulum': (.20, .17), 'Golgi_apparatus': (.17, .31)}
f, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={'width_ratios': [1, 1.25]})
for i, m in enumerate(order):
    mu = H[m].mean(); ci = tc * H[m].std() / np.sqrt(5)
    a.errorbar(mu, i, xerr=ci, fmt=MK[m], ms=6.5, color=K, mfc=FC[m], mec=K, capsize=2.5, lw=1.1)
    a.text(max(mu + ci, mu + .004) + .003, i, f'{mu:.3f}', va='center', fontsize=7.2, color=D1, zorder=6)
for v, ls, lab in [(np.mean([x[0] for x in dl.values()]), ':', 'DeepLoc 2.0 (ESM-1b)'), (np.mean([x[1] for x in dl.values()]), '-.', 'DeepLoc 2.0 (ProtT5)')]:
    a.axvline(v, color=D2, ls=ls, lw=1); a.text(v + .002, 6.55, lab, rotation=90, fontsize=6.6, color=D1, va='bottom')
a.set_yticks(range(len(order))); a.set_yticklabels([LB[m] for m in order]); a.invert_yaxis(); a.grid(axis='y', visible=False)
a.set_xlim(.25, .39); a.set_xlabel('Mean MCC over 6 compartments'); a.set_title('a  External HPA test set (1,717)')
hp = hpc[hpc.regime == 'tuned'].groupby(['label', 'method']).mcc.mean().unstack()
labs = ['Nucleus', 'Cytoplasm', 'Cell_membrane', 'Mitochondrion', 'Golgi_apparatus', 'Endoplasmic_reticulum']
x = np.arange(len(labs)); wd = .2
for j, (m, fc, h) in enumerate([('full', K, ''), ('SS-SA', D2, ''), ('L1_embedded', 'white', '////'), ('random', L, '')]):
    b.bar(x + (j - 1.5) * wd, hp.loc[labs, m], width=wd, color=fc, edgecolor=D1, hatch=h, lw=.6, label=LB[m].replace(' (1,280)', ''))
b.scatter(x, [dl[l][0] for l in labs], marker='_', s=140, color=K, lw=1.6, zorder=5, label='DeepLoc 2.0 (ESM-1b)')
b.set_xticks(x); b.set_xticklabels([l.replace('_', ' ').replace('Endoplasmic reticulum', 'ER').replace('Golgi apparatus', 'Golgi').replace('Cell membrane', 'Cell memb.') for l in labs], fontsize=7.2, rotation=30, ha='right')
b.set_ylabel('MCC'); b.set_title('b  Per compartment (C tuned)'); b.legend(fontsize=6.6, ncol=2, loc='upper right'); b.set_ylim(0, .72); b.grid(axis='x', visible=False)
f.tight_layout(); save(f, 'fig_external')

# ---------------- Supplementary: per-fold retention (tuned)
f, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, m in zip(axs, ['random', 'variance']):
    for fo in range(5):
        c = cvm[(cvm.regime == 'tuned') & (cvm.method == m) & (cvm.fold == fo)].sort_values('dims')
        ax.plot(np.r_[c.dims, 1280], np.r_[c.ret, 100], color=G, lw=.9)
    g = cvm[(cvm.regime == 'tuned') & (cvm.method == m)].groupby('dims').ret.mean()
    ax.plot(np.r_[g.index, 1280], np.r_[g.values, 100], color=K, lw=1.8, marker=MK[m], ms=4.5, mfc=FC[m], mec=K, label='Mean of 5 folds')
    for t, ls in [(100, '-'), (95, '--'), (90, ':')]: ax.axhline(t, color=D2, ls=ls, lw=.8)
    logx(ax); ax.set_xlabel('Retained dimensions (log scale)'); ax.set_title(('a  ' if m == 'random' else 'b  ') + LB[m] + 's'); ax.legend(loc='lower right')
axs[0].set_ylabel('MCC retained (% of full, C tuned)'); axs[0].set_ylim(45, 103)
f.tight_layout(); save(f, 'fig_retention_folds')

# ---------------- Supplementary: metric heatmap (tuned)
S = pf[pf.regime == 'tuned'].groupby('method')[met].mean()
M = np.array([[100 * S.loc[m, c] / S.loc['full', c] for c in met] for m in order])
f, ax = plt.subplots(figsize=(5.2, 2.9)); im = ax.imshow(M, cmap='Greys_r', vmin=80, vmax=100, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f'{M[i, j]:.1f}%', ha='center', va='center', fontsize=8, color='white' if M[i, j] < 91 else K)
ax.set_xticks(range(4)); ax.set_xticklabels(['Macro MCC', 'Macro F1', 'Micro F1', 'Jaccard'])
ax.set_yticks(range(len(order))); ax.set_yticklabels([LB[m] for m in order]); ax.grid(False); ax.tick_params(length=0)
[s.set_visible(False) for s in ax.spines.values()]
cb = f.colorbar(im, ax=ax, fraction=.04, pad=.02); cb.set_label('% of full-embedding score', fontsize=7.5); cb.outline.set_visible(False)
save(f, 'fig_metrics')

# ---------------- Supplementary: per-fold differences (tuned)
f, ax = plt.subplots(figsize=(5.6, 2.8)); x = np.arange(5)
for m, off in [('pca', -.15), ('L1_embedded', -.05), ('SS-SA', .05), ('mutual_info', .15)]:
    ax.plot(x + off, (P[m] - P['full']).values, MK[m], ls='-', lw=.9, color=K, mfc=FC[m], mec=K, ms=6, label=LB[m] + ' − full')
ax.axhline(0, color=K, lw=.9); ax.set_xticks(x); ax.set_xticklabels([f'Fold {i}' for i in x])
ax.set_ylabel('Δ macro MCC vs full (C tuned)'); ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); ax.grid(axis='x', visible=False)
save(f, 'fig_perfold')

# ---------------- Supplementary: C profiles
ct = pd.read_csv(V + 'V4_C_tuning_all.csv'); cols = [c for c in ct.columns if c.startswith('cv_C=')]
Cs = [float(c.split('=')[1]) for c in cols]
f, ax = plt.subplots(figsize=(5.2, 2.8))
for m, ls in [('full', '-'), ('pca', '--'), ('SS-SA', '-'), ('L1_embedded', ':'), ('random', '-.')]:
    sub = ct[ct.method == m][cols].mean()
    ax.plot(Cs, sub.values, ls=ls, marker=MK[m], color=K, mfc=FC[m], mec=K, ms=4.5, label=LB[m])
ax.axvline(1.0, color=G, lw=.8); ax.text(1.05, ax.get_ylim()[0] + .005, 'C = 1.0\n(fixed)', fontsize=7, color=D1)
ax.set_xscale('log'); ax.set_xlabel('Inverse regularisation strength C (log scale)'); ax.set_ylabel('Inner-CV macro MCC')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7)
save(f, 'fig_Cprofile')
print(f'figures written to {O}  (consensus dims: {o3}, permutation p = {pperm:.3f})')
