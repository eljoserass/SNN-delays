"""
Results figure: accuracy change vs baseline under increasing test-time
jitter, averaged across 3 seeds with +/- 1 std band.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ──────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────
rows = []
with open('analysis/full_sweep_summary/jitter_curves_long.csv') as f:
    for r in csv.DictReader(f):
        rows.append({
            'seed':        int(r['seed']),
            'sigma_train': float(r['sigma_train']),
            'sigma_test':  float(r['sigma_test']),
            'acc_mean':    float(r['acc_mean']),
        })

sigma_tests  = np.array([0.0, 0.05, 0.10, 0.25, 0.50])
sigma_trains = [0.00, 0.05, 0.10, 0.25, 0.50]
seeds        = sorted(set(r['seed'] for r in rows))

# baseline per seed
baselines = {}
for r in rows:
    if r['sigma_train'] == 0.0 and r['sigma_test'] == 0.0:
        baselines[r['seed']] = r['acc_mean']

# delta pp from own-seed baseline
mean_d, std_d = {}, {}
for st in sigma_trains:
    for j in sigma_tests:
        deltas = []
        for s in seeds:
            for r in rows:
                if (r['seed'] == s and
                    abs(r['sigma_train'] - st) < 1e-6 and
                    abs(r['sigma_test'] - j) < 1e-6):
                    deltas.append((r['acc_mean'] - baselines[s]) * 100)
                    break
        mean_d[(st, j)] = np.mean(deltas)
        std_d[(st, j)]  = np.std(deltas, ddof=0)

# ──────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

colors = {0.00: '#9CA3AF', 0.05: '#93C5FD', 0.10: '#1D4ED8',
          0.25: '#7C3AED', 0.50: '#F59E0B'}
labels = {0.00: 'baseline ($\\sigma_{drop}$ = 0)',
          0.05: '$\\sigma_{drop}$ = 0.05',
          0.10: '$\\sigma_{drop}$ = 0.10',
          0.25: '$\\sigma_{drop}$ = 0.25',
          0.50: '$\\sigma_{drop}$ = 0.50'}

for st in sigma_trains:
    mu = np.array([mean_d[(st, j)] for j in sigma_tests])
    sd = np.array([std_d[(st, j)]  for j in sigma_tests])

    emphasis = st in (0.00, 0.25)
    lw     = 2.4 if emphasis else 1.3
    alpha  = 1.0 if emphasis else 0.60
    ls     = '--' if st == 0.00 else '-'
    marker = 'o' if emphasis else None
    ms     = 4.5 if emphasis else 0

    ax.plot(sigma_tests, mu, color=colors[st], lw=lw, ls=ls, alpha=alpha,
            label=labels[st], marker=marker, ms=ms, zorder=3)
    ax.fill_between(sigma_tests, mu - sd, mu + sd,
                    color=colors[st], alpha=0.12 if emphasis else 0.06)

# baseline reference
ax.axhline(0, color='#9CA3AF', lw=1.0, alpha=0.5, zorder=0)
ax.text(0.51, 0.12, 'baseline level', fontsize=7.5, color='#9CA3AF',
        ha='right')

ax.set_xlim(-0.02, 0.53)
ax.set_xticks(sigma_tests)
ax.set_xticklabels(
    ['0\n(no jitter)', '0.05', '0.10', '0.25', '0.50\n(large jitter)'],
    fontsize=9)
ax.set_xlabel('jitter at test time  ($\\sigma_{test}$)', fontsize=10)
ax.set_ylabel('accuracy change vs. baseline  (pp)', fontsize=10)
ax.set_title(
    'Robustness under delay jitter  --  SSC sparse, '
    'mean $\\pm$ 1 std across 3 seeds',
    fontsize=11, fontweight='bold')

ax.legend(fontsize=8.5, ncol=3, loc='upper left', framealpha=0.85)

out = 'analysis/fig_results.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'saved -> {out}')
