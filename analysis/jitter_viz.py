"""
Delay dropout & jitter evaluation figure.
  Top row  — conceptual: what jitter does to a delayed spike signal.
  Bottom   — combined results across 3 seeds (delta pp from own baseline).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
})

C_SPIKE = '#E05252'
C_EXACT = '#3B82F6'
C_NOISY = '#93C5FD'
C_MISS  = '#FCA5A5'
C_ARROW = '#9CA3AF'

# ──────────────────────────────────────────────────────────────────────────
# Top row helpers — conceptual panels
# ──────────────────────────────────────────────────────────────────────────
T_MAX     = 14.0
t_arr     = np.linspace(0, T_MAX, 700)
t_spike   = 2.5
P         = 5.5
t_arrival = t_spike + P       # 8.0
k_sig     = 0.52              # blob width
FAR_THRESH = 1.5 * k_sig      # ~0.78

def gauss(x, mu, sig=k_sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


def draw_concept_panel(ax, sigma, col, n_copies=10, shift_sign=1):
    """Draw one conceptual panel (spike + shifting DCLS kernel, no window)."""

    # — presynaptic spike —
    ax.plot([t_spike, t_spike], [0, 0.85], color=C_SPIKE, lw=2.6,
            solid_capstyle='round', zorder=5)
    ax.plot(t_spike, 0.85, marker='^', color=C_SPIKE, ms=7, zorder=6)

    if col == 0:
        ax.text(t_spike, -0.12, 'spike', ha='center', va='top',
                color=C_SPIKE, fontsize=7.5)

    # — delay arrow (first panel only) —
    if col == 0:
        ax.annotate('', xy=(t_arrival - 0.08, 0.40),
                    xytext=(t_spike + 0.18, 0.40),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.1))
        ax.text((t_spike + t_arrival) / 2, 0.46,
                'learned delay  P', ha='center', fontsize=7.5, color='#6B7280')

    # — true-P reference line (when sigma > 0) —
    if sigma > 0:
        ax.axvline(t_arrival, color=C_EXACT, lw=0.9, ls=':', alpha=0.45)
        ax.text(t_arrival + 0.18, 0.92, 'true P', fontsize=6.8,
                color=C_EXACT, alpha=0.65)

    # — jittered copies (faint) —
    if sigma > 0:
        shifts = np.random.normal(0, sigma, n_copies)
        for sh in shifts:
            c = C_MISS if abs(sh) > FAR_THRESH else C_NOISY
            ax.plot(t_arr, gauss(t_arr, t_arrival + sh),
                    color=c, alpha=0.28, lw=1.0, zorder=1)

    # — one solid representative blob —
    rep_off = 0.0 if sigma == 0 else sigma * 0.65 * shift_sign
    solid_c = '#EF4444' if abs(rep_off) > FAR_THRESH else C_EXACT
    ax.fill_between(t_arr, gauss(t_arr, t_arrival + rep_off),
                    alpha=0.38, color=solid_c, zorder=2)
    ax.plot(t_arr, gauss(t_arr, t_arrival + rep_off),
            color=solid_c, lw=2.0, zorder=3)

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(-0.22, 1.15)
    ax.set_yticks([])
    ax.set_xlabel('time  (ms)', fontsize=8)


# ──────────────────────────────────────────────────────────────────────────
# Load results data (plain csv, no pandas)
# ──────────────────────────────────────────────────────────────────────────
rows = []
with open('analysis/full_sweep_summary/jitter_curves_long.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            'seed':        int(r['seed']),
            'sigma_train': float(r['sigma_train']),
            'sigma_test':  float(r['sigma_test']),
            'acc_mean':    float(r['acc_mean']),
        })

sigma_tests  = np.array([0.0, 0.05, 0.10, 0.25, 0.50])
sigma_trains = [0.00, 0.05, 0.10, 0.25, 0.50]
seeds        = sorted(set(r['seed'] for r in rows))

# baseline per seed (sigma_train=0, sigma_test=0)
baselines = {}
for s in seeds:
    for r in rows:
        if r['seed'] == s and r['sigma_train'] == 0.0 and r['sigma_test'] == 0.0:
            baselines[s] = r['acc_mean']

# compute delta pp for every (sigma_train, sigma_test) averaged across seeds
mean_deltas = {}
std_deltas  = {}
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
        key = (st, j)
        mean_deltas[key] = np.mean(deltas)
        std_deltas[key]  = np.std(deltas, ddof=0)

# ──────────────────────────────────────────────────────────────────────────
# Figure layout
# ──────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6.8))
gs = GridSpec(2, 3, height_ratios=[0.85, 1.15], hspace=0.50, wspace=0.28,
              left=0.06, right=0.97, top=0.89, bottom=0.09)

# ── Top row: conceptual panels ───────────────────────────────────────────
sigma_vals = [0.0, 0.25, 0.50]
titles = [
    'No jitter  ($\\sigma_{test}$ = 0)\nDCLS kernel at exact P',
    'Small jitter  ($\\sigma_{test}$ = 0.25)\nkernel shifts slightly each trial',
    'Large jitter  ($\\sigma_{test}$ = 0.5)\nkernel often lands far from P',
]

for col, (sig, title) in enumerate(zip(sigma_vals, titles)):
    ax = fig.add_subplot(gs[0, col])
    draw_concept_panel(ax, sig, col, n_copies=10,
                       shift_sign=(1 if col == 1 else -1))
    ax.set_title(title, fontsize=9.0, fontweight='bold', pad=5,
                 linespacing=1.45)

    if col == 0:
        ax.plot([], [], color=C_EXACT, lw=2,  label='DCLS kernel (one trial)')
        ax.plot([], [], color=C_NOISY, lw=1, alpha=.7, label='other jitter trials')
        ax.plot([], [], color=C_MISS,  lw=1, alpha=.7, label='trial far from P')
        ax.legend(fontsize=6.8, loc='upper right', framealpha=.85,
                  handlelength=1.5)

# ── Bottom: combined results ─────────────────────────────────────────────
ax_r = fig.add_subplot(gs[1, :])

colors = {0.00: '#9CA3AF', 0.05: '#93C5FD', 0.10: '#1D4ED8',
          0.25: '#7C3AED', 0.50: '#F59E0B'}
labels = {0.00: 'baseline  ($\\sigma_{drop}$ = 0)',
          0.05: '$\\sigma_{drop}$ = 0.05',
          0.10: '$\\sigma_{drop}$ = 0.10',
          0.25: '$\\sigma_{drop}$ = 0.25',
          0.50: '$\\sigma_{drop}$ = 0.50'}

for st in sigma_trains:
    x  = sigma_tests
    mu = np.array([mean_deltas[(st, j)] for j in sigma_tests])
    sd = np.array([std_deltas[(st, j)]  for j in sigma_tests])

    emphasis = st in (0.00, 0.25)
    lw     = 2.4 if emphasis else 1.2
    alpha  = 1.0 if emphasis else 0.60
    ls     = '--' if st == 0.00 else '-'
    marker = 'o' if emphasis else None
    ms     = 4 if emphasis else 0

    ax_r.plot(x, mu, color=colors[st], lw=lw, ls=ls, alpha=alpha,
              label=labels[st], marker=marker, ms=ms, zorder=3)
    ax_r.fill_between(x, mu - sd, mu + sd,
                      color=colors[st], alpha=0.10 if emphasis else 0.06)

# zero-line = baseline level
ax_r.axhline(0, color='#9CA3AF', lw=1.0, ls='-', alpha=0.5, zorder=0)
ax_r.text(0.51, 0.12, 'baseline level', fontsize=7.5, color='#9CA3AF',
          ha='right')

ax_r.set_xlim(-0.02, 0.53)
ax_r.set_xticks(sigma_tests)
ax_r.set_xticklabels(
    ['0\n(no jitter)', '0.05', '0.10', '0.25', '0.50\n(large jitter)'],
    fontsize=8.5)
ax_r.set_xlabel('jitter at test time  ($\\sigma_{test}$)', fontsize=9)
ax_r.set_ylabel('accuracy change vs. baseline  (pp)', fontsize=9)
ax_r.set_title(
    'Robustness under delay jitter  --  SSC sparse, mean across 3 seeds  '
    '($\\pm$ 1 std band)',
    fontsize=10, fontweight='bold')

ax_r.spines['left'].set_visible(True)
ax_r.spines['left'].set_alpha(0.3)
ax_r.legend(fontsize=8, ncol=3, loc='upper left', framealpha=0.85)

# ──────────────────────────────────────────────────────────────────────────
fig.suptitle('Delay Jitter Evaluation', fontsize=13.5, fontweight='bold',
             y=0.96)

out = 'analysis/jitter_viz.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'saved -> {out}')
