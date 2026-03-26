"""
Conceptual figure: how jitter shifts the DCLS kernel away from the true
learned position P.  Four panels at sigma = 0, 0.25, 0.5, 1.0 showing
the overlap shrinking to nothing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
})

C_SPIKE  = '#E05252'
C_EXACT  = '#3B82F6'
C_NOISY  = '#93C5FD'
C_MISS   = '#FCA5A5'
C_ARROW  = '#9CA3AF'

T_MAX     = 16.0
t_arr     = np.linspace(0, T_MAX, 800)
t_spike   = 2.5
P         = 5.5
t_arrival = t_spike + P           # 8.0
k_sig     = 0.52                  # DCLS kernel width (~ SIG at end of training)
FAR_THRESH = 1.5 * k_sig          # ~0.78

def gauss(x, mu, sig=k_sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)


# ──────────────────────────────────────────────────────────────────────────
# Layout: 1 row × 4 panels
# ──────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4),
                         gridspec_kw={'wspace': 0.22})

sigma_vals  = [0.0, 0.25, 0.50, 1.0]
titles = [
    '$\\sigma_{test}$ = 0\n(no jitter)',
    '$\\sigma_{test}$ = 0.25\n(small jitter)',
    '$\\sigma_{test}$ = 0.5\n(moderate jitter)',
    '$\\sigma_{test}$ = 1.0\n(large jitter)',
]
verdicts = [
    ('perfect overlap',   '#047857'),
    ('mostly overlaps',   '#92400E'),
    ('partial overlap',   '#B45309'),
    ('almost no overlap', '#991B1B'),
]

n_copies = 12

for col, (ax, sigma, title, (vtext, vcol)) in enumerate(
        zip(axes, sigma_vals, titles, verdicts)):

    # — true-P reference: the "ground truth" kernel in very faint fill —
    y_true = gauss(t_arr, t_arrival)
    ax.fill_between(t_arr, y_true, alpha=0.10, color=C_EXACT, zorder=0)
    ax.plot(t_arr, y_true, color=C_EXACT, lw=1.0, ls='--', alpha=0.40,
            zorder=1, label='true position P' if col == 0 else None)

    # — presynaptic spike —
    ax.plot([t_spike, t_spike], [0, 0.85], color=C_SPIKE, lw=2.4,
            solid_capstyle='round', zorder=5)
    ax.plot(t_spike, 0.85, marker='^', color=C_SPIKE, ms=6, zorder=6)
    if col == 0:
        ax.text(t_spike, -0.10, 'spike', ha='center', va='top',
                color=C_SPIKE, fontsize=7.5)

    # — delay arrow (first panel) —
    if col == 0:
        ax.annotate('', xy=(t_arrival - 0.08, 0.40),
                    xytext=(t_spike + 0.18, 0.40),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.0))
        ax.text((t_spike + t_arrival) / 2, 0.46,
                'delay P', ha='center', fontsize=7.5, color='#6B7280')

    # — jittered copies —
    if sigma > 0:
        shifts = np.random.normal(0, sigma, n_copies)
        for sh in shifts:
            c = C_MISS if abs(sh) > FAR_THRESH else C_NOISY
            ax.plot(t_arr, gauss(t_arr, t_arrival + sh),
                    color=c, alpha=0.25, lw=0.9, zorder=1)

    # — one solid representative blob —
    if sigma == 0:
        rep_off = 0.0
    elif sigma <= 0.5:
        rep_off = sigma * 0.7
    else:
        rep_off = sigma * 0.9
    solid_c = '#EF4444' if abs(rep_off) > FAR_THRESH else C_EXACT
    y_rep = gauss(t_arr, t_arrival + rep_off)
    ax.fill_between(t_arr, y_rep, alpha=0.35, color=solid_c, zorder=2)
    ax.plot(t_arr, y_rep, color=solid_c, lw=1.8, zorder=3,
            label='jittered kernel' if col > 0 else 'DCLS kernel')

    # — verdict —
    ax.text(0.5, -0.02, vtext, transform=ax.transAxes, ha='center',
            va='top', fontsize=8, color=vcol, fontstyle='italic')

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(-0.18, 1.12)
    ax.set_yticks([])
    ax.set_xlabel('time (ms)', fontsize=8)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=5,
                 linespacing=1.45)

    if col == 0:
        ax.legend(fontsize=6.8, loc='upper right', framealpha=0.85,
                  handlelength=1.5)
    elif col == 3:
        ax.plot([], [], color=C_NOISY, lw=1, alpha=.7, label='near P')
        ax.plot([], [], color=C_MISS,  lw=1, alpha=.7, label='far from P')
        ax.legend(fontsize=6.8, loc='upper right', framealpha=0.85,
                  handlelength=1.5)

fig.suptitle(
    'Delay Jitter — DCLS kernel overlap with true position P',
    fontsize=13, fontweight='bold', y=1.04)

out = 'analysis/fig_concept.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'saved -> {out}')
