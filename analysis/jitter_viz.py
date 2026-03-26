"""
Two-row figure:
  Row 1 — Training with delay dropout: how noise is injected into P each batch.
  Row 2 — Jitter evaluation: what happens when we perturb P at test time.
No results / accuracy curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
})

C_SPIKE  = '#E05252'
C_EXACT  = '#3B82F6'   # the exact learned position
C_NOISY  = '#93C5FD'   # perturbed copies
C_MISS   = '#FCA5A5'   # blobs that fall outside the window
C_WIN    = '#10B981'   # integration window
C_ARROW  = '#9CA3AF'

T_MAX      = 14.0
t_arr      = np.linspace(0, T_MAX, 700)
t_spike    = 2.5            # presynaptic spike fires here
P          = 5.5            # learned delay (ms)
t_arrival  = t_spike + P   # = 8.0 — where the signal should arrive
k_sig      = 0.52           # Gaussian blob width
window_hw  = 1.10           # half-width of integration window

def gauss(x, mu, sig=k_sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)

# --------------------------------------------------------------------------
# Layout — 2 rows × 3 cols
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 6.6))
gs = GridSpec(2, 3, height_ratios=[1, 1], hspace=0.62, wspace=0.28,
              left=0.05, right=0.97, top=0.88, bottom=0.09)

# Row labels (drawn as figure text, not axis titles)
fig.text(0.01, 0.735, 'Training\n(noise injection)', va='center', ha='left',
         fontsize=9.5, fontweight='bold', rotation=90, color='#374151')
fig.text(0.01, 0.285, 'Evaluation\n(jitter test)', va='center', ha='left',
         fontsize=9.5, fontweight='bold', rotation=90, color='#374151')

# A thin horizontal separator between the two rows
fig.add_artist(plt.Line2D([0.03, 0.97], [0.50, 0.50], color='#E5E7EB',
                           lw=1.2, transform=fig.transFigure))

# --------------------------------------------------------------------------
# Helper: draw one panel
# --------------------------------------------------------------------------
def draw_panel(ax, sigma, role, col,
               n_copies=8, rep_shift_sign=1):
    """
    role  : 'train' or 'eval'
    sigma : how much the blobs are spread (sigma_drop or sigma_test)
    col   : 0 / 1 / 2 (used only for annotation positioning)
    """

    # -- integration window -------------------------------------------------
    ax.axvspan(t_arrival - window_hw, t_arrival + window_hw,
               alpha=0.13, color=C_WIN, zorder=0)
    ax.axvline(t_arrival - window_hw, color=C_WIN, lw=0.9, ls='--', alpha=0.65)
    ax.axvline(t_arrival + window_hw, color=C_WIN, lw=0.9, ls='--', alpha=0.65)

    # window label (only leftmost panel of each row)
    if col == 0:
        ax.text(t_arrival, 1.28, 'integration\nwindow',
                ha='center', va='bottom', fontsize=7.0, color='#047857')

    # -- presynaptic spike --------------------------------------------------
    ax.plot([t_spike, t_spike], [0, 0.85], color=C_SPIKE, lw=2.6, zorder=5,
            solid_capstyle='round')
    ax.plot(t_spike, 0.85, marker='^', color=C_SPIKE, ms=7, zorder=6)

    if col == 0:
        ax.text(t_spike, -0.15, 'spike fires', ha='center', va='top',
                color=C_SPIKE, fontsize=7.2)
        # delay arrow
        ax.annotate('', xy=(t_arrival - 0.08, 0.40),
                    xytext=(t_spike + 0.18, 0.40),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.1))
        ax.text((t_spike + t_arrival) / 2, 0.46,
                'learned delay  P', ha='center', fontsize=7.2, color='#6B7280')

    # -- noisy / jittered copies --------------------------------------------
    if sigma > 0:
        rng = np.random.normal(0, sigma, n_copies)
        for sh in rng:
            center = t_arrival + sh
            outside = abs(sh) > window_hw
            c = C_MISS if outside else C_NOISY
            ax.plot(t_arr, gauss(t_arr, center), color=c, alpha=0.28, lw=1.0, zorder=1)

    # -- exact learned position (solid) -------------------------------------
    # for sigma > 0, offset slightly so both layers look different
    rep_off = 0.0 if sigma == 0 else sigma * 0.65 * rep_shift_sign
    center_solid = t_arrival + rep_off
    outside_solid = abs(rep_off) > window_hw
    solid_color = '#EF4444' if outside_solid else C_EXACT

    ax.fill_between(t_arr, gauss(t_arr, center_solid),
                    alpha=0.38, color=solid_color, zorder=2)
    ax.plot(t_arr, gauss(t_arr, center_solid),
            color=solid_color, lw=2.0, zorder=3)

    # -- exact P marker (dashed vertical, only when sigma > 0) --------------
    if sigma > 0:
        ax.axvline(t_arrival, color=C_EXACT, lw=0.8, ls=':', alpha=0.5, zorder=0)
        if col == 0:
            ax.text(t_arrival + 0.15, 0.70, 'true P',
                    fontsize=6.8, color=C_EXACT, alpha=0.7)

    # -- noise annotation ---------------------------------------------------
    if role == 'train' and sigma > 0:
        # double-headed arrow showing spread
        spread = sigma * 1.0
        ax.annotate('', xy=(t_arrival + spread, 0.18),
                    xytext=(t_arrival - spread, 0.18),
                    arrowprops=dict(arrowstyle='<->', color='#6B7280', lw=1.0))
        ax.text(t_arrival, 0.12,
                r'$P + \mathcal{N}(0,\,\sigma_\mathrm{drop})$' if sigma > 0 else '',
                ha='center', fontsize=7.2, color='#374151')

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(-0.30, 1.45)
    ax.set_yticks([])
    ax.set_xlabel('time  (ms)', fontsize=8)

# --------------------------------------------------------------------------
# Row 0 — Training: noise injection
# --------------------------------------------------------------------------
sigma_drop_vals = [0.0, 0.1, 0.5]
train_titles = [
    'Baseline  (σ_drop = 0)\nkernel always at exact P',
    'Delay dropout  (σ_drop = 0.1)\neach batch: P → P + small noise',
    'Delay dropout  (σ_drop = 0.5)\neach batch: P → P + large noise',
]

# per-panel legend handles collected for a shared legend
handles_train = []

for col, (sig, title) in enumerate(zip(sigma_drop_vals, train_titles)):
    ax = fig.add_subplot(gs[0, col])
    draw_panel(ax, sig, role='train', col=col, n_copies=9,
               rep_shift_sign=(1 if col == 1 else -1))
    ax.set_title(title, fontsize=9.0, fontweight='bold', pad=5, linespacing=1.5)

    if col == 0:
        ax.plot([], [], color=C_EXACT, lw=2, label='signal arrival (one training step)')
        ax.plot([], [], color=C_NOISY, lw=1, alpha=0.7, label='other training steps (noisy P)')
        ax.legend(fontsize=6.8, loc='upper right', framealpha=0.85, handlelength=1.5)

# --------------------------------------------------------------------------
# Row 1 — Evaluation: jitter test
# --------------------------------------------------------------------------
sigma_test_vals  = [0.0, 0.25, 0.50]
eval_subtitles   = [
    'No jitter  (σ_test = 0)\nsignal arrives exactly at P',
    'Small jitter  (σ_test = 0.25)\narrival slightly off, mostly OK',
    'Large jitter  (σ_test = 0.5)\narrival often misses the window',
]
verdicts = [
    ('signal hits every time',  '#047857'),
    ('mostly on target',        '#92400E'),
    ('often off target',        '#991B1B'),
]

for col, (sig, subtitle, (vtext, vcol)) in enumerate(
        zip(sigma_test_vals, eval_subtitles, verdicts)):
    ax = fig.add_subplot(gs[1, col])
    draw_panel(ax, sig, role='eval', col=col, n_copies=10,
               rep_shift_sign=(-1 if col == 2 else 1))
    ax.set_title(subtitle, fontsize=9.0, fontweight='bold', pad=5, linespacing=1.5)

    # verdict stamp at the bottom
    ax.text(0.5, -0.01, vtext, transform=ax.transAxes,
            ha='center', va='top', fontsize=7.5,
            color=vcol, fontstyle='italic')

    if col == 0:
        ax.plot([], [], color=C_EXACT, lw=2,  label='signal arrival (one trial)')
        ax.plot([], [], color=C_NOISY, lw=1, alpha=0.7, label='other random trials')
        ax.plot([], [], color=C_MISS,  lw=1, alpha=0.7, label='trial that misses window')
        ax.legend(fontsize=6.8, loc='upper right', framealpha=0.85, handlelength=1.5)

# --------------------------------------------------------------------------
# Super-title
# --------------------------------------------------------------------------
fig.suptitle('Delay Dropout & Jitter Evaluation — Concept',
             fontsize=13.5, fontweight='bold', y=0.96)

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
out = 'analysis/jitter_viz.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'saved → {out}')
