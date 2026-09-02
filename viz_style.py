"""
viz_style.py
────────────────────────────────────────────────────────────────────────────────
Shared visual style for every P08 figure: validated categorical palette (from
the dataviz skill's reference instance -- the first three slots of its
8-color order, which pass the all-pairs CVD/normal-vision floor for exactly
the 3-way comparison every figure here needs), larger type, and a reusable
"raincloud" plot (half-violin + jittered strip) -- a real, standard
statistical graphic (Allen et al. 2019), not a novelty: density where a
boxplot would flatten a bimodal or skewed distribution, individual subjects
still visible as points.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats as _stats

# ── Validated categorical palette (dataviz skill reference, slots 1-3) ──────
# blue / orange / aqua -- passes the ALL-PAIRS floor (worst pair CVD dE 9.2
# light, normal-vision 24.0 light), the correct check for 3 side-by-side
# distributions, not just adjacent pairs.
SITE_COLORS = {'CHBMP': '#2a78d6', 'SRM': '#eb6834', 'LEMON': '#1baf7a'}
BEFORE_COLOR = '#2a78d6'   # slot 1, blue -- "before"
AFTER_COLOR  = '#1baf7a'   # slot 3, aqua -- "after" (also reads as improvement)
INK          = '#0b0b0b'
MUTED_INK    = '#52514e'
GRID         = '#e1e0d9'

rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11.5,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'axes.edgecolor': MUTED_INK,
    'axes.labelcolor': INK,
    'text.color': INK,
    'xtick.color': MUTED_INK,
    'ytick.color': MUTED_INK,
})


def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def half_violin(ax, data, center, color, side='left', width=0.38, alpha=0.55):
    """Draws one half of a violin (density curve) at x=center."""
    if len(data) < 3 or np.std(data) == 0:
        return
    kde = _stats.gaussian_kde(data)
    ys = np.linspace(data.min(), data.max(), 200)
    dens = kde(ys)
    dens = dens / dens.max() * width
    xs = center - dens if side == 'left' else center + dens
    ax.fill_betweenx(ys, center, xs, color=color, alpha=alpha, linewidth=0, zorder=2)
    ax.plot(xs, ys, color=color, linewidth=1.2, alpha=min(alpha + 0.25, 1.0), zorder=3)


def raincloud(ax, data, center, color, jitter_width=0.16, point_alpha=0.65, point_size=22):
    """Half-violin (left) + jittered strip of individual points (right) +
    a slim median/IQR indicator -- the 'raincloud' pattern."""
    data = np.asarray(data)
    half_violin(ax, data, center, color, side='left')

    rng = np.random.default_rng(abs(hash(color)) % (2**32))
    jitter = rng.uniform(0.06, jitter_width, size=len(data))
    ax.scatter(center + jitter, data, s=point_size, color=color, alpha=point_alpha,
               edgecolors='white', linewidths=0.4, zorder=4)

    q1, med, q3 = np.percentile(data, [25, 50, 75])
    ax.plot([center - 0.03, center + 0.03], [med, med], color=INK, linewidth=2.4, zorder=5)
    ax.plot([center, center], [q1, q3], color=INK, linewidth=1.6, alpha=0.7, zorder=4)


def raincloud_group(ax, groups, labels, colors, title=None, ylabel=None, kruskal=True):
    """groups: list of arrays; one raincloud per group, evenly spaced."""
    centers = np.arange(1, len(groups) + 1)
    for c, g, color in zip(centers, groups, colors):
        raincloud(ax, g, c, color)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=13)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)

    title_text = title or ''
    if kruskal and all(len(g) >= 3 for g in groups):
        H, p = _stats.kruskal(*groups)
        title_text += f"\nH = {H:.2f}, {stars(p)}"
    if title_text:
        ax.set_title(title_text.strip(), fontsize=14, fontweight='bold')

    ax.grid(axis='y', color=GRID, alpha=0.8, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0.5, len(groups) + 0.7)
