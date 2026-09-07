"""
matching_contribution_v2.py -- isolates how much of the Step 5 improvement
came from matching itself vs. from harmonization, for slide 19.

Requested by Veronica (2026-09-06): given the honest N=226 comparison in
age_regression_matched_v2.py showed that Raw(matched) actually beats every
harmonization method on the matched sample, she asked for a direct,
same-method comparison (Site-only, with vs. without matching first) plus a
check on whether adjusting for sex by regression on the full N=333 sample
(instead of physically matching down to N=226) gets you the same result --
her own stated hypothesis was that it should be "almost equivalent". It is
not (see numbers below) -- a real, concrete "verify, don't assume" example.

Four bars, all real repeated_cv_r2() results (age_regression_v2.py), no
new modeling:
  Raw (N=333)                          -> DB_WIDE_DEMO_3SITES_RAW.xlsx
  Site-only (N=333, no matching)       -> DB_WIDE_DEMO_3SITES_SITEONLY.xlsx
  Site+Sex regression adjust (N=333)   -> DB_WIDE_DEMO_3SITES_RESIDNOAGE.xlsx
  Site-only (N=226, PSM sex-matched)   -> DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx
"""
from pathlib import Path
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from age_regression_v2 import repeated_cv_r2
from viz_style import GRID

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ('Raw\n(N=333)',                              'DB_WIDE_DEMO_3SITES_RAW.xlsx'),
    ('Site-only\n(N=333, no matching)',           'DB_WIDE_DEMO_3SITES_SITEONLY.xlsx'),
    ('Site+Sex regression\nadjust (N=333,\nno matching)', 'DB_WIDE_DEMO_3SITES_RESIDNOAGE.xlsx'),
    ('Site-only\n(N=226, PSM-matched)',           'DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx'),
]
COLORS = ['#898781', '#8fd9bb', '#ef9291', '#1baf7a']

if __name__ == '__main__':
    results = {}
    for label, fname in CONDITIONS:
        path = DATA_DIR / fname
        n = pd.read_excel(path).shape[0]
        r2s, maes = repeated_cv_r2(path)
        results[label] = {'r2s': r2s, 'n': n}
        print(f"[{label.replace(chr(10), ' ')}] N={n}, R²={r2s.mean():.3f} ± {r2s.std():.3f}")

    matched = results['Site-only\n(N=226, PSM-matched)']['r2s']
    print("\nPaired t-tests vs. Site-only (N=226, PSM-matched):")
    for label, _ in CONDITIONS[:-1]:
        t, p = stats.ttest_rel(matched, results[label]['r2s'])
        print(f"  vs. {label.replace(chr(10), ' ')}: t={t:.2f}, p={p:.6f}")

    fig, ax = plt.subplots(figsize=(11, 6.3))
    labels = [c[0] for c in CONDITIONS]
    means = [results[l]['r2s'].mean() for l in labels]
    sds = [results[l]['r2s'].std() for l in labels]

    x = range(1, len(labels) + 1)
    ax.bar(x, means, yerr=sds, color=COLORS, capsize=6, width=0.6,
           edgecolor='white', linewidth=1.2, error_kw={'linewidth': 1.8, 'ecolor': '#333'})
    for i, m in enumerate(means, start=1):
        ax.text(i, m + sds[i - 1] + 0.012, f"{m:.3f}", ha='center', fontsize=13, fontweight='bold')

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel('R² (age prediction, RidgeCV),\n20x repeated 10-fold CV', fontsize=13)
    ax.set_title('VERIFY, DON\'T ASSUME: does adjusting for sex by regression\nmatch physically matching for it?',
                 fontsize=15.5, fontweight='bold')
    ax.grid(axis='y', color=GRID, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "matching_contribution_chart.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out}")
