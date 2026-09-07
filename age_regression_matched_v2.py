"""
age_regression_matched_v2.py -- single-sample (N=226, sex-matched)
comparison of every harmonization method for age prediction.

Replaces the old 7-condition chart in age_regression_v2.py for slide 17,
which mixed N=333 conditions (Raw, Residualization x2, ComBat-no-age,
Site-only) with N=226 conditions (PSM+ComBat-no-age, PSM+Site-only) in one
bar chart -- confusing, and not a fair comparison. Requested by Veronica:
all five conditions here are computed on the exact same
DB_WIDE_DEMO_3SITES_PSM.xlsx matched sample (see psm_residualization_v2.py
for how the two new Residualization conditions were computed on it).

Same model (RidgeCV), same 20x-repeated 10-fold CV, same repeated_cv_r2()
function as age_regression_v2.py -- only the input files (and therefore
which sample they're computed on) differ.
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
    ('Raw\n(matched)',                   'DB_WIDE_DEMO_3SITES_PSM.xlsx'),
    ('Residualization\n(age+sex+site)',  'DB_WIDE_DEMO_3SITES_PSM_RESIDUALIZATION.xlsx'),
    ('Residualization\n(no age)',        'DB_WIDE_DEMO_3SITES_PSM_RESIDNOAGE.xlsx'),
    ('ComBat\n(no age)',                 'DB_WIDE_DEMO_3SITES_PSM_COMBATNOAGE.xlsx'),
    ('Site-only\nharmonization',         'DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx'),
]
COLORS = {'Raw\n(matched)': '#898781',
          'Residualization\n(age+sex+site)': '#e34948',
          'Residualization\n(no age)': '#ef9291',
          'ComBat\n(no age)': '#f2a97e',
          'Site-only\nharmonization': '#1baf7a'}


def plot_r2_boxplot(results, out_name="age_regression_r2_boxplot_matched.png"):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    labels = [c[0] for c in CONDITIONS]
    means = [results[label]['r2s'].mean() for label in labels]
    sds = [results[label]['r2s'].std() for label in labels]
    colors = [COLORS[label] for label in labels]

    x = range(1, len(labels) + 1)
    ax.bar(x, means, yerr=sds, color=colors, capsize=6, width=0.6,
           edgecolor='white', linewidth=1.2, error_kw={'linewidth': 1.8, 'ecolor': '#333'})
    for i, m in enumerate(means, start=1):
        va = 'bottom' if m >= 0 else 'top'
        offset = sds[i - 1] + 0.012
        ax.text(i, m + offset if m >= 0 else m - offset, f"{m:.3f}",
                ha='center', va=va, fontsize=12, fontweight='bold')

    ax.axhline(0, color='gray', linestyle=':', linewidth=1.3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_ylim(-0.2, 0.5)
    ax.set_ylabel('R² (age prediction, RidgeCV), 20x repeated 10-fold CV', fontsize=13.5)
    ax.set_title('Same N=226 sex-matched sample throughout - which harmonization wins?',
                 fontsize=16.5, fontweight='bold')
    ax.grid(axis='y', color=GRID, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    results = {}
    for label, fname in CONDITIONS:
        path = DATA_DIR / fname
        n = pd.read_excel(path).shape[0]
        r2s, maes = repeated_cv_r2(path)
        results[label] = {'r2s': r2s, 'maes': maes, 'n': n,
                           'r2_mean': r2s.mean(), 'r2_sd': r2s.std(),
                           'mae_mean': maes.mean()}
        print(f"[{label.replace(chr(10), ' ')}] N={n}, "
              f"R² = {r2s.mean():.3f} ± {r2s.std():.3f}, MAE = {maes.mean():.2f} years")

    raw_r2 = results['Raw\n(matched)']['r2s']
    site_r2 = results['Site-only\nharmonization']['r2s']
    print("\nPaired t-test vs. Raw (matched):")
    for label, fname in CONDITIONS:
        if label == 'Raw\n(matched)':
            continue
        t, p = stats.ttest_rel(raw_r2, results[label]['r2s'])
        print(f"  vs. {label.replace(chr(10), ' ')}: t={t:.2f}, p={p:.5f}")

    print("\nPaired t-test, Site-only vs. ComBat (no age):")
    t, p = stats.ttest_rel(site_r2, results['ComBat\n(no age)']['r2s'])
    print(f"  t={t:.2f}, p={p:.5f}")

    plot_r2_boxplot(results)

    summary = pd.DataFrame({k: {'N': v['n'], 'R2_mean': v['r2_mean'], 'R2_sd': v['r2_sd'],
                                  'MAE_mean': v['mae_mean']}
                             for k, v in results.items()}).T
    summary.to_csv(DATA_DIR / "age_regression_matched_summary.csv")
    print("\nSummary:")
    print(summary)
