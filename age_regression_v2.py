"""
age_regression_v2.py -- v2/N=333 counterpart of build/age_regression.py.
Identical logic, only paths point at data_v2/ and slides/precomputed_v2/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from viz_style import raincloud, GRID, AFTER_COLOR

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META   = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']
ALPHAS = np.logspace(-2, 3, 30)
N_REPEATS = 20

CONDITIONS = [
    ('Raw',                        'DB_WIDE_DEMO_3SITES_RAW.xlsx'),
    ('ComBat\n(age touched)',      'DB_WIDE_DEMO_3SITES_COMBAT.xlsx'),
    ('ComBat\n(no age)',           'DB_WIDE_DEMO_3SITES_COMBATNOAGE.xlsx'),
    ('Residualization\n(age+sex+site)', 'DB_WIDE_DEMO_3SITES_RESIDUALIZATION.xlsx'),
    ('Residualization\n(no age)',  'DB_WIDE_DEMO_3SITES_RESIDNOAGE.xlsx'),
    ('Site-only\nharmonization',   'DB_WIDE_DEMO_3SITES_SITEONLY.xlsx'),
    ('PSM (sex) +\nSite-only',     'DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx'),
]
COLORS = {'Raw': '#898781',
          'ComBat\n(age touched)': '#eb6834', 'ComBat\n(no age)': '#f2a97e',
          'Residualization\n(age+sex+site)': '#e34948', 'Residualization\n(no age)': '#ef9291',
          'Site-only\nharmonization': '#1baf7a', 'PSM (sex) +\nSite-only': '#2a78d6'}


def repeated_cv_r2(path, n_repeats=N_REPEATS):
    df = pd.read_excel(path)
    feat_cols = [c for c in df.columns if c not in META]
    X = df[feat_cols].fillna(df[feat_cols].median()).values
    y = df['age'].values
    r2s, maes = [], []
    for seed in range(n_repeats):
        cv = KFold(n_splits=10, shuffle=True, random_state=seed)
        y_pred = cross_val_predict(RidgeCV(alphas=ALPHAS), X, y, cv=cv)
        r2s.append(r2_score(y, y_pred))
        maes.append(mean_absolute_error(y, y_pred))
    return np.array(r2s), np.array(maes)


def single_split_for_plot(path):
    df = pd.read_excel(path)
    feat_cols = [c for c in df.columns if c not in META]
    X = df[feat_cols].fillna(df[feat_cols].median()).values
    y = df['age'].values
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    y_pred = cross_val_predict(RidgeCV(alphas=ALPHAS), X, y, cv=cv)
    return y, y_pred


def plot_r2_boxplot(results, out_name="age_regression_r2_boxplot.png"):
    fig, ax = plt.subplots(figsize=(15, 6.5))
    labels = [c[0] for c in CONDITIONS]
    data = [results[label]['r2s'] for label in labels]

    for i, (label, vals) in enumerate(zip(labels, data), start=1):
        raincloud(ax, vals, i, COLORS[label], jitter_width=0.14, point_size=28)

    ax.axhline(0, color='gray', linestyle=':', linewidth=1.3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_ylabel('R² (age prediction), 20x repeated 10-fold CV', fontsize=14)
    ax.set_title('What does harmonization actually buy you for AI?',
                 fontsize=18, fontweight='bold')
    ax.grid(axis='y', color=GRID, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_true_vs_pred(y, y_pred, r2, mae, label, out_name):
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.scatter(y, y_pred, alpha=0.75, s=65, color=AFTER_COLOR, edgecolors='white', linewidths=0.5)
    lims = [min(y.min(), y_pred.min()) - 3, max(y.max(), y_pred.max()) + 3]
    ax.plot(lims, lims, color='#898781', linestyle=':', linewidth=1.6, label='Perfect prediction')
    m, b = np.polyfit(y, y_pred, 1)
    xs = np.linspace(*lims, 100)
    ax.plot(xs, m * xs + b, color='#e34948', linewidth=2.6, label='Regression fit')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('True age (years)', fontsize=14); ax.set_ylabel('Predicted age (years)', fontsize=14)
    ax.set_title(f'{label}\nR² = {r2:.3f}, MAE = {mae:.1f} years (one representative split)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11.5, frameon=False)
    ax.grid(color=GRID, alpha=0.8)
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
        if not path.exists():
            print(f"[SKIP] {path.name} not found")
            continue
        r2s, maes = repeated_cv_r2(path)
        results[label] = {'r2s': r2s, 'maes': maes,
                           'r2_mean': r2s.mean(), 'r2_sd': r2s.std(),
                           'mae_mean': maes.mean()}
        print(f"[{label.replace(chr(10), ' ')}] "
              f"R² = {r2s.mean():.3f} ± {r2s.std():.3f}, MAE = {maes.mean():.2f} years")

    site_r2 = results['Site-only\nharmonization']['r2s']
    print("\nPaired t-test, Site-only harmonization vs. each condition:")
    for label, fname in CONDITIONS:
        if label == 'Site-only\nharmonization':
            continue
        t, p = stats.ttest_rel(site_r2, results[label]['r2s'])
        print(f"  vs. {label.replace(chr(10), ' ')}: t={t:.2f}, p={p:.5f}")

    plot_r2_boxplot(results)

    for label, fname, out_name in [
            ('Raw', 'DB_WIDE_DEMO_3SITES_RAW.xlsx', 'age_regression_raw.png'),
            ('Site-only harmonized', 'DB_WIDE_DEMO_3SITES_SITEONLY.xlsx', 'age_regression_siteonly.png'),
            ('PSM (sex) + Site-only harmonized', 'DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx', 'age_regression_psm_siteonly.png')]:
        y, y_pred = single_split_for_plot(DATA_DIR / fname)
        r2, mae = r2_score(y, y_pred), mean_absolute_error(y, y_pred)
        plot_true_vs_pred(y, y_pred, r2, mae, label, out_name)

    summary = pd.DataFrame({k: {'R2_mean': v['r2_mean'], 'R2_sd': v['r2_sd'],
                                  'MAE_mean': v['mae_mean']}
                             for k, v in results.items()}).T
    summary.to_csv(DATA_DIR / "age_regression_summary.csv")
    print("\nSummary:")
    print(summary)
