"""
site_classification_v2.py -- the mirror-image check to age_regression_v2.py.

Same 7 conditions, same repeated-CV design, but the target is Site (3-class)
instead of age. This is the "did we actually remove the bad signal?" check
that complements "did we keep the good signal?" (age_regression_v2.py):

  - Age is a noisy *biological* quantity -- even the best harmonized
    condition tops out at R^2~0.4, MAE~9.9yr, because brain age from resting
    EEG is inherently imprecise. That is not a harmonization failure.
  - Site is a *deterministic label* attached to hardware/protocol
    differences -- there is no biological noise floor. Raw data should
    classify it almost perfectly. Any condition that actually removes site
    effects (by construction: Site-only, Residualization, PSM+Site-only; by
    design: ComBat) should collapse toward chance (1/3 for 3 sites).

Multinomial logistic regression (not RidgeCV -- this is classification), same
feature columns, same repeated 10-fold stratified CV as age_regression_v2.
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from viz_style import raincloud, GRID

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']
N_REPEATS = 20
CHANCE = 1 / 3

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


def repeated_cv_acc(path, n_repeats=N_REPEATS):
    df = pd.read_excel(path)
    feat_cols = [c for c in df.columns if c not in META]
    X = df[feat_cols].fillna(df[feat_cols].median()).values
    y = df['Site'].values

    accs, bal_accs = [], []
    for seed in range(n_repeats):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        clf = make_pipeline(StandardScaler(),
                             LogisticRegressionCV(Cs=10, max_iter=2000,
                                                   scoring='accuracy', random_state=0))
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        accs.append(accuracy_score(y, y_pred))
        bal_accs.append(balanced_accuracy_score(y, y_pred))
    return np.array(accs), np.array(bal_accs)


def plot_acc_boxplot(results, out_name="site_classification_acc_boxplot.png"):
    fig, ax = plt.subplots(figsize=(15, 6.5))
    labels = [c[0] for c in CONDITIONS]
    data = [results[label]['bal_accs'] for label in labels]

    for i, (label, vals) in enumerate(zip(labels, data), start=1):
        raincloud(ax, vals, i, COLORS[label], jitter_width=0.14, point_size=28)

    ax.axhline(CHANCE, color='gray', linestyle=':', linewidth=1.6, label='Chance level (1/3)')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Balanced accuracy (site classification), 20x repeated 10-fold CV', fontsize=14)
    ax.set_title('The mirror check: can you still tell the sites apart?',
                 fontsize=18, fontweight='bold')
    ax.legend(fontsize=11.5, frameon=False, loc='upper right')
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
        if not path.exists():
            print(f"[SKIP] {path.name} not found")
            continue
        accs, bal_accs = repeated_cv_acc(path)
        results[label] = {'accs': accs, 'bal_accs': bal_accs,
                           'acc_mean': accs.mean(), 'acc_sd': accs.std(),
                           'bal_acc_mean': bal_accs.mean()}
        print(f"[{label.replace(chr(10), ' ')}] "
              f"Accuracy = {accs.mean():.3f} ± {accs.std():.3f} "
              f"(balanced = {bal_accs.mean():.3f}), chance = {CHANCE:.3f}")

    raw_bal_acc = results['Raw']['bal_accs']
    print("\nPaired t-test (balanced accuracy), Raw vs. each harmonized condition:")
    for label, fname in CONDITIONS:
        if label == 'Raw':
            continue
        t, p = stats.ttest_rel(raw_bal_acc, results[label]['bal_accs'])
        print(f"  vs. {label.replace(chr(10), ' ')}: t={t:.2f}, p={p:.5f}")

    plot_acc_boxplot(results)

    summary = pd.DataFrame({k: {'Acc_mean': v['acc_mean'], 'Acc_sd': v['acc_sd'],
                                  'BalAcc_mean': v['bal_acc_mean']}
                             for k, v in results.items()}).T
    summary.to_csv(DATA_DIR / "site_classification_summary.csv")
    print("\nSummary:")
    print(summary)
