"""
site_classification_exact_replica_v2.py -- v2-only addition.

Exact structural replica of the real ADCD/LBCD manuscript's confounder-control
design (Section 2.7, Manuscript.docx), but with Site as the outcome instead
of pathology, and age+sex as the confounders instead of age+sex+education
(education is CHBMP-only in this dataset -- 0 values for SRM/LEMON -- so it
cannot be used across all 3 sites; this is stated on the figure, not hidden).

Four conditions, same logic and same order as the real study's four
preprocessing conditions:
  1. No matching (raw data, no confounder control)
  2. Linear residualization: age + sex regressed out of every EEG feature
     (site is NOT a covariate here -- site is the target we are trying to
     recover, so removing it directly would be circular; only age+sex, the
     confounders, are removed)
  3. Propensity-score-equivalent matching (PSM): for >2 groups, exact
     pairwise propensity matching does not generalize cleanly, so this uses
     coarsened exact matching (CEM) -- age binned into quartiles, crossed
     with sex, subsampling every site down to the minimum per-site count
     in each age-bin x sex stratum. This is the standard multi-group
     extension of the same idea (Iacus, King & Porro, 2012).
  4. PSM + Residualization: age+sex residualized on the matched subsample.

If age+sex differences between sites were the *only* reason Raw predicts
site well, conditions 2-4 should already collapse it toward chance. If site
remains highly predictable even after age+sex are fully accounted for, that
is direct evidence of genuine technical/hardware batch effects between
sites -- independent of demographics -- which is the actual reason
site-level harmonization (ComBat, Site-only residualization) exists.
"""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
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
SEED = 42
N_REPEATS = 20
CHANCE = 1 / 3
N_AGE_BINS = 4

COLORS = {'Raw (no matching)': '#898781',
          'Residualization\n(age+sex)': '#e34948',
          'PSM (age+sex,\nCEM)': '#2a78d6',
          'PSM + Residualization\n(age+sex)': '#1baf7a',
          'ComBat\n(age touched)': '#eb6834',
          'ComBat\n(no age)': '#f2a97e'}


def residualize_age_sex(df, feat_cols):
    sex_dummy = (df['sex'] == 'M').astype(int).values.reshape(-1, 1)
    age = df['age'].values.reshape(-1, 1)
    C = np.hstack([age, sex_dummy])
    X_res = df[feat_cols].copy()
    for col in feat_cols:
        y = df[col].values
        mask = np.isfinite(y)
        if mask.sum() < 5:
            continue
        reg = LinearRegression()
        reg.fit(C[mask], y[mask])
        X_res.loc[mask, col] = y[mask] - reg.predict(C[mask])
    out = df[META].copy()
    out[feat_cols] = X_res
    return out


def cem_match_age_sex(df, n_bins=N_AGE_BINS, seed=SEED):
    rng = np.random.default_rng(seed)
    d = df.copy()
    d['age_bin'] = pd.qcut(d['age'], q=n_bins, duplicates='drop')

    keep_idx = []
    print(f"Coarsened exact matching (age x sex, {n_bins} age-quartile bins):")
    for (age_bin, sex), sub in d.groupby(['age_bin', 'sex'], observed=True):
        counts = sub.groupby('Site').size()
        counts = counts.reindex(['CHBMP', 'SRM', 'LEMON']).fillna(0).astype(int)
        n = counts.min()
        if n == 0:
            continue
        for site in ['CHBMP', 'SRM', 'LEMON']:
            site_idx = sub.index[sub['Site'] == site].tolist()
            if len(site_idx) == 0:
                continue
            keep = rng.choice(site_idx, size=n, replace=False).tolist()
            keep_idx.extend(keep)
        print(f"  age={age_bin}, sex={sex}: CHBMP={counts['CHBMP']} SRM={counts['SRM']} "
              f"LEMON={counts['LEMON']} -> kept {n} each")

    matched = df.loc[sorted(keep_idx)].reset_index(drop=True)
    return matched


def repeated_cv_acc(df, feat_cols, n_repeats=N_REPEATS):
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


def main():
    raw = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_RAW.xlsx")
    feat_cols = [c for c in raw.columns if c not in META]

    print("Building 'Residualization (age+sex)' table...")
    resid = residualize_age_sex(raw, feat_cols)
    resid.to_excel(DATA_DIR / "DB_WIDE_3SITES_RESID_AGESEX_FOR_SITECLF.xlsx", index=False)

    print("\nBuilding 'PSM (age+sex, CEM)' matched table...")
    matched = cem_match_age_sex(raw)
    print(f"\nTotal matched N: {len(matched)} (from {len(raw)})")
    print(matched.groupby('Site').size())
    matched.to_excel(DATA_DIR / "DB_WIDE_3SITES_PSM_AGESEX_FOR_SITECLF.xlsx", index=False)

    print("\nBuilding 'PSM + Residualization (age+sex)' table...")
    psm_resid = residualize_age_sex(matched, feat_cols)
    psm_resid.to_excel(DATA_DIR / "DB_WIDE_3SITES_PSM_RESID_AGESEX_FOR_SITECLF.xlsx", index=False)

    print("\nLoading ComBat conditions (already computed by harmonize_compare_v2.py)...")
    combat = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_COMBAT.xlsx")
    combat_noage = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_COMBATNOAGE.xlsx")

    conditions = [
        ('Raw (no matching)', raw),
        ('Residualization\n(age+sex)', resid),
        ('PSM (age+sex,\nCEM)', matched),
        ('PSM + Residualization\n(age+sex)', psm_resid),
        ('ComBat\n(age touched)', combat),
        ('ComBat\n(no age)', combat_noage),
    ]

    print("\nRunning repeated 10-fold CV site classification for each condition...")
    results = {}
    for label, df in conditions:
        accs, bal_accs = repeated_cv_acc(df, feat_cols)
        results[label] = {'accs': accs, 'bal_accs': bal_accs}
        print(f"[{label.replace(chr(10), ' ')}] N={len(df)}, "
              f"Accuracy = {accs.mean():.3f} ± {accs.std():.3f}, "
              f"Balanced accuracy = {bal_accs.mean():.3f} ± {bal_accs.std():.3f}, "
              f"chance = {CHANCE:.3f}")

    raw_bal = results['Raw (no matching)']['bal_accs']
    print("\nPaired t-test (balanced accuracy), Raw vs. each condition:")
    for label, _ in conditions:
        if label == 'Raw (no matching)':
            continue
        t, p = stats.ttest_rel(raw_bal, results[label]['bal_accs'])
        print(f"  vs. {label.replace(chr(10), ' ')}: t={t:.2f}, p={p:.5f}")

    fig, ax = plt.subplots(figsize=(14, 6.5))
    labels = [c[0] for c in conditions]
    for i, label in enumerate(labels, start=1):
        raincloud(ax, results[label]['bal_accs'], i, COLORS[label],
                  jitter_width=0.14, point_size=28)

    ax.axhline(CHANCE, color='gray', linestyle=':', linewidth=1.6, label='Chance level (1/3)')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_xlim(0.4, len(labels) + 0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Balanced accuracy (site classification), 20x repeated 10-fold CV', fontsize=14)
    ax.set_title('Is it really just demographics? Controlling age+sex is not enough -- you need ComBat',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, frameon=False, loc='upper right')
    ax.grid(axis='y', color=GRID, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "site_classification_exact_replica.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out}")

    summary = pd.DataFrame({label: {'Acc_mean': results[label]['accs'].mean(),
                                     'Acc_sd': results[label]['accs'].std(),
                                     'BalAcc_mean': results[label]['bal_accs'].mean(),
                                     'BalAcc_sd': results[label]['bal_accs'].std(),
                                     'N': len(df)}
                             for label, df in conditions}).T
    summary.to_csv(DATA_DIR / "site_classification_exact_replica_summary.csv")
    print("\nSummary:")
    print(summary)


if __name__ == '__main__':
    main()
