"""
harmonize_compare.py
────────────────────────────────────────────────────────────────────────────────
P08 ICCN 2026 workshop demo. Runs THREE real harmonization methods on the public
CHBMP+SRM+LEMON feature table (DB_WIDE_DEMO_3SITES.xlsx, produced by
extract_features_public.py) and precomputes everything the workshop needs:

  1. ComBat / neuroHarmonize   -- adapted from optional_neuroharmonize.py
                                   (PORTABLES project, Verónica + John Fredy Ochoa)
  2. Residualization (age+sex+Site) -- adapted from residualize_covariates()
                                   in ML_models_Classification.py (ADCD_LBCD /
                                   Vigilance project, Verónica + Matteo Carpi +
                                   Claudio Babiloni)
  3. Combined: ComBat first, then residualize remaining age/sex on the result

For each, the same diagnostic is computed: per-feature R^2 of "Site" (batch_r2,
adapted from plot_harmonization_diagnostics.py) and a 2D PCA colored by site,
fit on the RAW data and re-applied to every version so panels are comparable.

All outputs are saved to ../data/ and ../slides/precomputed/ so nothing in the
live workshop depends on this script running again in front of an audience.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from neuroHarmonize import harmonizationLearn, harmonizationApply

HERE     = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
FIG_DIR  = HERE.parent / "slides" / "precomputed"
FIG_DIR.mkdir(parents=True, exist_ok=True)

IN_XLSX  = DATA_DIR / "DB_WIDE_DEMO_3SITES.xlsx"
META_COLS = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']


# ─────────────────────────────────────────────────────────────────────────────
# 1) ComBat / neuroHarmonize -- adapted from PORTABLES/optional_neuroharmonize.py
#    (no "reference = controls only" split needed here: the whole demo dataset
#    IS a healthy-control cohort already, so ref == all)
# ─────────────────────────────────────────────────────────────────────────────

def run_combat(df, feat_cols, include_age=True, include_sex=False):
    """Default (include_age=True, include_sex=False) reproduces the
    original 'ComBat' condition exactly: covariates = SITE + age only --
    do not change these defaults, downstream verified numbers depend on it.

    include_age=False drops age from the covariate set (site [+ sex]) --
    isolates whether AGE specifically is what hurts downstream age-prediction,
    as opposed to any covariate adjustment in general."""
    X = df[feat_cols].values
    # log-transform (band powers / aperiodic offsets can be <=0 here since they
    # are aperiodic-corrected "flat" values, not raw PSD -- shift to positive
    # domain first with a per-column min-shift, then the same 0.001 epsilon
    # trick as the original PORTABLES script).
    shift = np.minimum(X.min(axis=0) - 0.001, 0)
    X_shifted = X - shift
    X_log = np.log(0.001 + X_shifted)
    X_log = np.nan_to_num(X_log, nan=0.0, posinf=0.0, neginf=0.0)

    covars_dict = {'SITE': df['Site'].values}
    if include_age:
        covars_dict['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median()).values
    if include_sex:
        sex_num = df['sex'].map({'M': 1.0, 'F': 0.0})
        covars_dict['sex'] = sex_num.fillna(sex_num.median()).values
    covars = pd.DataFrame(covars_dict)

    model, X_adj_log = harmonizationLearn(X_log, covars)
    X_adj = np.exp(X_adj_log) - 0.001 + shift
    return pd.DataFrame(X_adj, columns=feat_cols, index=df.index), model


# ─────────────────────────────────────────────────────────────────────────────
# 2) Residualization -- adapted from ML_models_Classification.py
#    (ADCD_LBCD / Vigilance): LinearRegression(feature ~ age + sex + Site),
#    fit on the whole demo set (no train/test CV split needed for a concept
#    demo -- the real project only splits per-fold to avoid leakage into a
#    classifier, which we are not training here).
# ─────────────────────────────────────────────────────────────────────────────

def build_covariate_matrix(df):
    cov = pd.DataFrame(index=df.index)
    cov['age'] = pd.to_numeric(df['age'], errors='coerce')
    cov['age'] = cov['age'].fillna(cov['age'].median())

    sex_map = {'M': 1.0, 'F': 0.0}
    cov['sex'] = df['sex'].map(sex_map)
    cov['sex'] = cov['sex'].fillna(cov['sex'].median())

    dummies = pd.get_dummies(df['Site'].astype(str), prefix='Site', drop_first=True, dtype=int)
    for c in dummies.columns:
        cov[c] = dummies[c].values
    return cov


def run_residualization(df, feat_cols, cov_df=None):
    if cov_df is None:
        cov_df = build_covariate_matrix(df)
    C = cov_df.values
    X_res = df[feat_cols].copy()
    for col in feat_cols:
        y = df[col].values
        mask = np.isfinite(y)
        if mask.sum() < 5:
            continue
        reg = LinearRegression()
        reg.fit(C[mask], y[mask])
        X_res.loc[mask, col] = y[mask] - reg.predict(C[mask])
    return X_res


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics -- adapted from plot_harmonization_diagnostics.py
# ─────────────────────────────────────────────────────────────────────────────

def batch_r2(X, site_labels):
    S = pd.get_dummies(site_labels, dtype=float).values
    preds = S @ np.linalg.lstsq(S, X, rcond=None)[0]
    ss_res = np.sum((X - preds) ** 2, axis=0)
    ss_tot = np.sum((X - X.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
    return np.clip(r2, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(in_xlsx=IN_XLSX, tag=''):
    df = pd.read_excel(in_xlsx)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Loaded {in_xlsx.name}: {df.shape[0]} subjects, {len(feat_cols)} features")
    print(df['Site'].value_counts().to_string())

    # Impute remaining NaNs per-feature with the column median (some IBF_* cols
    # are legitimately missing when no beta peak was detected -- same situation
    # the real pipelines handle with KNN imputation; median is the workshop-
    # simple stand-in).
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())

    site_labels = df['Site'].values

    # ── Raw (before) ──────────────────────────────────────────────────────────
    X_raw = df[feat_cols].values

    # ── 1) ComBat ────────────────────────────────────────────────────────────
    print("\nRunning ComBat/neuroHarmonize...")
    df_combat, combat_model = run_combat(df, feat_cols)
    X_combat = df_combat.values

    # ── 2) Residualization (age+sex+Site) ───────────────────────────────────
    print("Running Residualization (age+sex+Site)...")
    df_resid = run_residualization(df, feat_cols)
    X_resid = df_resid.values

    # ── 3) Combined: ComBat then residualize remaining age/sex ─────────────
    print("Running Combined (ComBat -> Residualization of age+sex only)...")
    cov_no_site = pd.DataFrame(index=df.index)
    cov_no_site['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())
    cov_no_site['sex'] = df['sex'].map({'M': 1.0, 'F': 0.0}).fillna(0.5)
    df_combined = run_residualization(df_combat.assign(**{c: df[c] for c in ['age', 'sex']}),
                                       feat_cols, cov_df=cov_no_site)
    X_combined = df_combined.values

    # ── 4) Site-only harmonization: residualize SITE ONLY, never touch the ──
    # covariates (age, sex) we might want to use as an ML target downstream.
    # This is the methodologically correct choice whenever the harmonized
    # table will feed a model predicting age (or anything correlated with
    # it) -- see age_regression.py, where this is empirically the best of
    # four conditions for predicting age (R^2 = 0.206 vs 0.074 raw, verified
    # over 20 repeated CV splits, p < 0.0001).
    print("Running Site-only harmonization (Site dummies only, age/sex protected)...")
    site_only_cov = pd.get_dummies(df['Site'].astype(str), prefix='Site', drop_first=True, dtype=int)
    df_siteonly = run_residualization(df, feat_cols, cov_df=site_only_cov)
    X_siteonly = df_siteonly.values

    # ── 5/6) No-age variants: does dropping ONLY age (keeping site + sex as
    # covariates) recover most of what Site-only gets, or is sex also part
    # of the problem? Asked directly after seeing the age-touched conditions
    # hurt age-prediction -- isolates the effect of age specifically.
    print("Running ComBat, no age (Site + sex only)...")
    df_combat_noage, _ = run_combat(df, feat_cols, include_age=False, include_sex=True)
    X_combat_noage = df_combat_noage.values

    print("Running Residualization, no age (Site + sex only)...")
    site_sex_cov = site_only_cov.copy()
    sex_num = df['sex'].map({'M': 1.0, 'F': 0.0})
    site_sex_cov['sex'] = sex_num.fillna(sex_num.median()).values
    df_resid_noage = run_residualization(df, feat_cols, cov_df=site_sex_cov)
    X_resid_noage = df_resid_noage.values

    # ── Diagnostics: R^2 of Site per feature ────────────────────────────────
    versions = {'Raw': X_raw, 'ComBat': X_combat,
                'Residualization': X_resid, 'Combined': X_combined,
                'SiteOnly': X_siteonly,
                'ComBatNoAge': X_combat_noage, 'ResidNoAge': X_resid_noage}

    r2_summary = {}
    for name, X in versions.items():
        r2 = batch_r2(X, site_labels)
        r2_summary[name] = {'mean_R2': float(np.mean(r2)), 'median_R2': float(np.median(r2))}
        print(f"  {name:16s}  mean site-R2={np.mean(r2):.4f}  median={np.median(r2):.4f}")

    r2_df = pd.DataFrame(r2_summary).T
    r2_df.to_csv(DATA_DIR / f"site_R2_summary{tag}.csv")

    # per-feature R^2 table too (for the notebook to show individual features)
    per_feat = {name: batch_r2(X, site_labels) for name, X in versions.items()}
    per_feat_df = pd.DataFrame(per_feat, index=feat_cols)
    per_feat_df.to_csv(DATA_DIR / f"site_R2_per_feature{tag}.csv")

    # NOTE: no PCA scatter here -- Babiloni's approved figure types are
    # spectra / boxplot+strip / mean+-SE / regression-line / SHAP only.
    # Site comparison is shown via plot_site_boxplots.py instead.

    # ── Save harmonized tables (precomputed, ready for the notebook) ───────
    for name, X in versions.items():
        out = df[META_COLS].copy()
        out[feat_cols] = X
        out.to_excel(DATA_DIR / f"DB_WIDE_DEMO_3SITES_{name.upper()}{tag}.xlsx", index=False)

    print(f"\nDone. R^2 summary saved to site_R2_summary{tag}.csv")
    print(r2_df)
    return r2_df, per_feat_df


if __name__ == '__main__':
    import sys
    if '--pilot' in sys.argv:
        main(in_xlsx=DATA_DIR / "PILOT_DB_WIDE_DEMO_3SITES.xlsx", tag='_PILOT')
    else:
        main()
