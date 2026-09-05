"""
psm_siteonly_v2.py -- v2-only addition, no v1 counterpart.

Adds a 7th harmonization condition to the v2 (N=333) demo: "PSM + Site-only".
Motivated by Veronica's question -- the real ADCD/LBCD classifier study uses
PSM (matching on age+sex) *before* residualizing, to remove the covariate-
target correlation that causes overadjustment bias (see
[[project_iccn26_symposium222_deck]] / Schisterman et al. 2009).

Here the target is age itself, so matching ON age is nonsensical (it would
homogenize the very signal being predicted). The only real between-site
confound left to match on is SEX (Step 1's own finding: sex differs
significantly by site). So this condition:

  1. Subsamples each site to a 50/50 F:M ratio (the site-level minimum of
     F/M counts sets how many of the other sex get kept) -- this is
     mathematically identical to a formal propensity-score match here,
     because the confounder is a single binary variable (sex): balancing
     the raw counts IS the propensity-matched result, no logistic-regression
     propensity model needed for one binary predictor.
  2. On that matched subset, applies the same Site-only residualization
     (site dummies only, age untouched) as the existing SiteOnly condition.

Output: DB_WIDE_DEMO_3SITES_PSM.xlsx (matched, unharmonized -- for reference)
        DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx (matched + site-only harmonized)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
SEED = 42

df = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_RAW.xlsx")
meta_cols = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']
feat_cols = [c for c in df.columns if c not in meta_cols]

rng = np.random.default_rng(SEED)

matched_idx = []
print("Sex-balancing (50/50) per site:")
for site, sub in df.groupby('Site'):
    f_idx = sub.index[sub['sex'] == 'F'].tolist()
    m_idx = sub.index[sub['sex'] == 'M'].tolist()
    n = min(len(f_idx), len(m_idx))
    f_keep = rng.choice(f_idx, size=n, replace=False).tolist()
    m_keep = rng.choice(m_idx, size=n, replace=False).tolist()
    matched_idx.extend(f_keep + m_keep)
    print(f"  {site}: {len(f_idx)}F/{len(m_idx)}M -> kept {n}F/{n}M = {2*n}")

df_psm = df.loc[sorted(matched_idx)].reset_index(drop=True)
print(f"\nTotal matched N: {len(df_psm)} (from {len(df)})")
print(pd.crosstab(df_psm['Site'], df_psm['sex']))

df_psm.to_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM.xlsx", index=False)
print(f"\nSaved: DB_WIDE_DEMO_3SITES_PSM.xlsx")

# -- Site-only residualization on the matched subset (age untouched) --------
site_only_cov = pd.get_dummies(df_psm['Site'].astype(str), prefix='Site', drop_first=True, dtype=int)
C = site_only_cov.values
X_res = df_psm[feat_cols].copy()
for col in feat_cols:
    y = df_psm[col].values
    mask = np.isfinite(y)
    if mask.sum() < 5:
        continue
    reg = LinearRegression()
    reg.fit(C[mask], y[mask])
    X_res.loc[mask, col] = y[mask] - reg.predict(C[mask])

out = df_psm[meta_cols].copy()
out[feat_cols] = X_res
out.to_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx", index=False)
print(f"Saved: DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx")
