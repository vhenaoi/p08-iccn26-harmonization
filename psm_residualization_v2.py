"""
psm_residualization_v2.py -- v2-only addition.

Adds the matched-sample (N=226, sex-matched) counterparts of Full
Residualization (age+sex+site) and Residualization (no age), so that the
Step 5 R^2 comparison (slide 17) can show every harmonization method
computed on the exact same N=226 sex-matched sample, instead of mixing
N=333 and N=226 conditions in one chart. Requested by Veronica so the
chart compares apples to apples: same sample, same conditions, throughout.

Reuses harmonize_compare_v2.py's own run_residualization() so the
covariate handling is byte-identical to the N=333 versions of these same
two conditions -- only the input rows differ (the already sex-matched
DB_WIDE_DEMO_3SITES_PSM.xlsx from psm_siteonly_v2.py, N=226).

Output: DB_WIDE_DEMO_3SITES_PSM_RESIDUALIZATION.xlsx (matched, full resid.)
        DB_WIDE_DEMO_3SITES_PSM_RESIDNOAGE.xlsx (matched, resid. no age)
"""
from pathlib import Path
import pandas as pd
from harmonize_compare_v2 import run_residualization

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
META_COLS = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']

df = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM.xlsx")
feat_cols = [c for c in df.columns if c not in META_COLS]
print(f"Loaded matched sample: {len(df)} subjects")

print("Running Residualization (age+sex+site) on matched sample...")
df_resid = run_residualization(df, feat_cols)
out = df[META_COLS].copy()
out[feat_cols] = df_resid
out.to_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM_RESIDUALIZATION.xlsx", index=False)
print("Saved: DB_WIDE_DEMO_3SITES_PSM_RESIDUALIZATION.xlsx")

print("Running Residualization (no age: site+sex only) on matched sample...")
site_sex_cov = pd.get_dummies(df['Site'].astype(str), prefix='Site', drop_first=True, dtype=int)
sex_num = df['sex'].map({'M': 1.0, 'F': 0.0})
site_sex_cov['sex'] = sex_num.fillna(sex_num.median()).values
df_resid_noage = run_residualization(df, feat_cols, cov_df=site_sex_cov)
out2 = df[META_COLS].copy()
out2[feat_cols] = df_resid_noage
out2.to_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM_RESIDNOAGE.xlsx", index=False)
print("Saved: DB_WIDE_DEMO_3SITES_PSM_RESIDNOAGE.xlsx")
