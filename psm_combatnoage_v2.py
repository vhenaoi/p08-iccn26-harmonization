"""
psm_combatnoage_v2.py -- v2-only addition, no v1 counterpart.

Adds an 8th harmonization condition to the v2 (N=333) demo: "PSM (sex) +
ComBat (no age)". Companion to psm_siteonly_v2.py, which pairs the same
50/50 sex-matched subset with Site-only residualization instead.

Motivated by Veronica's question (2026-09-05): does sex-matching help
ComBat the same way it helps Site-only harmonization? Answer: no --
ComBat(no age) alone already gets R^2=0.372, and matching first barely
moves it (0.373). This is the opposite of Site-only, where matching first
gives a real, significant improvement (0.380 -> 0.406). Likely reason:
matching shrinks the sample from 333 to 226 *before* ComBat ever sees it,
and ComBat's empirical-Bayes per-site estimation benefits from the larger
N -- Site-only residualization has no such sample-size sensitivity.

Reuses the already-matched, unharmonized subset from psm_siteonly_v2.py
(DB_WIDE_DEMO_3SITES_PSM.xlsx) so the matching itself is not redone here.

Output: DB_WIDE_DEMO_3SITES_PSM_COMBATNOAGE.xlsx
"""
from pathlib import Path
import pandas as pd

from harmonize_compare_v2 import run_combat

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
META_COLS = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']

df_psm = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM.xlsx")
feat_cols = [c for c in df_psm.columns if c not in META_COLS]

print(f"Loaded matched subset: {len(df_psm)} subjects (from psm_siteonly_v2.py's PSM)")
print("Running ComBat, no age (Site + sex only), on the matched subset...")
df_combat_noage, _ = run_combat(df_psm, feat_cols, include_age=False, include_sex=True)

out = df_psm[META_COLS].copy()
out[feat_cols] = df_combat_noage
out.to_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM_COMBATNOAGE.xlsx", index=False)
print("Saved: DB_WIDE_DEMO_3SITES_PSM_COMBATNOAGE.xlsx")
