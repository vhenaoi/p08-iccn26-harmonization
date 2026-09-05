"""
plot_site_boxplots_v2.py -- v2/N=111 counterpart of build/plot_site_boxplots.py.
Identical logic (including the grand-mean-restore fix for site-only), only
paths point at data_v2/ and slides/precomputed_v2/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from viz_style import SITE_COLORS, raincloud_group, stars, GRID

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

KEY_FEATURES = ['Global_Exp', 'Global_Off', 'Global_IAF', 'Global_Alpha2_pow',
                 'Global_Theta_pow', 'Global_Beta_pow']
SITE_ORDER   = ['CHBMP', 'SRM', 'LEMON']


def plot_before_after(df_before, df_after, before_label, after_label,
                       out_name, features=KEY_FEATURES):
    n = len(features)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.6))

    for col, feat in enumerate(features):
        groups_before = [df_before.loc[df_before.Site == s, feat].dropna().values for s in SITE_ORDER]
        raincloud_group(axes[0, col], groups_before, SITE_ORDER,
                         [SITE_COLORS[s] for s in SITE_ORDER],
                         title=feat, ylabel=(before_label if col == 0 else None))

        groups_after = [df_after.loc[df_after.Site == s, feat].dropna().values for s in SITE_ORDER]
        raincloud_group(axes[1, col], groups_after, SITE_ORDER,
                         [SITE_COLORS[s] for s in SITE_ORDER],
                         title=None, ylabel=(after_label if col == 0 else None))

    fig.suptitle(f'Before ({before_label}) vs. after ({after_label}) harmonization',
                 fontsize=18, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_single(df, title, out_name, features=KEY_FEATURES):
    n = len(features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features):
        groups = [df.loc[df.Site == s, feat].dropna().values for s in SITE_ORDER]
        raincloud_group(ax, groups, SITE_ORDER, [SITE_COLORS[s] for s in SITE_ORDER], title=feat)

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(title, fontsize=17, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    raw      = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_RAW.xlsx")
    combat   = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_COMBAT.xlsx")
    resid    = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_RESIDUALIZATION.xlsx")
    siteonly = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_SITEONLY.xlsx")

    plot_single(raw, "Site comparison -- Raw (before any correction)", "boxplot_site_raw.png")
    plot_single(combat, "Site comparison -- ComBat", "boxplot_site_combat.png")
    plot_single(resid, "Site comparison -- Residualization (age+sex+site)", "boxplot_site_residualization.png")

    # grand-mean-restore fix (same as build/plot_site_boxplots.py) -- SiteOnly
    # is OLS residualization, mean-zero by construction; add back the raw
    # grand mean before plotting so axes stay physically interpretable.
    siteonly_display = siteonly.copy()
    siteonly_display[KEY_FEATURES] = siteonly_display[KEY_FEATURES] + raw[KEY_FEATURES].mean()

    plot_before_after(raw, siteonly_display, "Raw", "Site-only harmonized",
                       "boxplot_before_after_siteonly.png")
    plot_before_after(raw, combat, "Raw", "ComBat",
                       "boxplot_before_after_combat.png")
