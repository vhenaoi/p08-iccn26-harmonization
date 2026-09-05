"""
donoghue_comparison_v2.py -- external validation for v2.

Replicates Donoghue et al. 2020 (Nat Neurosci), Fig. 5e as closely as our
data allows: aperiodic Offset and Exponent, channel Cz, compared between a
"younger" group (20-30 yr) and an "older" group (60-70 yr) -- Donoghue's own
age bands (their n=16 younger / n=14 older), not a median split. We use the
'C' region (C3/Cz/C4 average) as the closest available proxy to their
single-channel Cz.

Shown twice -- Raw and PSM(sex)+Site-only harmonized -- so the same external
benchmark also serves as a harmonization sanity check: if harmonization were
destroying real signal, this age effect would shrink or disappear in the
harmonized panel. It should not, because sex/site harmonization never
touches age.

Known caveat (printed and annotated on the figure): CHBMP has only 1 subject
in the 60-70yr band, so the "older" group is dominated by LEMON and SRM.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sstats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from viz_style import SITE_COLORS, GRID, INK, MUTED_INK, stars

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

YOUNGER_RANGE = (20, 30)   # Donoghue et al. 2020: n=16, 8F
OLDER_RANGE   = (60, 70)   # Donoghue et al. 2020: n=14, 7F
GROUP_COLORS  = {'Younger (20-30yr)': '#2a78d6', 'Older (60-70yr)': '#e34948'}

FEATURES = [('C_Off', 'Offset (channel C3/Cz/C4 avg.)'),
            ('C_Exp', 'Exponent (channel C3/Cz/C4 avg.)')]


def split_groups(df):
    younger = df[(df.age >= YOUNGER_RANGE[0]) & (df.age <= YOUNGER_RANGE[1])]
    older   = df[(df.age >= OLDER_RANGE[0])   & (df.age <= OLDER_RANGE[1])]
    return younger, older


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_sd


def jitter_scatter(ax, values, sites, center, color, rng):
    jitter = rng.uniform(-0.14, 0.14, size=len(values))
    for site in SITE_COLORS:
        mask = (sites.values == site)
        if mask.sum() == 0:
            continue
        ax.scatter(center + jitter[mask], values.values[mask], s=34,
                   color=SITE_COLORS[site], alpha=0.75, edgecolors='white',
                   linewidths=0.4, zorder=4, label=site)
    mean = values.mean()
    ax.plot([center - 0.18, center + 0.18], [mean, mean], color=color, linewidth=3.0, zorder=5)


def panel(ax, df, feat_col, feat_label, condition_label, rng):
    younger, older = split_groups(df)
    y_vals, o_vals = younger[feat_col].dropna(), older[feat_col].dropna()

    jitter_scatter(ax, y_vals, younger.loc[y_vals.index, 'Site'], 1, GROUP_COLORS['Younger (20-30yr)'], rng)
    jitter_scatter(ax, o_vals, older.loc[o_vals.index, 'Site'], 2, GROUP_COLORS['Older (60-70yr)'], rng)

    t, p = sstats.ttest_ind(y_vals, o_vals, equal_var=False)
    d = cohens_d(y_vals, o_vals)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Younger\n(20-30yr, n={len(y_vals)})', f'Older\n(60-70yr, n={len(o_vals)})'])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylabel(feat_label, fontsize=12.5)
    ax.set_title(f'{condition_label}\nt={t:.2f}, {stars(p)} (p={p:.4f}), Cohen\'s d={d:.2f}',
                 fontsize=12.5, fontweight='bold')
    ax.grid(axis='y', color=GRID, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    return {'condition': condition_label, 'feature': feat_col,
            'n_younger': len(y_vals), 'n_older': len(o_vals),
            'mean_younger': y_vals.mean(), 'mean_older': o_vals.mean(),
            't': t, 'p': p, 'cohens_d': d}


def main():
    rng = np.random.default_rng(42)
    raw = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_RAW.xlsx")
    harm = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx")

    conditions = [('Raw (unharmonized)', raw), ('PSM (sex) + Site-only harmonized', harm)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    rows = []
    for row_i, (cond_label, df) in enumerate(conditions):
        for col_i, (feat_col, feat_label) in enumerate(FEATURES):
            ax = axes[row_i, col_i]
            row = panel(ax, df, feat_col, feat_label, cond_label, rng)
            rows.append(row)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3,
               frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Do we replicate the known aging pattern? (cf. Donoghue et al. 2020, Fig. 5e)',
                  fontsize=16, fontweight='bold', y=1.01)
    fig.text(0.5, -0.06,
              'Age bands match Donoghue et al. 2020 exactly (younger 20-30yr, older 60-70yr; not a median split).\n'
              'Caveat: CHBMP contributes only 1 subject to the older band -- that group is mostly LEMON + SRM.',
              ha='center', fontsize=10, color=MUTED_INK)
    fig.tight_layout()
    out = FIG_DIR / "donoghue_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(DATA_DIR / "donoghue_comparison_stats.csv", index=False)
    print("\nStats:")
    print(stats_df.to_string(index=False))

    print("\nDonoghue 2020 reference values (channel Cz, their n=16/14):")
    print("  Offset:   younger=-11.1 uV^2, older=-11.9 uV^2, t(28)=6.75, P<0.0001, d=2.45")
    print("  Exponent: younger=1.43 uV^2/Hz^2, older=0.75 uV^2/Hz^2, t(28)=7.19, P<0.0001, d=2.63")


if __name__ == '__main__':
    main()
