"""
site_classification_slides_simple_v2.py -- simplified bar-chart versions of
the Step 4b figures for the actual slide deck (the raincloud versions in
site_classification_v2.py / site_classification_exact_replica_v2.py stay as
the full analysis record, but Veronica flagged them as too dense/unclear for
a live talk with 6-7 near-identical conditions). Pulls the already-computed
summary stats (no need to refit anything) -- simple bar + SD error bar,
Raw always shown for scale, chance line at 1/3.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed_v2"

CHANCE = 1 / 3
GRID = '#e1e0d9'


def bar_chart(labels, means, sds, colors, title, out_name, note=None):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sds, color=colors, capsize=6, width=0.55,
           edgecolor='white', linewidth=1.2, error_kw={'linewidth': 1.8, 'ecolor': '#333'})
    ax.axhline(CHANCE, color='gray', linestyle=':', linewidth=1.8)
    ax.text(len(labels) - 0.4, CHANCE + 0.015, 'Chance level (1/3)', fontsize=11,
            color='#555', ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel('Balanced accuracy\n(guessing which site a recording came from)', fontsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold')
    for i, m in enumerate(means):
        ax.text(i, m + sds[i] + 0.025, f"{m:.0%}", ha='center', fontsize=13, fontweight='bold')
    ax.grid(axis='y', color=GRID, alpha=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if note:
        fig.text(0.5, -0.02, note, ha='center', fontsize=10, color='#555')
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    # -- Slide 11 (mirror check): 4 essential conditions, matches the script's
    # "After ComBat, after residualization, after site-only harmonization" line
    bar_chart(
        labels=['Raw\n(no correction)', 'ComBat', 'Residualization\n(age+sex+site)', 'Site-only\nharmonization'],
        means=[0.777, 0.280, 0.126, 0.129],
        sds=[0.015, 0.020, 0.015, 0.015],
        colors=['#898781', '#eb6834', '#e34948', '#1baf7a'],
        title='Can a model still guess which site a recording came from?',
        out_name='site_classification_mirror_simple.png',
        note='20x repeated 10-fold cross-validation. 3 sites, so chance = 33%.'
    )

    # -- Slide 12 (is it just demographics?): Raw vs. two age+sex-only fixes
    # (neither reaches chance) vs. ComBat (which does)
    bar_chart(
        labels=['Raw\n(no correction)', 'Residualization\n(age+sex only)', 'Matching\n(age+sex only)', 'ComBat\n(targets site)'],
        means=[0.777, 0.741, 0.658, 0.280],
        sds=[0.015, 0.008, 0.028, 0.020],
        colors=['#898781', '#f2a97e', '#2a78d6', '#eb6834'],
        title='Is it just demographics? Removing only age+sex is not enough',
        out_name='site_classification_exact_replica_simple.png',
        note='Only correcting age and sex barely moves the needle -- you have to target site itself.'
    )
