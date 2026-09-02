"""
normative_age_model.py
────────────────────────────────────────────────────────────────────────────────
A reduced, real implementation of the normative-modeling method used in
Verónica's AAIC26 poster (methodology from Luisa Zapata's UdeA master's
thesis: per-feature Bayesian Linear Regression, closed-form, type-II ML
hyperparameter optimization -- exactly what sklearn.BayesianRidge fits --
trained on healthy controls only, non-linear age basis, z-score deviations).

The real AAIC26 use of this method: fit feature ~ f(age) per feature in
controls, then compute z-score DEVIATIONS for disease groups (ACr/SCr/AD+MCI)
as biomarkers -- theta-band deviations were the most consistent marker.

IMPORTANT COURSE CORRECTION (2026-08-14): an earlier version of this script
inverted the fitted curves to estimate a subject's AGE (multivariate
argmin-of-summed-z² search) and compared that R² against Step 5's Ridge
model. That comparison was DROPPED after actually running it: it made the
real, published normative-modeling method look artificially bad (R² = -0.75
vs Ridge's 0.18), because the method was never designed or validated to
predict age -- the real AAIC26 use takes age as a KNOWN input and computes
deviation z-scores FOR DISEASE GROUPS, it does not invert the curve to
recover age from healthy controls. Forcing it to do a task it wasn't built
for and then reporting that it "loses" would be an unfair test of
Verónica's own real, peer-reviewed method -- exactly the kind of
not-actually-useful comparison to avoid.

What this script does instead: fit the real per-feature normative curves
(faithful to the thesis method) and show ONE illustrative curve -- feature
vs. age, mean + 95% predictive interval, real data points colored by site --
as an honest bridge: "this is the same real age-relationship in the data
that Step 5's Ridge model exploits, viewed through your own group's real
normative-modeling lens, which asks a different (and here, not directly
comparable) scientific question." No numeric "which one wins" claim is made.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from viz_style import GRID, SITE_COLORS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']


def age_basis(age):
    """Non-linear age basis, faithful to the thesis method ('non-linear age
    basis' -- quadratic expansion): [age, age^2]."""
    age = np.asarray(age, dtype=float).reshape(-1, 1)
    return np.hstack([age, age ** 2])


def fit_normative_curves(train_df, feat_cols):
    """One closed-form BayesianRidge per feature (type-II ML hyperparameter
    optimization, exactly matching the thesis method), feature ~ [age, age^2]."""
    X_age = age_basis(train_df['age'].values)
    models = {}
    for col in feat_cols:
        y = train_df[col].values
        mask = np.isfinite(y)
        if mask.sum() < 10:
            continue
        m = BayesianRidge()
        m.fit(X_age[mask], y[mask])
        models[col] = m
    return models


def plot_normative_curve_example(df, feat_cols, feature='Global_Exp',
                                  out_name="normative_curve_example.png"):
    """The classic normative-modeling figure: one feature's fitted curve
    (mean + 95% CI) vs. age, with individual subjects as points colored by
    site -- the same figure family as the AAIC26 poster."""
    models = fit_normative_curves(df, feat_cols)
    m = models[feature]
    ages_fine = np.linspace(df['age'].min(), df['age'].max(), 200)
    mean, std = m.predict(age_basis(ages_fine), return_std=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ages_fine, mean, color='#4a3aa7', linewidth=2.6, label='Normative curve (mean)')
    ax.fill_between(ages_fine, mean - 1.96 * std, mean + 1.96 * std,
                     color='#4a3aa7', alpha=0.15, label='95% predictive interval')
    for site, color in SITE_COLORS.items():
        sub = df[df.Site == site]
        ax.scatter(sub['age'], sub[feature], color=color, s=45, alpha=0.75,
                   edgecolors='white', linewidths=0.4, label=site)

    ax.set_xlabel('Age (years)', fontsize=14)
    ax.set_ylabel(feature, fontsize=14)
    ax.set_title(f'Normative curve: {feature} ~ age\n(Bayesian Ridge, quadratic age basis -- same method as the AAIC26 poster)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, frameon=False, ncol=2)
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
    df = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_SITEONLY.xlsx")
    feat_cols = [c for c in df.columns if c not in META]

    print("Fitting the real normative-modeling method (per-feature BayesianRidge, "
          "quadratic age basis) and plotting one illustrative curve...")
    plot_normative_curve_example(df, feat_cols, feature='Global_Exp')
    print("Done. (No age-prediction comparison performed -- see module docstring "
          "for why that comparison was dropped as unfair to the real method.)")
