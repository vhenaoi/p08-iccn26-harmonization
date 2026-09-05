"""
normative_age_model_v2.py -- v2 counterpart of build/normative_age_model.py.
Same method (per-feature BayesianRidge, quadratic age basis), now illustrated
on the PSM(sex) + Site-only harmonized table -- the same data v2's winning
Ridge model (age_regression_v2.py) actually sees, for full consistency with
the "same real age-relationship" narrative in Step 5b.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from viz_style import GRID, SITE_COLORS

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']


def age_basis(age):
    age = np.asarray(age, dtype=float).reshape(-1, 1)
    return np.hstack([age, age ** 2])


def fit_normative_curves(train_df, feat_cols):
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
    ax.set_title(f'What is "normal" for this age? ({feature})\nPurple line = expected value | Shaded band = 95% of healthy people this age',
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
    df = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx")
    feat_cols = [c for c in df.columns if c not in META]

    print("Fitting the real normative-modeling method (per-feature BayesianRidge, "
          "quadratic age basis) and plotting one illustrative curve...")
    plot_normative_curve_example(df, feat_cols, feature='Global_Exp')
    print("Done.")
