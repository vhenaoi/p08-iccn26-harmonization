"""
age_regression_sage.py
────────────────────────────────────────────────────────────────────────────────
SAGE (Shapley Additive Global importancE) for the age-regression model --
the main-manuscript explainability method in the real Vigilance project
(age_regression_shap.py runs the supplementary-material SHAP version of the
same analysis). SAGE is a permutation-based global-importance method: it
asks how much the model's overall accuracy drops when a feature is removed,
rather than how a feature moves any one prediction -- a different question
from SHAP, so the two methods are expected to disagree on which features
rank highest, and both are shown together in the workshop deck/notebook for
exactly that reason.

Fits on the SITE-ONLY harmonized features -- see age_regression.py: this is
the condition that actually won the 4-way repeated-CV comparison (R²=0.182,
beating raw, ComBat, and full residualization, all p<0.0001), so it is the
correct final model to explain, not a placeholder.

Markedly heavier than SHAP -- permutation estimation over 70 features took
~3 minutes even for this modest 90-subject table, too slow for a live
40-minute browser demo, so this is precomputed offline like the SHAP figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sage
from sklearn.linear_model import RidgeCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META   = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']
ALPHAS = np.logspace(-2, 3, 30)


def main():
    df = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_SITEONLY.xlsx")
    feat_cols = [c for c in df.columns if c not in META]
    X = df[feat_cols].fillna(df[feat_cols].median()).values
    y = df['age'].values

    model = RidgeCV(alphas=ALPHAS)
    model.fit(X, y)
    print(f"Final model alpha: {model.alpha_:.3f}, in-sample R²: {model.score(X, y):.3f}")

    imputer = sage.MarginalImputer(model, X)
    estimator = sage.PermutationEstimator(imputer, loss='mse')
    sage_values = estimator(X, y, batch_size=32, thresh=0.05, verbose=True)

    values = pd.Series(sage_values.values, index=feat_cols)
    stds = pd.Series(sage_values.std, index=feat_cols)
    order = values.sort_values(ascending=False).index[:15]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    y_pos = np.arange(len(order))[::-1]
    ax.barh(y_pos, values[order], xerr=stds[order], color='#2a78d6',
            edgecolor='white', height=0.7, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, fontsize=12.5)
    ax.set_xlabel('SAGE value (reduction in MSE)', fontsize=14)
    ax.axvline(0, color='#52514e', linewidth=0.8)
    ax.set_title('What drives the age prediction? (SAGE, top 15 features)\nSite-only harmonized model',
                 fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "age_regression_sage_barplot.png"
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")

    values.sort_values(ascending=False).head(15).to_csv(DATA_DIR / "age_regression_sage_top15.csv")
    print("\nTop 5 features by SAGE value:")
    print(values.sort_values(ascending=False).head(5))


if __name__ == '__main__':
    main()
