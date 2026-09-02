"""
age_regression_shap.py
────────────────────────────────────────────────────────────────────────────────
Explainable AI for the age-regression model, adapted from the real Vigilance
project's explainability code (generate_combined_shap.py / SHAP_comb.py use
SHAP in supplementary material; SAGE is the main-manuscript method in
Figure2_SAGE_v3_avgref.png but is markedly heavier to compute -- multi-MB logs,
minutes per run in the real pipeline -- so this 40-minute precomputed demo
uses SHAP, the same tool the real project already uses as its supplementary
explainability method, not an invented substitute.

Fits on the SITE-ONLY harmonized features -- see age_regression.py: this is
the condition that actually won the 4-way repeated-CV comparison (R²=0.182,
beating raw, ComBat, and full residualization, all p<0.0001), so it is the
correct final model to explain, not a placeholder.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
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
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df['age'].values

    model = RidgeCV(alphas=ALPHAS)
    model.fit(X.values, y)
    print(f"Final model alpha: {model.alpha_:.3f}, in-sample R²: {model.score(X.values, y):.3f}")

    explainer = shap.LinearExplainer(model, X.values)
    shap_values = explainer(X.values)
    shap_values.feature_names = feat_cols

    fig = plt.figure(figsize=(10, 8.5))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    ax = plt.gca()
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.tick_params(labelsize=12.5)
    plt.title('What drives the age prediction? (SHAP, top 15 features)\nSite-only harmonized model',
               fontsize=16, fontweight='bold')
    plt.tight_layout()
    out = FIG_DIR / "age_regression_shap_beeswarm.png"
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=feat_cols
    ).sort_values(ascending=False)
    mean_abs_shap.head(15).to_csv(DATA_DIR / "age_regression_shap_top15.csv")
    print("\nTop 5 features by mean |SHAP|:")
    print(mean_abs_shap.head(5))


if __name__ == '__main__':
    main()
