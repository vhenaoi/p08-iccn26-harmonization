"""
age_regression_shap_v2.py -- v2 counterpart of build/age_regression_shap.py.

Fits on the PSM(sex) + Site-only harmonized features -- this is the condition
that won the 7-way repeated-CV comparison in age_regression_v2.py (R²=0.406,
beating plain Site-only [R²=0.380, p<0.0001] and Raw [R²=0.378]), so it is
the correct v2 final model to explain, not plain Site-only (that was v1's
winner).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import RidgeCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data_v2"
FIG_DIR  = Path(__file__).resolve().parent.parent / "slides" / "precomputed_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META   = ['Subject', 'Site', 'N_epochs', 'age', 'sex', 'education']
ALPHAS = np.logspace(-2, 3, 30)


def main():
    df = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_PSM_SITEONLY.xlsx")
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
    plt.title('What drives the age prediction? (SHAP, top 15 features)\nPSM (sex) + Site-only harmonized model',
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
