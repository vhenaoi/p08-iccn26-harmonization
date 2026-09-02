"""
make_demographics_table.py
────────────────────────────────────────────────────────────────────────────────
Table 1 (demographics), the standard first table of any clinical EEG paper --
built directly from DB_WIDE_DEMO_3SITES.xlsx (age, sex; education where the
site actually reports it).

Honesty note: education is only reported for CHBMP (categorical school level,
e.g. "High School"). SRM's public participants.tsv does not include it, and
LEMON's encoding is a different, non-comparable school-system scale that this
project does not currently extract. Rather than inventing a cross-site
years-of-education number, education is shown as CHBMP-only, explicitly
labeled -- the same "don't manufacture missing precision" standard used
throughout this project.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_XLSX  = DATA_DIR / "DB_WIDE_DEMO_3SITES.xlsx"


def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def build_table1(df):
    rows = []
    for site in ['CHBMP', 'SRM', 'LEMON']:
        sub = df[df.Site == site]
        n = len(sub)
        age_mean, age_sd = sub['age'].mean(), sub['age'].std()
        n_f = (sub['sex'] == 'F').sum()
        n_m = (sub['sex'] == 'M').sum()
        rows.append({
            'Site': site,
            'N': n,
            'Age (mean ± SD, years)': f"{age_mean:.1f} ± {age_sd:.1f}",
            'Sex (F:M)': f"{n_f}:{n_m}",
        })
    return pd.DataFrame(rows)


def group_tests(df):
    """Real between-site tests -- do NOT skip these: a demographics table
    without them just shows numbers look different, not whether they are."""
    groups = [df.loc[df.Site == s, 'age'].dropna().values for s in ['CHBMP', 'SRM', 'LEMON']]
    H, p_age = stats.kruskal(*groups)

    ct = pd.crosstab(df['Site'], df['sex'])
    chi2, p_sex, dof, _ = stats.chi2_contingency(ct)

    return pd.DataFrame([
        {'Test': 'Age (Kruskal-Wallis)', 'Statistic': f"H = {H:.2f}", 'p': f"{p_age:.4f}", 'Sig.': stars(p_age)},
        {'Test': 'Sex (Chi-square)',     'Statistic': f"Chi2 = {chi2:.2f}", 'p': f"{p_sex:.4f}", 'Sig.': stars(p_sex)},
    ])


def education_breakdown_chbmp(df):
    sub = df[(df.Site == 'CHBMP') & df['education'].notna()]
    if sub.empty:
        return None
    return sub['education'].value_counts().to_dict()


if __name__ == '__main__':
    df = pd.read_excel(IN_XLSX)
    table1 = build_table1(df)
    print(table1.to_string(index=False))
    table1.to_csv(DATA_DIR / "table1_demographics.csv", index=False)
    print(f"\nSaved: {DATA_DIR / 'table1_demographics.csv'}")

    tests = group_tests(df)
    print("\nBetween-site tests:")
    print(tests.to_string(index=False))
    tests.to_csv(DATA_DIR / "table1_group_tests.csv", index=False)
    print(f"Saved: {DATA_DIR / 'table1_group_tests.csv'}")

    edu = education_breakdown_chbmp(df)
    print("\nEducation (CHBMP only -- SRM/LEMON not extracted, see module docstring):")
    print(edu)
    with open(DATA_DIR / "table1_education_chbmp_note.txt", "w") as f:
        f.write("Education level, CHBMP only (counts):\n")
        for level, n in (edu or {}).items():
            f.write(f"  {level}: {n}\n")
        f.write("\nSRM and LEMON education not shown: SRM's public participants.tsv\n"
                 "does not include it; LEMON encodes a different, non-comparable\n"
                 "school-system scale not extracted in this pipeline.\n")
