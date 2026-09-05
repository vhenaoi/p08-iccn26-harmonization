# P08 — AI for Harmonizing Multicenter EEG Biomarkers

Analysis code behind the ICCN 2026 pre-congress workshop **"AI for
Harmonizing Multicenter EEG Biomarkers"** (P08, September 8 2026,
Cartagena).

**Live interactive workshop notebook:** https://p08-iccn26-harmonization.vercel.app
(runs entirely in your browser via JupyterLite/Pyodide — no install, no
account)

**This is v2 of the workshop (N=111/site, 333 total).** The original v1
(N=30/site, 90 total) is preserved as-is at the tag
[`v1-n30-workshop`](https://github.com/vhenaoi/p08-iccn26-harmonization/releases/tag/v1-n30-workshop)
— same repo, earlier state, kept for reference and reproducibility of what
was originally submitted/planned for this workshop.

This repo holds the scripts that extracted features from raw EEG, compared
harmonization methods, and produced every number and figure used in the
workshop and the notebook above. They are shared for transparency and
reproducibility, not as a one-command pipeline — running them end to end
requires the raw datasets below, and the file paths are hardcoded to the
author's local setup.

Install with `pip install -r requirements.txt` — the versions are pinned
because `specparam` is pre-1.0 release-candidate software whose API has
changed between RCs.

## What's new in v2

With N=111/site instead of N=30/site, three findings changed in ways worth
knowing before reusing this code:

- **Site-only harmonization alone no longer clearly beats raw data** for
  age prediction (R² 0.380 vs. 0.378, not significant) — a genuinely more
  honest result than v1's N=30 finding, where it looked like a clear win.
  Fixing this required also matching on sex (`psm_siteonly_v2.py`) before
  harmonizing, which recovers a real, significant improvement (R² = 0.406,
  p<0.0001).
- **A new, more direct verification** (`site_classification_v2.py`,
  `site_classification_exact_replica_v2.py`): instead of only checking
  whether a per-feature statistical test says "not significant" after
  harmonization, these scripts test whether a classifier can still guess
  *which site* a recording came from. Raw data: ~78% accuracy (chance =
  33%). Every harmonized condition: chance level or below — except
  removing only age+sex (not site itself), which stays at 66–74%,
  demonstrating the site effect is a genuine technical/hardware difference,
  not just a demographic proxy.
- **External validation** (`donoghue_comparison_v2.py`): replicates
  Donoghue et al. (2020, *Nature Neuroscience*, Fig. 5e)'s exact age-band
  comparison (20–30yr vs. 60–70yr) on this workshop's own data, confirming
  the aperiodic offset/exponent age effect SHAP/SAGE flag as important
  tracks a real, independently-published finding — and that the effect
  does not weaken after harmonization.

## Datasets used

Three real, open-access, healthy-control resting-state EEG datasets — no
patient data, nothing private:

| Site  | Dataset                                    | Access |
|-------|---------------------------------------------|--------|
| CHBMP | Cuban Human Brain Mapping Project           | [chbmp-open.loris.ca](https://chbmp-open.loris.ca/) — [Hernández-González et al., 2021, *Scientific Data*](https://doi.org/10.1038/s41597-021-00829-7) |
| SRM   | SRM resting-state EEG                        | [openneuro.org/datasets/ds003775](https://openneuro.org/datasets/ds003775) |
| LEMON | Leipzig Mind-Body-Emotion (MPI-Leipzig)      | [openneuro.org/datasets/ds000221](https://openneuro.org/datasets/ds000221) (BIDS version used here) / [study info](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) — [Babayan et al., 2019, *Scientific Data*](https://doi.org/10.1038/sdata.2018.308) |

**111 subjects per site (333 total):** SRM's 111 subjects available is the
real ceiling. CHBMP (248 available) was matched down to 111 via
age-decade-stratified sampling (`selected_chbmp_111.txt`). LEMON's usable
set in the original 2022-era processing run was smaller than that, so it
was reprocessed end-to-end with the current pipeline to reach the real 111
— validated against the earlier processing at the signal/feature level
before trusting it (r=0.975 between old and new features) given the
~4-year pipeline-version gap between LEMON's reprocessing and
CHBMP/SRM's original processing; this is a known limitation of the v2
dataset, not hidden.

## Preprocessing

Raw EEG was preprocessed and ICA-cleaned with **sovaharmony**
(GRUNECO, Universidad de Antioquia):
https://github.com/GRUNECO/eeg_harmonization

Despite the name, this stage harmonizes the *preprocessing pipeline* — the
same filtering/referencing/ICA steps applied uniformly across sites — not
the statistical batch effects in the extracted features. Statistical
harmonization (ComBat, residualization) is a separate, later step, run by
`harmonize_compare_v2.py` below, and is exactly what the workshop
demonstrates live.

Method reference: Henao Isaza, V., et al. (2023). Tackling EEG Test-Retest
Reliability with a Pre-Processing Pipeline based on ICA and Wavelet-ICA.
*Authorea Preprints*. https://doi.org/10.22541/au.168570191.12788016/v1

## Pipeline order

1. **`extract_features_public_v2.py`** — loads the ICA-cleaned continuous
   EEG, epochs it, fits specparam/FOOOF per subject and region, extracts
   periodic (IAF/IBF, bandwidth, power) and aperiodic (offset, exponent)
   features. Produces `DB_WIDE_DEMO_3SITES.xlsx`.
2. **`make_demographics_table_v2.py`** — Table 1: age/sex tests per site
   (Kruskal-Wallis, chi-square).
3. **`plot_spectra_by_site_v2.py`** — raw and aperiodic-fit spectra by
   site, before/after harmonization.
4. **`harmonize_compare_v2.py`** — runs ComBat/neuroHarmonize, regression
   residualization, and the combined variant; computes per-feature site-R²
   diagnostics for every version.
5. **`plot_site_boxplots_v2.py`** — raincloud plots, before/after
   harmonization, per feature.
6. **`psm_siteonly_v2.py`** — matches sex 50/50 within each site (the
   confound found significant in Step 1), then applies site-only
   harmonization on the matched subset — the condition that actually wins
   Step 6 below.
7. **`site_classification_v2.py`** / **`site_classification_exact_replica_v2.py`**
   / **`site_classification_slides_simple_v2.py`** — the direct
   verification: can a classifier still guess the recording site after
   harmonization? Tests all conditions, plus an age+sex-only control to
   rule out a purely demographic explanation for the site effect.
8. **`age_regression_v2.py`** — the AI: predicts age from EEG features,
   comparing raw / ComBat / residualization / site-only / matched+site-only
   harmonization, with 20× repeated 10-fold cross-validation.
9. **`age_regression_shap_v2.py`** / **`age_regression_sage_v2.py`** —
   explainability on the winning matched + site-only harmonized model
   (SHAP: supplementary-material method; SAGE: main-manuscript method).
10. **`donoghue_comparison_v2.py`** — external validation against
    Donoghue et al. (2020)'s published age-band comparison.
11. **`normative_age_model_v2.py`** — the normative-modeling method from
    the AAIC 2026 poster (per-feature Bayesian Ridge, healthy-control-only,
    non-linear age basis), shown as an alternative lens on the same
    age–EEG relationship.
12. **`viz_style.py`** — shared color palette and plot style used across
    every figure.

## Author

Verónica Henao Isaza — Sapienza University of Rome / Universidad de
Antioquia. veronica.henaoisaza@uniroma1.it
