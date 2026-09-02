# P08 — AI for Harmonizing Multicenter EEG Biomarkers

Analysis code behind the ICCN 2026 pre-congress workshop **"AI for
Harmonizing Multicenter EEG Biomarkers"** (P08, September 8 2026,
Cartagena).

**Live interactive workshop notebook:** https://p08-iccn26-harmonization.vercel.app
(runs entirely in your browser via JupyterLite/Pyodide — no install, no
account)

This repo holds the scripts that extracted features from raw EEG, compared
harmonization methods, and produced every number and figure used in the
workshop and the notebook above. They are shared for transparency and
reproducibility, not as a one-command pipeline — running them end to end
requires the raw datasets below, and the file paths are hardcoded to the
author's local setup.

**Verified 2026-09-01:** `extract_features_public.py` and
`harmonize_compare.py` were tested by cloning this exact repo fresh and
running them on real CHBMP/SRM/LEMON data (`--pilot` mode, 5 subjects/site).
Install with `pip install -r requirements.txt` — the versions are pinned
because `specparam` is pre-1.0 release-candidate software whose API has
changed between RCs.

## Datasets used

Three real, open-access, healthy-control resting-state EEG datasets — no
patient data, nothing private:

| Site  | Dataset                                    | Access |
|-------|---------------------------------------------|--------|
| CHBMP | Cuban Human Brain Mapping Project           | [chbmp-open.loris.ca](https://chbmp-open.loris.ca/) — [Hernández-González et al., 2021, *Scientific Data*](https://doi.org/10.1038/s41597-021-00829-7) |
| SRM   | SRM resting-state EEG                        | [openneuro.org/datasets/ds003775](https://openneuro.org/datasets/ds003775) |
| LEMON | Leipzig Mind-Body-Emotion (MPI-Leipzig)      | [openneuro.org/datasets/ds000221](https://openneuro.org/datasets/ds000221) (BIDS version used here) / [study info](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) — [Babayan et al., 2019, *Scientific Data*](https://doi.org/10.1038/sdata.2018.308) |

30 subjects per site: LEMON's 30 usable subjects (after preprocessing) is
the real ceiling — CHBMP (248 available) and SRM (111 available) were
matched down to it so all three groups stayed balanced.

## Preprocessing

Raw EEG was preprocessed and ICA-cleaned with **sovaharmony**
(GRUNECO, Universidad de Antioquia):
https://github.com/GRUNECO/eeg_harmonization

Despite the name, this stage harmonizes the *preprocessing pipeline* — the
same filtering/referencing/ICA steps applied uniformly across sites — not
the statistical batch effects in the extracted features. Statistical
harmonization (ComBat, residualization) is a separate, later step, run by
`harmonize_compare.py` below, and is exactly what the workshop demonstrates
live.

Method reference: Henao Isaza, V., et al. (2023). Tackling EEG Test-Retest
Reliability with a Pre-Processing Pipeline based on ICA and Wavelet-ICA.
*Authorea Preprints*. https://doi.org/10.22541/au.168570191.12788016/v1

## Pipeline order

1. **`extract_features_public.py`** — loads the ICA-cleaned continuous EEG,
   epochs it, fits specparam/FOOOF per subject and region, extracts
   periodic (IAF/IBF, bandwidth, power) and aperiodic (offset, exponent)
   features. Produces `DB_WIDE_DEMO_3SITES.xlsx`.
2. **`make_demographics_table.py`** — Table 1: age/sex tests per site
   (Kruskal-Wallis, chi-square).
3. **`plot_spectra_by_site.py`** — raw and aperiodic-fit spectra by site,
   before/after harmonization.
4. **`harmonize_compare.py`** — runs ComBat/neuroHarmonize, regression
   residualization, and the combined variant; computes per-feature site-R²
   diagnostics for every version.
5. **`plot_site_boxplots.py`** — raincloud plots, before/after
   harmonization, per feature.
6. **`age_regression.py`** — the AI: predicts age from EEG features,
   comparing raw / ComBat / residualization / site-only harmonization, with
   20× repeated 10-fold cross-validation.
7. **`age_regression_shap.py`** / **`age_regression_sage.py`** —
   explainability on the winning site-only harmonized model (SHAP:
   supplementary-material method; SAGE: main-manuscript method).
8. **`normative_age_model.py`** — the normative-modeling method from the
   AAIC 2026 poster (per-feature Bayesian Ridge, healthy-control-only,
   non-linear age basis), shown as an alternative lens on the same
   age–EEG relationship.
9. **`viz_style.py`** — shared color palette and plot style used across
   every figure.

## Author

Verónica Henao Isaza — Sapienza University of Rome / Universidad de
Antioquia. veronica.henaoisaza@uniroma1.it
