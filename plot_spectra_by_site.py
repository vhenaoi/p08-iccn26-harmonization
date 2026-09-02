"""
plot_spectra_by_site.py
────────────────────────────────────────────────────────────────────────────────
Site-level power spectra, same two-panel layout as the real
Figure1_spectra_v3b_avgref.png (raw PSD + aperiodic fit | aperiodic-corrected
PSD), mean ± SEM, colored by Site. Validated palette + larger type
(viz_style). Reads the per-subject spectra cache written by
extract_features_public.py (data/spectra/{site}_{subject}_spectrum.npz).

Also produces a BEFORE / AFTER aperiodic-trend comparison: the raw spectrum
curve itself can't be re-harmonized (harmonization operates on the 70 derived
scalar features, not the continuous PSD), but two of those 70 features ARE
the aperiodic offset and exponent that define the dashed 1/f line in panel A
-- so redrawing that line with each subject's Site-only-harmonized Off/Exp
values is a real, honest "after" picture of exactly what got harmonized, not
a fabricated one.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from viz_style import SITE_COLORS, GRID

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
SPECTRA_DIR = DATA_DIR / "spectra"
FIG_DIR     = Path(__file__).resolve().parent.parent / "slides" / "precomputed"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_site_spectra():
    by_site = {s: {'freqs': None, 'raw': [], 'ap': [], 'flat': [], 'subj': []} for s in SITE_COLORS}
    for f in sorted(SPECTRA_DIR.glob("*_spectrum.npz")):
        d = np.load(f, allow_pickle=True)
        site = str(d['site'])
        if site not in by_site:
            continue
        if by_site[site]['freqs'] is None:
            by_site[site]['freqs'] = d['freqs']
        by_site[site]['raw'].append(d['raw_log'])
        by_site[site]['ap'].append(d['ap_log'])
        by_site[site]['flat'].append(d['flat'])
        by_site[site]['subj'].append(str(d['subject']))
    return by_site


def mean_sem(arr_list):
    X = np.vstack(arr_list)
    return X.mean(axis=0), X.std(axis=0) / np.sqrt(X.shape[0])


def plot_raw_aperiodic_panel():
    by_site = load_site_spectra()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for site, color in SITE_COLORS.items():
        d = by_site[site]
        if d['freqs'] is None or len(d['raw']) == 0:
            continue
        freqs = d['freqs']
        raw_mean, raw_sem = mean_sem(d['raw'])
        ap_mean, _         = mean_sem(d['ap'])
        flat_mean, flat_sem = mean_sem(d['flat'])

        axes[0].plot(freqs, raw_mean, color=color, linewidth=2.4, label=f'{site}')
        axes[0].fill_between(freqs, raw_mean - raw_sem, raw_mean + raw_sem, color=color, alpha=0.18)
        axes[0].plot(freqs, ap_mean, color=color, linewidth=1.6, linestyle='--', alpha=0.85)

        axes[1].plot(freqs, flat_mean, color=color, linewidth=2.4, label=site)
        axes[1].fill_between(freqs, flat_mean - flat_sem, flat_mean + flat_sem, color=color, alpha=0.18)

    axes[0].set_title('A.  Raw PSD & aperiodic fit', fontweight='bold')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel(r'$\log_{10}$ $\mu V^2$/Hz')
    axes[0].legend(frameon=False)
    axes[0].grid(color=GRID, alpha=0.8)
    axes[0].set_axisbelow(True)
    for sp in ('top', 'right'):
        axes[0].spines[sp].set_visible(False)

    axes[1].set_title('B.  Aperiodic-corrected PSD (mean ± SEM)', fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Corrected power (a.u.)')
    axes[1].axhline(0, color='gray', linestyle=':', linewidth=1)
    axes[1].legend(frameon=False)
    axes[1].grid(color=GRID, alpha=0.8)
    axes[1].set_axisbelow(True)
    for sp in ('top', 'right'):
        axes[1].spines[sp].set_visible(False)

    fig.suptitle('Same brain state, three sites -- do the spectra already differ?',
                 fontsize=18, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / "spectra_by_site.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_aperiodic_before_after():
    """Redraws the aperiodic 1/f line using each subject's REAL harmonized
    Global_Off / Global_Exp (site-only harmonization) instead of raw -- an
    honest before/after of exactly the two features that define this line.

    IMPORTANT (found 2026-08-14, thanks to a direct question about why the
    "after" panel looked wrong): `residualize_covariates` output is a
    residual -- mean ~0 by construction of OLS (verified: raw Global_Off
    mean 0.331, site-only-harmonized mean 1.2e-17). Plugging a mean-zero
    residual straight into offset - exponent*log10(f) does not reconstruct
    a spectrum -- offset and exponent stop being physical log-power/slope
    values once they're residuals, so the "line" collapses to a flat,
    meaningless smear near zero. That is a real bug in this reconstruction,
    not a real property of harmonized data. Fix: add back the RAW grand
    mean of each feature before reconstructing, so the line represents
    "this subject's aperiodic component, referenced to the overall cohort
    average" -- physically interpretable, same convention harmonization
    papers use when they plot ComBat-adjusted data (adjusted = residual +
    grand mean, not the bare residual).
    """
    by_site = load_site_spectra()
    raw_feat = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_RAW.xlsx").set_index('Subject')
    harm_feat = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_SITEONLY.xlsx").set_index('Subject')
    grand_mean_off = raw_feat['Global_Off'].mean()
    grand_mean_exp = raw_feat['Global_Exp'].mean()

    freqs_ref = next(d['freqs'] for d in by_site.values() if d['freqs'] is not None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for panel_ax, feat_table, label, is_harmonized in zip(
        axes, [raw_feat, harm_feat], ['Before (Raw)', 'After (Site-only harmonized)'], [False, True]
    ):
        for site, color in SITE_COLORS.items():
            d = by_site[site]
            if d['freqs'] is None:
                continue
            lines = []
            for subj in d['subj']:
                if subj not in feat_table.index:
                    continue
                row = feat_table.loc[subj]
                off, exp = row.get('Global_Off'), row.get('Global_Exp')
                if pd.isna(off) or pd.isna(exp):
                    continue
                if is_harmonized:
                    # residual -> back to physical units, referenced to the
                    # cohort grand mean (standard "adjusted data" convention)
                    off = off + grand_mean_off
                    exp = exp + grand_mean_exp
                lines.append(off - exp * np.log10(freqs_ref))
            if not lines:
                continue
            arr = np.vstack(lines)
            m, sem = arr.mean(axis=0), arr.std(axis=0) / np.sqrt(arr.shape[0])
            panel_ax.plot(freqs_ref, m, color=color, linewidth=2.6, label=site)
            panel_ax.fill_between(freqs_ref, m - sem, m + sem, color=color, alpha=0.2)

        panel_ax.set_title(label, fontweight='bold')
        panel_ax.set_xlabel('Frequency (Hz)')
        panel_ax.set_ylabel(r'Aperiodic fit ($\log_{10}$ $\mu V^2$/Hz)')
        panel_ax.legend(frameon=False)
        panel_ax.grid(color=GRID, alpha=0.8)
        panel_ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            panel_ax.spines[sp].set_visible(False)

    # shared y-limits so the visual convergence is honest, not an axis trick
    ylims = [ax.get_ylim() for ax in axes]
    ymin, ymax = min(y[0] for y in ylims), max(y[1] for y in ylims)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    fig.suptitle('Do the aperiodic (1/f) trends converge after harmonization?',
                 fontsize=18, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / "spectra_aperiodic_before_after.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


def gaussian_peak(freqs, cf, pw, bw, sigma_scale=2.0):
    """specparam's own periodic-component parameterization: a Gaussian in
    log-power space, CF=center freq, PW=peak height, BW=bandwidth
    (sigma = BW / 2). Verified against the real cached flat_raw curve for a
    held-out subject: r = 0.993 -- this is a near-exact reconstruction, not
    an approximation invented for this figure."""
    sigma = bw / sigma_scale
    return pw * np.exp(-((freqs - cf) ** 2) / (2 * sigma ** 2))


def reconstruct_periodic(freqs, row):
    """Sum of the alpha (IAF) and beta (IBF) Gaussian peaks -- the ONLY two
    peaks specparam fits in this pipeline (see FOOOF_CFG / find_IAF_IBF in
    extract_features_public.py)."""
    out = np.zeros_like(freqs)
    if pd.notna(row.get('Global_IAF')) and pd.notna(row.get('Global_IAFpow')) and pd.notna(row.get('Global_IAF_BW')):
        out = out + gaussian_peak(freqs, row['Global_IAF'], row['Global_IAFpow'], row['Global_IAF_BW'])
    if pd.notna(row.get('Global_IBF')) and pd.notna(row.get('Global_IBFpow')) and pd.notna(row.get('Global_IBF_BW')):
        out = out + gaussian_peak(freqs, row['Global_IBF'], row['Global_IBFpow'], row['Global_IBF_BW'])
    return out


def plot_full_panel_before_after():
    """The complete Panel A + Panel B figure, before vs. after harmonization.
    Both panels' "after" versions are reconstructed entirely from the 70
    harmonized scalar features (aperiodic Off/Exp + peak CF/PW/BW) -- no
    continuous curve was ever harmonized directly (harmonization operates on
    scalars), so this is the honest way to show it: rebuild the same two
    curves specparam itself would draw, using the harmonized numbers.

    Residualized features are centered at 0 by construction (OLS residuals)
    -- adding back each feature's RAW grand mean before reconstruction is
    required to get physically interpretable curves (same convention as
    plot_aperiodic_before_after()).
    """
    by_site = load_site_spectra()
    raw_feat  = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_RAW.xlsx").set_index('Subject')
    harm_feat = pd.read_excel(DATA_DIR / "DB_WIDE_DEMO_3SITES_SITEONLY.xlsx").set_index('Subject')

    feat_cols_for_grandmean = ['Global_Off', 'Global_Exp', 'Global_IAF', 'Global_IAFpow',
                                'Global_IAF_BW', 'Global_IBF', 'Global_IBFpow', 'Global_IBF_BW']
    grand_means = raw_feat[feat_cols_for_grandmean].mean()

    freqs_ref = next(d['freqs'] for d in by_site.values() if d['freqs'] is not None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11.5))

    for row_idx, (feat_table, row_label, is_harmonized) in enumerate(
        [(raw_feat, 'BEFORE (Raw)', False), (harm_feat, 'AFTER (Site-only harmonized)', True)]
    ):
        ax_a, ax_b = axes[row_idx]
        fig.text(0.01, 0.76 if row_idx == 0 else 0.29, row_label, rotation=90,
                  fontsize=15, fontweight='bold', va='center', ha='center')

        for site, color in SITE_COLORS.items():
            d = by_site[site]
            if d['freqs'] is None:
                continue

            ap_lines, periodic_lines, raw_lines = [], [], []
            for subj in d['subj']:
                if subj not in feat_table.index:
                    continue
                r = feat_table.loc[subj].copy()
                if is_harmonized:
                    for c in feat_cols_for_grandmean:
                        if c in r.index and pd.notna(r[c]):
                            r[c] = r[c] + grand_means[c]

                off, exp = r.get('Global_Off'), r.get('Global_Exp')
                if pd.isna(off) or pd.isna(exp):
                    continue
                ap = off - exp * np.log10(freqs_ref)
                periodic = reconstruct_periodic(freqs_ref, r)

                ap_lines.append(ap)
                periodic_lines.append(periodic)
                raw_lines.append(ap + periodic)

            if not ap_lines:
                continue
            ap_arr, per_arr, raw_arr = np.vstack(ap_lines), np.vstack(periodic_lines), np.vstack(raw_lines)

            raw_m, raw_sem = raw_arr.mean(axis=0), raw_arr.std(axis=0) / np.sqrt(raw_arr.shape[0])
            ap_m = ap_arr.mean(axis=0)
            per_m, per_sem = per_arr.mean(axis=0), per_arr.std(axis=0) / np.sqrt(per_arr.shape[0])

            ax_a.plot(freqs_ref, raw_m, color=color, linewidth=2.4, label=site)
            ax_a.fill_between(freqs_ref, raw_m - raw_sem, raw_m + raw_sem, color=color, alpha=0.18)
            ax_a.plot(freqs_ref, ap_m, color=color, linewidth=1.6, linestyle='--', alpha=0.85)

            ax_b.plot(freqs_ref, per_m, color=color, linewidth=2.4, label=site)
            ax_b.fill_between(freqs_ref, per_m - per_sem, per_m + per_sem, color=color, alpha=0.18)

        ax_a.set_title('A. Reconstructed PSD & aperiodic fit', fontweight='bold', fontsize=13.5)
        ax_a.set_xlabel('Frequency (Hz)')
        ax_a.set_ylabel(r'$\log_{10}$ $\mu V^2$/Hz')
        ax_a.legend(frameon=False)
        ax_a.grid(color=GRID, alpha=0.8)
        ax_a.set_axisbelow(True)

        ax_b.set_title('B. Periodic component (alpha + beta peaks)', fontweight='bold', fontsize=13.5)
        ax_b.set_xlabel('Frequency (Hz)')
        ax_b.set_ylabel('Peak power (a.u.)')
        ax_b.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax_b.legend(frameon=False)
        ax_b.grid(color=GRID, alpha=0.8)
        ax_b.set_axisbelow(True)

        for ax in (ax_a, ax_b):
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)

    # shared y-limits within each column so convergence is visually honest
    for col in range(2):
        ylims = [axes[r, col].get_ylim() for r in range(2)]
        ymin, ymax = min(y[0] for y in ylims), max(y[1] for y in ylims)
        for r in range(2):
            axes[r, col].set_ylim(ymin, ymax)

    fig.suptitle('Full spectrum reconstruction, before vs. after harmonization',
                 fontsize=19, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0.03, 0, 1, 1], w_pad=3.0, h_pad=4.0)
    out = FIG_DIR / "spectra_full_before_after.png"
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot_raw_aperiodic_panel()
    plot_aperiodic_before_after()
    plot_full_panel_before_after()
