"""
extract_features_public_v2.py
────────────────────────────────────────────────────────────────────────────────
"v2" / "version 111" of extract_features_public.py -- same exact feature-
extraction method (REGIONS / FOOOF_CFG / EPOCH_LEN / FEAT_ORDER / helper
functions, byte-identical to build/extract_features_public.py), but N=111
per site instead of N=30, with:

  - CHBMP: 111 subjects, age-stratified sample of the 248 available (see
    select_chbmp_111.py / selected_chbmp_111.txt) -- same 2022 sovaharmony
    pipeline as before, unchanged.
  - SRM: all 111 available subjects (that IS the site's real ceiling --
    no selection needed).
  - LEMON: 111 subjects, age-stratified sample of the 213 available,
    reprocessed from scratch on tvb-gpu-01 (2026 sovaharmony pipeline,
    2026-09-03/04) -- see derivatives_v2_111/.

KNOWN LIMITATION (documented per Veronica's decision 2026-09-04): CHBMP and
SRM were preprocessed in 2022; LEMON's 111 were preprocessed in 2026 with a
newer sovaharmony version pulled fresh from GRUNECO's public GitHub repos
(the original 2022 gitfront-hosted install is no longer reachable). This
means "site" and "sovaharmony pipeline version" are perfectly confounded for
LEMON vs. the other two sites in this v2 dataset. Validated directly before
accepting this (sub-032301, reprocessed with both pipeline versions):
signal correlation r=0.975 (51/54 channels r>0.95), and the two features
that matter most for this workshop (Global_IAF, Global_IBF) agree within
0.1% between versions. The drift is real but small relative to genuine
site effects the workshop already detects -- documented here rather than
hidden, not eliminated.

This script writes to data_v2/ (sibling of data/), never touching the N=30
production dataset used for the live Sept 8 workshop.

Usage:
    python extract_features_public_v2.py --pilot   # 5 subjects/site, quick check
    python extract_features_public_v2.py            # full run (111/site)
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import mne
from scipy.signal import welch
import specparam as _specparam_mod
from specparam import SpectralModel

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE_ROOT = Path(r"E:\Academico\Universidad\Posgrado\Tesis\Datos\BASESDEDATOS")
CHBMP_DIR   = SOURCE_ROOT / "CHBMP" / "derivatives" / "sovaharmony"
SRM_DIR     = SOURCE_ROOT / "SRM" / "derivatives" / "sovaharmony"
LEMON_DIR   = SOURCE_ROOT / "LEMON_BIDS" / "derivatives_v2_111"          # NEW pipeline, 111 subjects
CHBMP_DEMO  = Path(r"E:\Academico\Universidad\Posgrado\Tesis\Datos\OTRASBASESDEDATOS\CHBMP\Demographic_data.csv")
SRM_PARTIC  = SOURCE_ROOT / "SRM" / "participants.tsv"
LEMON_DEMO  = Path(r"E:\Academico\Universidad\Posgrado\Tesis\Datos\OTRASBASESDEDATOS\LEMON\Behavioural_Data_MPILMBB_LEMON"
                    r"\META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")

CHBMP_SELECTED_LIST = Path(__file__).resolve().parent / "selected_chbmp_111.txt"

SITES = ('CHBMP', 'SRM', 'LEMON')

OUT_DIR     = Path(__file__).resolve().parent.parent / "data_v2"   # sibling of data/, never touches v1
SPECTRA_DIR = OUT_DIR / "spectra"
N_PER_SITE  = 111  # full-run target; --pilot overrides to 5

# ── Region definitions (IDENTICAL to build/extract_features_public.py) ─────────
REGIONS = {
    'Global': ['Fp1','Fp2','F7','F3','Fz','F4','F8',
               'T7','C3','Cz','C4','T8',
               'P7','P3','Pz','P4','P8','O1','O2'],
    'F':      ['Fp1','Fp2','F7','F3','Fz','F4','F8'],
    'C':      ['C3','Cz','C4'],
    'P':      ['P3','Pz','P4','P7','P8'],
    'O':      ['O1','O2'],
}

EPOCH_LEN = 2.0  # seconds -> nperseg = sfreq * 2 -> df = 0.5 Hz

FOOOF_FREQ_RANGE = [1, 30]
FOOOF_CFG = dict(
    peak_width_limits = [1.0, 12.0],
    max_n_peaks       = 8,
    min_peak_height   = 0.05,
    aperiodic_mode    = 'fixed',
    verbose           = False,
)

_IAF_LO, _IAF_HI = 4.0, 14.0
_IBF_LO, _IBF_HI = 14.0, 22.0

FEAT_ORDER = [
    'Exp', 'Off',
    'IAF', 'IAFpow', 'IAF_BW',
    'IBF', 'IBFpow', 'IBF_BW',
    'Delta_pow', 'Theta_pow', 'Alpha1_pow', 'Alpha2_pow', 'Alpha3_pow', 'Beta_pow',
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS -- byte-identical to build/extract_features_public.py
# ─────────────────────────────────────────────────────────────────────────────

def compute_region_psd(epochs_data, ch_names, region_chs, sfreq, nperseg):
    ch_lower  = [c.lower() for c in ch_names]
    reg_lower = [c.lower() for c in region_chs]
    indices   = [ch_lower.index(c) for c in reg_lower if c in ch_lower]
    if not indices:
        return None, None, 0
    seg = epochs_data[:, indices, :]
    f, pxx = welch(seg, fs=sfreq, nperseg=nperseg, nfft=nperseg, noverlap=0, axis=-1)
    return f, pxx.mean(axis=(0, 1)), len(indices)


def run_specparam(freqs, psd_linear):
    mask  = (freqs >= FOOOF_FREQ_RANGE[0]) & (freqs <= FOOOF_FREQ_RANGE[1])
    f_fit = freqs[mask]
    p_fit = np.clip(psd_linear[mask], 1e-30, None)

    fm = SpectralModel(**FOOOF_CFG)
    offset = exponent = r2 = error = np.nan
    flat_raw    = None
    peak_params = None

    try:
        fm.fit(f_fit, p_fit)
        if not fm.results.has_model:
            raise RuntimeError("no model")
        offset  = float(fm.results.get_params('aperiodic', 'offset'))
        exponent = float(fm.results.get_params('aperiodic', 'exponent'))
        metrics = fm.results.metrics.results
        r2    = float(metrics['gof_rsquared'])
        error = float(metrics['error_mae'])

        ap_fit   = offset - exponent * np.log10(f_fit)
        flat_raw = np.asarray(fm.data.power_spectrum, dtype=float) - ap_fit

        pp = fm.results.get_params('peak')
        if pp is not None and np.size(pp) > 0:
            peak_params = np.atleast_2d(pp)
    except Exception as e:
        print(f"         [FOOOF fit error] {type(e).__name__}: {e}")

    if flat_raw is None:
        log_p    = np.log10(p_fit)
        flat_raw = np.where(np.isfinite(log_p), log_p, 0.0)

    return {
        'offset': offset, 'exponent': exponent, 'r2': r2, 'error': error,
        'freqs_fit': f_fit, 'psd_fit': p_fit, 'flat_raw': flat_raw,
        'peak_params': peak_params,
    }


def find_IAF_IBF(peak_params):
    IAF = IAFpow = IAF_BW = np.nan
    IBF = IBFpow = IBF_BW = np.nan
    if peak_params is None:
        return IAF, IAFpow, IAF_BW, IBF, IBFpow, IBF_BW

    mask_iaf = (peak_params[:, 0] >= _IAF_LO) & (peak_params[:, 0] <= _IAF_HI)
    cands = peak_params[mask_iaf]
    if len(cands):
        best = cands[np.argmax(cands[:, 1])]
        IAF, IAFpow, IAF_BW = float(best[0]), float(best[1]), float(best[2])

    mask_ibf = (peak_params[:, 0] >= _IBF_LO) & (peak_params[:, 0] <= _IBF_HI)
    cands = peak_params[mask_ibf]
    if len(cands):
        best = cands[np.argmax(cands[:, 1])]
        IBF, IBFpow, IBF_BW = float(best[0]), float(best[1]), float(best[2])

    return IAF, IAFpow, IAF_BW, IBF, IBFpow, IBF_BW


def individual_bands_v3b(IAF, IAF_BW, IBF, IBF_BW):
    if np.isnan(IAF) or np.isnan(IAF_BW):
        return None, np.nan, np.nan, np.nan, np.nan

    s_BW     = max(IAF - IAF_BW / 2.0, 1.0)
    e_BW     = IAF + IAF_BW / 2.0
    theta_lo = max(s_BW - 2.0, 1.0)
    mid_al   = (s_BW + IAF) / 2.0

    if not (np.isnan(IBF) or np.isnan(IBF_BW)):
        beta_lo = max(IBF - IBF_BW / 2.0, _IBF_LO)
        beta_hi = min(IBF + IBF_BW / 2.0, _IBF_HI)
        if beta_lo >= beta_hi:
            beta_lo, beta_hi = _IBF_LO, _IBF_HI
    else:
        beta_lo, beta_hi = _IBF_LO, _IBF_HI

    bands = {
        'Delta':  (1.0,      theta_lo),
        'Theta':  (theta_lo, s_BW),
        'Alpha1': (s_BW,     mid_al),
        'Alpha2': (mid_al,   IAF),
        'Alpha3': (IAF,      e_BW),
        'Beta':   (beta_lo,  beta_hi),
    }
    return bands, s_BW, e_BW, beta_lo, beta_hi


def flat_band_power(freqs_fit, flat_raw, flo, fhi):
    if flo >= fhi:
        return np.nan
    mask = (freqs_fit >= flo) & (freqs_fit <= fhi)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(flat_raw[mask]))


# ─────────────────────────────────────────────────────────────────────────────
# Subject selection -- the only real logic difference from v1
# ─────────────────────────────────────────────────────────────────────────────

def _load_chbmp_selected_ids():
    with open(CHBMP_SELECTED_LIST, encoding="utf-8") as f:
        return set(l.strip() for l in f if l.strip())


def list_site_subjects(site):
    if site == 'CHBMP':
        selected = _load_chbmp_selected_ids()
        files = sorted(CHBMP_DIR.glob("sub-*/eeg/*_desc-wica_eeg.fif"))
        files = [f for f in files if f.parents[1].name in selected]
    elif site == 'SRM':
        # prefer ses-t1 only (baseline), avoid double-counting longitudinal subjects
        # -- all 111 available subjects go in, this IS the site's real ceiling.
        files = sorted(SRM_DIR.glob("sub-*/ses-t1/eeg/*_desc-wica_eeg.fif"))
    elif site == 'LEMON':
        files = sorted(LEMON_DIR.glob("sub-*/eeg/*_desc-wica_eeg.fif"))
    else:
        raise ValueError(site)
    return files


def subject_id_from_path(fpath, site):
    if site == 'CHBMP':
        return fpath.parents[1].name
    if site == 'SRM':
        return fpath.parents[2].name
    if site == 'LEMON':
        return fpath.parents[1].name
    raise ValueError(site)


def load_epochs_from_fif(fpath, epoch_len=EPOCH_LEN):
    """Amplitude-scale fix identical to v1 -- see build/extract_features_public.py
    for the full rationale (CHBMP/SRM sovaharmony outputs disagree on units
    despite both FIFF headers claiming Volts; auto-detected per file)."""
    raw = mne.io.read_raw_fif(str(fpath), preload=True, verbose=False)
    raw.pick_types(eeg=True)

    native_std = np.std(raw.get_data())
    if native_std >= 1e-2:
        raw.apply_function(lambda x: x / 1e6, channel_wise=False)

    raw.set_eeg_reference('average', projection=False, verbose=False)
    sfreq = raw.info['sfreq']
    events = mne.make_fixed_length_events(raw, duration=epoch_len)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_len,
                         baseline=None, preload=True, verbose=False)
    return epochs, raw.ch_names, sfreq


# ── Demographics ────────────────────────────────────────────────────────────

def load_chbmp_demo():
    df = pd.read_csv(CHBMP_DEMO, skiprows=1)
    df = df.rename(columns={'Code': 'code', 'Gender': 'sex', 'Age': 'age',
                             'Education Level ': 'education'})
    df = df[['code', 'sex', 'age', 'education']].dropna(subset=['code'])
    df['Subject'] = 'sub-' + df['code'].astype(str)
    return df[['Subject', 'sex', 'age', 'education']]


def load_srm_demo():
    df = pd.read_csv(SRM_PARTIC, sep='\t')
    df = df.rename(columns={'participant_id': 'Subject'})
    return df[['Subject', 'age', 'sex']]


def load_lemon_demo():
    """LEMON reports age as a 5-year privacy bin -- same handling as v1."""
    df = pd.read_csv(LEMON_DEMO)
    df = df.rename(columns={df.columns[0]: 'Subject',
                             'Gender_ 1=female_2=male': 'sex_code',
                             'Age': 'age_bin'})
    df['Subject'] = df['Subject'].astype(str)

    def bin_to_age(b):
        try:
            lo, hi = str(b).split('-')
            return (float(lo) + float(hi)) / 2.0
        except Exception:
            return np.nan

    df['age'] = df['age_bin'].apply(bin_to_age)
    df['sex'] = df['sex_code'].map({1: 'F', 2: 'M'})
    return df[['Subject', 'age', 'sex']]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN -- identical logic to v1, different N_PER_SITE / OUT_DIR / subject lists
# ─────────────────────────────────────────────────────────────────────────────

def main(n_per_site, out_name):
    mne.set_log_level('ERROR')
    print("=" * 60)
    print("  extract_features_public_v2.py")
    print(f"  specparam : {_specparam_mod.__version__}")
    print(f"  mne       : {mne.__version__}")
    print(f"  scipy     : {scipy.__version__}")
    print(f"  numpy     : {np.__version__}")
    print(f"  n_per_site: {n_per_site}")
    print("=" * 60)

    rows, skipped = [], []

    for site in SITES:
        files = list_site_subjects(site)[:n_per_site]
        print(f"\n[{site}] {len(files)} files selected")

        for i, fp in enumerate(files):
            subj = subject_id_from_path(fp, site)
            print(f"  [{i+1:3d}/{len(files)}] {subj}", flush=True)

            try:
                epochs, ch_names, sfreq = load_epochs_from_fif(fp)
                data_uv = epochs.get_data() * 1e6
                n_epochs = len(epochs)
            except Exception as e:
                print(f"           [SKIP] load error: {e}")
                skipped.append((site, subj, str(e)))
                continue

            if n_epochs < 10:
                print(f"           [SKIP] only {n_epochs} epochs")
                skipped.append((site, subj, f"only {n_epochs} epochs"))
                continue

            nperseg = int(sfreq * EPOCH_LEN)
            row = {'Subject': subj, 'Site': site, 'N_epochs': n_epochs}

            for reg_name in REGIONS:
                reg_chs = ch_names if reg_name == 'Global' else REGIONS[reg_name]
                freqs_raw, psd_raw, n_ch = compute_region_psd(
                    data_uv, ch_names, reg_chs, sfreq, nperseg
                )
                if freqs_raw is None or n_ch == 0:
                    for feat in FEAT_ORDER:
                        row[f'{reg_name}_{feat}'] = np.nan
                    continue

                sp = run_specparam(freqs_raw, psd_raw)
                freqs_fit, flat_raw = sp['freqs_fit'], sp['flat_raw']
                row[f'{reg_name}_Exp'] = sp['exponent']
                row[f'{reg_name}_Off'] = sp['offset']

                IAF, IAFpow, IAF_BW, IBF, IBFpow, IBF_BW = find_IAF_IBF(sp['peak_params'])
                row[f'{reg_name}_IAF']    = IAF
                row[f'{reg_name}_IAFpow'] = IAFpow
                row[f'{reg_name}_IAF_BW'] = IAF_BW
                row[f'{reg_name}_IBF']    = IBF
                row[f'{reg_name}_IBFpow'] = IBFpow
                row[f'{reg_name}_IBF_BW'] = IBF_BW

                bands, *_ = individual_bands_v3b(IAF, IAF_BW, IBF, IBF_BW)
                band_names = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Alpha3', 'Beta']
                if bands is None:
                    for b in band_names:
                        row[f'{reg_name}_{b}_pow'] = np.nan
                else:
                    for b_name, (flo, fhi) in bands.items():
                        row[f'{reg_name}_{b_name}_pow'] = flat_band_power(
                            freqs_fit, flat_raw, flo, fhi
                        )

                if reg_name == 'Global' and np.isfinite(sp['exponent']):
                    SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
                    ap_fit = sp['offset'] - sp['exponent'] * np.log10(freqs_fit)
                    raw_log = np.log10(np.clip(sp['psd_fit'], 1e-30, None))
                    np.savez_compressed(
                        SPECTRA_DIR / f"{site}_{subj}_spectrum.npz",
                        freqs=freqs_fit, raw_log=raw_log, ap_log=ap_fit,
                        flat=flat_raw, site=site, subject=subj,
                    )

            rows.append(row)

    df = pd.DataFrame(rows)
    meta_cols   = ['Subject', 'Site', 'N_epochs']
    region_cols = [f'{r}_{f}' for r in REGIONS for f in FEAT_ORDER]
    df = df[meta_cols + [c for c in region_cols if c in df.columns]]

    demo_chbmp = load_chbmp_demo()
    demo_srm   = load_srm_demo()
    demo_lemon = load_lemon_demo()
    demo = pd.concat([demo_chbmp, demo_srm, demo_lemon], ignore_index=True)
    demo['sex'] = demo['sex'].astype(str).str.upper().str[0]
    df = df.merge(demo, on='Subject', how='left')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    df.to_excel(out_path, index=False)

    print(f"\n{'-'*60}")
    print(f"Done. Processed: {len(rows)}  Skipped: {len(skipped)}")
    print(f"Output: {out_path}  shape={df.shape}")
    print(df.groupby('Site')['Subject'].count())
    if skipped:
        print("\nSkipped:")
        for s in skipped:
            print(" ", s)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='5 subjects/site quick check')
    args = parser.parse_args()

    if args.pilot:
        main(n_per_site=5, out_name='PILOT_DB_WIDE_DEMO_3SITES.xlsx')
    else:
        main(n_per_site=N_PER_SITE, out_name='DB_WIDE_DEMO_3SITES.xlsx')
