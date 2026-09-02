"""
extract_features_public.py
────────────────────────────────────────────────────────────────────────────────
P08 ICCN 2026 workshop demo data. Re-implements Veronica's real Sapienza feature
extraction (same REGIONS / FOOOF_CFG / EPOCH_LEN / FEAT_ORDER / helper functions
as build_individual_bands_db_v3b_avgref.py, ADCD_LBCD project) but points at
PUBLIC, open EEG instead of the private PDWAVES clinical consortium data:

  - CHBMP (Cuban Human Brain Mapping Project)
  - SRM
  - LEMON (Leipzig Mind-Body-Emotion, Max Planck Institute)

All three already preprocessed/ICA-cleaned via sovaharmony (GRUNECO/UdeA
harmonization pipeline) -- we read their "desc-wica_eeg.fif" continuous output,
epoch it ourselves (2 s, matching EPOCH_LEN), then run the identical
FOOOF/specparam feature logic. No raw PDWAVES data is used anywhere here.

Usage:
    python extract_features_public.py --pilot        # 5 subjects/site, quick check
    python extract_features_public.py                # full run (N_PER_SITE each)
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
LEMON_DIR   = SOURCE_ROOT / "LEMON_BIDS" / "derivatives" / "sovaharmony"
CHBMP_DEMO  = Path(r"E:\Academico\Universidad\Posgrado\Tesis\Datos\OTRASBASESDEDATOS\CHBMP\Demographic_data.csv")
SRM_PARTIC  = SOURCE_ROOT / "SRM" / "participants.tsv"
LEMON_DEMO  = Path(r"E:\Academico\Universidad\Posgrado\Tesis\Datos\OTRASBASESDEDATOS\LEMON\Behavioural_Data_MPILMBB_LEMON"
                    r"\META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")

SITES = ('CHBMP', 'SRM', 'LEMON')

OUT_DIR     = Path(__file__).resolve().parent.parent / "data"
SPECTRA_DIR = OUT_DIR / "spectra"  # per-subject Global spectra cache, for the raw+aperiodic figure
N_PER_SITE  = 30  # full-run target; --pilot overrides to 5

# ── Region definitions (IDENTICAL to build_individual_bands_db_v3b_avgref.py) ──
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
# HELPER FUNCTIONS -- copied verbatim from build_individual_bands_db_v3b_avgref.py
# (ADCD_LBCD, Vigilance project) so the feature *method* stays exactly Sapienza's.
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
    """
    NOTE on specparam API: the original Sapienza script (build_individual_bands_db_
    v3b_avgref.py) was written against a specparam release exposing a nested
    `fm.results.model...` object. The environment here has specparam 2.0.0rc3,
    whose SpectralModel exposes flat attributes instead (`aperiodic_params_`,
    `peak_params_`, `r_squared_`, `error_`, `power_spectrum` already in log10).
    The maths are identical -- 'fixed' aperiodic mode is offset - exponent*log10(f)
    -- so flat_raw (aperiodic-removed spectrum) is reconstructed manually instead
    of relying on a private nested attribute. Verified against a synthetic
    spectrum before use (manual reconstruction matches within the peak-fit
    residual, as expected).
    """
    mask  = (freqs >= FOOOF_FREQ_RANGE[0]) & (freqs <= FOOOF_FREQ_RANGE[1])
    f_fit = freqs[mask]
    p_fit = np.clip(psd_linear[mask], 1e-30, None)

    fm = SpectralModel(**FOOOF_CFG)
    offset = exponent = r2 = error = np.nan
    flat_raw    = None
    peak_params = None

    try:
        fm.fit(f_fit, p_fit)
        if not fm.has_model:
            raise RuntimeError("no model")
        offset, exponent = float(fm.aperiodic_params_[0]), float(fm.aperiodic_params_[1])
        r2    = float(fm.r_squared_)
        error = float(fm.error_)

        ap_fit   = offset - exponent * np.log10(f_fit)
        flat_raw = np.asarray(fm.power_spectrum, dtype=float) - ap_fit

        pp = fm.peak_params_
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
# NEW (not in the original script): find + epoch the public sovaharmony files.
# The original script reads already-epoched EEGLAB .set files (epoching was done
# upstream in MATLAB for PDWAVES). Public data here is continuous, ICA-cleaned
# .fif -- we epoch it ourselves, same EPOCH_LEN, dropping the first/last 2 s to
# avoid filter edge artifacts.
# ─────────────────────────────────────────────────────────────────────────────

def list_site_subjects(site):
    if site == 'CHBMP':
        files = sorted(CHBMP_DIR.glob("sub-*/eeg/*_desc-wica_eeg.fif"))
    elif site == 'SRM':
        # prefer ses-t1 only (baseline), avoid double-counting longitudinal subjects
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
    """
    Loads a sovaharmony wICA-cleaned continuous .fif and epochs it.

    Amplitude-scale fix: CHBMP and SRM's sovaharmony outputs disagree on what
    unit their data is actually stored in, despite both FIFF headers claiming
    Volts. Verified directly (2026-08-13 pilot run): CHBMP std ~= 7e-6 (true
    Volts -> needs x1e6 to get uV, a normal ~7 uV EEG amplitude). SRM std ~= 7
    (already uV-scale, mislabeled as Volts in the header -- multiplying by 1e6
    again would inflate power by 1e12 and fabricate a fake "site effect" of
    ~12 in the aperiodic offset, which is not real physiology). Auto-detect
    per file instead of trusting the header.
    """
    raw = mne.io.read_raw_fif(str(fpath), preload=True, verbose=False)
    raw.pick_types(eeg=True)

    # If native std >= 1e-2, the data is already uV-scale despite the Volts
    # header -- rescale back down so it matches MNE's true-Volts convention,
    # which the rest of the pipeline (average ref, "* 1e6" in main()) assumes.
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
    """
    LEMON reports age as a 5-year privacy bin (e.g. '20-25'), not an exact
    value -- this is the dataset's own real anonymization scheme, not
    something we introduced. We use the bin midpoint as a numeric proxy for
    the age covariate (documented limitation, fine for a teaching demo).
    """
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
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(n_per_site, out_name):
    mne.set_log_level('ERROR')
    print("=" * 60)
    print("  extract_features_public.py")
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

                # ── cache the Global spectrum for the raw+aperiodic figure ──────
                # (same triplet the real Vigilance pipeline saves: raw log10 PSD,
                # the aperiodic fit reconstructed from offset/exponent, and the
                # aperiodic-corrected "flat" spectrum)
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

    # merge demographics
    demo_chbmp = load_chbmp_demo()
    demo_srm   = load_srm_demo()
    demo_lemon = load_lemon_demo()
    demo = pd.concat([demo_chbmp, demo_srm, demo_lemon], ignore_index=True)
    demo['sex'] = demo['sex'].astype(str).str.upper().str[0]  # M/F
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
