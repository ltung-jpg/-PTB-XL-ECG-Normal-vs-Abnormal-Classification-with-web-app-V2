"""Feature extraction - a faithful port of the notebook (cell 12 + cell 10).

The algorithms, constants and the ORDER of the produced features are identical to the notebook.  The only
change is that the sampling rate is an explicit argument (default 100 Hz = the rate the model was trained at)
instead of a global.  With fs = 100 the output is bit-for-bit identical to the notebook
(verified in tests/test_feature_equivalence.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal, stats

from . import config as C
from .preprocessing import preprocess_signal


# ----------------------------------------------------------------------------- R-peak detection
def _qrs_sos(fs: float):
    return signal.butter(C.QRS_DETECTION["band_order"], C.QRS_DETECTION["band_hz"], "bandpass", fs=fs, output="sos")


def detect_r_peaks(x: np.ndarray, fs: float = C.MODEL_FS) -> np.ndarray:
    """Notebook `detect_r_peaks` (Pan-Tompkins-style) on ONE filtered lead."""
    q = signal.sosfiltfilt(_qrs_sos(fs), x)
    w = int(0.12 * fs)
    energy = np.convolve(np.diff(q, prepend=q[0]) ** 2, np.ones(w) / w, mode="same")
    cand, _ = signal.find_peaks(energy, height=0.35 * np.percentile(energy, 99), distance=int(0.25 * fs))
    half, peaks = int(0.06 * fs), []
    for c in cand:                                           # refine to the actual R/S extremum
        a, b = max(c - half, 0), min(c + half + 1, len(x))
        peaks.append(a + int(np.argmax(np.abs(x[a:b]))))
    peaks = np.unique(peaks)
    return peaks[np.r_[True, np.diff(peaks) > int(0.2 * fs)]] if len(peaks) else peaks


def find_peaks_any_lead(xf: np.ndarray, fs: float = C.MODEL_FS) -> np.ndarray:
    """Notebook `find_peaks_any_lead`: try leads II, I, V5, V2 until >= 3 peaks are found."""
    for k in C.QRS_DETECTION["peak_leads"]:
        if np.std(xf[:, k]) > 1e-6:
            p = detect_r_peaks(xf[:, k], fs)
            if len(p) >= 3:
                return p
    return np.array([], dtype=int)


# ----------------------------------------------------------------------------- rhythm / HRV
def rhythm_features(peaks: np.ndarray, fs: float = C.MODEL_FS) -> dict:
    """Notebook `rhythm_features` - 8 features, same key order."""
    f = dict.fromkeys(C.RHYTHM_FEATURES, np.nan)
    f["n_beats"] = len(peaks)
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        d = np.diff(rr)
        f.update(hr_bpm=60 / rr.mean(), mean_rr_s=rr.mean(), sdnn_s=rr.std(), rmssd_s=np.sqrt(np.mean(d ** 2)),
                 pnn50=np.mean(np.abs(d) > 0.05), rr_cv=rr.std() / rr.mean(), rr_range_s=rr.max() - rr.min())
    return f


# ----------------------------------------------------------------------------- median-beat morphology
def _sl(r: int, a: float, b: float, fs: float) -> slice:
    return slice(r + int(round(a * fs)), r + int(round(b * fs)) + 1)


def _width_above(a: np.ndarray, center: int, thr: float) -> int:
    l = r = center
    while l > 0 and a[l - 1] > thr: l -= 1
    while r < len(a) - 1 and a[r + 1] > thr: r += 1
    return r - l + 1


def template_features(t, fs: float = C.MODEL_FS) -> dict:
    """Notebook `template_features` - 7 features of ONE lead's median beat (None -> all NaN)."""
    f = dict.fromkeys(C.TEMPLATE_FEATURES, np.nan)
    if t is None:
        return f
    r = int(0.25 * fs)                                       # PRE
    tc = t - np.median(t[_sl(r, -0.08, -0.04, fs)])          # baseline = PR segment
    qrs = tc[_sl(r, -0.06, 0.08, fs)]
    f["qrs_max"], f["qrs_min"] = qrs.max(), qrs.min()
    f["qrs_area"] = np.abs(qrs).sum() / fs
    win = np.abs(tc[_sl(r, -0.12, 0.12, fs)])
    if win.max() > 1e-4:
        f["qrs_width_ms"] = 1000 * _width_above(win, int(np.argmax(win)), C.TEMPLATE["width_fraction"] * win.max()) / fs
    f["st_dev"] = tc[_sl(r, 0.08, 0.12, fs)].mean()
    for name, (a, b) in {"t_amp": (0.15, 0.40), "p_amp": (-0.20, -0.10)}.items():
        seg = tc[_sl(r, a, b, fs)]
        f[name] = seg[np.argmax(np.abs(seg))]
    return f


@dataclass
class WaveformInfo:
    """Intermediate results the UI can reuse (not model inputs)."""
    filtered: np.ndarray            # (n, 12) band-passed signal
    peaks: np.ndarray               # R-peak sample indices
    template: np.ndarray | None     # (pre+post+1, 12) median beat of every lead, or None
    n_valid_beats: int
    fs: float


def extract_waveform_features(xf: np.ndarray, fs: float = C.MODEL_FS) -> tuple[dict, WaveformInfo]:
    """Notebook `record_features` after the file has been read and filtered.

    `xf` is the band-passed (n_samples, 12) signal.  Returns the 140 waveform features in notebook order.
    """
    pre, post = int(0.25 * fs), int(0.45 * fs)
    peaks = find_peaks_any_lead(xf, fs)
    feats = rhythm_features(peaks, fs)
    valid = peaks[(peaks - pre >= 0) & (peaks + post < len(xf))]
    tm = None
    if len(valid) >= 2:
        idx = valid[:, None] + np.arange(-pre, post + 1)[None, :]
        tm = np.median(xf[idx, :], axis=0)                   # (window, 12) median beat of every lead
    for k, lead in enumerate(C.LEADS):
        x = xf[:, k]
        feats[f"{lead}_std"], feats[f"{lead}_ptp"] = x.std(), np.ptp(x)
        feats[f"{lead}_skew"], feats[f"{lead}_kurt"] = stats.skew(x), stats.kurtosis(x)
        for name, v in template_features(None if tm is None else tm[:, k], fs).items():
            feats[f"{lead}_{name}"] = v
    return feats, WaveformInfo(filtered=xf, peaks=peaks, template=tm, n_valid_beats=len(valid), fs=fs)


def waveform_features_from_100hz(sig100: np.ndarray) -> tuple[dict, WaveformInfo]:
    """Raw 100 Hz (n, 12) signal -> notebook preprocessing -> 140 waveform features."""
    return extract_waveform_features(preprocess_signal(sig100, C.MODEL_FS), C.MODEL_FS)


# ----------------------------------------------------------------------------- demographics
def _in_range(v: float, lo: float, hi: float) -> float:
    return v if (np.isfinite(v) and lo <= v <= hi) else np.nan


def build_demographic_features(age=None, sex=None, height_cm=None, weight_kg=None) -> dict:
    """The 5 demographic model inputs, cleaned exactly like notebook cell 10.

    Unknown values stay NaN so the model's ORIGINAL fitted median imputer fills them.  Nothing is invented.
    `sex` must be "male" / "female" (case-insensitive), the PTB-XL codes 0 / 1, or None.

    One deliberate difference from the notebook: pandas `Series.where(age <= 120, 90)` turns a *missing* age
    into 90 - a quirk that never mattered in training (no training record lacked an age).  In deployment a
    missing age stays missing and is median-imputed, as the specification requires.
    """
    D = C.DEMOGRAPHICS

    def num(v):
        try:
            return np.nan if v is None else float(v)
        except (TypeError, ValueError):
            return np.nan

    a = num(age)
    if np.isfinite(a):
        a = a if a <= D["age_cap_threshold"] else float(D["age_cap_value"])

    if sex is None or (isinstance(sex, float) and np.isnan(sex)):
        s = np.nan
    elif isinstance(sex, str):
        key = sex.strip().lower()
        if key not in D["sex_coding"]:
            raise ValueError(f"sex must be 'male' or 'female' (or None), got {sex!r}")
        s = float(D["sex_coding"][key])
    else:
        if int(sex) not in (0, 1):
            raise ValueError(f"numeric sex must be the PTB-XL code 0 (male) or 1 (female), got {sex!r}")
        s = float(int(sex))

    h = _in_range(num(height_cm), *D["height_cm_valid_range"])
    w = _in_range(num(weight_kg), *D["weight_kg_valid_range"])
    bmi = w / (h / 100) ** 2 if (np.isfinite(h) and np.isfinite(w)) else np.nan
    bmi = _in_range(bmi, *D["bmi_valid_range"])
    return {"age": a, "sex": s, "height": h, "weight": w, "bmi": bmi}


# ----------------------------------------------------------------------------- schema assembly
class FeatureSchemaError(ValueError):
    pass


def assemble_feature_frame(waveform_feats: dict, demographic_feats: dict, feature_names: list[str]) -> pd.DataFrame:
    """One-row DataFrame whose columns are EXACTLY `feature_names`, in that order (the saved schema)."""
    pool = {**waveform_feats, **demographic_feats}
    unknown = [n for n in feature_names if n not in pool]
    if unknown:
        raise FeatureSchemaError(f"Model expects features this pipeline cannot produce: {unknown[:8]}"
                                 f"{' ...' if len(unknown) > 8 else ''}")
    row = {n: pool[n] for n in feature_names}
    df = pd.DataFrame([row], columns=list(feature_names)).astype(float)
    return df.replace([np.inf, -np.inf], np.nan)
