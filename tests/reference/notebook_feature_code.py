"""VERBATIM copy of the feature-extraction code from the executed notebook
`ptbxl_ecg_binary_classification_4th.ipynb` (code cell 12).  It is the *reference oracle* for the
deployment tests and must not be edited.  Only the imports and FS were added so it can be imported.
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
FS = 100          # records100 = 100 Hz (sampling rate of the recorded final experiment)

# ----------------------------- notebook cell 12 (verbatim) -----------------------------
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]   # PTB-XL lead order
BP_SOS = signal.butter(2, [0.5, 40], "bandpass", fs=FS, output="sos")                  # baseline + noise
QRS_SOS = signal.butter(2, [5, 15], "bandpass", fs=FS, output="sos")                   # QRS energy band
PRE, POST = int(0.25 * FS), int(0.45 * FS)                                              # beat window around R
PEAK_LEADS = [1, 0, 10, 7]                                                              # II, I, V5, V2 (fallbacks)

def fill_nans(sig):
    sig = sig.copy()
    for k in range(sig.shape[1]):
        bad = ~np.isfinite(sig[:, k])
        if bad.any():
            good = ~bad
            sig[:, k] = np.interp(np.arange(len(sig)), np.where(good)[0], sig[good, k]) if good.sum() > 1 else 0.0
    return sig

def preprocess_signal(sig):
    # (n_samples, 12) raw -> baseline-wander- and noise-filtered signal
    return signal.sosfiltfilt(BP_SOS, fill_nans(sig), axis=0)

def detect_r_peaks(x):
    q = signal.sosfiltfilt(QRS_SOS, x)
    w = int(0.12 * FS)
    energy = np.convolve(np.diff(q, prepend=q[0]) ** 2, np.ones(w) / w, mode="same")
    cand, _ = signal.find_peaks(energy, height=0.35 * np.percentile(energy, 99), distance=int(0.25 * FS))
    half, peaks = int(0.06 * FS), []
    for c in cand:                                           # refine to the actual R/S extremum
        a, b = max(c - half, 0), min(c + half + 1, len(x))
        peaks.append(a + int(np.argmax(np.abs(x[a:b]))))
    peaks = np.unique(peaks)
    return peaks[np.r_[True, np.diff(peaks) > int(0.2 * FS)]] if len(peaks) else peaks

def find_peaks_any_lead(xf):
    for k in PEAK_LEADS:
        if np.std(xf[:, k]) > 1e-6:
            p = detect_r_peaks(xf[:, k])
            if len(p) >= 3:
                return p
    return np.array([], dtype=int)

def rhythm_features(peaks):
    names = ["hr_bpm", "mean_rr_s", "sdnn_s", "rmssd_s", "pnn50", "rr_cv", "rr_range_s", "n_beats"]
    f = dict.fromkeys(names, np.nan)
    f["n_beats"] = len(peaks)
    if len(peaks) >= 3:
        rr = np.diff(peaks) / FS
        d = np.diff(rr)
        f.update(hr_bpm=60 / rr.mean(), mean_rr_s=rr.mean(), sdnn_s=rr.std(), rmssd_s=np.sqrt(np.mean(d ** 2)),
                 pnn50=np.mean(np.abs(d) > 0.05), rr_cv=rr.std() / rr.mean(), rr_range_s=rr.max() - rr.min())
    return f

def sl(r, a, b):                                             # slice of a template, from a to b seconds around R
    return slice(r + int(round(a * FS)), r + int(round(b * FS)) + 1)

def width_above(a, center, thr):
    l = r = center
    while l > 0 and a[l - 1] > thr: l -= 1
    while r < len(a) - 1 and a[r + 1] > thr: r += 1
    return r - l + 1

TEMPLATE_FEATS = ["qrs_max", "qrs_min", "qrs_area", "qrs_width_ms", "st_dev", "t_amp", "p_amp"]
def template_features(t):
    f = dict.fromkeys(TEMPLATE_FEATS, np.nan)
    if t is None:
        return f
    r = PRE
    tc = t - np.median(t[sl(r, -0.08, -0.04)])               # baseline = PR segment
    qrs = tc[sl(r, -0.06, 0.08)]
    f["qrs_max"], f["qrs_min"] = qrs.max(), qrs.min()
    f["qrs_area"] = np.abs(qrs).sum() / FS
    win = np.abs(tc[sl(r, -0.12, 0.12)])
    if win.max() > 1e-4:
        f["qrs_width_ms"] = 1000 * width_above(win, int(np.argmax(win)), 0.1 * win.max()) / FS   # width at 10% of peak height
    f["st_dev"] = tc[sl(r, 0.08, 0.12)].mean()
    for name, (a, b) in {"t_amp": (0.15, 0.40), "p_amp": (-0.20, -0.10)}.items():
        seg = tc[sl(r, a, b)]
        f[name] = seg[np.argmax(np.abs(seg))]
    return f

def record_features(path):
    # All features of ONE record. Returns None if the file cannot be read.
    import wfdb
    try:
        sig, fields = wfdb.rdsamp(path)
        if fields["fs"] != FS or sig.shape[1] != 12:
            return None
        xf = preprocess_signal(sig)
    except Exception:
        return None
    peaks = find_peaks_any_lead(xf)
    feats = rhythm_features(peaks)
    valid = peaks[(peaks - PRE >= 0) & (peaks + POST < len(xf))]
    tm = None
    if len(valid) >= 2:
        idx = valid[:, None] + np.arange(-PRE, POST + 1)[None, :]
        tm = np.median(xf[idx, :], axis=0)                   # (window, 12) median beat of every lead
    for k, lead in enumerate(LEADS):
        x = xf[:, k]
        feats[f"{lead}_std"], feats[f"{lead}_ptp"] = x.std(), np.ptp(x)
        feats[f"{lead}_skew"], feats[f"{lead}_kurt"] = stats.skew(x), stats.kurtosis(x)
        for name, v in template_features(None if tm is None else tm[:, k]).items():
            feats[f"{lead}_{name}"] = v
    return feats
