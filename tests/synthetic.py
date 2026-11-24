"""Synthetic 12-lead ECG generator with KNOWN ground-truth intervals.

Used ONLY for tests (pipeline plumbing, delineation accuracy, the flagged test-stub model).  It is not
physiologically realistic enough to say anything about real patients - never use it to judge model quality.

Every wave is a raised-cosine (Hann) bump with an explicit onset and offset, so the true durations are
defined by construction.
"""
from __future__ import annotations

import numpy as np

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# amplitude scaling of the single "cardiac vector" into each lead (sign-flips and sizes roughly typical)
LEAD_GAIN = np.array([0.9, 1.2, 0.5, -1.0, 0.3, 0.85, -0.5, 0.2, 0.8, 1.3, 1.2, 0.9])


def _bump(t: np.ndarray, onset: float, dur: float, amp: float) -> np.ndarray:
    x = np.zeros_like(t)
    m = (t >= onset) & (t <= onset + dur)
    x[m] = amp * 0.5 * (1 - np.cos(2 * np.pi * (t[m] - onset) / dur))
    return x


def make_ecg(fs: float = 100.0, dur: float = 10.0, hr: float = 70.0, *, p_dur=0.09, pr=0.16, qrs_dur=0.09,
             st_seg=0.08, t_dur=0.18, st_dev=0.0, t_amp=0.30, afib=False, noise=0.015, wander=0.08,
             seed: int = 0, gains: np.ndarray | None = None):
    """Returns (signal (n,12) in mV, truth dict).  `pr` = P onset -> QRS onset."""
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    t = np.arange(n) / fs
    vec = np.zeros(n)
    rr0 = 60.0 / hr
    beats, tr = [], 0.45
    truth = {"p_onset": [], "p_offset": [], "qrs_onset": [], "qrs_offset": [], "t_onset": [], "t_offset": [], "r_peak": []}
    while tr < dur - 0.7:
        qrs_on = tr - 0.4 * qrs_dur
        qrs_off = qrs_on + qrs_dur
        p_on = qrs_on - pr
        t_on = qrs_off + st_seg
        # QRS: Q (first 15 %), R (15-65 %), S (65-100 %)
        vec += _bump(t, qrs_on, 0.15 * qrs_dur, -0.10)
        vec += _bump(t, qrs_on + 0.15 * qrs_dur, 0.50 * qrs_dur, 1.00)
        vec += _bump(t, qrs_on + 0.65 * qrs_dur, 0.35 * qrs_dur, -0.25)
        if not afib:
            vec += _bump(t, p_on, p_dur, 0.12)
        vec += _bump(t, t_on, t_dur, t_amp)
        if st_dev:                                       # ST deviation: plateau between QRS end and T end
            vec += np.where((t >= qrs_off) & (t <= t_on + t_dur), st_dev, 0.0)
        for k, v in dict(p_onset=p_on, p_offset=p_on + p_dur, qrs_onset=qrs_on, qrs_offset=qrs_off, t_onset=t_on,
                         t_offset=t_on + t_dur, r_peak=tr).items():
            truth[k].append(v)
        rr = rr0 * (rng.uniform(0.6, 1.4) if afib else rng.uniform(0.985, 1.015))
        tr += rr
    g = LEAD_GAIN if gains is None else gains
    sig = vec[:, None] * g[None, :]
    sig += wander * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6))[:, None] * rng.uniform(0.5, 1.0, size=(1, 12))
    sig += noise * rng.normal(size=sig.shape)
    truth = {k: np.array(v) for k, v in truth.items()}
    rrs = np.diff(truth["r_peak"])
    truth.update(
        hr_bpm=60.0 / rrs.mean() if len(rrs) else np.nan, rr_s=rrs.mean() if len(rrs) else np.nan,
        p_dur_s=p_dur if not afib else np.nan, pr_s=pr if not afib else np.nan, qrs_dur_s=qrs_dur,
        qt_s=(qrs_dur + st_seg + t_dur), t_dur_s=t_dur, st_seg_s=st_seg, st_dev_mv=st_dev * g[1])
    return sig, truth


def make_dataset(n: int = 400, fs: float = 100.0, seed: int = 0):
    """A labelled synthetic set for the TEST-STUB model: 0 = 'normal', 1 = 'abnormal' (wide QRS / ST shift / AF)."""
    rng = np.random.default_rng(seed)
    sigs, labels, demos = [], [], []
    for i in range(n):
        ab = int(rng.random() < 0.55)
        kind = rng.integers(0, 3) if ab else -1
        kw = dict(hr=float(rng.uniform(55, 95)), seed=int(rng.integers(1e9)))
        if ab and kind == 0:
            kw["qrs_dur"] = float(rng.uniform(0.12, 0.16))
        if ab and kind == 1:
            kw["st_dev"] = float(rng.choice([-1, 1]) * rng.uniform(0.12, 0.3)); kw["t_amp"] = float(rng.uniform(-0.2, 0.1))
        if ab and kind == 2:
            kw["afib"] = True
        sig, _ = make_ecg(fs=fs, **kw)
        sigs.append(sig); labels.append(ab)
        age = float(np.clip(rng.normal(50 + 15 * ab, 15), 18, 90))
        demos.append({"age": age, "sex": float(rng.integers(0, 2)),
                      "height": float(rng.normal(168, 9)) if rng.random() > 0.6 else np.nan,
                      "weight": float(rng.normal(72, 12)) if rng.random() > 0.5 else np.nan})
    return sigs, np.array(labels), demos
