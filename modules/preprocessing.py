"""Signal preprocessing.

`fill_nans` and `preprocess_signal` are the notebook's functions (cell 12) - same operations, same filter
design - so that a 100 Hz recording gets EXACTLY the same filtered signal as in training.

Everything else here (resampling, quality checks, window selection) is deployment logic that did not exist in
the notebook, because PTB-XL's 100 Hz files were already uniform.  Those parts are clearly marked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import signal

from . import config as C


# ============================================================================ notebook-equivalent
@lru_cache(maxsize=8)
def _bandpass_sos(fs: float, low_hz: float, high_hz: float, order: int):
    return signal.butter(order, [low_hz, high_hz], "bandpass", fs=fs, output="sos")


def fill_nans(sig: np.ndarray) -> np.ndarray:
    """Notebook `fill_nans`: linear interpolation of non-finite samples, lead by lead."""
    sig = sig.copy()
    for k in range(sig.shape[1]):
        bad = ~np.isfinite(sig[:, k])
        if bad.any():
            good = ~bad
            sig[:, k] = np.interp(np.arange(len(sig)), np.where(good)[0], sig[good, k]) if good.sum() > 1 else 0.0
    return sig


def preprocess_signal(sig: np.ndarray, fs: float = C.MODEL_FS) -> np.ndarray:
    """Notebook `preprocess_signal`: (n_samples, 12) raw -> zero-phase Butterworth band-pass 0.5-40 Hz.

    With fs = 100 (the model rate) this is bit-for-bit the notebook's filter.  For other rates (used only by
    the independent morphology module) the upper corner is limited to 0.45 * fs to stay below Nyquist.
    """
    b = C.BANDPASS
    high = min(b["high_hz"], 0.45 * fs)
    sos = _bandpass_sos(float(fs), float(b["low_hz"]), float(high), int(b["order"]))
    return signal.sosfiltfilt(sos, fill_nans(sig), axis=0)


# ============================================================================ deployment logic (new)
class SignalError(ValueError):
    """The signal cannot be processed reliably."""


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float = C.MODEL_FS) -> np.ndarray:
    """Polyphase (anti-aliased, zero-phase FIR) resampling of a (n_samples, n_leads) array.

    NOT part of the trained pipeline: PTB-XL's 100 Hz files were prepared by the dataset authors.  Other rates
    are converted with `scipy.signal.resample_poly`, whose Kaiser low-pass prevents aliasing.  Non-finite
    samples are interpolated first.
    """
    if fs_in <= 0 or not np.isfinite(fs_in):
        raise SignalError(f"Invalid sampling frequency: {fs_in}")
    if abs(fs_in - fs_out) < 1e-9:
        return x
    ratio = (Fraction(fs_out) / Fraction(fs_in)).limit_denominator(1000)
    if abs(float(ratio) * fs_in - fs_out) / fs_out > 1e-3:
        raise SignalError(f"Cannot resample {fs_in} Hz to {fs_out} Hz accurately (rational approximation too coarse).")
    x = fill_nans(x)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0, padtype="line")


@dataclass
class QualityReport:
    errors: list[str] = field(default_factory=list)       # make the recording unusable
    warnings: list[str] = field(default_factory=list)     # usable but questionable
    missing_fraction: dict[str, float] = field(default_factory=dict)
    flat_leads: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def assess_signal_quality(sig_mv: np.ndarray, fs: float, lead_names: list[str] | None = None) -> QualityReport:
    """Signal-level checks on a physical (mV) recording BEFORE resampling."""
    P = C.POLICY
    names = lead_names or C.LEADS
    q = QualityReport()
    if sig_mv.ndim != 2 or sig_mv.shape[1] != C.N_LEADS:
        q.errors.append(f"Expected an (n_samples x 12) array, got shape {sig_mv.shape}.")
        return q
    dur = sig_mv.shape[0] / fs
    if dur < C.MIN_DURATION_S:
        q.errors.append(f"Recording is {dur:.1f} s long; at least {C.MIN_DURATION_S:.0f} s are required "
                        f"(the model was trained on {C.WINDOW_S:.0f} s recordings).")
    elif abs(dur - C.WINDOW_S) > 0.5:
        q.warnings.append(f"Recording is {dur:.1f} s long, the model was trained on {C.WINDOW_S:.0f} s recordings; "
                          "rhythm features may be less comparable.")
    fin = np.isfinite(sig_mv)
    for k, name in enumerate(names):
        frac = 1.0 - fin[:, k].mean()
        q.missing_fraction[name] = float(frac)
        if frac > P["max_missing_fraction_per_lead"]:
            q.errors.append(f"Lead {name}: {frac:.0%} of samples are missing/invalid (limit {P['max_missing_fraction_per_lead']:.0%}).")
        elif frac > 0:
            q.warnings.append(f"Lead {name}: {frac:.1%} of samples missing; they were interpolated.")
        col = sig_mv[fin[:, k], k]
        if col.size == 0 or np.std(col) <= P["flat_lead_std_mv"]:
            q.flat_leads.append(name)
            q.errors.append(f"Lead {name} is flat (no signal) - electrode disconnected or wrong channel.")
    col = sig_mv[fin[:, 1], 1]                                         # amplitude plausibility on lead II
    if col.size:
        ptp = float(np.ptp(col))
        lo, hi = P["implausible_ptp_mv"]
        if ptp < lo or ptp > hi:
            q.warnings.append(f"Lead II peak-to-peak amplitude is {ptp:.3g} mV (typical ECG: about 0.5-5 mV). "
                              "Check that the signal units are correct.")
    return q


def prepare_for_model(sig_mv: np.ndarray, fs: float, window_start_s: float = 0.0):
    """Physical recording -> raw 100 Hz window the model expects.  Returns (signal_100hz, notes).

    Band-pass filtering is NOT applied here (use `preprocess_signal`).  Policy: recordings longer than 10 s are
    analysed over ONE 10-s window starting at `window_start_s` (reported, never truncated silently).
    """
    notes: list[str] = []
    x = np.asarray(sig_mv, dtype=float)
    if abs(fs - C.MODEL_FS) > 1e-9:
        x = resample_signal(x, fs, C.MODEL_FS)
        notes.append(f"Resampled from {fs:g} Hz to {C.MODEL_FS} Hz (anti-aliased polyphase filter).")
    n_win = int(round(C.WINDOW_S * C.MODEL_FS))
    start = int(round(window_start_s * C.MODEL_FS))
    if x.shape[0] > n_win:
        if start + n_win > x.shape[0]:
            start = x.shape[0] - n_win
        x = x[start:start + n_win]
        notes.append(f"Recording longer than {C.WINDOW_S:.0f} s: analysed the {C.WINDOW_S:.0f}-s window "
                     f"starting at {start / C.MODEL_FS:.1f} s.")
    return x, notes
