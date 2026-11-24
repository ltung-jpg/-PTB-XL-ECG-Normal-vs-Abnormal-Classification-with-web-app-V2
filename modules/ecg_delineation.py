"""ECG morphology: wave/interval measurements.  INDEPENDENT of the classifier (it never sees model output).

Method: NeuroKit2 discrete-wavelet-transform delineation (Martinez et al., 2004) on one lead (default II), run at >= 250 Hz.
A recording at 100 Hz is up-sampled to 500 Hz first; that does NOT add information, so the time resolution is limited to
one source sample (10 ms at 100 Hz) and the result says so.  Every measure is reported per beat, summarised by the
median, and flagged `measured` / `not_measurable` - missing waves are never filled in.

Definitions
    P duration   P onset -> P offset            PR interval  P onset -> QRS onset
    QRS duration QRS onset -> QRS offset        QT interval  QRS onset -> T offset
    RR interval  R -> next R                    QTc          Bazett QT/sqrt(RR) and Fridericia QT/RR^(1/3)
    T duration   T onset -> T offset
    PR segment   P offset -> QRS onset          ST segment   QRS offset -> T onset      TP segment  T offset -> next P onset
    ST level     mean amplitude 60-80 ms after QRS offset minus PR-segment baseline (mV)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from . import config as C
from .preprocessing import fill_nans

MEASURES = ["rr_s", "hr_bpm", "p_duration_ms", "pr_interval_ms", "qrs_duration_ms", "qt_ms", "qtc_bazett_ms",
            "qtc_fridericia_ms", "t_duration_ms", "pr_segment_ms", "st_segment_ms", "tp_segment_ms", "st_level_mv"]


@dataclass
class Measurement:
    name: str
    value: float | None                     # median over valid beats
    n_beats: int                            # beats that contributed (single lead) / leads that contributed (consensus)
    spread: float | None = None             # IQR over beats (single lead) or over leads (consensus), same unit
    status: str = "not_measurable"          # "measured" | "low_agreement" | "not_measurable"
    note: str = ""


@dataclass
class MorphologyResult:
    ok: bool
    lead: str
    method: str
    fs_analysis: float
    resolution_ms: float
    measures: dict[str, Measurement] = field(default_factory=dict)
    fiducials: dict[str, np.ndarray] = field(default_factory=dict)       # seconds, per beat, NaN = not found
    signal: np.ndarray | None = None                                     # analysed lead (mV) at fs_analysis
    time: np.ndarray | None = None
    beat_table: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    per_lead: dict = field(default_factory=dict)


def _bandpass(x, fs):
    hi = min(40.0, 0.45 * fs)
    sos = sps.butter(2, [0.5, hi], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def _median_iqr(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None, 0
    q1, q3 = np.percentile(v, [25, 75])
    return float(np.median(v)), float(q3 - q1), int(v.size)


def delineate(signal_mv: np.ndarray, fs: float, lead: str = "II", lead_names: list[str] | None = None) -> MorphologyResult:
    """signal_mv: (n,12) recording in mV (native sampling rate)."""
    import neurokit2 as nk
    names = lead_names or C.LEADS
    res = MorphologyResult(ok=False, lead=lead, method="NeuroKit2 DWT delineation", fs_analysis=0.0, resolution_ms=0.0)
    if lead not in names:
        res.message = f"Lead {lead} not available."
        return res
    x = fill_nans(np.asarray(signal_mv, float))[:, names.index(lead)]
    x = _bandpass(x, fs)
    fs_a = fs
    if fs < 250:
        x = sps.resample_poly(x, 500, int(round(fs)) if abs(fs - round(fs)) < 1e-9 else int(round(fs)))
        fs_a = 500.0 * fs / int(round(fs))
        res.warnings.append(f"Recording at {fs:g} Hz was up-sampled to {fs_a:g} Hz for delineation. Timing resolution remains "
                            f"{1000 / fs:.0f} ms, so wave boundaries are approximate (+/- one source sample).")
    res.fs_analysis, res.resolution_ms = float(fs_a), 1000.0 / fs
    t = np.arange(len(x)) / fs_a
    res.signal, res.time = x, t
    try:
        _, rp = nk.ecg_peaks(x, sampling_rate=fs_a)
        r = np.asarray(rp["ECG_R_Peaks"], int)
    except Exception as e:
        res.message = f"R-peak detection failed: {e}"
        return res
    if len(r) < 3:
        res.message = f"Only {len(r)} R peaks found - at least 3 beats are needed."
        return res
    try:
        _, w = nk.ecg_delineate(x, r, sampling_rate=fs_a, method="dwt")
    except Exception as e:
        res.message = f"Wave delineation failed: {e}"
        return res

    def arr(k):
        v = np.array(w.get(k, [np.nan] * len(r)), float)
        return v if len(v) == len(r) else np.full(len(r), np.nan)

    f = {"r_peak": r.astype(float), "p_onset": arr("ECG_P_Onsets"), "p_peak": arr("ECG_P_Peaks"), "p_offset": arr("ECG_P_Offsets"),
         "qrs_onset": arr("ECG_R_Onsets"), "qrs_offset": arr("ECG_R_Offsets"), "t_peak": arr("ECG_T_Peaks"),
         "t_onset": arr("ECG_T_Onsets"), "t_offset": arr("ECG_T_Offsets")}
    res.fiducials = {k: v / fs_a for k, v in f.items()}                  # seconds (NaN preserved)
    s = lambda a, b: (f[b] - f[a]) / fs_a                               # seconds between two fiducials (per beat)

    rr = np.diff(f["r_peak"]) / fs_a
    rr_beat = np.append(rr, np.nan)                                      # RR following each beat
    rr_prev = np.insert(rr, 0, np.nan)                                   # RR preceding each beat (used for QTc)
    per = {
        "rr_s": rr,
        "hr_bpm": 60.0 / rr,
        "p_duration_ms": 1000 * s("p_onset", "p_offset"),
        "pr_interval_ms": 1000 * s("p_onset", "qrs_onset"),
        "qrs_duration_ms": 1000 * s("qrs_onset", "qrs_offset"),
        "qt_ms": 1000 * s("qrs_onset", "t_offset"),
        "t_duration_ms": 1000 * s("t_onset", "t_offset"),
        "pr_segment_ms": 1000 * s("p_offset", "qrs_onset"),
        "st_segment_ms": 1000 * s("qrs_offset", "t_onset"),
    }
    qt_s = per["qt_ms"] / 1000.0
    per["qtc_bazett_ms"] = 1000 * qt_s / np.sqrt(rr_prev)
    per["qtc_fridericia_ms"] = 1000 * qt_s / np.cbrt(rr_prev)
    per["tp_segment_ms"] = 1000 * (np.append(f["p_onset"][1:], np.nan) - f["t_offset"]) / fs_a

    # ST level relative to the PR-segment baseline
    st = np.full(len(r), np.nan)
    for i in range(len(r)):
        try:
            a, b = int(f["p_offset"][i]), int(f["qrs_onset"][i])
            j = int(f["qrs_offset"][i])
            if b - a >= 2:
                base = np.mean(x[a:b])
                lo, hi = j + int(0.06 * fs_a), j + int(0.08 * fs_a)
                if hi < len(x):
                    st[i] = np.mean(x[lo:hi + 1]) - base
        except (ValueError, OverflowError):
            pass
    per["st_level_mv"] = st

    # physiologic sanity limits - values outside are treated as delineation failures, not as findings
    limits = {"p_duration_ms": (30, 220), "pr_interval_ms": (60, 400), "qrs_duration_ms": (40, 250), "qt_ms": (200, 700),
              "t_duration_ms": (60, 400), "pr_segment_ms": (0, 300), "st_segment_ms": (0, 300), "tp_segment_ms": (0, 1500),
              "qtc_bazett_ms": (250, 750), "qtc_fridericia_ms": (250, 750)}
    for k, (lo, hi) in limits.items():
        v = per[k]; v[(v < lo) | (v > hi)] = np.nan
    for name in MEASURES:
        med, iqr, n = _median_iqr(per[name])
        need = 1 if name in ("rr_s", "hr_bpm") else 3
        ok = med is not None and n >= need
        note = "" if ok else (f"found in only {n} beat(s)" if n else "wave not detected")
        res.measures[name] = Measurement(name, med if ok else None, n, iqr if ok else None, "measured" if ok else "not_measurable", note)
    # missing / absent P waves: say so instead of leaving a silent blank
    if res.measures["p_duration_ms"].status != "measured":
        res.measures["p_duration_ms"].note = "no reliable P wave (absent, hidden or irregular rhythm) - PR interval not measurable"
    rr_arr = rr[np.isfinite(rr)]
    if rr_arr.size and np.std(rr_arr) / np.mean(rr_arr) > 0.15:
        res.warnings.append("RR intervals vary by more than 15 % (irregular rhythm): QTc corrections assume a steady rate and are less reliable, "
                            "and P waves cannot be identified reliably by the automatic method, so P/PR measures are withheld.")
        for k in ("p_duration_ms", "pr_interval_ms", "pr_segment_ms", "tp_segment_ms"):
            res.measures[k] = Measurement(k, None, res.measures[k].n_beats, None, "not_measurable",
                                          "withheld: irregular rhythm - P waves not reliably identifiable automatically")
    res.beat_table = [{"beat": i + 1, "r_time_s": float(f["r_peak"][i] / fs_a), **{k: (None if not np.isfinite(per[k][i]) else float(per[k][i]))
                       for k in MEASURES if k not in ("rr_s", "hr_bpm") and len(per[k]) == len(r)}} for i in range(len(r))]
    res.ok = True
    return res


def delineation_for_display(res: MorphologyResult) -> dict:
    """Plain dict of the per-measure results (for the report/UI)."""
    return {k: {"value": m.value, "n_beats": m.n_beats, "iqr": m.spread, "status": m.status, "note": m.note}
            for k, m in res.measures.items()}


# ---------------------------------------------------------------------------------------------------------------
# Multi-lead consensus.  A single-lead wavelet delineation is not reliable enough on its own (on the three example
# PTB-XL records the QRS duration of the SAME record differed by up to 76 ms between leads), so the reported value is
# the median over several leads and it is only called "measured" when the leads agree.
CONSENSUS_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
MIN_LEADS = 4
AGREEMENT_TOL_MS = {"p_duration_ms": 30, "pr_interval_ms": 30, "qrs_duration_ms": 25, "qt_ms": 40, "qtc_bazett_ms": 45,
                    "qtc_fridericia_ms": 45, "t_duration_ms": 50, "pr_segment_ms": 30, "st_segment_ms": 40, "tp_segment_ms": 200}


def delineate_consensus(signal_mv, fs, leads=None, display_lead="II", lead_names=None) -> MorphologyResult:
    """Run `delineate` on several leads; value = median over leads, spread = IQR over leads, status from lead agreement."""
    leads = leads or CONSENSUS_LEADS
    per_lead = {}
    for ld in leads:
        r = delineate(signal_mv, fs, ld, lead_names)
        if r.ok:
            per_lead[ld] = r
    disp = per_lead.get(display_lead) or (next(iter(per_lead.values())) if per_lead else None)
    if disp is None or len(per_lead) < MIN_LEADS:
        out = disp or delineate(signal_mv, fs, display_lead, lead_names)
        out.ok = False
        out.message = out.message or f"Delineation succeeded on only {len(per_lead)} lead(s); at least {MIN_LEADS} are needed."
        return out
    out = MorphologyResult(ok=True, lead=f"consensus of {len(per_lead)} leads (display: {disp.lead})",
                           method=disp.method + ", median over leads", fs_analysis=disp.fs_analysis,
                           resolution_ms=disp.resolution_ms, fiducials=disp.fiducials, signal=disp.signal, time=disp.time,
                           beat_table=disp.beat_table, warnings=list(disp.warnings))
    for name in MEASURES:
        vals = [(ld, r.measures[name].value) for ld, r in per_lead.items() if r.measures[name].status == "measured"]
        v = np.array([x for _, x in vals], float)
        if name in ("rr_s", "hr_bpm"):                                   # rhythm: lead independent
            out.measures[name] = per_lead[display_lead if display_lead in per_lead else next(iter(per_lead))].measures[name]
            continue
        if len(v) < MIN_LEADS:
            note = per_lead[next(iter(per_lead))].measures[name].note if len(v) == 0 else f"measured on only {len(v)} of {len(per_lead)} leads"
            out.measures[name] = Measurement(name, None, len(v), None, "not_measurable", note or "wave not detected")
            continue
        med = float(np.median(v)); q1, q3 = np.percentile(v, [25, 75]); iqr = float(q3 - q1)
        tol = AGREEMENT_TOL_MS.get(name)
        if name == "st_level_mv":
            tol = None
        agree = tol is None or iqr <= tol
        out.measures[name] = Measurement(name, med, len(v), iqr, "measured" if agree else "low_agreement",
                                         "" if agree else f"leads disagree (inter-lead IQR {iqr:.0f} ms > {tol} ms): value unreliable")
    # propagate rhythm-based withholding (irregular rhythm -> no P/PR)
    for k in ("p_duration_ms", "pr_interval_ms", "pr_segment_ms", "tp_segment_ms"):
        if all(per_lead[l].measures[k].status == "not_measurable" and "irregular" in per_lead[l].measures[k].note for l in per_lead):
            out.measures[k] = per_lead[next(iter(per_lead))].measures[k]
    out.warnings = sorted(set(out.warnings + [w for r in per_lead.values() for w in r.warnings]))
    out.per_lead = {ld: {k: m.value for k, m in r.measures.items()} for ld, r in per_lead.items()}
    return out
