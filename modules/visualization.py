"""Matplotlib figures for the UI and the report (each function returns a Figure)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ui import tokens as T

from . import config as C

# presentation only: colours/fonts come from ui/tokens.py ("Clinical Burgundy & White"); no value shown in a chart is changed here
# reference-range status is NOT a clinical verdict: no green "good" / red "bad" colours (within = neutral blue, outside = amber)
STATUS_COLORS = {"within": T.INFO, "below": T.WARNING, "above": T.WARNING, "indeterminate": T.TEXT_MUTED}
_RC = {"figure.facecolor": T.CHART_BG, "axes.facecolor": T.CHART_BG, "savefig.facecolor": T.CHART_BG, "text.color": T.TEXT,
       "axes.labelcolor": T.TEXT_2, "axes.edgecolor": T.BORDER, "xtick.color": T.TEXT_2, "ytick.color": T.TEXT_2,
       "axes.titlecolor": T.TEXT, "grid.color": T.GRID_MAJOR, "font.family": "sans-serif",
       "font.sans-serif": ["Source Sans 3", "Source Sans Pro", "Segoe UI", "Arial", "DejaVu Sans"], "axes.spines.top": False,
       "axes.spines.right": False, "axes.titlesize": 11, "axes.labelsize": 9}


def _styled(fn):
    """Run a plotting function inside the app's matplotlib style (white chart background, no dark panels)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        with plt.rc_context(_RC):
            return fn(*a, **k)
    return wrapper


def _ecg_grid(ax):
    """Light ECG-paper style grid: major + minor lines."""
    ax.minorticks_on()
    ax.grid(which="major", color=T.GRID_MAJOR, lw=0.7)
    ax.grid(which="minor", color=T.GRID_MINOR, lw=0.5)
    ax.set_axisbelow(True)


LAYOUT = [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"]]


@_styled
def plot_12_lead(signal_mv, fs, title="12-lead ECG", t_max=10.0, peaks_s=None, missing_mask=None):
    """Standard 3x4 layout.  Samples that were missing in the source are shaded."""
    n = min(signal_mv.shape[0], int(t_max * fs))
    t = np.arange(n) / fs
    fig, axes = plt.subplots(3, 4, figsize=(15, 7), sharex=True)
    for r, row in enumerate(LAYOUT):
        for c, lead in enumerate(row):
            ax, k = axes[r, c], C.LEADS.index(lead)
            ax.plot(t, signal_mv[:n, k], lw=0.8, color=T.ECG_LINE)
            if missing_mask is not None and missing_mask[:n, k].any():
                m = missing_mask[:n, k]
                ax.fill_between(t, *ax.get_ylim(), where=m, color=T.WARNING, alpha=0.25, step="mid")
            if peaks_s is not None:
                pk = peaks_s[peaks_s < t[-1]]
                ax.plot(pk, np.interp(pk, t, signal_mv[:n, k]), "v", ms=4, color=T.R_PEAK, markeredgecolor=T.TEXT_2, markeredgewidth=0.3)
            ax.set_title(lead, loc="left", fontsize=10, fontweight="bold")
            _ecg_grid(ax); ax.tick_params(labelsize=8, which="both")
            if r == 2:
                ax.set_xlabel("s", fontsize=8)
            if c == 0:
                ax.set_ylabel("mV", fontsize=8)
    fig.suptitle(title, fontsize=12, color=T.TEXT); fig.tight_layout()
    return fig


@_styled
def plot_preprocessing(raw_mv, filtered, fs, lead="II", t_max=6.0):
    k = C.LEADS.index(lead)
    n = min(len(filtered), int(t_max * fs)); t = np.arange(n) / fs
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    ax[0].plot(t, raw_mv[:n, k], color=T.SECONDARY_WAVE, lw=0.9); ax[0].set_title(f"Lead {lead}: input (100 Hz window)")
    ax[1].plot(t, filtered[:n, k], color=T.ECG_LINE, lw=0.9)
    ax[1].set_title(f"Lead {lead}: after {C.BANDPASS['low_hz']}-{C.BANDPASS['high_hz']} Hz band-pass")
    for a in ax:
        _ecg_grid(a); a.set_ylabel("mV")
    ax[1].set_xlabel("s"); fig.tight_layout()
    return fig


@_styled
def plot_beat_with_fiducials(morph, n_beats=1, pad_s=0.35):
    """Analysed lead around one beat with the delineated wave boundaries and a measurement summary."""
    x, t, f = morph.signal, morph.time, morph.fiducials
    fig, ax = plt.subplots(figsize=(11, 4.2))
    if not morph.ok:
        ax.text(0.5, 0.5, morph.message or "Delineation failed", ha="center", va="center"); ax.axis("off"); return fig
    i = len(f["r_peak"]) // 2
    r = f["r_peak"][i]
    m = (t > r - 0.45) & (t < r + 0.75)
    ax.plot(t[m], x[m], color=T.ECG_LINE, lw=1.4)
    ax.margins(y=0.18)   # headroom so the wave labels are not clipped
    marks = [("p_onset", "P on", T.ANNOTATION), ("p_offset", "P off", T.ANNOTATION), ("qrs_onset", "QRS on", T.ACCENT_HOVER),
             ("qrs_offset", "QRS off", T.ACCENT_HOVER), ("t_onset", "T on", T.SECONDARY_WAVE), ("t_offset", "T off", T.SECONDARY_WAVE)]
    for key, lab, col in marks:
        v = f[key][i]
        if np.isfinite(v):
            ax.axvline(v, color=col, ls="--", lw=0.9, alpha=0.8)
            ax.text(v, ax.get_ylim()[1], lab, color=col, fontsize=8, rotation=90, va="top", ha="right")
    ax.set_title(f"Estimated wave boundaries - one beat, lead {morph.lead}", loc="left")
    ax.set_xlabel("s"); ax.set_ylabel("mV"); _ecg_grid(ax)
    fig.tight_layout()
    return fig


@_styled
def plot_reference_comparison(comparisons):
    """Horizontal range bars: shaded reference range, marker for the measured value (colour = status)."""
    rows = [c for c in comparisons if c.value is not None and (c.lower is not None or c.upper is not None)]
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 1.5)); ax.text(0.5, 0.5, "No measurable value with a configured reference range", ha="center"); ax.axis("off")
        return fig
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 0.95 * len(rows) + 0.6))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, rows):
        lo = c.lower if c.lower is not None else min(c.value, c.upper) * 0.6
        hi = c.upper if c.upper is not None else max(c.value, c.lower) * 1.4
        span = hi - lo
        xmin, xmax = min(lo - 0.6 * span, c.value - 0.1 * span), max(hi + 0.6 * span, c.value + 0.1 * span)
        ax.axvspan(lo if c.lower is not None else xmin, hi if c.upper is not None else xmax, color=T.INFO_BG, zorder=0)
        ax.plot([c.value], [0], "o", ms=11, color=STATUS_COLORS[c.status], zorder=3)
        ax.set_xlim(xmin, xmax); ax.set_yticks([]); ax.set_ylim(-1, 1)
        ax.set_title(f"{c.label}: {c.value_text}  ->  {c.status}", loc="left", fontsize=9)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


@_styled
def plot_probability(p_abnormal, threshold):
    fig, ax = plt.subplots(figsize=(7, 1.3))
    ax.barh([0], [1], color=T.SURFACE_2); ax.barh([0], [p_abnormal], color=T.ACCENT if p_abnormal > threshold else T.SECONDARY_WAVE)
    ax.axvline(threshold, color=T.TEXT, lw=1.5); ax.text(threshold, 0.62, f"threshold {threshold:g}", ha="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.9); ax.set_yticks([]); ax.set_xlabel("model P(Abnormal)")
    fig.tight_layout()
    return fig


@_styled
def plot_contributions(table, title="Local explanation (top features)"):
    t = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(t) + 1))
    ax.barh(t["feature"], t["contribution"], color=[T.ACCENT if v > 0 else T.SECONDARY_WAVE for v in t["contribution"]])
    ax.axvline(0, color=T.TEXT, lw=0.8); ax.set_title(title, fontsize=10)
    ax.set_xlabel("pushes towards Abnormal  >  0  <  pushes towards Normal")
    fig.tight_layout()
    return fig


@_styled
def plot_global_importance(df, title="Global permutation importance (test set)"):
    t = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1))
    ax.barh(t.index, t["importance"], xerr=t["std"], color=T.SECONDARY_WAVE)
    ax.set_title(title, fontsize=10); ax.set_xlabel("drop in ROC-AUC when the feature is shuffled")
    fig.tight_layout()
    return fig
