"""Interactive ECG viewer (Plotly). DISPLAY ONLY: nothing here feeds the analysis.

* raw = the samples of the file converted to mV (exactly what ingestion returned)
* filtered = zero-phase 0.5-40 Hz band-pass computed AT THE FILE'S OWN SAMPLING RATE for display (the model uses its own 100 Hz pipeline)
* long records: the visible window is min/max-decimated for display so that peaks are never smoothed away
* R-peaks (marker) and estimated wave boundaries (dashed lines) are drawn with different shapes AND colours
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules import config as C
from modules.preprocessing import preprocess_signal

from . import tokens as T

MODES = ("Raw", "Filtered", "Raw vs filtered")
MAX_POINTS = 4000
FILTER_NOTE = ("Filtered view: zero-phase 0.5-40 Hz band-pass applied at the file's own sampling rate, for display only. "
               "The model uses a separate 100 Hz pipeline.")


def decimate_minmax(t: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS):
    """Display-only reduction that keeps the min AND max of every bucket (so R-peaks survive). Identity if short enough."""
    n = len(t)
    if n <= max_points:
        return t, y
    n_b = max_points // 2
    edges = np.linspace(0, n, n_b + 1).astype(int)
    tt, yy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        if np.all(np.isnan(seg)):
            tt += [t[a], t[b - 1]]; yy += [np.nan, np.nan]; continue
        i0, i1 = int(np.nanargmin(seg)), int(np.nanargmax(seg))
        for i in sorted({i0, i1}):
            tt.append(t[a + i]); yy.append(seg[i])
    return np.asarray(tt), np.asarray(yy)


def display_filtered(sig_mv: np.ndarray, fs: float) -> np.ndarray:
    return preprocess_signal(np.nan_to_num(sig_mv), fs)


def snap_r_peaks(t: np.ndarray, y: np.ndarray, peaks_s, half_window_s: float = 0.10) -> np.ndarray:
    """DISPLAY ONLY. Move each detected R-peak time onto the true R extremum of the lead that is being shown.

    The detector works on one lead (II, I, V5 or V2) on a 100 Hz grid. Drawn as is on another lead, a marker can sit on the
    first small bump of a notched QRS, or on the wrong side of a small or inverted complex (typical in lead III). Here each
    marker is re-placed on the highest extremum of the lead's dominant polarity within +/- half_window_s of the detected time,
    then walked uphill so it never stops on a shoulder. The detector, the 145 model features and the measurements are unchanged."""
    peaks_s = np.asarray(peaks_s, float)
    if len(t) < 5 or len(peaks_s) == 0 or not np.isfinite(y).any():
        return peaks_s
    fs = (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 1.0
    yy = np.nan_to_num(np.asarray(y, float), nan=float(np.nanmedian(y)))
    base = float(np.median(yy))
    hw = max(1, int(round(half_window_s * fs)))
    idx = np.clip(np.round((peaks_s - t[0]) * fs).astype(int), 0, len(t) - 1)
    ext = []
    for i in idx:
        a, b = max(0, i - hw), min(len(t), i + hw + 1)
        seg = yy[a:b] - base
        ext.append(a + int(np.argmax(np.abs(seg))))
    ext = np.asarray(ext)
    sign = 1.0 if np.median(np.sign(yy[ext] - base)) >= 0 else -1.0       # dominant polarity of this lead
    out = []
    for i in idx:
        a, b = max(0, i - hw), min(len(t), i + hw + 1)
        j = a + int(np.argmax(sign * (yy[a:b] - base)))
        while j + 1 < len(yy) and sign * yy[j + 1] > sign * yy[j]:            # walk uphill: never stop on a shoulder
            j += 1
        while j - 1 >= 0 and sign * yy[j - 1] > sign * yy[j]:
            j -= 1
        out.append(j)
    out = sorted(set(out))
    # a marker is kept only where there is a real complex: drop extrema far smaller than the typical one (no clear peak there),
    # and when two markers fall within 0.25 s keep the larger one (two markers on one spike)
    amp = np.array([sign * (yy[j] - base) for j in out])
    if len(out) >= 3:
        keep = amp >= 0.35 * np.median(amp)
        out = [j for j, k in zip(out, keep) if k]
    final = []
    for j in out:
        if final and t[j] - t[final[-1]] < 0.25:
            if sign * yy[j] > sign * yy[final[-1]]:
                final[-1] = j
        else:
            final.append(j)
    return t[np.asarray(final, int)]


def window_slice(fs: float, n: int, t0: float, t1: float) -> slice:
    return slice(max(0, int(round(t0 * fs))), min(n, int(round(t1 * fs))))


def build_figure(rec, filtered, leads: list[str], mode: str, t0: float, t1: float, peaks_s=None, morph=None,
                 show_peaks: bool = True, show_bounds: bool = True, fiducials_by_lead: dict | None = None):
    """Return (figure, displayed_traces) where displayed_traces maps (lead, kind) -> (t, y) exactly as plotted."""
    n_rows = max(1, len(leads))
    height = int(np.clip(110 * n_rows + 120, 420, 520)) if n_rows <= 3 else 80 * n_rows + 130
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=min(0.06, 0.25 / n_rows))
    sl = window_slice(rec.fs, rec.signal.shape[0], t0, t1)
    t = rec.time_s[sl]
    shown = {}
    for r, lead in enumerate(leads, 1):
        k = rec.lead_names.index(lead)
        raw = rec.signal[sl, k]
        traces = []
        if mode in ("Raw", "Raw vs filtered"):
            traces.append(("raw", raw, T.SECONDARY_WAVE if mode == "Raw vs filtered" else T.ECG_LINE, "Raw" if mode == "Raw vs filtered" else lead))
        if mode in ("Filtered", "Raw vs filtered") and filtered is not None:
            traces.append(("filtered", filtered[sl, k], T.ECG_LINE, "Filtered" if mode == "Raw vs filtered" else lead))
        for kind, y, color, name in traces:
            td, yd = decimate_minmax(t, y)
            shown[(lead, kind)] = (td, yd)
            fig.add_trace(go.Scatter(x=td, y=yd, mode="lines", name=name, line=dict(color=color, width=1.3 if kind == "filtered" or mode != "Raw vs filtered" else 1.0),
                                     legendgroup=kind, showlegend=(r == 1 and mode == "Raw vs filtered"),
                                     hovertemplate=f"{lead} {kind}<br>%{{x:.3f}} s<br>%{{y:.3f}} mV<extra></extra>"), row=r, col=1)
        ref_y = (filtered[sl, k] if (mode != "Raw" and filtered is not None) else raw)
        if show_peaks and peaks_s is not None and len(peaks_s):
            pk = peaks_s[(peaks_s >= t[0] - 0.1) & (peaks_s <= t[-1] + 0.1)] if len(t) else []
            if len(pk):
                pk = snap_r_peaks(t, ref_y, pk)
                pk = pk[(pk >= t[0]) & (pk <= t[-1])]
            if len(pk):
                fig.add_trace(go.Scatter(x=pk, y=np.interp(pk, t, ref_y), mode="markers", name="R-peaks", legendgroup="peaks",
                                         showlegend=(r == 1), marker=dict(symbol="triangle-down", size=10, color=T.R_PEAK, line=dict(color=T.TEXT_2, width=1)),
                                         hovertemplate="R-peak<br>%{x:.3f} s<extra></extra>"), row=r, col=1)
        fid = (fiducials_by_lead or {}).get(lead)
        if fid is None and morph is not None and morph.ok and lead == morph.lead.split("(display: ")[-1].rstrip(")"):
            fid = morph.fiducials
        if show_bounds and fid is not None:
            ymin, ymax = np.nanmin(ref_y), np.nanmax(ref_y)
            pad = 0.08 * (ymax - ymin or 1.0)
            names = {"p_onset": "P onset", "p_offset": "P offset", "qrs_onset": "QRS onset", "qrs_offset": "QRS offset", "t_onset": "T onset", "t_offset": "T offset"}
            first = not any(getattr(t, "legendgroup", "") == "bounds" and t.showlegend for t in fig.data)      # one legend entry for all leads
            for key, nm in names.items():
                xs, ys, cd = [], [], []
                for v in fid[key]:
                    if np.isfinite(v) and t[0] <= v <= t[-1]:
                        xs += [v, v, v, None]; ys += [ymin - pad, float(np.interp(v, t, ref_y)), ymax + pad, None]; cd += [nm, nm, nm, None]
                if xs:
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Estimated wave boundaries", legendgroup="bounds", showlegend=first,
                                             customdata=cd, line=dict(color=T.ANNOTATION, width=1, dash="dash"),
                                             marker=dict(symbol="diamond", size=7, color=T.ANNOTATION),
                                             hovertemplate="%{customdata} (estimated)<br>%{x:.3f} s<extra></extra>"), row=r, col=1)
                    first = False
        fig.update_yaxes(title_text=f"{lead} (mV)", row=r, col=1, gridcolor=T.GRID_MAJOR, minor=dict(showgrid=True, gridcolor=T.GRID_MINOR),
                         zeroline=False, linecolor=T.BORDER, title_font=dict(size=12, color=T.TEXT_2))
    fig.update_xaxes(gridcolor=T.GRID_MAJOR, minor=dict(showgrid=True, gridcolor=T.GRID_MINOR), zeroline=False, linecolor=T.BORDER)
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1, range=[t0, t1])
    fig.update_layout(height=height, paper_bgcolor=T.CHART_BG, plot_bgcolor=T.CHART_BG, margin=dict(l=70, r=20, t=80, b=50),
                      font=dict(family=T.FONT_STACK, color=T.TEXT, size=13), dragmode="zoom", hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(255,255,255,0)"))
    return fig, shown


def export_png(rec, filtered, leads: list[str], mode: str, t0: float, t1: float, peaks_s=None, title: str = "") -> bytes:
    """PNG of the CHART ONLY (the selected leads, the visible time range, R-peak markers): no page, sidebar or numbers around it."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sl = window_slice(rec.fs, rec.signal.shape[0], t0, t1)
    t = rec.time_s[sl]
    n = max(1, len(leads))
    fig, axes = plt.subplots(n, 1, figsize=(11, max(2.2, 1.9 * n)), sharex=True, squeeze=False)
    for ax, lead in zip(axes[:, 0], leads):
        k = rec.lead_names.index(lead)
        use_f = mode != "Raw" and filtered is not None
        if mode in ("Raw", "Raw vs filtered") or not use_f:
            ax.plot(t, rec.signal[sl, k], color=T.SECONDARY_WAVE if mode == "Raw vs filtered" else T.ECG_LINE, lw=1.0, label="Raw")
        if use_f:
            ax.plot(t, filtered[sl, k], color=T.ECG_LINE, lw=1.1, label="Filtered")
        ref = filtered[sl, k] if use_f else rec.signal[sl, k]
        if peaks_s is not None and len(peaks_s) and len(t):
            pk = snap_r_peaks(t, ref, np.asarray(peaks_s)[(np.asarray(peaks_s) >= t[0] - 0.1) & (np.asarray(peaks_s) <= t[-1] + 0.1)])
            pk = pk[(pk >= t[0]) & (pk <= t[-1])]
            ax.plot(pk, np.interp(pk, t, ref), "v", color=T.R_PEAK, markeredgecolor=T.TEXT_2, ms=7, ls="none")
        ax.set_ylabel(f"{lead} (mV)"); ax.grid(True, color=T.GRID_MAJOR, lw=.6); ax.margins(x=0)
    axes[-1, 0].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150); plt.close(fig)
    return buf.getvalue()


def plot_config(filename: str) -> dict:
    return {"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2}}


def export_frame(rec, leads: list[str], t0: float, t1: float) -> pd.DataFrame:
    """Native-rate samples of the selected leads in the visible window (mV)."""
    sl = window_slice(rec.fs, rec.signal.shape[0], t0, t1)
    df = pd.DataFrame({"time_s": rec.time_s[sl]})
    for lead in leads:
        df[f"{lead}_mV"] = rec.signal[sl, rec.lead_names.index(lead)]
    return df
