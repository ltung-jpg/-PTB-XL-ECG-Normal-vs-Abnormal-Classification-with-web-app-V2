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


def window_slice(fs: float, n: int, t0: float, t1: float) -> slice:
    return slice(max(0, int(round(t0 * fs))), min(n, int(round(t1 * fs))))


def build_figure(rec, filtered, leads: list[str], mode: str, t0: float, t1: float, peaks_s=None, morph=None,
                 show_peaks: bool = True, show_bounds: bool = True):
    """Return (figure, displayed_traces) where displayed_traces maps (lead, kind) -> (t, y) exactly as plotted."""
    n_rows = max(1, len(leads))
    height = int(np.clip(110 * n_rows + 70, 360, 480)) if n_rows <= 3 else 80 * n_rows + 80
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
            pk = peaks_s[(peaks_s >= t[0]) & (peaks_s <= t[-1])] if len(t) else []
            if len(pk):
                fig.add_trace(go.Scatter(x=pk, y=np.interp(pk, t, ref_y), mode="markers", name="R-peaks (model detector)", legendgroup="peaks",
                                         showlegend=(r == 1), marker=dict(symbol="triangle-down", size=10, color=T.R_PEAK, line=dict(color=T.TEXT_2, width=1)),
                                         hovertemplate="R-peak<br>%{x:.3f} s<extra></extra>"), row=r, col=1)
        if show_bounds and morph is not None and morph.ok and lead == morph.lead.split("(display: ")[-1].rstrip(")"):
            ymin, ymax = np.nanmin(ref_y), np.nanmax(ref_y)
            pad = 0.08 * (ymax - ymin or 1.0)
            groups = (("P-wave boundaries", ("p_onset", "p_offset"), T.ANNOTATION), ("QRS boundaries", ("qrs_onset", "qrs_offset"), T.ANNOTATION),
                      ("T-wave boundaries", ("t_onset", "t_offset"), T.ANNOTATION))
            for gi, (gname, keys, col) in enumerate(groups):
                xs, ys = [], []
                for key in keys:
                    for v in morph.fiducials[key]:
                        if np.isfinite(v) and t[0] <= v <= t[-1]:
                            xs += [v, v, None]; ys += [ymin - pad, ymax + pad, None]
                if xs:
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Estimated wave boundaries", legendgroup="bounds",
                                             showlegend=(gi == 0), line=dict(color=col, width=1, dash="dash"), hoverinfo="skip"), row=r, col=1)
        fig.update_yaxes(title_text=f"{lead} (mV)", row=r, col=1, gridcolor=T.GRID_MAJOR, minor=dict(showgrid=True, gridcolor=T.GRID_MINOR),
                         zeroline=False, linecolor=T.BORDER, title_font=dict(size=12, color=T.TEXT_2))
    fig.update_xaxes(gridcolor=T.GRID_MAJOR, minor=dict(showgrid=True, gridcolor=T.GRID_MINOR), zeroline=False, linecolor=T.BORDER)
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1, range=[t0, t1])
    fig.update_layout(height=height, paper_bgcolor=T.CHART_BG, plot_bgcolor=T.CHART_BG, margin=dict(l=70, r=20, t=30, b=50),
                      font=dict(family=T.FONT_STACK, color=T.TEXT, size=13), dragmode="zoom", hovermode="closest",
                      legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(255,255,255,0)"))
    return fig, shown


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
