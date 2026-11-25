"""Signal Analysis: interactive ECG viewer and measurement cards."""
import streamlit as st

from modules import config as C
from ui import context as X
from ui import inspection as INS
from ui import theme as UI
from ui import viewer as VW

from modules.ecg_delineation import delineate


def lead_fiducials(a, leads):
    """Wave boundaries found on EACH shown lead by its own delineation (never copied from another lead). Cached per input."""
    cache = st.session_state.setdefault("ei_fid", {})
    out = {}
    for ld in leads:
        k = (st.session_state.get("ei_sig"), a.name, ld)
        if k not in cache:
            r = delineate(a.rec.signal, a.rec.fs, ld, a.rec.lead_names)
            cache[k] = r.fiducials if r.ok else None
        if cache[k] is not None:
            out[ld] = cache[k]
    return out


a = X.require_analysis()
rec, res = a.rec, a.res
state, _ = INS.validation_status(rec)

st.markdown(f"## Signal analysis - {a.name}")
st.markdown(UI.badge_html(state.replace("_", " "), INS.VALIDATION_KIND[state]) +
            f" &nbsp; {rec.source_format} &middot; {rec.original_fs:g} Hz &middot; {rec.duration_s:.1f} s &middot; {len(rec.lead_names)} leads",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- model output: its own box, above the viewer, full width
if res.status == "ok":
    ab = res.label_name == "Abnormal"
    UI.kpi_card("Model output", res.label_name, tag="PREDICTED", tag_kind="predicted", key="model",
                extra_chip=("MODEL: ABNORMAL", "fail") if ab else ("MODEL: NORMAL", "info"),
                abnormality={"title": "MODEL: ABNORMAL", "range": "", "high": False, "lines_title": "", "lines": [],
                             "deviation": "The research model classed this recording as Abnormal (any PTB-XL diagnosis other than normal sinus rhythm). "
                                          "It does not say which condition, and it is not a diagnosis."} if ab else None)
else:
    UI.kpi_card("Model output", None, tag="PREDICTED", tag_kind="predicted", key="model")
    UI.notice("error", f"No prediction is shown ({res.status}): " + "; ".join(res.reasons))

# ---------------------------------------------------------------- viewer controls
with UI.card("ECG viewer", key="viewer"):
    c1, c2, c3 = st.columns([2.4, 1.8, 1.8])
    with c1:
        leads = st.multiselect("Leads", rec.lead_names, default=["II"], key="sv_leads", disabled=st.session_state.get("sv_all", False))
        all12 = st.checkbox("Show all 12 leads", value=False, key="sv_all")
        leads = rec.lead_names if all12 else leads
    with c2:
        mode = st.radio("Signal", VW.MODES, key="sv_mode")
    with c3:
        show_peaks = st.checkbox("Mark R-peaks", value=True, key="sv_peaks")
        show_bounds = st.checkbox("Mark estimated wave boundaries", value=False, key="sv_bounds")
    dur = float(rec.duration_s)
    t0, t1 = st.slider("Time range (s)", 0.0, dur, (0.0, min(10.0, dur)), step=0.1, key="sv_window") if dur > 1 else (0.0, dur)
    if not leads:
        UI.notice("info", "Select at least one lead.")
    else:
        morph = a.morph()
        peaks_s = res.waveform_info.peaks / C.MODEL_FS if res.waveform_info is not None else None
        fig, shown = VW.build_figure(rec, a.filtered() if mode != "Raw" else None, leads, mode, t0, max(t1, t0 + 0.5), peaks_s=peaks_s,
                                     morph=morph, show_peaks=show_peaks, show_bounds=show_bounds,
                                     fiducials_by_lead=lead_fiducials(a, leads) if show_bounds else None)
        st.plotly_chart(fig, config=VW.plot_config(f"{a.name}_{'-'.join(leads)}"), width="stretch", key="sv_chart")
        st.caption("Zoom: drag a box or scroll; pan: shift-drag; double-click resets. Triangles = R-peaks, placed on the highest point of each "
                   "QRS in the lead shown; hover a dashed boundary or its diamond to read its time in seconds (estimated on each lead separately, not validated; a boundary is not drawn where that lead gives none).")
        if mode != "Raw":
            st.caption(VW.FILTER_NOTE)
        d1, d2, _sp = st.columns([1.15, 1.45, 3.4])
        with d1:
            st.download_button("Download chart only (PNG)", VW.export_png(rec, a.filtered() if mode != "Raw" else None, leads, mode, t0, max(t1, t0 + 0.5),
                                                                       peaks_s=peaks_s, title=f"{a.name}  |  {', '.join(leads)}  |  {t0:g}-{t1:g} s"),
                               file_name=f"{a.name}_{'-'.join(leads)}_{t0:g}-{t1:g}s.png", mime="image/png", key="sv_png",
                               help="Only the chart: the selected leads and time range, without the page or the numbers around it.")
        d2.download_button("Download visible samples (CSV)", VW.export_frame(rec, leads, t0, max(t1, t0 + 0.5)).to_csv(index=False),
                           file_name=f"{a.name}_{t0:g}-{t1:g}s.csv", mime="text/csv", key="sv_csv")
    for w in rec.warnings + (rec.quality.warnings if rec.quality else []):
        UI.notice("warning", w)

# ---------------------------------------------------------------- Key measurements (flagged ones first)
role = X.role()
st.markdown("### Key measurements")
morph = a.morph()
if not morph.ok:
    UI.notice("warning", f"Waveform measurements could not be obtained: {morph.message}")
by_measure = {c.measure: c for c in a.cmp()}
cards = [INS.peaks_card(res)] + ([INS.add_abnormality(INS.measurement_card(n, morph.measures.get(n), morph.method), by_measure.get(n), role)
                                  for n in INS.KPI_ORDER + INS.EXTRA_ORDER] if morph.ok else [])
cards = [{k: v for k, v in c.items() if not (k == "abnormality" and v is None)} for c in cards]
cards.sort(key=lambda c: 0 if c.get("abnormality") else 1)       # abnormal first (stable)
n_flag = sum(1 for c in cards if c.get("abnormality"))
if n_flag:
    st.caption(f"{n_flag} of {len(cards)} values show an abnormality. Open 'Abnormality' on a card.")
for i in range(0, len(cards), 3):
    cols = st.columns(3)
    for col, card in zip(cols, cards[i:i + 3]):
        with col:
            UI.kpi_card(**card)
