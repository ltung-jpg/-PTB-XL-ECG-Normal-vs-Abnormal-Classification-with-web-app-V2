"""Signal Analysis: interactive ECG viewer and measurement cards."""
import streamlit as st

from modules import config as C
from ui import context as X
from ui import inspection as INS
from ui import theme as UI
from ui import viewer as VW

a = X.require_analysis()
rec, res = a.rec, a.res
state, _ = INS.validation_status(rec)

st.markdown(f"## Signal analysis - {a.name}")
st.markdown(UI.badge_html(state.replace("_", " "), INS.VALIDATION_KIND[state]) +
            f" &nbsp; {rec.source_format} &middot; {rec.original_fs:g} Hz &middot; {rec.duration_s:.1f} s &middot; {len(rec.lead_names)} leads",
            unsafe_allow_html=True)

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
    t0, t1 = st.slider("Time window (s)", 0.0, dur, (0.0, min(10.0, dur)), step=0.1, key="sv_window") if dur > 1 else (0.0, dur)
    if not leads:
        UI.notice("info", "Select at least one lead.")
    else:
        morph = a.morph()
        peaks_s = res.waveform_info.peaks / C.MODEL_FS if res.waveform_info is not None else None
        fig, shown = VW.build_figure(rec, a.filtered() if mode != "Raw" else None, leads, mode, t0, max(t1, t0 + 0.5), peaks_s=peaks_s,
                                     morph=morph, show_peaks=show_peaks, show_bounds=show_bounds)
        st.plotly_chart(fig, config=VW.plot_config(f"{a.name}_{'-'.join(leads)}"), width="stretch", key="sv_chart")
        st.caption("Zoom: drag a box or scroll; pan: shift-drag or the toolbar; double-click resets. Camera icon = PNG export. "
                   "Triangles = R-peaks of the model's detector (100 Hz grid); dashed lines = estimated wave boundaries (not validated).")
        if mode != "Raw":
            st.caption(VW.FILTER_NOTE)
        st.download_button("Download visible samples (CSV)", VW.export_frame(rec, leads, t0, max(t1, t0 + 0.5)).to_csv(index=False),
                           file_name=f"{a.name}_{t0:g}-{t1:g}s.csv", mime="text/csv", key="sv_csv")
    for w in rec.warnings + (rec.quality.warnings if rec.quality else []):
        UI.notice("warning", w)

# ---------------------------------------------------------------- KPI cards
st.markdown("### Measurements")
morph = a.morph()
if not morph.ok:
    UI.notice("warning", f"Waveform measurements could not be obtained: {morph.message}")
cards = [INS.peaks_card(res)] + ([INS.measurement_card(n, morph.measures.get(n), morph.method) for n in INS.KPI_ORDER] if morph.ok else [])
for i in range(0, len(cards), 3):
    cols = st.columns(3)
    for col, card in zip(cols, cards[i:i + 3]):
        with col:
            UI.kpi_card(**card)
UI.notice("info", "MEASURED = read from the signal by a detector. ESTIMATED = automatic delineation, not validated against expert annotation. "
                  "DERIVED = computed from other values and inherits their uncertainty. Not available = could not be obtained.")
