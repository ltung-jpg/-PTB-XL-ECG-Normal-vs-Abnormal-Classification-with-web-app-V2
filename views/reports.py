"""Reports: self-contained HTML report with full provenance."""
import streamlit as st

from modules import visualization as V
from modules.report import build_report, image_html
from ui import context as X
from ui import theme as UI

a = X.require_analysis()
rec, res, morph, cmp = a.rec, a.res, a.morph(), a.cmp()
st.markdown(f"## Report - {a.name}")
with UI.card("Generate report", key="rp"):
    st.write("The report contains the recording and input details, the three independent analyses (model output, waveform measurements, "
             "published reference ranges), the model information and the limitations. It is generated only from the current analysis.")
    st.caption("Format: HTML (self-contained). PDF export is not available in this version.")
    if st.button("Generate report", type="primary", key="rp_btn"):
        expl = a.expl()
        figs = {"ecg": image_html(V.plot_12_lead(rec.signal, rec.fs, title=a.name, missing_mask=rec.missing_mask))}
        if res.status == "ok":
            figs["probability"] = image_html(V.plot_probability(res.probability_abnormal, res.threshold))
            if expl and expl["table"] is not None:
                figs["explanation"] = image_html(V.plot_contributions(expl["table"]))
        if morph.ok:
            figs["delineation"] = image_html(V.plot_beat_with_fiducials(morph))
        if cmp:
            figs["reference"] = image_html(V.plot_reference_comparison(cmp))
        info = {"Source format": rec.source_format, "Original sampling rate (Hz)": rec.original_fs, "Duration (s)": round(rec.duration_s, 1),
                "Original units": rec.original_units, "Lead names in file": ", ".join(rec.original_lead_names),
                "Derived leads": ", ".join(rec.derived_leads) or "none", "Input signature (sha256, first 12)": (st.session_state.get("ei_sig") or "")[:12]}
        st.session_state["report_html"] = (a.name, build_report(record_id=a.name, source_info=info, demographics=dict(a.demographics), prediction=res,
                                                                model_info=a.clf.model_info(), morph=morph, comparisons=cmp, figures=figs, explanation=expl))
rep = st.session_state.get("report_html")
if rep and rep[0] == a.name:
    st.download_button("Download report (HTML)", rep[1], file_name=f"ecg_report_{a.name}.html", mime="text/html", key="rp_dl")
    with st.expander("Preview", expanded=True):
        st.components.v1.html(rep[1], height=700, scrolling=True)
