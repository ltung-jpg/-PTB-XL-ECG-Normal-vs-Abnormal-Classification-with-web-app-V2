"""Reports: self-contained HTML report with full provenance."""
import streamlit as st

from modules import visualization as V
from modules.report import build_report, html_to_pdf, image_html
from ui import context as X
from ui import theme as UI

a = X.require_analysis()
rec, res, morph, cmp = a.rec, a.res, a.morph(), a.cmp()
role, setting = X.role(), st.session_state.get("ei_setting", "Not specified")
tri = a.triage()
st.markdown(f"## Report - {a.name}")
with UI.card("Generate report", key="rp"):
    st.write("The report contains the recording and input details, the three independent analyses (model output, waveform measurements, "
             "published reference ranges), the model information and the limitations. It is generated only from the current analysis.")
    fmt = st.radio("Format", ["PDF", "HTML"], horizontal=True, key="rp_fmt", help="HTML is self-contained and always available; PDF is built with reportlab.")
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
        html_doc = (build_report(record_id=a.name, source_info=info, demographics=dict(a.demographics), prediction=res,
                                                                model_info=a.clf.model_info(), morph=morph, comparisons=cmp, figures=figs, explanation=expl,
                                                                commentary=None, triage=tri, role=role, setting=None if setting == "Not specified" else setting))
        st.session_state["report_html"] = (a.name, html_doc, None, None)          # the PDF is made only when the PDF format is chosen
rep = st.session_state.get("report_html")
if rep and rep[0] == a.name:
    if fmt == "PDF":
        if rep[2] is None and rep[3] is None:
            try:
                rep = (rep[0], rep[1], html_to_pdf(rep[1]), None)
            except Exception as exc:
                rep = (rep[0], rep[1], None, str(exc))
            st.session_state["report_html"] = rep
        if rep[3]:
            UI.notice("error", f"The PDF could not be created. {rep[3]}")
        if rep[2]:
            from modules import report_pdf as _rp
            if _rp.LAST_ENGINE:
                st.caption(f"PDF layout: {_rp.LAST_ENGINE}")
            st.download_button("Download report (PDF)", rep[2], file_name=f"ecg_report_{a.name}.pdf", mime="application/pdf", key="rp_dl")
    else:
        st.download_button("Download report (HTML)", rep[1], file_name=f"ecg_report_{a.name}.html", mime="text/html", key="rp_dl_html")
    with st.expander("Preview", expanded=True):
        st.components.v1.html(rep[1], height=700, scrolling=True)
