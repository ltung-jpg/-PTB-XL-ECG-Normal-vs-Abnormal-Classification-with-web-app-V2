"""Assessment: key findings first, then A signal measurements, B model prediction, C published reference ranges (kept separate).
Sections that are normal are collapsed; the reader is told for each section whether it needs reading."""
import streamlit as st

from modules import explainability as EX
from modules import visualization as V
from modules.reference_comparison import REFERENCES
from ui import context as X
from ui import inspection as INS
from ui import theme as UI
from ui import triage as TR

a = X.require_analysis()
res, morph, cmp = a.res, a.morph(), a.cmp()
tri = a.triage()
role = st.session_state.get("ei_role", "Doctor")
doctor = role == "Doctor"

st.markdown(f"## Assessment - {a.name}")
UI.notice("info", "Sections A, B and C are independent analyses. None of them is a diagnosis, and none changes another.")


def head(title: str, status: tuple[str, str]) -> None:
    st.markdown(f'<div class="ei-cardtitle" style="font-size:1.2rem">{title} &nbsp; {UI.badge_html(*status)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ key findings
with UI.card("Key findings" + (" - what to verify" if doctor else " - what it means"), key="as-key"):
    flagged = [f for f in tri.findings if f.severity in ("high", "abnormal")]
    to_check = [f for f in tri.findings if f.severity == "check"]
    if not tri.findings:
        UI.notice("success", "None of the three analyses flagged anything. Nothing needs a detailed read in this section.")
    for f in flagged + to_check:
        kind = {"high": ("HIGH RISK", "fail"), "abnormal": ("FLAGGED", "fail"), "check": ("TO CHECK", "needs_review")}[f.severity]
        st.markdown(f'<div style="margin-top:.6rem"><b style="font-size:1.1rem">{f.title}</b> &nbsp; {UI.badge_html(*kind)} '
                    f'{UI.badge_html(f.source, "neutral")}</div><div class="ei-kpi-line">{f.detail}</div>', unsafe_allow_html=True)
        if doctor and f.doctor:
            st.markdown("\n".join(f"- {x}" for x in f.doctor))
        elif not doctor and f.student:
            st.markdown(f.student)
    st.caption(TR.DISCLAIMER)

# ------------------------------------------------------------------ A. measurements
with UI.card(None, key="as-a"):
    if not morph.ok:
        head("A. Signal measurements", ("NOT AVAILABLE", "unavailable"))
        UI.notice("warning", f"Not available: {morph.message}")
    else:
        by = {c.measure: c for c in cmp}
        status = TR.section_status(tri, "Measurement")
        head("A. Signal measurements", status)
        rows_ranged, rows_noref = [], []
        for name, m in morph.measures.items():
            c = INS.measurement_card(name, m, morph.method)
            cm = by.get(name)
            unit = c["unit"]
            value = "Not available" if c["value"] is None else f"{c['value']} {unit}"
            if cm is not None and not cm.no_range and c["value"] is not None:
                dv = TR.deviation(cm, unit)["text"]
                res_lab = {"above": ("ABOVE RANGE", "fail"), "below": ("BELOW RANGE", "fail"), "within": ("WITHIN RANGE", "info"),
                           "indeterminate": ("INDETERMINATE", "neutral")}[cm.status]
                rows_ranged.append((0 if cm.status in ("above", "below") else 1, {"Measure": c["label"], "Value": value, "Published range": f"{cm.range_text} {unit}",
                                                                                   "Result": res_lab, "Outside the range by": dv or "-"}))
            else:
                rows_noref.append({"Measure": c["label"], "Value": value, "Evidence": ("UNAVAILABLE", "unavailable") if c["value"] is None else (c["tag"], c["tag_kind"])})
        rows_ranged.sort(key=lambda r: r[0])
        UI.html_table(["Measure", "Value", "Published range", "Result", "Outside the range by"], [r for _, r in rows_ranged], badge_cols=("Result",))
        st.caption(f"{morph.method}; timing resolution {morph.resolution_ms:.0f} ms. Automatic estimates, not validated against expert annotations here. "
                   "A value outside a range is a deviation from a population range, not a diagnosis.")
        for w in morph.warnings:
            UI.notice("warning", w)
        with st.expander("Measures without a published range, and how each value was obtained", expanded=not doctor):
            if rows_noref:
                UI.html_table(["Measure", "Value", "Evidence"], rows_noref, badge_cols=("Evidence",))
            st.caption("No population reference range from a reliable published source was identified for these measures; they are shown for information only.")
            UI.html_table(["Measure", "Source", "Validity"], [{"Measure": INS.MEASURE_META.get(n, (n,))[0], "Source": INS.measurement_card(n, m, morph.method)["source"],
                                                              "Validity": INS.measurement_card(n, m, morph.method)["validity"]} for n, m in morph.measures.items()])

# ------------------------------------------------------------------ B. ML prediction
with UI.card(None, key="as-b"):
    if res.status != "ok":
        head("B. Machine-learning prediction", ("NO PREDICTION", "unavailable"))
        UI.notice("error", f"No prediction is shown ({res.status}). The model input is not suitable:")
        for r in res.reasons:
            st.write("-", r)
    else:
        head("B. Machine-learning prediction", ("MODEL: ABNORMAL", "fail") if res.label_name == "Abnormal" else ("MODEL: NORMAL", "info"))
        c1, c2 = st.columns(2)
        with c1:
            UI.kpi_card("Model output", res.label_name, tag="PREDICTED", tag_kind="predicted", note="Research model - not a diagnosis", key="label",
                        flag=("MODEL: ABNORMAL", "fail") if res.label_name == "Abnormal" else ("MODEL: NORMAL", "info"))
        with c2:
            UI.kpi_card("P(Abnormal)", f"{res.probability_abnormal:.3f}", tag="PREDICTED", tag_kind="predicted",
                        note=f"Raw model output (not a calibrated risk); decision threshold {res.threshold:g}", key="prob",
                        range_text=f"threshold {res.threshold:g}")
        st.pyplot(V.plot_probability(res.probability_abnormal, res.threshold))
        st.caption("'Abnormal' = any PTB-XL diagnosis other than normal sinus rhythm. It says nothing about which condition, if any, is present.")
    for w in res.warnings:
        UI.notice("warning", w)
    for n in res.notes:
        UI.notice("info", n)
    if res.feature_frame is not None:
        with st.expander("Model details: the 145 input features, importance and per-recording explanation"):
            st.dataframe(res.feature_frame.T.rename(columns={0: "value"}), width="stretch")
            gi = EX.global_importance(a.clf.bundle, 15)
            if gi is not None:
                st.pyplot(V.plot_global_importance(gi))
                st.caption(gi.attrs.get("method", ""))
            expl = a.expl()
            if expl and expl["table"] is not None:
                st.pyplot(V.plot_contributions(expl["table"]))
                st.caption(expl["method"])
                unusual = EX.unusual_features(a.clf, res.feature_frame)
                if len(unusual):
                    UI.notice("warning", f"{len(unusual)} input features lie outside the 1st-99th percentile of the training data (input-quality hint).")
                    st.dataframe(unusual, hide_index=True)
            st.info(EX.DISCLAIMER)

# ------------------------------------------------------------------ C. published reference ranges
with UI.card(None, key="as-c"):
    if not cmp:
        head("C. Comparison with published reference ranges", ("NOT AVAILABLE", "unavailable"))
        UI.notice("info", "No waveform measurements are available to compare.")
    else:
        flagged_c = [c for c in cmp if c.status in ("above", "below")]
        head("C. Comparison with published reference ranges", (f"{len(flagged_c)} OUTSIDE RANGE", "fail") if flagged_c else ("NONE OUTSIDE RANGE", "info"))
        if flagged_c:
            st.pyplot(V.plot_reference_comparison(cmp))
        else:
            st.write("Every measure that has a published range lies within it.")
        used = sorted({c.reference_id for c in cmp if c.reference_id})
        links = " &middot; ".join(UI.link_html(REFERENCES[r]["url"], REFERENCES[r]["citation"].split(".")[0][:60] + " (source)") for r in used if r in REFERENCES)
        st.markdown(f'<div class="ei-kpi-line">Sources (open in a new tab): {links}</div>', unsafe_allow_html=True)
        with st.expander("Full comparison table with comments", expanded=False):
            label = {"within": ("WITHIN RANGE", "info"), "above": ("ABOVE RANGE", "needs_review"), "below": ("BELOW RANGE", "needs_review"),
                     "indeterminate": ("INDETERMINATE", "neutral")}
            _res = lambda c: ("NO RANGE", "neutral") if c.no_range else label[c.status]
            UI.html_table(["Measure", "Value", "Range", "Result", "Source", "Comment"],
                          [{"Measure": c.label, "Value": c.value_text, "Range": c.range_text, "Result": _res(c),
                            "Source": UI.link_html(c.url, c.reference_id) if c.url else "", "Comment": c.reason or ""} for c in cmp],
                          badge_cols=("Result",), html_cols=("Source",))

# ------------------------------------------------------------------ D. dataset labels
with st.expander("D. Dataset reference labels"):
    st.markdown(UI.badge_html("UNAVAILABLE", "unavailable"), unsafe_allow_html=True)
    st.write("No dataset label was provided, and this version does not read them. Nothing here is compared with ground truth.")
