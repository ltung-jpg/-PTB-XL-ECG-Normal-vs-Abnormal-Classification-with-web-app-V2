"""Assessment: A signal measurements, B model prediction, C published reference ranges, D dataset labels - kept separate."""
import pandas as pd
import streamlit as st

from modules import explainability as EX
from modules import visualization as V
from modules.reference_comparison import reference_table
from ui import context as X
from ui import inspection as INS
from ui import theme as UI

a = X.require_analysis()
res, morph, cmp = a.res, a.morph(), a.cmp()
st.markdown(f"## Assessment - {a.name}")
UI.notice("info", "Sections A, B and C are independent analyses. None of them is a diagnosis, and none changes another.")

# ------------------------------------------------------------------ A. measurements
with UI.card("A. Signal measurements", key="as-a"):
    if not morph.ok:
        UI.notice("warning", f"Not available: {morph.message}")
    else:
        rows = []
        for name, m in morph.measures.items():
            c = INS.measurement_card(name, m, morph.method)
            rows.append({"Measure": c["label"], "Value": "Not available" if c["value"] is None else c["value"], "Unit": c["unit"],
                         "Evidence": ("UNAVAILABLE", "unavailable") if c["value"] is None else (c["tag"], c["tag_kind"]),
                         "Validity": c["validity"], "Inter-lead spread": "" if m.spread is None or c["value"] is None else f"{m.spread:.3g} {c['unit']}"})
        UI.html_table(["Measure", "Value", "Unit", "Evidence", "Validity", "Inter-lead spread"], rows, badge_cols=("Evidence",))
        st.caption(f"{morph.method}; timing resolution {morph.resolution_ms:.0f} ms. Automatic estimates - not validated against expert annotations here.")
        for w in morph.warnings:
            UI.notice("warning", w)

# ------------------------------------------------------------------ B. ML prediction
with UI.card("B. Machine-learning prediction", key="as-b"):
    if res.status != "ok":
        UI.notice("error", f"No prediction is shown ({res.status}). The model input is not suitable:")
        for r in res.reasons:
            st.write("-", r)
    else:
        c1, c2 = st.columns(2)
        with c1:
            UI.kpi_card("Model output", res.label_name, tag="PREDICTED", tag_kind="predicted", note="Research model - not a diagnosis", key="label")
        with c2:
            UI.kpi_card("P(Abnormal)", f"{res.probability_abnormal:.3f}", tag="PREDICTED", tag_kind="predicted",
                        note=f"Raw model output (not a calibrated risk); decision threshold {res.threshold:g}", key="prob")
        st.pyplot(V.plot_probability(res.probability_abnormal, res.threshold))
        st.caption("'Abnormal' = any PTB-XL diagnosis other than normal sinus rhythm. This output says nothing about which condition, "
                   "if any, is present, and it must not be read as a statement about the person's health.")
    for w in res.warnings:
        UI.notice("warning", w)
    for n in res.notes:
        UI.notice("info", n)
    if res.feature_frame is not None:
        with st.expander("All 145 model input features (saved schema order)"):
            st.dataframe(res.feature_frame.T.rename(columns={0: "value"}), width="stretch")

# ------------------------------------------------------------------ C. published reference ranges
with UI.card("C. Comparison with published reference ranges", key="as-c"):
    if not cmp:
        UI.notice("info", "No waveform measurements are available to compare.")
    else:
        st.pyplot(V.plot_reference_comparison(cmp))
        label = {"within": ("WITHIN RANGE", "info"), "above": ("ABOVE RANGE", "needs_review"), "below": ("BELOW RANGE", "needs_review"),
                 "indeterminate": ("INDETERMINATE", "neutral")}
        _res = lambda c: ("NO RANGE", "neutral") if c.no_range else label[c.status]
        UI.html_table(["Measure", "Value", "Range", "Result", "Reference", "Comment"],
                      [{"Measure": c.label, "Value": c.value_text,
                        "Range": c.range_text,
                        "Result": _res(c), "Reference": c.reference_id or "", "Comment": c.reason or ""} for c in cmp],
                      badge_cols=("Result",))
        flagged = [c for c in cmp if c.status in ("above", "below")]
        UI.notice("info", "A value outside a range is a deviation from a published population range, not a diagnosis. "
                          + ("Outside the range: " + "; ".join(f"{c.label} ({c.status})" for c in flagged) if flagged else "No measure is clearly outside its range."))
        with st.expander("All configured references (traceable)"):
            st.dataframe(pd.DataFrame(reference_table()), width="stretch", hide_index=True)

# ------------------------------------------------------------------ D. dataset labels
with UI.card("D. Dataset reference labels", key="as-d"):
    st.markdown(UI.badge_html("UNAVAILABLE", "unavailable"), unsafe_allow_html=True)
    st.write("No dataset label was provided, and this version does not read them. Nothing here is compared with ground truth.")

# ------------------------------------------------------------------ explainability
with st.expander("Explainability (model only)"):
    gi = EX.global_importance(a.clf.bundle, 15)
    if gi is not None:
        st.pyplot(V.plot_global_importance(gi))
        st.caption(gi.attrs.get("method", ""))
        st.dataframe(EX.group_importance(a.clf.bundle), width="stretch")
    else:
        st.info("This model bundle contains no global importances.")
    expl = a.expl()
    if res.status == "ok" and expl and expl["table"] is not None:
        st.pyplot(V.plot_contributions(expl["table"]))
        st.caption(expl["method"])
        st.dataframe(expl["table"], width="stretch", hide_index=True)
        unusual = EX.unusual_features(a.clf, res.feature_frame)
        if len(unusual):
            UI.notice("warning", f"{len(unusual)} input features lie outside the 1st-99th percentile of the training data (input-quality hint).")
            st.dataframe(unusual, hide_index=True)
    elif res.status == "ok":
        st.info("SHAP explanation is not available for this model.")
    st.info(EX.DISCLAIMER)
