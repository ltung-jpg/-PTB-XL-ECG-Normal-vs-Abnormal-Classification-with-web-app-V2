"""Upload and data inspection: validation badges, field states, eligibility."""
import os

import streamlit as st

from modules import config as C
from modules.prediction import ECGClassifier
from ui import context as X
from ui import inspection as INS
from ui import theme as UI

st.markdown("## Upload and data inspection")
loaded = st.session_state.get("ei_loaded") or []
if not loaded:
    UI.empty_state("No recording loaded", "Add a file in the sidebar. WFDB needs both the .hea and the .dat file of a record.")
    st.write("")
    UI.workflow([("Upload", "WFDB pair, CSV, TXT or EDF"), ("Inspect", "Validation and field states appear here"), ("Analyze", "Press Analyze Recording")])
    st.stop()

# ---- validation overview
rows, details = [], []
for name, obj in loaded:
    state, reasons = INS.validation_status(obj)
    ok = hasattr(obj, "signal")
    rows.append({"Recording": name, "Validation": (state.replace("_", " "), INS.VALIDATION_KIND[state]),
                 "Format": obj.source_format if ok else "", "Sampling rate": f"{obj.original_fs:g} Hz" if ok else "",
                 "Duration": f"{obj.duration_s:.1f} s" if ok else "", "Size": f"{sum(sz for f, sz in st.session_state['ei_sizes'].items() if f == name or os.path.splitext(f)[0] == name) / 1024:.1f} KB"})
    if reasons:
        details.append((name, state, reasons))
with UI.card("Validation", key="validation"):
    UI.html_table(["Recording", "Validation", "Format", "Sampling rate", "Duration", "Size"], rows, badge_cols=("Validation",))
    for name, state, reasons in details:
        UI.notice("error" if state == "FAIL" else "warning", f"{name}: " + " | ".join(reasons))

good = st.session_state.get("ei_good") or {}
if not good:
    st.stop()

# ---- field states of the selected recording
sel = st.session_state["ei_selected"]
with UI.card(f"Field states - {sel}", key="fields"):
    UI.html_table(["Field", "State", "Value", "Note"], INS.field_states(good[sel], st.session_state["ei_demo"]), badge_cols=("State",))

# ---- eligibility (needs the analysis)
a = X.current()
with UI.card("Analysis eligibility", key="eligibility"):
    if a is None:
        st.write("Eligibility of each analysis is evaluated when you press **Analyze Recording**.")
    else:
        UI.html_table(["Analysis", "Status", "Reason"], INS.eligibility(a.rec, a.res, a.morph(), a.cmp(), a.expl_available()), badge_cols=("Status",))

if a is not None and len(good) > 1:
    with UI.card("Batch overview (model output only)", key="batch"):
        results = [x.res for x in st.session_state["ei_cache"].values()]
        st.dataframe(ECGClassifier.summarize(results), width="stretch", hide_index=True)
