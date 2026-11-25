"""Upload: files, file options, optional patient information, validation, field states and the mandatory 'Who is reading' choice."""
import os

import streamlit as st

from modules.prediction import ECGClassifier
from ui import context as X
from ui import inspection as INS
from ui import theme as UI
from ui import triage as TR

st.markdown("## Upload")
clf = st.session_state["ei_clf"]

# ---- who is reading (required)
with UI.card("Who is reading", key="up-role"):
    st.radio("Choose one to continue", TR.ROLES, index=None, horizontal=True, key="ei_role",
             help="Doctor: the results list what to verify next. Student: they explain what each flagged value means. "
                  "This also sets how the report is written.")
    if X.role() is None:
        UI.notice("warning", "Choose Doctor or Student. The Signal Analysis and Reports pages stay locked until you do.")
    else:
        UI.notice("success", f"Reading as: {X.role()}.")

SETTINGS = ["Not specified", "Ambulance (pre-hospital)", "Emergency department", "Ward or clinic"]

left, right = st.columns([3, 2], gap="large")
with left:
    with UI.card("ECG files", key="up-files"):
        files = st.file_uploader("WFDB (.hea + .dat), CSV, TXT, EDF, MAT or XML; several files allowed", accept_multiple_files=True,
                                 type=["hea", "dat", "csv", "txt", "tsv", "edf", "mat", "xml"], key="ei_files")
        names = [f.name for f in files] if files else []
        raw_prev = st.session_state.get("ei_raw") or ()
        if st.session_state.get("ei_up_last") is None:              # just arrived on this page: an empty uploader does not mean 'remove'
            if files:
                st.session_state["ei_raw"] = tuple((f.name, f.getvalue()) for f in files)
        elif names != st.session_state["ei_up_last"]:                # the reader added or removed files on this page
            st.session_state["ei_raw"] = tuple((f.name, f.getvalue()) for f in files)
        st.session_state["ei_up_last"] = names
        raw = st.session_state.get("ei_raw") or ()
        if raw and not files:
            st.caption("Loaded earlier: " + ", ".join(n for n, _ in raw))
            if st.button("Remove the loaded files", key="ei_clear"):
                st.session_state["ei_raw"] = ()
                st.rerun()
        with st.expander("File options (advanced)"):
            fs_in = st.number_input("Sampling frequency (Hz) - only for CSV/TXT without time column", min_value=0.0, value=0.0, step=1.0, key="ei_fs")
            unit_in = st.selectbox("Signal units - only if not stated in the file", ["(from file)", "mV", "uV", "V"], key="ei_units")
            std_order = st.checkbox("CSV/TXT without lead names: columns are in standard order (I, II, III, aVR, aVL, aVF, V1-V6)", key="ei_std")
            derive = st.checkbox("Derive III, aVR, aVL, aVF from I and II if missing (exact formulas; flagged in the report)", key="ei_derive")
with right:
    with UI.card("Patient information (optional)", key="up-patient"):
        setting = st.selectbox("Recording made in", SETTINGS, key="ei_setting",
                               help="Only used for the history. It does not change the model or any measurement.")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 1.0, 120.0, value=None, step=1.0, format="%.0f", placeholder="Not provided", key="ei_age")
            height = st.number_input("Height (cm)", 1.0, 230.0, value=None, step=1.0, format="%.0f", placeholder="Not provided", key="ei_height")
        with c2:
            sex = st.selectbox("Sex", ["not provided", "male", "female"], key="ei_sex_w")
            weight = st.number_input("Weight (kg)", 1.0, 250.0, value=None, step=1.0, format="%.1f", placeholder="Not provided", key="ei_weight")
        st.caption("Missing values are never invented; the model's original median imputer handles them (and the result is flagged).")

demographics = {"age": age, "sex": None if sex == "not provided" else (0 if sex == "male" else 1), "height_cm": height, "weight_kg": weight}
sex_txt = None if sex == "not provided" else sex
opts = tuple(sorted({"fs": fs_in or None, "units": None if unit_in == "(from file)" else unit_in,
                     "assume_standard_order": std_order, "derive_limb_leads": derive}.items()))
X.process_inputs(clf, tuple(raw), opts, demographics, sex_txt, setting)

loaded = st.session_state.get("ei_loaded") or []
if not loaded:
    UI.empty_state("No recording loaded", "Add a file above. WFDB needs both the .hea and the .dat file of a record.")
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
if len(good) > 1:
    st.selectbox("Recording to inspect", list(good), key="ei_pick")
    st.session_state["ei_selected"] = st.session_state["ei_pick"]
sel = st.session_state["ei_selected"]

# ---- field states of the selected recording
with UI.card(f"Field states - {sel}", key="fields"):
    UI.html_table(["Field", "State", "Value", "Note"], INS.field_states(good[sel], st.session_state["ei_demo"]), badge_cols=("State",))

if len(good) > 1 and st.session_state.get("ei_cache"):
    with UI.card("Batch overview (model output only)", key="batch"):
        st.dataframe(ECGClassifier.summarize([x.res for x in st.session_state["ei_cache"].values()]), width="stretch", hide_index=True)

if X.role() is not None:
    st.page_link("views/signal_analysis.py", label="Open Signal Analysis")
