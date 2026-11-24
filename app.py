"""ECG Insight - Computer-Aided ECG Analysis Platform (Streamlit).   Run:  streamlit run app.py

Entry script: theme, header, model loading, the persistent sidebar (upload, options, patient data, Analyze) and the top navigation.
Pages live in views/. The model is loaded once (st.cache_resource) and is never retrained or modified by an upload.
"""
import streamlit as st

from modules.data_ingestion import ECGRecording
from modules.prediction import BundleError
from ui import context as X
from ui import inspection as INS
from ui import theme as UI

st.set_page_config(page_title="ECG Insight - ECG analysis", layout="wide", initial_sidebar_state="expanded")
UI.inject_css()
UI.header()

# ------------------------------------------------------------------------------------------------ model
path, using_stub = X.pick_model_path()
if path is None:
    st.error("No model found. Run the export cell of the notebook and put `ptbxl_binary_model.joblib` in `models/` "
             "(see README), or set ECG_MODEL_PATH.")
    st.stop()
try:
    clf = X.get_classifier(path)
except BundleError as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()
st.session_state["ei_clf"] = clf
if clf.is_test_stub or using_stub:
    st.error("TEST-STUB MODEL ACTIVE: trained on synthetic ECGs for software testing. Predictions are meaningless. "
             "Place the real exported model in `models/ptbxl_binary_model.joblib`.")
for w in clf.model_info()["load_warnings"]:
    st.warning(w)

# ------------------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    UI.section_label("Recording")
    c_upload, c_select, c_button, c_status = st.container(), st.container(), st.container(), st.container()
    with st.expander("File options (advanced)"):
        fs_in = st.number_input("Sampling frequency (Hz) - only for CSV/TXT without time column", min_value=0.0, value=0.0, step=1.0)
        unit_in = st.selectbox("Signal units - only if not stated in the file", ["(from file)", "mV", "uV", "V"])
        std_order = st.checkbox("CSV/TXT without lead names: columns are in standard order (I, II, III, aVR, aVL, aVF, V1-V6)")
        derive = st.checkbox("Derive III, aVR, aVL, aVF from I and II if missing (exact formulas; flagged in the report)")
    with st.expander("Patient information (optional)"):
        age = st.number_input("Age (years)", 1.0, 120.0, value=None, step=1.0, placeholder="Not provided")
        sex = st.selectbox("Sex", ["not provided", "male", "female"])
        height = st.number_input("Height (cm)", 1.0, 230.0, value=None, placeholder="Not provided")
        weight = st.number_input("Weight (kg)", 1.0, 250.0, value=None, placeholder="Not provided")
        st.caption("Missing values are never invented; the model's original median imputer handles them (and the result is flagged).")
    _given = [n for n, v in (("age", age), ("sex", None if sex == "not provided" else sex), ("height", height), ("weight", weight)) if v]
    st.caption("Patient information: " + (", ".join(_given) + " provided" if _given else "none provided"))

    with c_upload:
        files = st.file_uploader("ECG files (WFDB .hea + .dat pairs, CSV, TXT, EDF); several allowed", accept_multiple_files=True,
                                 type=["hea", "dat", "csv", "txt", "tsv", "edf"], key="ei_files")

demographics = {"age": age, "sex": None if sex == "not provided" else (0 if sex == "male" else 1), "height_cm": height, "weight_kg": weight}
sex_txt = None if sex == "not provided" else sex

# ------------------------------------------------------------------------------------------------ parse uploads
if files:
    opts = tuple(sorted({"fs": fs_in or None, "units": None if unit_in == "(from file)" else unit_in,
                         "assume_standard_order": std_order, "derive_limb_leads": derive}.items()))
    raw = tuple((f.name, f.getvalue()) for f in files)
    loaded = X.parse_uploads(raw, opts)
    sig = X.signature(raw, opts, demographics)
    sizes = {n: len(b) for n, b in raw}
else:
    loaded, sig, sizes = [], None, {}
# only recordings that pass validation are analyzed (FAIL = not fit for analysis, with the reason shown on the Upload page)
good = {n: o for n, o in loaded if isinstance(o, ECGRecording) and INS.validation_status(o)[0] != "FAIL"}
st.session_state.update(ei_loaded=loaded, ei_good=good, ei_sig=sig, ei_demo=demographics, ei_sex=sex_txt, ei_sizes=sizes)

with c_select:
    if len(good) > 1:
        picked = st.selectbox("Recording to inspect", list(good), key="ei_pick")
    else:
        picked = next(iter(good), None)
st.session_state["ei_selected"] = picked

with c_button:
    clicked = st.button("Analyze Recording", type="primary", key="analyze", disabled=not good, width="stretch")
if clicked:
    with st.spinner("Analyzing ..."):
        cache = X.run_analysis(clf, good, demographics, sex_txt)
        cache[picked].morph(); cache[picked].cmp()
    st.session_state["ei_cache"] = cache
    st.session_state["ei_analyzed_sig"] = sig
# ------------------------------------------------------------------------------------------------ navigation
pages = [st.Page("views/overview.py", title="Overview", url_path="overview", default=True),
         st.Page("views/upload.py", title="Upload", url_path="upload"),
         st.Page("views/signal_analysis.py", title="Signal Analysis", url_path="signal"),
         st.Page("views/assessment.py", title="Assessment", url_path="assessment"),
         st.Page("views/reports.py", title="Reports", url_path="reports")]
nav = st.navigation(pages, position="top")
with c_status:                    # (after st.navigation so that page links resolve)
    if not good:
        st.caption("No valid recording loaded.")
    elif X.is_analyzed():
        st.success("Analysis is up to date.")
        st.page_link("views/signal_analysis.py", label="Open Signal Analysis")
        st.page_link("views/assessment.py", label="Open Assessment")
    elif st.session_state.get("ei_analyzed_sig") is not None:
        st.caption("Files or settings changed - analyze again to refresh the results.")
    else:
        st.caption("Not analyzed yet.")
nav.run()
