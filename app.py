"""ECG Insight - Computer-Aided ECG Analysis Platform (Streamlit).   Run:  streamlit run app.py

Entry script: theme, header, model loading and the top navigation. There is no sidebar: every control sits on the page where it is needed
(uploading, file options, patient information and "Who is reading" are on the Upload page).
Pages live in views/. The model is loaded once (st.cache_resource) and is never retrained or modified by an upload.
"""
import streamlit as st

from modules.prediction import BundleError
from ui import context as X
from ui import theme as UI

st.set_page_config(page_title="ECG Insight - ECG analysis", layout="wide", initial_sidebar_state="collapsed")
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

X.keep_widget_state()                                  # widget values survive a change of page
pages = [st.Page("views/overview.py", title="Overview", url_path="overview", default=True),
         st.Page("views/upload.py", title="Upload", url_path="upload"),
         st.Page("views/signal_analysis.py", title="Signal Analysis", url_path="signal"),
         st.Page("views/reports.py", title="Reports", url_path="reports"),
         st.Page("views/history.py", title="History", url_path="history")]
nav = st.navigation(pages, position="top")
if st.session_state.get("ei_last_page") != nav.title:   # arriving on a page: the uploader there starts empty (files are kept in ei_raw)
    st.session_state["ei_last_page"] = nav.title
    st.session_state["ei_up_last"] = None
nav.run()
