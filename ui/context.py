"""Session plumbing shared by all pages: model loading, upload parsing, and a per-recording analysis cache.
Analyses call the unchanged backend; results are cached in st.session_state keyed by a signature of ALL inputs
(file bytes, file options, demographics), so a changed input can never show a stale result."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import streamlit as st

from modules import explainability as EX
from modules.data_ingestion import ECGRecording
from modules.ingestion_extra import load_uploaded_ext as load_uploaded
from modules.ecg_delineation import delineate_consensus
from modules.prediction import ECGClassifier
from modules.reference_comparison import comparisons_from_morphology

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("ECG_MODEL_PATH", os.path.join(HERE, "models", "ptbxl_binary_model.joblib"))
STUB_PATH = os.path.join(HERE, "models", "test_stub", "ptbxl_binary_model.joblib")


def pick_model_path():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH, False
    if os.path.exists(STUB_PATH):
        return STUB_PATH, True
    return None, False


@st.cache_resource(show_spinner="Loading model ...")
def get_classifier(path: str):
    return ECGClassifier.from_path(path)


@st.cache_data(show_spinner=False)
def parse_uploads(files: tuple, opts: tuple):
    return load_uploaded(list(files), **dict(opts))


def signature(files: tuple, opts: tuple, demographics: dict) -> str:
    h = hashlib.sha256()
    for name, data in files:
        h.update(name.encode()); h.update(hashlib.sha256(data).digest())
    h.update(repr(opts).encode()); h.update(repr(sorted(demographics.items(), key=lambda kv: kv[0])).encode())
    return h.hexdigest()


@dataclass
class Analysis:
    """Results for one recording. Heavy parts are computed once, on first use."""
    name: str
    rec: ECGRecording
    res: object
    demographics: dict
    sex_txt: str | None
    clf: object = field(repr=False, default=None)
    _morph: object = field(default=None, repr=False)
    _cmp: object = field(default=None, repr=False)
    _expl: object = field(default=None, repr=False)
    _expl_done: bool = False
    _filt: object = field(default=None, repr=False)
    _triage: object = field(default=None, repr=False)

    def filtered(self):
        """Display-only band-pass at the file's own sampling rate (see ui/viewer.py)."""
        if self._filt is None:
            from .viewer import display_filtered
            self._filt = display_filtered(self.rec.signal, self.rec.fs)
        return self._filt

    def morph(self):
        if self._morph is None:
            self._morph = delineate_consensus(self.rec.signal, self.rec.fs)
        return self._morph

    def cmp(self):
        if self._cmp is None:
            m = self.morph()
            self._cmp = comparisons_from_morphology(m, self.sex_txt) if m.ok else []
        return self._cmp

    def triage(self):
        """Attention level + findings (ui/triage.py). Re-reads the three finished analyses; changes none of them."""
        if self._triage is None:
            from . import inspection as INS
            from . import triage as TR
            state, reasons = INS.validation_status(self.rec)
            self._triage = TR.compute_findings(self.res, self.morph(), self.cmp(), state, reasons)
        return self._triage

    def expl(self):
        if not self._expl_done:
            self._expl = EX.local_explanation(self.clf, self.res.feature_frame) if self.res.status == "ok" else None
            self._expl_done = True
        return self._expl

    def expl_available(self) -> bool:
        e = self.expl()
        return bool(e) and e.get("table") is not None


def run_analysis(clf, good: dict, demographics: dict, sex_txt) -> dict:
    out = {}
    for name, rec in good.items():
        res = clf.predict_recording(rec, demographics)
        res.record_id = name
        out[name] = Analysis(name, rec, res, demographics, sex_txt, clf)
    return out


# widget keys whose values must survive a change of page (Streamlit drops the state of widgets that are not on the current page)
PERSIST_KEYS = ("ei_role", "ei_setting", "ei_fs", "ei_units", "ei_std", "ei_derive", "ei_age", "ei_sex_w", "ei_height", "ei_weight", "ei_pick")


def keep_widget_state() -> None:
    for k in PERSIST_KEYS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def state_get(key, default=None):
    return st.session_state.get(key, default)


def is_analyzed() -> bool:
    return st.session_state.get("ei_analyzed_sig") is not None and st.session_state.get("ei_analyzed_sig") == st.session_state.get("ei_sig")


def current() -> Analysis | None:
    """Analysis of the selected recording if it is up to date, else None."""
    if not is_analyzed():
        return None
    return (st.session_state.get("ei_cache") or {}).get(st.session_state.get("ei_selected"))


def role() -> str | None:
    """'Doctor' or 'Student'; None until the reader has chosen on the Upload page."""
    return st.session_state.get("ei_role")


def process_inputs(clf, raw: tuple, opts: tuple, demographics: dict, sex_txt, setting: str, force: bool = False) -> None:
    """Parse the files, validate, and analyze every valid recording as soon as it is loaded. Results go to st.session_state."""
    from modules.data_ingestion import ECGRecording
    from . import history as HIST
    from . import inspection as INS
    if raw:
        loaded = parse_uploads(raw, opts)
        sig = signature(raw, opts, demographics)
        sizes = {n: len(b) for n, b in raw}
    else:
        loaded, sig, sizes = [], None, {}
    good = {n: o for n, o in loaded if isinstance(o, ECGRecording) and INS.validation_status(o)[0] != "FAIL"}
    st.session_state.update(ei_loaded=loaded, ei_good=good, ei_sig=sig, ei_demo=demographics, ei_sex=sex_txt, ei_sizes=sizes)
    pick = st.session_state.get("ei_pick")
    st.session_state["ei_selected"] = pick if pick in good else next(iter(good), None)
    if good and (force or st.session_state.get("ei_analyzed_sig") != sig):
        with st.spinner("Analyzing ..."):
            cache = run_analysis(clf, good, demographics, sex_txt)
            for a in cache.values():
                a.morph(); a.cmp(); a.triage()
        st.session_state["ei_cache"] = cache
        st.session_state["ei_analyzed_sig"] = sig
        for n, a in cache.items():
            HIST.add(HIST.make_entry(name=n, signature=sig, role=role() or "not chosen", setting=setting, demographics=demographics, sex_txt=sex_txt,
                                     triage=a.triage(), res=a.res, cmp=a.cmp()))
    elif not good:
        st.session_state["ei_analyzed_sig"] = None
        st.session_state["ei_cache"] = {}


def require_analysis(need_role: bool = True) -> Analysis:
    """For pages that need results: explain what is missing and stop instead of showing empty or stale numbers."""
    from . import theme as UI
    a = current()
    if a is None:
        if not st.session_state.get("ei_good"):
            UI.empty_state("No recording loaded", "Go to the Upload page and add an ECG file. It is analyzed automatically.")
        else:
            UI.empty_state("Inputs changed", "Open the Upload page: the recording is analyzed again when it is loaded there.")
        st.page_link("views/upload.py", label="Go to Upload")
        st.stop()
    if need_role and role() is None:
        UI.empty_state("Choose who is reading", "On the Upload page, choose Doctor or Student. This sets how the results and the report are written.")
        st.page_link("views/upload.py", label="Go to Upload")
        st.stop()
    return a
