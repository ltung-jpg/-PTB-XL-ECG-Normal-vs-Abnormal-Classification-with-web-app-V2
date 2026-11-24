"""Session plumbing shared by all pages: model loading, upload parsing, and a per-recording analysis cache.
Analyses call the unchanged backend; results are cached in st.session_state keyed by a signature of ALL inputs
(file bytes, file options, demographics), so a changed input can never show a stale result."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import streamlit as st

from modules import explainability as EX
from modules.data_ingestion import ECGRecording, load_uploaded
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


def state_get(key, default=None):
    return st.session_state.get(key, default)


def is_analyzed() -> bool:
    return st.session_state.get("ei_analyzed_sig") is not None and st.session_state.get("ei_analyzed_sig") == st.session_state.get("ei_sig")


def current() -> Analysis | None:
    """Analysis of the selected recording if it is up to date, else None."""
    if not is_analyzed():
        return None
    return (st.session_state.get("ei_cache") or {}).get(st.session_state.get("ei_selected"))


def require_analysis() -> Analysis:
    """For pages that need results: explain what is missing and stop instead of showing empty or stale numbers."""
    from . import theme as UI
    a = current()
    if a is not None:
        return a
    if not st.session_state.get("ei_good"):
        UI.empty_state("No recording loaded", "Upload an ECG in the sidebar, then press Analyze Recording.")
    elif st.session_state.get("ei_analyzed_sig") is not None:
        UI.empty_state("Inputs changed", "The files or settings changed since the last analysis. Press Analyze Recording to refresh the results.")
    else:
        UI.empty_state("Ready to analyze", "Press Analyze Recording in the sidebar to run the analysis on the loaded recording.")
    st.stop()
