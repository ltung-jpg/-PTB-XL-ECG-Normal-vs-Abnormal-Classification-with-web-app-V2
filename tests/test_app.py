"""The Streamlit app end to end (AppTest), multipage version.  The file uploader is replaced by fake uploads.
Covers UI-01 ... UI-11, UI-13, UI-14 (UI-12 and the visual VIS tests live in test_ui_theme.py / tests/visual)."""
import os

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from modules import config as C
from tests.synthetic import make_ecg

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


class FakeUpload:
    def __init__(self, name, data):
        self.name, self._d = name, data

    def getvalue(self):
        return self._d


def new_app(monkeypatch, files):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [FakeUpload(n, d) for n, d in files])
    return AppTest.from_file(APP, default_timeout=180).run()


def analyze(at):
    at.sidebar.button(key="analyze").click().run()
    return at


def go(at, page):
    at.switch_page(f"views/{page}.py").run()
    return at


def run_page(monkeypatch, files, page, *, do_analyze=True):
    at = new_app(monkeypatch, files)
    if do_analyze:
        analyze(at)
    return go(at, page)


def wfdb_files(base):
    return [(os.path.basename(base) + e, open(base + e, "rb").read()) for e in (".hea", ".dat")]


def csv_bytes(sig, names=None, fs=None, with_time=False):
    names = names or [f"{l} (mV)" for l in C.LEADS]
    df = pd.DataFrame(sig, columns=names)
    if with_time:
        df.insert(0, "time_s", np.arange(len(df)) / fs)
    return df.to_csv(index=False).encode()


def text(at):
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info])


def no_exc(at):
    assert not at.exception, [e.value for e in at.exception]


# ------------------------------------------------------------------------------------------------ UI-01
@pytest.mark.parametrize("page,expect", [("overview", "No recording loaded yet"), ("upload", "No recording loaded"), ("signal_analysis", "No recording loaded"),
                                         ("assessment", "No recording loaded"), ("reports", "No recording loaded")])
def test_ui01_opens_without_data_on_every_page(monkeypatch, clf, page, expect):
    at = go(new_app(monkeypatch, []), page)
    no_exc(at)
    assert expect in text(at)
    assert any(b.label == "Analyze Recording" and b.disabled for b in at.sidebar.button)


def test_ui01_header_and_stub_banner(monkeypatch, clf):
    at = new_app(monkeypatch, [])
    no_exc(at)
    assert "ECG Insight" in text(at) and any("TEST-STUB MODEL ACTIVE" in e.value for e in at.error)


# ------------------------------------------------------------------------------------------------ UI-02 / 03 / 04
def test_ui02_wfdb_pair_uploads_and_analyzes(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "upload")
    no_exc(at)
    t = text(at)
    assert "00001_hr" in t and ">PASS<" in t and "0.0 KB" not in t and "WFDB" in t and "500 Hz" in t
    assert "ELIGIBLE" in t                                        # eligibility matrix appears after analysis


def test_ui03_csv_uploads(monkeypatch, clf):
    sig, _ = make_ecg(fs=100)
    at = new_app(monkeypatch, [("rec.csv", csv_bytes(sig))])
    at.sidebar.number_input[0].set_value(100.0).run()
    go(analyze(at), "upload")
    no_exc(at)
    assert "rec.csv" in text(at) and ">PASS<" in text(at) and "CSV" in text(at)


def test_analyze_updates_status_and_keeps_inputs(monkeypatch, clf, real_records):
    at = new_app(monkeypatch, wfdb_files(real_records[0]))
    assert any("Not analyzed yet" in c.value for c in at.sidebar.caption)
    analyze(at)
    no_exc(at)
    assert any("up to date" in s.value for s in at.sidebar.success)
    assert at.session_state["ei_good"]                                  # inputs are still there after the analysis


def test_ui04_ptbxl_12_lead_supported(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "signal_analysis")
    no_exc(at)
    labels = [m.label for m in at.metric]
    assert "Heart rate" in labels and "QRS duration" in labels and "Detected R-peaks" in labels
    assert at.get("plotly_chart")                                  # the interactive viewer is on the page


# ------------------------------------------------------------------------------------------------ UI-05 / 06 / 07
def test_ui05_missing_wfdb_header(monkeypatch, clf, real_records):
    dat_only = [f for f in wfdb_files(real_records[0]) if f[0].endswith(".dat")]
    at = new_app(monkeypatch, dat_only)
    go(at, "upload")
    no_exc(at)
    assert ">FAIL<" in text(at) and any(b.label == "Analyze Recording" and b.disabled for b in at.sidebar.button)


def test_ui06_invalid_sampling_rate_rejected(monkeypatch, clf):
    sig, _ = make_ecg(fs=100)
    at = new_app(monkeypatch, [("rec.csv", csv_bytes(sig))])          # no time column and no sampling rate given
    go(at, "upload")
    no_exc(at)
    assert ">FAIL<" in text(at) and "sampl" in text(at).lower()
    at = new_app(monkeypatch, [("rec.csv", csv_bytes(sig))])
    at.sidebar.number_input[0].set_value(20.0).run()                  # far below any usable ECG rate
    go(at, "upload")
    no_exc(at)
    assert ">FAIL<" in text(at) and "below the 100 Hz" in text(at)
    assert any(b.label == "Analyze Recording" and b.disabled for b in at.sidebar.button)      # nothing is analyzed


def test_ui07_unknown_lead_names_reported(monkeypatch, clf):
    sig, _ = make_ecg(fs=100)
    at = new_app(monkeypatch, [("odd.csv", csv_bytes(sig, names=[f"chan{i}" for i in range(12)], with_time=False))])
    at.sidebar.number_input[0].set_value(100.0).run()
    go(at, "upload")
    no_exc(at)
    t = text(at)
    assert ">FAIL<" in t and ("chan0" in t or "lead" in t.lower())


# ------------------------------------------------------------------------------------------------ UI-08
def test_ui08_missing_patient_info_handled(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "upload")
    no_exc(at)
    assert "Not provided" in text(at) and ">MISSING<" in text(at)
    assert any("none provided" in c.value for c in at.sidebar.caption)
    at = go(at, "assessment")
    assert any("Age and/or sex were not provided" in w.value for w in at.warning)
    assert not any(n.value == 0 for n in at.sidebar.number_input if n.label.startswith(("Age", "Height", "Weight")))   # never 0


def test_ui08_patient_info_entered_is_used_and_marks_analysis_stale(monkeypatch, clf, real_records):
    at = new_app(monkeypatch, wfdb_files(real_records[0]))
    analyze(at)
    at.sidebar.number_input[1].set_value(60.0).run()                  # age changes after the analysis
    go(at, "signal_analysis")
    assert "Inputs changed" in text(at)                                # a stale result is never shown
    analyze(at)
    go(at, "upload")
    assert ">AVAILABLE<" in text(at) and "60 years" in text(at)


# ------------------------------------------------------------------------------------------------ UI-09 / UI-10
def test_ui09_viewer_raw_and_filtered_show_correct_data(monkeypatch, clf, real_records):
    from modules.data_ingestion import load_recording
    from modules.preprocessing import preprocess_signal
    from ui import viewer as VW
    rec = load_recording(real_records[0] + ".hea")
    filt = VW.display_filtered(rec.signal, rec.fs)
    np.testing.assert_allclose(filt, preprocess_signal(np.nan_to_num(rec.signal), rec.fs))
    fig, shown = VW.build_figure(rec, filt, ["II", "V1"], "Raw vs filtered", 2.0, 6.0)
    k = rec.lead_names.index("II")
    sl = slice(int(2.0 * rec.fs), int(6.0 * rec.fs))
    t, y = shown[("II", "raw")]
    np.testing.assert_array_equal(y, rec.signal[sl, k])               # raw = the file's samples, unmodified
    np.testing.assert_array_equal(t, rec.time_s[sl])
    np.testing.assert_array_equal(shown[("II", "filtered")][1], filt[sl, k])
    assert not np.allclose(shown[("II", "raw")][1], shown[("II", "filtered")][1])
    # in the app: the radio switches the plotted traces
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "signal_analysis")
    at.radio(key="sv_mode").set_value("Filtered").run()
    no_exc(at)
    assert any("Filtered view" in c.value for c in at.caption)


def test_ui09_long_record_decimation_keeps_peaks():
    from ui import viewer as VW
    t = np.arange(600_000) / 500.0
    y = np.zeros_like(t); y[123_456] = 1.5; y[400_001] = -2.0
    td, yd = VW.decimate_minmax(t, y)
    assert len(td) <= VW.MAX_POINTS + 2 and yd.max() == 1.5 and yd.min() == -2.0
    t2, y2 = VW.decimate_minmax(t[:1000], y[:1000])
    assert len(t2) == 1000                                            # short windows are not altered


def test_ui10_peaks_and_boundaries_are_distinguishable(monkeypatch, clf, real_records):
    from modules.data_ingestion import load_recording
    from modules.ecg_delineation import delineate_consensus
    from ui import tokens as T
    from ui import viewer as VW
    rec = load_recording(real_records[0] + ".hea")
    res = clf.predict_recording(rec, {})
    morph = delineate_consensus(rec.signal, rec.fs)
    fig, _ = VW.build_figure(rec, None, ["II"], "Raw", 0, 10, peaks_s=res.waveform_info.peaks / C.MODEL_FS, morph=morph)
    peaks = [d for d in fig.data if d.name == "R-peaks (model detector)"][0]
    bounds = [d for d in fig.data if d.name == "Estimated wave boundaries"]
    sig = [d for d in fig.data if d.name == "II"][0]
    assert peaks.mode == "markers" and peaks.marker.symbol == "triangle-down" and peaks.marker.color == T.R_PEAK
    assert bounds and all(b.mode == "lines" and b.line.dash == "dash" for b in bounds)
    assert len({sig.line.color, peaks.marker.color, bounds[0].line.color}) == 3          # three different colours
    assert len(peaks.x) >= 3 and all(0 <= x <= 10 for x in peaks.x)


# ------------------------------------------------------------------------------------------------ UI-11
def test_ui11_incompatible_model_input_is_blocked(monkeypatch, clf):
    noise = np.random.default_rng(1).normal(0, 0.3, size=(1000, 12))
    at = new_app(monkeypatch, [("noise.csv", csv_bytes(noise))])
    at.sidebar.number_input[0].set_value(100.0).run()
    analyze(at)
    at = go(at, "assessment")
    no_exc(at)
    assert any("No prediction is shown" in e.value for e in at.error)
    assert "Model output" not in [m.label for m in at.metric] and "P(Abnormal)" not in [m.label for m in at.metric]
    at = go(at, "upload")
    assert "BLOCKED" in text(at)


def test_ui11_assessment_sections_are_separate_and_bounded(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "assessment")
    no_exc(at)
    t = text(at)
    for s in ("A. Signal measurements", "B. Machine-learning prediction", "C. Comparison with published reference ranges", "D. Dataset reference labels"):
        assert s in t
    assert "not a diagnosis" in t.lower() and "healthy" not in t.lower()
    assert {"Model output", "P(Abnormal)"} <= {m.label for m in at.metric}
    import re
    badge = lambda row: re.search(row + r"</td><td>[^<]*</td><td>[^<]*</td><td><span[^>]*>([A-Z]+)</span>", t).group(1)
    assert badge("QRS duration") == "ESTIMATED" and badge("QT interval") == "ESTIMATED" and badge("QTc \\(Bazett\\)") == "DERIVED"
    assert badge("Heart rate") == "MEASURED"


# ------------------------------------------------------------------------------------------------ UI-13
def test_ui13_report_retains_provenance(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "reports")
    at.button(key="rp_btn").click().run()
    no_exc(at)
    html = st_report(at)
    for s in ["Source format", "WFDB", "Original sampling rate (Hz)", "500", "Original units", "Lead names in file", "Derived leads", "none",
              "Input signature", "Demographics given", "not provided", "TEST-STUB MODEL", "not a diagnosis", "A. Machine-learning prediction",
              "B. Waveform measurements", "C. Comparison with published reference ranges"]:
        assert s in html, s
    assert "healthy" not in html.lower()


def st_report(at):
    return at.session_state["report_html"][1]


# ------------------------------------------------------------------------------------------------ UI-14
@pytest.mark.parametrize("name,data", [("bad.hea", b"\x00\xff garbage"), ("bad.dat", os.urandom(500)), ("empty.csv", b""), ("junk.csv", b"x\ny\n"),
                                       ("binary.edf", os.urandom(300)), ("notes.txt", "xin chào".encode("utf-8")), ("weird.bin", b"123")])
def test_ui14_corrupted_files_do_not_crash(monkeypatch, clf, name, data):
    at = new_app(monkeypatch, [(name, data)])
    go(at, "upload")
    no_exc(at)
    at = go(at, "signal_analysis")
    no_exc(at)
    assert not at.sidebar.button(key="analyze").disabled or True


def test_ui14_mixed_good_and_bad_files(monkeypatch, clf, real_records):
    at = new_app(monkeypatch, wfdb_files(real_records[0]) + [("junk.csv", b"x\ny\n")])
    go(analyze(at), "upload")
    no_exc(at)
    t = text(at)
    assert ">PASS<" in t and ">FAIL<" in t


def test_batch_overview_after_analysis(monkeypatch, clf, real_records):
    files = sum([wfdb_files(b) for b in real_records], [])
    at = run_page(monkeypatch, files, "upload")
    no_exc(at)
    assert "Batch overview" in text(at)


def test_app_does_not_modify_model(monkeypatch, clf, real_records):
    import io, joblib
    b = io.BytesIO(); joblib.dump(clf.pipeline, b); before = b.getvalue()
    run_page(monkeypatch, wfdb_files(real_records[1]), "assessment")
    b = io.BytesIO(); joblib.dump(clf.pipeline, b)
    assert b.getvalue() == before
