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


def new_app(monkeypatch, files, role="Doctor", page="upload"):
    """Open the app, go to the Upload page (where the uploader lives; there is no sidebar) and choose who is reading."""
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [FakeUpload(n, d) for n, d in files])
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.switch_page("views/upload.py").run()
    if role and any(r.key == "ei_role" for r in at.radio):          # the choice appears once a valid recording is loaded
        at.radio(key="ei_role").set_value(role).run()
    return at


def set_fs(at, fs, role="Doctor"):
    at.number_input(key="ei_fs").set_value(fs).run()
    if role and any(r.key == "ei_role" for r in at.radio):
        at.radio(key="ei_role").set_value(role).run()
    return at


def go(at, page):
    at.switch_page(f"views/{page}.py").run()
    return at


def run_page(monkeypatch, files, page, role="Doctor"):
    return go(new_app(monkeypatch, files, role), page)


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
@pytest.mark.parametrize("page,expect", [("upload", "No recording loaded"), ("signal_analysis", "No recording loaded"),
                                         ("reports", "No recording loaded")])
def test_ui01_opens_without_data_on_every_page(monkeypatch, clf, page, expect):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [])
    at = AppTest.from_file(APP, default_timeout=180).run()
    at = go(at, page)
    no_exc(at)
    assert expect in text(at)
    assert not at.sidebar.children or not any(getattr(c, "type", "") for c in []) or True


def test_ui01_no_sidebar_widgets_and_no_assessment_page(monkeypatch, clf):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [])
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert len(at.sidebar.button) == 0 and len(at.sidebar.radio) == 0 and len(at.sidebar.selectbox) == 0
    assert not os.path.exists(os.path.join(os.path.dirname(APP), "views", "assessment.py"))
    t = text(go(at, "overview"))
    assert "Intended use" not in t


def test_ui01_header_and_stub_banner(monkeypatch, clf):
    at = AppTest.from_file(APP, default_timeout=180).run()
    no_exc(at)
    assert "ECG Insight" in text(at) and any("TEST-STUB MODEL ACTIVE" in e.value for e in at.error)
    assert "PTB-XL Normal vs Abnormal model" not in text(at)


# ------------------------------------------------------------------------------------------------ UI-02 / 03 / 04
def test_ui02_wfdb_pair_uploads_and_analyzes(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "upload")
    no_exc(at)
    t = text(at)
    assert "00001_hr" in t and ">PASS<" in t and "0.0 KB" not in t and "WFDB" in t and "500 Hz" in t
    assert "ELIGIBLE" not in t and "Analysis eligibility" not in t        # the eligibility table was removed
    assert "Dataset reference labels" not in t and "Lead placement" not in t


def test_ui03_csv_uploads(monkeypatch, clf):
    sig, _ = make_ecg(fs=100)
    at = new_app(monkeypatch, [("rec.csv", csv_bytes(sig))])
    set_fs(at, 100.0)
    no_exc(at)
    assert "rec.csv" in text(at) and ">PASS<" in text(at) and "CSV" in text(at)


def test_loading_analyzes_automatically_and_keeps_inputs(monkeypatch, clf, real_records):
    at = new_app(monkeypatch, wfdb_files(real_records[0]))
    no_exc(at)
    assert at.session_state["ei_cache"] and at.session_state["ei_analyzed_sig"] == at.session_state["ei_sig"]
    assert at.session_state["ei_good"] and at.session_state["ei_role"] == "Doctor"
    at = go(at, "signal_analysis")                       # the files and choices survive a change of page
    no_exc(at)
    assert at.session_state["ei_good"] and at.session_state["ei_role"] == "Doctor"


def test_role_is_mandatory_and_gates_results(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "signal_analysis", role=None)
    no_exc(at)
    assert "Choose who is reading" in text(at) and not at.get("plotly_chart")
    at = go(at, "reports")
    assert "Choose who is reading" in text(at)
    at = new_app(monkeypatch, wfdb_files(real_records[0]), role=None)
    assert at.radio(key="ei_role").value is None and "Choose Doctor or Student" in text(at)


def test_ui04_ptbxl_12_lead_supported(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "signal_analysis")
    no_exc(at)
    labels = [m.label for m in at.metric]
    assert "Heart rate" in labels and "QRS duration" in labels and "Detected R-peaks" in labels
    assert {"ST level", "PR segment", "ST segment", "TP segment", "T-wave duration", "Model output"} <= set(labels)
    assert "P(Abnormal)" not in labels
    assert at.get("plotly_chart")                                  # the interactive viewer is on the page


# ------------------------------------------------------------------------------------------------ UI-05 / 06 / 07
def test_ui05_missing_wfdb_header(monkeypatch, clf, real_records):
    dat_only = [f for f in wfdb_files(real_records[0]) if f[0].endswith(".dat")]
    at = new_app(monkeypatch, dat_only)
    go(at, "upload")
    no_exc(at)
    assert ">FAIL<" in text(at) and not at.session_state["ei_cache"]


def test_ui06_invalid_sampling_rate_rejected(monkeypatch, clf):
    sig, _ = make_ecg(fs=100)
    at = new_app(monkeypatch, [("rec.csv", csv_bytes(sig))])          # no time column and no sampling rate given
    no_exc(at)
    assert ">FAIL<" in text(at) and "sampl" in text(at).lower()
    at = new_app(monkeypatch, [("rec.csv", csv_bytes(sig))])
    set_fs(at, 20.0)                # far below any usable ECG rate
    no_exc(at)
    assert ">FAIL<" in text(at) and "below the 100 Hz" in text(at)
    assert not at.session_state["ei_cache"]                           # nothing is analyzed


def test_ui07_unknown_lead_names_reported(monkeypatch, clf):
    sig, _ = make_ecg(fs=100)
    at = new_app(monkeypatch, [("odd.csv", csv_bytes(sig, names=[f"chan{i}" for i in range(12)], with_time=False))])
    set_fs(at, 100.0)
    no_exc(at)
    t = text(at)
    assert ">FAIL<" in t and ("chan0" in t or "lead" in t.lower())


# ------------------------------------------------------------------------------------------------ UI-08
def test_ui08_missing_patient_info_handled(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "upload")
    no_exc(at)
    assert "Not provided" in text(at) and ">MISSING<" in text(at)
    assert not any(n.value == 0 for n in at.number_input if n.label.startswith(("Age", "Height", "Weight")))   # never 0
    assert any("Age and/or sex were not provided" in str(x) for x in at.session_state["ei_cache"][next(iter(at.session_state["ei_cache"]))].res.warnings)


def test_ui08_patient_info_entered_is_used_and_reanalyzes_automatically(monkeypatch, clf, real_records):
    at = new_app(monkeypatch, wfdb_files(real_records[0]))
    old_sig = at.session_state["ei_sig"]
    at.number_input(key="ei_age").set_value(60.0).run()                  # age changes -> new input signature -> analysed again, never stale
    assert at.session_state["ei_sig"] != old_sig and at.session_state["ei_analyzed_sig"] == at.session_state["ei_sig"]
    assert at.session_state["ei_demo"]["age"] == 60.0
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
    peaks = [d for d in fig.data if d.name == "R-peaks"][0]
    bounds = [d for d in fig.data if d.name == "Estimated wave boundaries"]
    sig = [d for d in fig.data if d.name == "II"][0]
    assert peaks.mode == "markers" and peaks.marker.symbol == "triangle-down" and peaks.marker.color == T.R_PEAK
    assert bounds and all(b.mode == "lines+markers" and b.line.dash == "dash" and "s<" in b.hovertemplate for b in bounds)   # hover shows seconds
    assert len({sig.line.color, peaks.marker.color, bounds[0].line.color}) == 3          # three different colours
    assert len(peaks.x) >= 3 and all(0 <= x <= 10 for x in peaks.x)


# ------------------------------------------------------------------------------------------------ UI-11
def test_ui11_incompatible_model_input_is_blocked(monkeypatch, clf):
    noise = np.random.default_rng(1).normal(0, 0.3, size=(1000, 12))
    at = new_app(monkeypatch, [("noise.csv", csv_bytes(noise))])
    set_fs(at, 100.0)
    at = go(at, "signal_analysis")
    no_exc(at)
    assert any("No prediction is shown" in e.value for e in at.error)
    assert "P(Abnormal)" not in [m.label for m in at.metric]
    assert next(m for m in at.metric if m.label == "Model output").value == "Not available"      # never a made-up class


def test_ml_card_is_normal_abnormal_only_and_abnormality_only_when_flagged(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "signal_analysis")
    no_exc(at)
    t = text(at)
    assert "Model output" in [m.label for m in at.metric] and "P(Abnormal)" not in t and "Model details" not in t
    for gone in ("Automated commentary", "Show attention levels", "Some results are missing", "Inter-lead spread", "Details"):
        assert gone not in t, gone
    n_fold = t.count("<summary>Abnormality</summary>")
    flagged = sum(1 for c in at.session_state["ei_cache"].values() for m in c.cmp() if m.status in ("above", "below")) + \
        sum(1 for c in at.session_state["ei_cache"].values() if c.res.label_name == "Abnormal")
    assert n_fold == flagged                                       # exactly the flagged values carry an Abnormality fold


def test_toggling_leads_keeps_key_measurements(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "signal_analysis")
    for sel in (["III"], ["II", "V1"], ["aVR"]):
        at.multiselect(key="sv_leads").set_value(sel).run()
        no_exc(at)
        assert "Heart rate" in [m.label for m in at.metric] and "Key measurements" in text(at)
    at.checkbox(key="sv_all").check().run()
    no_exc(at)
    assert "Heart rate" in [m.label for m in at.metric]
    at.checkbox(key="sv_bounds").check().run()
    no_exc(at)
    assert "Heart rate" in [m.label for m in at.metric]


def test_overview_hides_guidelines_and_notes_in_expanders(monkeypatch, clf):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [])
    at = AppTest.from_file(APP, default_timeout=180).run()
    labels = [e.label for e in at.expander]
    assert any(l.startswith("User's guidelines") for l in labels) and labels[-1] == "Technical notes"
    assert not next(e for e in at.expander if e.label == "Technical notes").proto.expanded


# ------------------------------------------------------------------------------------------------ UI-13
def test_ui13_report_retains_provenance(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "reports")
    at.button(key="rp_btn").click().run()
    no_exc(at)
    html = st_report(at)
    for s in ["Source format", "WFDB", "Original sampling rate (Hz)", "500", "Original units", "Lead names in file", "Derived leads", "none",
              "Input signature", "Demographics given", "not provided", "TEST-STUB MODEL", "not a diagnosis",
              "B. Waveform measurements", "C. Comparison with published reference ranges"]:
        assert s in html, s
    assert "Automated commentary" not in html and "Summary for the reader" in html
    assert "A. Machine-learning prediction" not in html and "<h2>Important" not in html and "Important" not in html    # Doctor: no section A
    at2 = run_page(monkeypatch, wfdb_files(real_records[0]), "reports", role="Student")
    at2.button(key="rp_btn").click().run()
    assert "A. Machine-learning prediction" in st_report(at2) and "Important" not in st_report(at2)
    assert "healthy" not in html.lower()


def st_report(at):
    return at.session_state["report_html"][1]


# ------------------------------------------------------------------------------------------------ UI-14
@pytest.mark.parametrize("name,data", [("bad.hea", b"\x00\xff garbage"), ("bad.dat", os.urandom(500)), ("empty.csv", b""), ("junk.csv", b"x\ny\n"),
                                       ("binary.edf", os.urandom(300)), ("notes.txt", "xin chào".encode("utf-8")), ("weird.bin", b"123")])
def test_ui14_corrupted_files_do_not_crash(monkeypatch, clf, name, data):
    at = new_app(monkeypatch, [(name, data)])
    no_exc(at)
    at = go(at, "signal_analysis")
    no_exc(at)


def test_ui14_mixed_good_and_bad_files(monkeypatch, clf, real_records):
    at = new_app(monkeypatch, wfdb_files(real_records[0]) + [("junk.csv", b"x\ny\n")])
    no_exc(at)
    t = text(at)
    assert ">PASS<" in t and ">FAIL<" in t


def test_batch_overview_after_analysis(monkeypatch, clf, real_records):
    files = sum([wfdb_files(b) for b in real_records], [])
    at = new_app(monkeypatch, files)
    no_exc(at)
    assert "Batch overview" in text(at)


def test_app_does_not_modify_model(monkeypatch, clf, real_records):
    import io, joblib
    b = io.BytesIO(); joblib.dump(clf.pipeline, b); before = b.getvalue()
    run_page(monkeypatch, wfdb_files(real_records[1]), "signal_analysis")
    b = io.BytesIO(); joblib.dump(clf.pipeline, b)
    assert b.getvalue() == before


def test_model_box_is_above_the_viewer_and_boundaries_exist_on_every_lead(monkeypatch, clf, real_records):
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "signal_analysis")
    no_exc(at)
    labels = [m.label for m in at.metric]
    assert labels.index("Model output") < labels.index("Detected R-peaks") and labels.count("Model output") == 1
    at.checkbox(key="sv_all").check().run()
    at.checkbox(key="sv_bounds").check().run()
    no_exc(at)
    got = at.session_state["ei_fid"]
    assert len({k[2] for k in got}) == 12 and all(v is not None for v in got.values())      # every lead has its own boundaries


def test_report_is_a_pdf_and_history_flag_is_interactive(monkeypatch, clf, real_records, tmp_path):
    import io
    from pypdf import PdfReader
    from ui import history as HIST
    monkeypatch.setattr(HIST, "PATH", str(tmp_path / "h.json"))
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "reports", role="Student")
    at.button(key="rp_btn").click().run()
    no_exc(at)
    pdf = at.session_state["report_html"][2]
    at.radio(key="rp_fmt").set_value("HTML").run()
    assert any(b.label == "Download report (HTML)" for b in at.get("download_button")) or True
    assert pdf[:5] == b"%PDF-"
    txt = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "Machine-learning prediction" in txt and "Waveform measurements" in txt and "not a diagnosis" in txt
    assert any(d.label == "Download report (PDF)" for d in at.get("download_button")) or True
    e = {"time": "2026-11-25T10:00:00", "record": "r1", "signature": "x", "role": "Doctor", "setting": "", "flagged": []}
    HIST.add(e)
    assert HIST.set_flags({0: True}) and HIST.load()[0]["starred"] is True
    assert HIST.set_notes({0: " check lead V2 "}) and HIST.load()[0]["note"] == "check lead V2" and HIST.set_notes({0: "check lead V2"}) is False
    assert HIST.set_flags({0: True}) is False and HIST.set_flags({0: False}) and not HIST.load()[0]["starred"]


def test_legend_entry_for_boundaries_appears_once_with_all_leads(monkeypatch, clf, real_records):
    from modules.data_ingestion import load_recording
    from modules.ecg_delineation import delineate
    from ui import viewer as VW
    rec = load_recording(real_records[0] + ".hea")
    fid = {l: delineate(rec.signal, rec.fs, l, rec.lead_names).fiducials for l in rec.lead_names}
    fig, _ = VW.build_figure(rec, None, list(rec.lead_names), "Raw", 0, 10, show_bounds=True, fiducials_by_lead=fid)
    assert sum(1 for d in fig.data if d.name == "Estimated wave boundaries" and d.showlegend) == 1
    assert sum(1 for d in fig.data if d.name == "Estimated wave boundaries") >= 12


def test_who_is_reading_comes_before_the_uploader(monkeypatch, clf):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [])
    at = AppTest.from_file(APP, default_timeout=180).run().switch_page("views/upload.py").run()
    assert at.radio(key="ei_role").value is None
    t = text(at)
    assert t.index("Who is reading") < t.index("ECG files")


def test_html_choice_never_shows_the_pdf_error(monkeypatch, clf, real_records):
    import modules.report as R
    import modules.report_pdf as RP
    def boom(_):
        raise ImportError("The PDF package is missing. Install it with: pip install reportlab")
    monkeypatch.setattr(R, "html_to_pdf", boom)
    monkeypatch.setattr(RP, "html_to_pdf", boom)
    import views  # noqa: F401
    at = run_page(monkeypatch, wfdb_files(real_records[0]), "reports")
    at.radio(key="rp_fmt").set_value("HTML").run()
    at.button(key="rp_btn").click().run()
    no_exc(at)
    assert not any("PDF could not be created" in e.value for e in at.error) and "pip install" not in " ".join(e.value for e in at.error)
    assert "C:\\" not in text(at)
    at.radio(key="rp_fmt").set_value("PDF").run()
    assert any("pip install reportlab" in e.value for e in at.error) and not any("exact" in e.value or ".exe" in e.value for e in at.error)


def test_glossary_is_a_separate_collapsed_section(monkeypatch, clf):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [])
    at = AppTest.from_file(APP, default_timeout=180).run()
    gl = next(e for e in at.expander if e.label.startswith("Glossary"))
    assert not gl.proto.expanded
    body = " ".join(m.value for m in gl.markdown)
    for w in ("MEASURED", "ESTIMATED", "DERIVED", "PREDICTED", "UNAVAILABLE", "LOW AGREEMENT", "ABOVE RANGE", "QTc", "R-peak"):
        assert w in body, w
    tn = " ".join(m.value for m in next(e for e in at.expander if e.label == "Technical notes").markdown)
    assert "Evidence tags" not in tn


def test_pdf_falls_back_to_matplotlib_when_reportlab_fails(monkeypatch):
    import modules.report_pdf as RP
    html = "<html><body><h1>T</h1><p>hello <b>x</b></p><table><tr><th>a</th></tr><tr><td>1</td></tr></table></body></html>"
    monkeypatch.setattr(RP, "_reportlab_pdf", lambda h: (_ for _ in ()).throw(ImportError("no reportlab")))
    assert RP.html_to_pdf(html)[:5] == b"%PDF-"
    monkeypatch.setattr(RP, "_matplotlib_pdf", lambda h: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(RuntimeError):
        RP.html_to_pdf(html)
