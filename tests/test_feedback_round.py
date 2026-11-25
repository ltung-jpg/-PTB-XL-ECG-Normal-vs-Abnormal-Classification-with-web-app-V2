"""Tests for the changes made after the user's test round (22 items): triage, deviation, history, R-peak display snapping,
chart-only PNG, new input formats, links in a new tab, step sizes, and the solid header."""
import base64
import io
import os

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from modules import config as C
from modules.reference_comparison import Comparison
from ui import history as HIST
from ui import theme as UI
from ui import triage as TR
from ui import viewer as VW


# ---------------------------------------------------------------------------------------------- deviation + triage
def _cmp(measure, value, status, lo, hi):
    return Comparison(measure=measure, label=measure, value=value, status=status, lower=lo, upper=hi, reference_id="mason2007")


def test_deviation_is_distance_beyond_the_nearest_limit():
    d = TR.deviation(_cmp("qrs_duration_ms", 124.0, "above", 69, 109), "ms")
    assert d["amount"] == pytest.approx(15.0) and d["pct"] == pytest.approx(100 * 15 / 109) and "above the upper limit (109 ms)" in d["text"]
    d = TR.deviation(_cmp("hr_bpm", 40.0, "below", 48, 98), "bpm")
    assert d["amount"] == pytest.approx(8.0) and "below the lower" in d["text"]
    assert TR.deviation(_cmp("hr_bpm", 70.0, "within", 48, 98), "bpm")["text"] == ""          # inside: nothing invented
    assert TR.deviation(_cmp("hr_bpm", None, "indeterminate", 48, 98), "bpm")["text"] == ""


class _Res:
    status = "ok"; label_name = "Normal"; probability_abnormal = 0.1; threshold = 0.5; reasons = []; waveform_info = None


class _M:
    def __init__(self, ok=True, status="measured"):
        self.ok = ok; self.measures = {"qt_ms": type("m", (), {"status": status})()}


def test_triage_levels_never_call_missing_data_routine():
    ok = [_cmp("hr_bpm", 70.0, "within", 48, 98)]
    assert TR.compute_findings(_Res(), _M(), ok, "PASS").level == "ROUTINE"
    assert TR.compute_findings(_Res(), _M(), [], "PASS").level == "INCOMPLETE"                      # nothing could be compared
    assert TR.compute_findings(_Res(), _M(ok=False), ok, "PASS").level == "INCOMPLETE"
    r = _Res(); r.label_name = "Abnormal"
    assert TR.compute_findings(r, _M(), ok, "PASS").level == "ATTENTION"
    out = [_cmp("pr_interval_ms", 250.0, "above", 113, 212)]
    t = TR.compute_findings(_Res(), _M(), out, "PASS")
    assert t.level == "ATTENTION" and any("PR interval" in x for x in t.reasons) and t.findings[0].doctor and t.findings[0].student
    assert TR.compute_findings(_Res(), _M(), ok, "NEEDS_REVIEW", ["noise"]).level == "REVIEW"
    assert TR.compute_findings(_Res(), _M(status="low_agreement"), ok, "PASS").level in ("REVIEW", "INCOMPLETE")      # never ROUTINE


def test_qtc_over_500_is_marked_high_risk():
    t = TR.compute_findings(_Res(), _M(), [_cmp("qtc_bazett_ms", 520.0, "above", None, 450)], "PASS")
    assert t.findings[0].severity == "high"
    t = TR.compute_findings(_Res(), _M(), [_cmp("qtc_bazett_ms", 480.0, "above", None, 450)], "PASS")
    assert t.findings[0].severity == "abnormal"


def test_commentary_is_template_text_from_the_numbers_and_role_dependent():
    cmp = [_cmp("qrs_duration_ms", 124.0, "above", 69, 109)]
    t = TR.compute_findings(_Res(), _M(), cmp, "PASS")
    doc = " ".join(TR.commentary("Doctor", _Res(), _M(), cmp, t, "Ambulance (pre-hospital)"))
    stu = " ".join(TR.commentary("Student", _Res(), _M(), cmp, t))
    assert "124" in doc and "Points to verify" in doc and "ambulance" in doc.lower() and "not a diagnosis" in doc.lower()
    assert "Points to verify" not in stu and "ventricular depolarisation" in stu
    assert "healthy" not in doc.lower()


# ---------------------------------------------------------------------------------------------- history
def test_history_saves_facts_only_dedupes_and_clears(tmp_path):
    p = str(tmp_path / "h" / "hist.json")
    t = TR.compute_findings(_Res(), _M(), [_cmp("hr_bpm", 70.0, "within", 48, 98)], "PASS")
    e = HIST.make_entry(name="rec1", signature="a" * 40, role="Doctor", setting="Ambulance (pre-hospital)", demographics={"age": 60.0, "sex": None},
                        sex_txt=None, triage=t, res=_Res(), cmp=[_cmp("hr_bpm", 70.0, "within", 48, 98)])
    assert HIST.add(e, p) is True and HIST.add(dict(e), p) is False           # same recording + inputs: not saved twice
    got = HIST.load(p)
    assert len(got) == 1 and got[0]["level"] == "ROUTINE" and got[0]["sex"] is None and got[0]["age"] == 60.0
    assert not any(k in got[0] for k in ("signal", "samples", "data"))
    HIST.clear(p)
    assert HIST.load(p) == []


# ---------------------------------------------------------------------------------------------- R-peak display
def _notched(fs=500, polarity=1.0):
    t = np.arange(0, 6, 1 / fs); y = np.zeros_like(t); centres = np.arange(0.5, 6, 1.0)
    for c in centres:       # first small bump, true (higher) R 40 ms later
        y += 0.3 * np.exp(-((t - (c - 0.04)) / 0.010) ** 2) + 1.0 * np.exp(-((t - c) / 0.010) ** 2)
    return t, polarity * y, centres


@pytest.mark.parametrize("polarity", [1.0, -0.4])
def test_r_peak_marker_moves_from_first_bump_to_true_peak_and_handles_inverted_lead(polarity):
    t, y, centres = _notched(polarity=polarity)
    snapped = VW.snap_r_peaks(t, y, centres - 0.04)          # detector sat on the first bump (or on the wrong side)
    assert len(snapped) == len(centres) and np.all(np.abs(snapped - centres) <= 0.004)


def test_snap_does_not_change_the_model_detector_or_features(clf, real_records):
    from modules.data_ingestion import load_recording
    rec = load_recording(real_records[0] + ".hea")
    a = clf.predict_recording(rec, {}); b = clf.predict_recording(rec, {})
    assert np.array_equal(a.waveform_info.peaks, b.waveform_info.peaks)
    peaks = a.waveform_info.peaks / C.MODEL_FS
    VW.snap_r_peaks(rec.time_s, rec.signal[:, 2], peaks)      # display helper only
    assert np.array_equal(a.waveform_info.peaks, b.waveform_info.peaks)


def test_chart_png_contains_only_the_chart(clf, real_records):
    from modules.data_ingestion import load_recording
    from PIL import Image
    rec = load_recording(real_records[0] + ".hea")
    png = VW.export_png(rec, None, ["II", "III"], "Raw", 0, 5, peaks_s=None, title="t")
    im = Image.open(io.BytesIO(png))
    assert im.format == "PNG" and 1000 < im.size[0] < 2200 and im.size[1] < 1200          # a chart, not a screen capture


# ---------------------------------------------------------------------------------------------- UI pieces
def test_links_open_in_new_tab():
    h = UI.link_html("https://pubmed.ncbi.nlm.nih.gov/17276451", "Mason")
    assert 'target="_blank"' in h and "noopener" in h
    from modules.report import build_report  # report links too
    import inspect
    src = inspect.getsource(build_report)
    assert src.count("target='_blank'") >= 2


def test_header_is_solid_red_and_inputs_hint_is_hidden():
    css = UI.css()
    assert '[data-testid="stHeader"] { background: #9D3141 !important' in css
    assert "InputInstructions" in css and "display: none" in css


def test_upload_page_number_steps_are_usable(monkeypatch, real_records):
    import streamlit as st
    from tests.test_app import new_app, wfdb_files
    at = new_app(monkeypatch, wfdb_files(real_records[0]), role=None)
    by = {n.label: n for n in at.number_input}
    assert by["Height (cm)"].step == 1.0 and by["Weight (kg)"].step == 1.0 and by["Age (years)"].step == 1.0
    assert at.radio(key="ei_role").options == list(TR.ROLES) and at.radio(key="ei_role").value is None


def test_kpi_card_abnormality_fold_only_when_abnormal():
    def app():
        from ui import theme as UI
        UI.kpi_card("QRS duration", 124, unit="ms", tag="ESTIMATED", tag_kind="estimated", key="qrs",
                    abnormality={"title": "ABOVE RANGE", "range": "69 - 109 ms", "deviation": "15 ms above the upper limit (109 ms), 14 % beyond it",
                                 "high": False, "lines_title": "What to verify", "lines": ["Check the QRS on a second lead"]})
        UI.kpi_card("Heart rate", 63, unit="bpm", tag="MEASURED", tag_kind="measured", key="hr")
    at = AppTest.from_function(app).run()
    t = " ".join(m.value for m in at.markdown)
    assert t.count("<summary>Abnormality</summary>") == 1 and "69 - 109 ms" in t and "15 ms above" in t and "Check the QRS" in t
    assert [m.value for m in at.metric] == ["124 ms", "63 bpm"]


def test_snap_drops_marker_without_peak_and_merges_double_markers():
    from ui import viewer as VW
    t = np.arange(0, 6, 0.01); y = 0.02 * np.sin(t * 3)
    for c in (1.0, 2.0, 3.0, 4.0, 5.0):
        y += np.exp(-((t - c) / 0.015) ** 2)
    out = VW.snap_r_peaks(t, y, [1.0, 2.02, 2.06, 3.0, 3.5, 4.0, 5.0])       # 2.02/2.06: one spike; 3.5: nowhere near a peak
    assert len(out) == 5 and all(min(abs(out - c)) < 0.03 for c in (1, 2, 3, 4, 5))


# ---------------------------------------------------------------------------------------------- new input formats
def _sig(fs=500):
    t = np.arange(fs * 10) / fs; y = np.zeros_like(t)
    for c in np.arange(0.4, 10, 0.8):
        y += np.exp(-((t - c) / 0.012) ** 2)
    return np.stack([y * (0.5 + 0.1 * k) for k in range(12)], 1)


def test_decimal_comma_semicolon_csv_is_read():
    from modules.ingestion_extra import load_uploaded_ext
    s = _sig()[:2000]
    text = "Time (ms);" + ";".join(f"{l} (mV)" for l in C.LEADS) + "\n" + "\n".join(
        f"{i * 2};" + ";".join(f"{v:.4f}".replace(".", ",") for v in r) for i, r in enumerate(s))
    (name, rec), = load_uploaded_ext([("dc.csv", text.encode("latin-1"))])
    assert hasattr(rec, "signal") and rec.signal.shape == (2000, 12) and abs(rec.original_fs - 500) < 1


def test_mat_with_hea_and_mat_without_units():
    from scipy.io import savemat
    from modules.ingestion_extra import load_uploaded_ext
    s = _sig(); adc = np.round(s.T * 1000).astype(np.int16)
    buf = io.BytesIO(); savemat(buf, {"val": adc})
    hea = "rec 12 500 5000\n" + "\n".join(f"rec.mat 16+24 1000/mV 16 0 {int(adc[k, 0])} 0 0 {l}" for k, l in enumerate(C.LEADS))
    (name, rec), = load_uploaded_ext([("rec.mat", buf.getvalue()), ("rec.hea", hea.encode())])
    assert hasattr(rec, "signal") and rec.fs == 500 and np.nanmax(rec.signal[:, 0]) == pytest.approx(0.5, abs=0.01)
    (name, err), = load_uploaded_ext([("rec.mat", buf.getvalue())])
    assert isinstance(err, Exception) and "units" in str(err).lower()          # never guessed


def test_hl7_and_muse_xml_and_doctype_refused():
    from modules.ingestion_extra import load_uploaded_ext
    s = _sig()
    def seq(code, y):
        d = " ".join(str(int(round(v * 1000))) for v in y)
        return (f'<component><sequence><code code="{code}"/><value xsi:type="SLIST_PQ"><origin value="0" unit="uV"/><scale value="1" unit="uV"/>'
                f'<increment value="2" unit="ms"/><digits>{d}</digits></value></sequence></component>')
    xml = ('<?xml version="1.0"?><AnnotatedECG xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><component><series>'
           + "".join(seq("MDC_ECG_LEAD_" + l.upper(), s[:, i]) for i, l in enumerate(C.LEADS)) + "</series></component></AnnotatedECG>")
    (n, rec), = load_uploaded_ext([("hl7.xml", xml.encode())])
    assert rec.fs == 500 and np.nanmax(rec.signal[:, 1]) == pytest.approx(0.6, abs=0.01)

    def ld(l, y):
        raw = np.round(y * 1000 / 4.88).astype("<i2").tobytes()
        return (f"<LeadData><LeadID>{l}</LeadID><LeadAmplitudeUnitsPerBit>4.88</LeadAmplitudeUnitsPerBit><LeadAmplitudeUnits>MICROVOLTS</LeadAmplitudeUnits>"
                f"<WaveFormData>{base64.b64encode(raw).decode()}</WaveFormData></LeadData>")
    ls = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    muse = ("<RestingECG><Waveform><WaveformType>Rhythm</WaveformType><SampleBase>500</SampleBase><SampleExponent>0</SampleExponent>"
            + "".join(ld(l, s[:, C.LEADS.index(l)]) for l in ls) + "</Waveform></RestingECG>")
    (n, err), = load_uploaded_ext([("m.xml", muse.encode())])
    assert isinstance(err, Exception) and "derive" in str(err).lower()          # 8 leads: III/aVR/aVL/aVF only if the user allows deriving
    (n, rec), = load_uploaded_ext([("m.xml", muse.encode())], derive_limb_leads=True)
    assert rec.derived_leads == ["III", "AVR", "AVL", "AVF"] and np.nanmax(rec.signal[:, 1]) == pytest.approx(0.6, abs=0.01)
    (n, err), = load_uploaded_ext([("x.xml", b"<!DOCTYPE x [<!ENTITY a 'b'>]><x/>")])
    assert isinstance(err, Exception) and "refused" in str(err)
