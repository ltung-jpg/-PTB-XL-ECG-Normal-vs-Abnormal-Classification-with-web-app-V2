"""Stage 3/4 tests: feature equivalence with the notebook oracle, prediction behaviour, robustness."""
import os
import numpy as np
import pandas as pd
import pytest
import wfdb
from scipy import signal as sps

from modules import config as C
from modules.data_ingestion import (ECGRecording, IncompatibleLeadsError, IngestionError, load_recording,
                                    load_uploaded, normalize_lead_name)
from modules.feature_extraction import waveform_features_from_100hz
from modules.preprocessing import assess_signal_quality
from tests.synthetic import make_ecg
from tests.reference import trained_oracle as NB


def _make_rec(sig, fs, rid="syn"):
    from modules.data_ingestion import _finish
    return _finish(sig, fs, "TEST", rid, fs, "mV", C.LEADS, [], [])


# ------------------------------------------------------------------ feature equivalence with the notebook
def test_feature_schema():
    names = C.default_feature_names()
    assert len(names) == 145 and len(set(names)) == 145
    assert len(C.waveform_feature_names()) == 140


def _oracle_features(sig100, tmp_path, name="rec"):
    """Write a 100 Hz WFDB record, run the notebook's own record_features(path) on it, and return
    (notebook features, the signal as the notebook read it back)."""
    wfdb.wrsamp(name, fs=100, units=["mV"] * 12, sig_name=C.LEADS, p_signal=sig100, fmt=["16"] * 12, write_dir=str(tmp_path))
    path = os.path.join(str(tmp_path), name)
    return NB.record_features(path), wfdb.rdsamp(path)[0]


def test_features_bit_identical_to_notebook(real_records, tmp_path):
    for base in real_records:
        r = wfdb.rdrecord(base)
        sig100 = sps.resample_poly(r.p_signal, 1, 5, axis=0)         # 500 -> 100 Hz, same as the notebook's records100
        ref, read_back = _oracle_features(sig100, tmp_path, os.path.basename(base))
        mine, _ = waveform_features_from_100hz(read_back)
        assert list(mine) == list(ref) == C.waveform_feature_names()
        a, b = np.array(list(mine.values()), float), np.array(list(ref.values()), float)
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0)


def test_features_identical_on_synthetic(tmp_path):
    for seed, afib in [(1, False), (2, True)]:
        sig, _ = make_ecg(fs=100, seed=seed, afib=afib)
        ref, read_back = _oracle_features(sig, tmp_path, f"syn{seed}")
        mine, _ = waveform_features_from_100hz(read_back)
        a, b = np.array(list(mine.values()), float), np.array(list(ref.values()), float)
        assert np.allclose(a, b, rtol=0, atol=0, equal_nan=True)


# ------------------------------------------------------------------ predictions
def test_single_recording_probabilities_consistent(clf, real_records):
    rec = load_recording([real_records[0] + ".hea", real_records[0] + ".dat"])
    r = clf.predict_recording(rec, {"age": 56, "sex": 1})
    assert r.status == "ok" and r.feature_frame.shape == (1, 145)
    assert list(r.feature_frame.columns) == clf.feature_names
    assert abs(r.probability_abnormal + r.probability_normal - 1) < 1e-12
    assert r.label == int(r.probability_abnormal > clf.threshold)
    # direct call on the loaded pipeline reproduces the app's probability exactly
    assert r.probability_abnormal == pytest.approx(float(clf.pipeline.predict_proba(r.feature_frame)[0, 1]), abs=1e-12)


def test_different_recordings_give_different_features(clf, real_records):
    res = [clf.predict_recording(load_recording([b + ".hea", b + ".dat"]), {"age": 50, "sex": 0}) for b in real_records]
    assert all(r.ok for r in res)
    hrs = [r.qc["hr_bpm"] for r in res]
    assert len({round(h, 1) for h in hrs}) == len(hrs)


def test_multiple_recordings_match_single(clf, real_records):
    recs = [load_recording([b + ".hea", b + ".dat"]) for b in real_records]
    demos = [{"age": 40 + i, "sex": i % 2} for i in range(len(recs))]
    many = clf.predict_many(recs, demos)
    for r, d, m in zip(recs, demos, many):
        s = clf.predict_recording(r, d)
        assert s.probability_abnormal == m.probability_abnormal and s.label == m.label
    assert len(clf.summarize(many)) == len(recs)


def test_missing_demographics_stay_nan_and_are_imputed(clf, real_records):
    rec = load_recording([real_records[0] + ".hea", real_records[0] + ".dat"])
    r = clf.predict_recording(rec, None)
    assert r.ok
    assert set(r.missing_inputs) == {"age", "sex", "height", "weight", "bmi"}
    assert r.feature_frame[["age", "sex", "height", "weight", "bmi"]].isna().all().all()     # NOT imputed by the app
    assert any("median" in n for n in r.notes) and any("less reliable" in w for w in r.warnings)
    r2 = clf.predict_recording(rec, {"age": 60, "sex": 1})
    assert set(r2.missing_inputs) == {"height", "weight", "bmi"}
    # bmi is derived only when height AND weight are both supplied
    r3 = clf.predict_recording(rec, {"age": 60, "sex": 1, "height_cm": 170, "weight_kg": 70})
    assert r3.feature_frame["bmi"].iloc[0] == pytest.approx(70 / 1.7 ** 2)


def test_demographic_cleaning_matches_notebook(clf):
    from modules.feature_extraction import build_demographic_features as bd
    d = bd(130, 1, 300, 70)                                          # age >=120 -> 90 ; height out of range -> NaN
    assert d["age"] == 90 and np.isnan(d["height"]) and np.isnan(d["bmi"])


def test_model_not_modified_by_prediction(clf, real_records):
    import joblib, io
    before = io.BytesIO(); joblib.dump(clf.pipeline, before)
    rec = load_recording([real_records[0] + ".hea", real_records[0] + ".dat"])
    clf.predict_recording(rec, {"age": 33, "sex": 0})
    after = io.BytesIO(); joblib.dump(clf.pipeline, after)
    assert before.getvalue() == after.getvalue()


# ------------------------------------------------------------------ sampling rates
@pytest.mark.parametrize("fs", [100, 250, 360, 500, 1000])
def test_sampling_rates_equivalent(clf, fs):
    sig, _ = make_ecg(fs=fs, dur=10, hr=72, seed=5)
    r = clf.predict_recording(_make_rec(sig, fs), {"age": 55, "sex": 1})
    assert r.ok, r.reasons
    assert r.qc["hr_bpm"] == pytest.approx(72, abs=1.5)
    assert r.qc["resampled"] == (fs != 100)


def test_resampled_close_to_native_100hz(clf):
    s100, _ = make_ecg(fs=100, hr=70, seed=7, noise=0.0, wander=0.0)
    s500, _ = make_ecg(fs=500, hr=70, seed=7, noise=0.0, wander=0.0)
    a = clf.predict_recording(_make_rec(s100, 100), {"age": 50, "sex": 0})
    b = clf.predict_recording(_make_rec(s500, 500), {"age": 50, "sex": 0})
    assert abs(a.qc["hr_bpm"] - b.qc["hr_bpm"]) < 0.5


def test_unknown_sampling_rate_rejected(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame(np.random.randn(500, 12), columns=C.LEADS).to_csv(p, index=False)
    with pytest.raises(IngestionError, match="[Ss]ampling frequency"):
        load_recording(str(p), units="mV")


# ------------------------------------------------------------------ invalid / incompatible input
def test_invalid_files(tmp_path):
    bad = tmp_path / "bad.csv"; bad.write_text("hello\nworld\n")
    with pytest.raises(IngestionError):
        load_recording(str(bad), fs=100, units="mV")
    with pytest.raises(IngestionError, match="matching .hea"):
        load_recording(str(tmp_path / "a.dat"))
    with pytest.raises(IngestionError):
        load_recording(str(tmp_path / "a.xyz"))
    junk = tmp_path / "junk.edf"; junk.write_bytes(b"not an edf")
    with pytest.raises(IngestionError):
        load_recording(str(junk))
    hea = tmp_path / "r.hea"; hea.write_text("r 12 500 5000\n")
    with pytest.raises(IngestionError):
        load_recording(str(hea))


def test_wfdb_pair_incomplete(real_records, tmp_path):
    import shutil
    shutil.copy(real_records[0] + ".hea", tmp_path / "r.hea")
    with pytest.raises(IngestionError, match="incomplete"):
        load_recording([str(tmp_path / "r.hea")])


def test_incompatible_leads_are_rejected_not_fabricated(tmp_path):
    sig, _ = make_ecg(fs=100)
    df = pd.DataFrame(sig[:, :8], columns=[f"{l} (mV)" for l in C.LEADS[:8]])
    p = tmp_path / "eight.csv"; df.to_csv(p, index=False)
    with pytest.raises(IncompatibleLeadsError, match="Missing"):
        load_recording(str(p), fs=100)
    single = tmp_path / "one.csv"; pd.DataFrame({"II (mV)": sig[:, 1]}).to_csv(single, index=False)
    with pytest.raises(IncompatibleLeadsError):
        load_recording(str(single), fs=100)


def test_derived_leads_only_when_opted_in_and_flagged(tmp_path):
    sig, _ = make_ecg(fs=100)
    keep = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    df = pd.DataFrame(sig[:, [C.LEADS.index(l) for l in keep]], columns=[f"{l} (mV)" for l in keep])
    p = tmp_path / "eight.csv"; df.to_csv(p, index=False)
    with pytest.raises(IncompatibleLeadsError, match="derive"):
        load_recording(str(p), fs=100)
    rec = load_recording(str(p), fs=100, derive_limb_leads=True)
    assert rec.derived_leads == ["III", "AVR", "AVL", "AVF"] and any("DERIVED" in w for w in rec.warnings)
    assert np.allclose(rec.signal[:, 2], sig[:, 1] - sig[:, 0])           # III = II - I (exact identity)
    assert np.allclose(rec.signal[:, 3], -(sig[:, 0] + sig[:, 1]) / 2)    # aVR


def test_lead_order_and_units_preserved(tmp_path):
    sig, _ = make_ecg(fs=100, seed=3)
    order = list(reversed(C.LEADS))
    df = pd.DataFrame(sig[:, ::-1] * 1000, columns=[f"{l} (uV)" for l in order])    # shuffled order, microvolts
    p = tmp_path / "x.csv"; df.to_csv(p, index=False)
    rec = load_recording(str(p), fs=100)
    assert np.allclose(rec.signal, sig, atol=1e-9)                        # back in standard order, converted to mV
    assert rec.original_units == "uV"


def test_csv_without_lead_names_requires_confirmation(tmp_path):
    sig, _ = make_ecg(fs=100)
    p = tmp_path / "raw.csv"; pd.DataFrame(sig).to_csv(p, index=False, header=False)
    with pytest.raises(IncompatibleLeadsError, match="not guessed"):
        load_recording(str(p), fs=100, units="mV")
    rec = load_recording(str(p), fs=100, units="mV", assume_standard_order=True)
    assert any("standard order" in w for w in rec.warnings)


def test_csv_missing_values_and_time_column(tmp_path):
    sig, _ = make_ecg(fs=100)
    df = pd.DataFrame(sig, columns=[f"{l} (mV)" for l in C.LEADS]); df.insert(0, "time (s)", np.arange(len(df)) / 100)
    df.iloc[200:230, 3] = np.nan
    p = tmp_path / "t.txt"; df.to_csv(p, index=False, sep=";")
    rec = load_recording(str(p), units=None)
    assert rec.fs == pytest.approx(100) and rec.missing_mask.sum() == 30 and rec.source_format == "TXT"


def test_edf_roundtrip(tmp_path):
    import pyedflib
    sig, _ = make_ecg(fs=250, seed=4)
    path = str(tmp_path / "e.edf")
    w = pyedflib.EdfWriter(path, 12, file_type=pyedflib.FILETYPE_EDFPLUS)
    w.setSignalHeaders([{"label": l, "dimension": "mV", "sample_frequency": 250, "physical_min": -10, "physical_max": 10,
                         "digital_min": -32768, "digital_max": 32767, "transducer": "", "prefilter": ""} for l in C.LEADS])
    w.writeSamples([np.ascontiguousarray(sig[:, i]) for i in range(12)]); w.close()
    rec = load_recording(path)
    assert rec.fs == 250 and np.allclose(rec.signal, sig, atol=1e-3) and rec.source_format == "EDF"


def test_quality_gates(clf):
    flat = np.zeros((1000, 12))
    r = clf.predict_recording(_make_rec(flat + 1e-9 * np.random.randn(1000, 12), 100), {"age": 50, "sex": 1})
    assert r.status in ("error", "indeterminate") and r.label is None and r.probability_abnormal is None
    short, _ = make_ecg(fs=100, dur=2.0)
    r = clf.predict_recording(_make_rec(short, 100))
    assert r.status == "error" and r.label is None
    noise = np.random.default_rng(0).normal(size=(1000, 12))
    r = clf.predict_recording(_make_rec(noise, 100))
    assert r.status in ("error", "indeterminate") and r.label is None


def test_long_recording_uses_one_window(clf):
    sig, _ = make_ecg(fs=100, dur=30, hr=65, seed=9)
    r = clf.predict_recording(_make_rec(sig, 100), {"age": 60, "sex": 0})
    assert r.ok and r.qc["duration_analysed_s"] == pytest.approx(10.0)


def test_upload_batch_isolates_failures(real_records):
    files = []
    for ext in (".hea", ".dat"):
        files.append((os.path.basename(real_records[0]) + ext, open(real_records[0] + ext, "rb").read()))
    files.append(("broken.csv", b"a,b\n1,2\n"))
    out = load_uploaded(files)
    kinds = {n: isinstance(v, ECGRecording) for n, v in out}
    assert kinds["00001_hr"] is True and kinds["broken.csv"] is False


def test_lead_name_normalisation():
    assert [normalize_lead_name(x) for x in ["ECG II", "lead_aVR", " v1 ", "AVF"]] == ["II", "AVR", "V1", "AVF"]
