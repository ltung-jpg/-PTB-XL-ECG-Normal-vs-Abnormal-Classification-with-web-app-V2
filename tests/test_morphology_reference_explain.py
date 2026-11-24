"""Stage 6 tests: morphology, reference comparison, explainability, report."""
import numpy as np
import pytest

from modules import config as C
from modules import explainability as EX
from modules.ecg_delineation import delineate, delineate_consensus
from modules.reference_comparison import REFERENCES, RANGES, compare_measurements, comparisons_from_morphology
from modules.report import build_report, image_html
from tests.synthetic import make_ecg


def test_delineation_against_known_truth():
    sig, tr = make_ecg(fs=500, hr=70, seed=1, noise=0.01)
    m = delineate_consensus(sig, 500)
    assert m.ok
    got = {k: v.value for k, v in m.measures.items()}
    assert got["hr_bpm"] == pytest.approx(tr["hr_bpm"], abs=1.5)
    assert got["qrs_duration_ms"] == pytest.approx(tr["qrs_dur_s"] * 1000, abs=25)
    assert got["qt_ms"] == pytest.approx(tr["qt_s"] * 1000, abs=40)
    assert got["pr_interval_ms"] == pytest.approx(tr["pr_s"] * 1000, abs=30)


def test_irregular_rhythm_withholds_pr():
    sig, _ = make_ecg(fs=500, afib=True, seed=2)
    m = delineate_consensus(sig, 500)
    assert m.measures["pr_interval_ms"].status == "not_measurable" and any("irregular" in w for w in m.warnings)


def test_delineation_handles_unusable_input():
    m = delineate(np.random.default_rng(0).normal(size=(1000, 12)) * 1e-4, 100)
    assert (not m.ok) or all(x.status != "measured" or x.value is not None for x in m.measures.values())
    short = delineate(make_ecg(fs=500, dur=1.5)[0], 500)
    assert not short.ok and short.message


def test_reference_comparison_logic():
    c = {x.measure: x for x in compare_measurements({"hr_bpm": 40, "hr_x": 1, "qrs_duration_ms": 160, "pr_interval_ms": 160,
                                                     "qtc_bazett_ms": 500, "t_duration_ms": 100, "qt_ms": None},
                                                    sex="male", resolution_ms=2)}
    assert c["hr_bpm"].status == "below" and c["qrs_duration_ms"].status == "above"
    assert c["pr_interval_ms"].status == "within" and c["qtc_bazett_ms"].status == "above"
    assert c["t_duration_ms"].status == "indeterminate" and c["t_duration_ms"].no_range and "No population reference range" in c["t_duration_ms"].reason
    assert c["qt_ms"].status == "indeterminate"
    near = compare_measurements({"qrs_duration_ms": 112}, resolution_ms=2)[0]
    assert near.status == "indeterminate"                                   # within uncertainty of the 109 ms limit
    f = compare_measurements({"qtc_bazett_ms": 455}, sex="female", resolution_ms=2)[0]
    assert f.upper == 460 and f.reference_id == "rautaharju2009"


def test_every_range_is_traceable():
    for rules in RANGES.values():
        for r in rules:
            ref = REFERENCES[r["ref"]]
            assert ref["citation"] and ref["url"].startswith("http")


def test_shap_additivity(clf, real_records):
    from modules.data_ingestion import load_recording
    rec = load_recording([real_records[0] + ".hea", real_records[0] + ".dat"])
    r = clf.predict_recording(rec, {"age": 60, "sex": 1})
    import shap
    Z = EX._transform(clf.pipeline, r.feature_frame[clf.feature_names])
    ex = shap.TreeExplainer(clf.pipeline.named_steps["model"])
    sv = np.asarray(ex.shap_values(Z)); sv = sv[1] if sv.ndim == 3 else sv
    logit = float(np.log(r.probability_abnormal / (1 - r.probability_abnormal)))
    assert float(np.ravel(ex.expected_value)[-1]) + sv[0].sum() == pytest.approx(logit, abs=1e-6)
    loc = EX.local_explanation(clf, r.feature_frame)
    assert len(loc["table"]) == 12 and "not causal" in EX.DISCLAIMER


def test_global_importance_comes_from_bundle_only(clf):
    gi = EX.global_importance(clf.bundle, 5)
    stored = clf.bundle["global_importance"]["importances"]
    assert all(gi.loc[n, "importance"] == stored[n]["mean"] for n in gi.index)
    assert EX.global_importance({"model": None}) is None


def test_report_separates_sections(clf, real_records):
    from modules.data_ingestion import load_recording
    from modules import visualization as V
    rec = load_recording([real_records[0] + ".hea", real_records[0] + ".dat"])
    res = clf.predict_recording(rec, {"age": 50, "sex": 0}); m = delineate_consensus(rec.signal, rec.fs)
    cmp = comparisons_from_morphology(m, "male")
    html = build_report(record_id="t", source_info={"fs": rec.fs}, demographics={"age": 50}, prediction=res, model_info=clf.model_info(),
                        morph=m, comparisons=cmp, figures={"reference": image_html(V.plot_reference_comparison(cmp))})
    for s in ["A. Machine-learning prediction", "B. Waveform measurements", "C. Comparison with published reference ranges",
              "TEST-STUB MODEL", "not a diagnosis", "pubmed.ncbi.nlm.nih.gov"]:
        assert s in html


def test_new_ranges_p_duration_and_rr():
    c = {x.measure: x for x in compare_measurements({"p_duration_ms": 160, "rr_s": 0.80, "st_level_mv": 0.02, "tp_segment_ms": 300}, resolution_ms=2)}
    assert c["p_duration_ms"].status == "above" and c["p_duration_ms"].upper == 120 and c["p_duration_ms"].lower is None
    assert c["p_duration_ms"].range_text == "<= 120"
    assert c["rr_s"].status == "within" and (c["rr_s"].lower, c["rr_s"].upper) == (0.61, 1.25) and c["rr_s"].value_text == "0.80"
    assert abs(60 / 98 - 0.61) < 0.005 and abs(60 / 48 - 1.25) < 0.005           # RR range really is the converted Mason HR range
    near = compare_measurements({"p_duration_ms": 110}, resolution_ms=2)[0]
    assert near.status == "indeterminate" and not near.no_range                   # inside the +/-30 ms uncertainty band of the limit
    assert compare_measurements({"rr_s": 1.5}, resolution_ms=2)[0].status == "above"
    for m in ("st_level_mv", "tp_segment_ms"):                                      # still no invented range for these
        assert c[m].no_range and c[m].status == "indeterminate" and c[m].lower is None and c[m].upper is None
    assert c["st_level_mv"].reason != c["tp_segment_ms"].reason                     # specific explanation per measure


def test_small_st_level_is_not_displayed_as_zero():
    c = compare_measurements({"st_level_mv": 0.021}, resolution_ms=2)[0]
    assert c.value_text == "0.021"
