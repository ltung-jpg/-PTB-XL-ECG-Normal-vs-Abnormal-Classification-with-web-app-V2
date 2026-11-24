"""Stage 2 tests: the exported bundle is complete, valid and loads once; broken bundles are rejected."""
import json
import os
import joblib
import numpy as np
import pytest

from modules import config as C
from modules.bundle_builder import build_bundle, save_bundle, convert_legacy
from modules.prediction import BundleError, load_bundle


def test_bundle_contents(stub_path):
    b = joblib.load(stub_path)
    for k in ["model", "feature_names", "threshold", "class_mapping", "sampling_frequency_hz", "n_leads", "lead_names",
              "waveform_mode", "demographics_config", "inference_config", "metadata", "training_feature_summary", "global_importance"]:
        assert k in b, k
    assert b["feature_names"] == C.default_feature_names() and len(b["feature_names"]) == 145
    assert b["sampling_frequency_hz"] == 100 and b["lead_names"] == C.LEADS and b["threshold"] == 0.5
    assert b["class_mapping"] == {0: "Normal", 1: "Abnormal"}
    m = b["metadata"]
    for k in ["library_versions", "hyperparameters", "limitations", "created_utc", "performance_recorded_in_notebook"]:
        assert k in m
    assert m["hyperparameters"]["learning_rate"] == 0.1 and m["hyperparameters"]["max_leaf_nodes"] == 31
    assert b["inference_config"]["bandpass"]["low_hz"] == 0.5 and b["inference_config"]["bandpass"]["high_hz"] == 40


def test_metadata_json_matches_bundle(stub_path):
    j = json.load(open(os.path.join(os.path.dirname(stub_path), "model_metadata.json")))
    assert j["n_features"] == 145 and j["is_test_stub"] is True


def test_reference_vectors_reproduce_probabilities(stub_path):
    import pandas as pd
    d = os.path.dirname(stub_path)
    ref = pd.read_csv(os.path.join(d, "reference_test_vectors.csv"))
    b = joblib.load(stub_path)
    p = b["model"].predict_proba(ref[b["feature_names"]])[:, 1]
    assert np.allclose(p, ref["p_abnormal"], rtol=0, atol=1e-12)


def test_recorded_performance_is_untouched():
    assert {k: v for k, v in C.RECORDED_TEST_PERFORMANCE.items() if isinstance(v, float)} == {"accuracy": 0.8531, "balanced_accuracy": 0.8611, "precision": 0.9257,
                                           "sensitivity": 0.8171, "specificity": 0.9051, "f1": 0.8680, "roc_auc": 0.9297}


def _tamper(stub_path, tmp_path, fn):
    b = joblib.load(stub_path); fn(b)
    p = tmp_path / "bad.joblib"; joblib.dump(b, p); return str(p)


def test_load_rejects_bad_bundles(stub_path, tmp_path):
    with pytest.raises(BundleError, match="not found"):
        load_bundle(str(tmp_path / "none.joblib"))
    junk = tmp_path / "junk.joblib"; junk.write_bytes(b"xx")
    with pytest.raises(BundleError):
        load_bundle(str(junk))
    with pytest.raises(BundleError, match="schema"):
        load_bundle(_tamper(stub_path, tmp_path, lambda b: b.update(feature_names=b["feature_names"][:-1] + ["foo"])))
    with pytest.raises(BundleError, match="configuration"):
        load_bundle(_tamper(stub_path, tmp_path, lambda b: b["inference_config"]["bandpass"].update(high_hz=45)))
    with pytest.raises(BundleError, match="OLD notebook export"):
        load_bundle(_tamper(stub_path, tmp_path, lambda b: [b.pop(k) for k in ["class_mapping", "inference_config", "metadata", "lead_names", "n_leads", "sampling_frequency_hz"]]))


def test_legacy_conversion_no_retraining(stub_path, tmp_path):
    b = joblib.load(stub_path)
    old = tmp_path / "old.joblib"
    joblib.dump({"model": b["model"], "feature_names": b["feature_names"], "threshold": 0.5, "waveform_mode": True}, old)
    new = convert_legacy(str(old))
    assert new["model"] is not None and new["feature_names"] == b["feature_names"]
    paths = save_bundle(new, str(tmp_path / "out"))
    again = load_bundle(paths["bundle"])
    X = joblib.load(stub_path)["training_feature_summary"] and None
    import pandas as pd
    ref = pd.read_csv(os.path.join(os.path.dirname(stub_path), "reference_test_vectors.csv"))
    assert np.allclose(again["model"].predict_proba(ref[b["feature_names"]])[:, 1], ref["p_abnormal"], atol=1e-12)
