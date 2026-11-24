"""Build / save / convert the deployment bundle (`models/ptbxl_binary_model.joblib` + `model_metadata.json`).

Used by `scripts/export_model_bundle.py` (convert an EXISTING trained model - no retraining) and by the tests.
The notebook's export cell (`scripts/notebook_export_cell.py`) is a self-contained twin of `build_bundle` for use
inside Colab; `tests/test_export.py` proves that both produce bundles `prediction.load_bundle` accepts.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
from importlib import metadata as importlib_metadata

import joblib
import numpy as np
import pandas as pd

from . import config as C

FORMAT_VERSION = 2

LIMITATIONS = [
    "Research/educational model trained on PTB-XL; not validated for clinical decision making.",
    "'Abnormal' merges many different diagnoses; per-diagnosis performance was not analysed.",
    "Trained on 10-s, 12-lead recordings at 100 Hz; other sampling rates are resampled (not part of the original pipeline).",
    "Age is among the strongest predictors; predictions without age/sex are less reliable.",
    "The R-peak detector can miss wide ectopic (PVC) beats, affecting rhythm features.",
    "No external validation (other hospitals/devices) has been performed.",
]


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ["scikit-learn", "numpy", "scipy", "pandas", "joblib", "wfdb", "pyedflib", "shap", "matplotlib"]:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def training_feature_summary(X: pd.DataFrame) -> dict:
    """Per-feature distribution of the TRAINING features (used to flag unusual values at inference time)."""
    q = X.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T
    return {n: {"mean": float(X[n].mean()), "std": float(X[n].std()), "p01": float(q.loc[n, 0.01]),
                "p05": float(q.loc[n, 0.05]), "p50": float(q.loc[n, 0.5]), "p95": float(q.loc[n, 0.95]),
                "p99": float(q.loc[n, 0.99]), "missing_fraction": float(X[n].isna().mean())} for n in X.columns}


def permutation_importance_dict(model, X: pd.DataFrame, y, *, scoring="roc_auc", n_repeats=5, random_state=42, dataset="test"):
    """Model-agnostic permutation importance on held-out data (drop in `scoring` when a feature is shuffled)."""
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
    return {"method": "permutation importance (sklearn.inspection.permutation_importance)", "scoring": scoring,
            "n_repeats": n_repeats, "dataset": dataset, "n_samples": int(len(X)),
            "importances": {n: {"mean": float(m), "std": float(s)} for n, m, s in zip(X.columns, r.importances_mean, r.importances_std)}}


def make_metadata(model, feature_names, threshold, *, training_info=None, performance=None, notes=None, is_test_stub=False) -> dict:
    est = model.named_steps["model"]
    params = est.get_params()
    keep = ["learning_rate", "max_leaf_nodes", "max_iter", "max_depth", "min_samples_leaf", "l2_regularization",
            "class_weight", "random_state", "early_stopping"]
    return {
        "model_name": "PTB-XL binary ECG classifier (Normal vs Abnormal)",
        "model_version": "1.0.0",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "is_test_stub": bool(is_test_stub),
        "task": "Binary classification of 12-lead ECG: 0 = Normal, 1 = Abnormal",
        "algorithm": type(est).__name__,
        "hyperparameters": {k: params[k] for k in keep if k in params},
        "pipeline_steps": C.PIPELINE_STEPS,
        "selection": {"procedure": "5-fold patient-grouped stratified cross-validation, model comparison + grid search",
                      "metric": "balanced_accuracy"},
        "decision_threshold": float(threshold),
        "class_mapping": {str(k): v for k, v in C.CLASS_MAPPING.items()},
        "sampling_frequency_hz": C.MODEL_FS, "n_leads": C.N_LEADS, "lead_names": C.LEADS,
        "n_features": len(feature_names), "feature_names": list(feature_names),
        "training_info": training_info or {},
        "performance_recorded_in_notebook": performance if performance is not None else C.RECORDED_TEST_PERFORMANCE,
        "library_versions": library_versions(),
        "limitations": LIMITATIONS,
        "notes": notes or [],
    }


def build_bundle(model, feature_names, threshold=C.DEFAULT_THRESHOLD, *, X_train=None, X_test=None, y_test=None,
                 compute_importance=False, training_info=None, performance=None, notes=None,
                 is_test_stub=False, random_state=42) -> dict:
    feature_names = list(feature_names)
    return {
        "format_version": FORMAT_VERSION,
        "model": model,
        "feature_names": feature_names,
        "threshold": float(threshold),
        "class_mapping": dict(C.CLASS_MAPPING),
        "sampling_frequency_hz": C.MODEL_FS,
        "n_leads": C.N_LEADS,
        "lead_names": list(C.LEADS),
        "waveform_mode": True,
        "demographics_config": C.DEMOGRAPHICS,
        "inference_config": C.inference_config(),
        "metadata": make_metadata(model, feature_names, threshold, training_info=training_info, performance=performance,
                                  notes=notes, is_test_stub=is_test_stub),
        "training_feature_summary": training_feature_summary(X_train[feature_names]) if X_train is not None else None,
        "global_importance": (permutation_importance_dict(model, X_test[feature_names], y_test, random_state=random_state)
                              if compute_importance and X_test is not None else None),
    }


def save_bundle(bundle: dict, out_dir: str = "models", reference_vectors: pd.DataFrame | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    bpath, mpath = os.path.join(out_dir, "ptbxl_binary_model.joblib"), os.path.join(out_dir, "model_metadata.json")
    joblib.dump(bundle, bpath, compress=3)
    meta = dict(bundle["metadata"])
    meta["has_global_importance"] = bundle.get("global_importance") is not None
    meta["has_training_feature_summary"] = bundle.get("training_feature_summary") is not None
    with open(mpath, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    paths = {"bundle": bpath, "metadata": mpath}
    gi = bundle.get("global_importance")
    if gi:
        ipath = os.path.join(out_dir, "permutation_importance.csv")
        pd.DataFrame(gi["importances"]).T.sort_values("mean", ascending=False).to_csv(ipath)
        paths["importance"] = ipath
    if reference_vectors is not None:
        rpath = os.path.join(out_dir, "reference_test_vectors.csv")
        reference_vectors.to_csv(rpath, index=False)
        paths["reference_vectors"] = rpath
    return paths


def convert_legacy(path: str, **kw) -> dict:
    """Wrap the OLD notebook export {model, feature_names, threshold, waveform_mode} - no retraining."""
    old = joblib.load(path)
    need = {"model", "feature_names", "threshold"}
    if not isinstance(old, dict) or not need <= set(old):
        raise ValueError(f"{path} is not a legacy notebook export (expected keys {sorted(need)}).")
    names = list(old["feature_names"])
    if set(names) != set(C.default_feature_names()):
        raise ValueError("The legacy model's feature names do not match the 145 expected features.")
    return build_bundle(old["model"], names, float(old["threshold"]), **kw)
