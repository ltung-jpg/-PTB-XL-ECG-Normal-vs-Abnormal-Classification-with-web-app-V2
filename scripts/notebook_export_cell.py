# ======================================================================================================
#  EXPORT: complete deployment bundle for the Streamlit CAD application   (replaces the old export cell)
#  Nothing is retrained: this cell only packages the model that was fitted above, plus everything the
#  inference pipeline needs to reproduce this notebook's preprocessing and feature extraction.
# ======================================================================================================
import os, json, platform, datetime as dt
from importlib import metadata as _md
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

EXPORT_DIR = os.path.join(DATA_DIR, "models")           # copy this folder into the app project  (models/)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---- 1. inference configuration (snapshot of the feature-extraction cell above; identical to modules/config.py)
INFERENCE_CONFIG = json.loads(r"""{
 "model_fs_hz": 100,
 "n_leads": 12,
 "lead_names": [
  "I",
  "II",
  "III",
  "AVR",
  "AVL",
  "AVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6"
 ],
 "window_s": 10.0,
 "bandpass": {
  "kind": "butterworth_bandpass",
  "order": 2,
  "low_hz": 0.5,
  "high_hz": 40.0,
  "zero_phase": true,
  "implementation": "scipy.signal.sosfiltfilt (sos)"
 },
 "qrs_detection": {
  "algorithm": "Pan-Tompkins-style (custom, from notebook)",
  "band_hz": [
   5,
   15
  ],
  "band_order": 2,
  "smoothing_window_s": 0.12,
  "threshold_fraction": 0.35,
  "threshold_percentile": 99,
  "min_distance_s": 0.25,
  "refine_half_window_s": 0.06,
  "min_peak_spacing_s": 0.2,
  "peak_leads": [
   1,
   0,
   10,
   7
  ],
  "min_peaks": 3
 },
 "template": {
  "pre_s": 0.25,
  "post_s": 0.45,
  "min_beats": 2,
  "baseline_window_s": [
   -0.08,
   -0.04
  ],
  "qrs_window_s": [
   -0.06,
   0.08
  ],
  "width_window_s": [
   -0.12,
   0.12
  ],
  "width_fraction": 0.2,
  "st_window_s": [
   0.08,
   0.12
  ],
  "t_window_s": [
   0.15,
   0.4
  ],
  "p_window_s": [
   -0.2,
   -0.1
  ]
 },
 "rhythm_features": [
  "hr_bpm",
  "mean_rr_s",
  "sdnn_s",
  "rmssd_s",
  "pnn50",
  "rr_cv",
  "rr_range_s",
  "n_beats"
 ],
 "lead_stat_features": [
  "std",
  "ptp",
  "skew",
  "kurt"
 ],
 "template_features": [
  "qrs_max",
  "qrs_min",
  "qrs_area",
  "qrs_width_ms",
  "st_dev",
  "t_amp",
  "p_amp"
 ],
 "demographics": {
  "feature_order": [
   "age",
   "sex",
   "height",
   "weight",
   "bmi"
  ],
  "age_cap_threshold": 120,
  "age_cap_value": 90,
  "height_cm_valid_range": [
   50,
   230
  ],
  "weight_kg_valid_range": [
   3,
   250
  ],
  "bmi_valid_range": [
   8,
   70
  ],
  "sex_coding": {
   "male": 0,
   "female": 1
  },
  "median_imputation_of_missing": true
 }
}""")
CLASS_MAPPING = {0: "Normal", 1: "Abnormal"}
LIMITATIONS = json.loads(r"""[
 "Research/educational model trained on PTB-XL; not validated for clinical decision making.",
 "'Abnormal' merges many different diagnoses; per-diagnosis performance was not analysed.",
 "Trained on 10-s, 12-lead recordings at 100 Hz; other sampling rates are resampled (not part of the original pipeline).",
 "Age is among the strongest predictors; predictions without age/sex are less reliable.",
 "The R-peak detector can miss wide ectopic (PVC) beats, affecting rhythm features.",
 "No external validation (other hospitals/devices) has been performed."
]""")

# ---- 2. feature schema: the exact names and order the model was trained on, verified against the config
FEATURE_NAMES = list(X_train.columns)
expected = list(INFERENCE_CONFIG["rhythm_features"])
for _lead in INFERENCE_CONFIG["lead_names"]:
    expected += [f"{_lead}_{n}" for n in INFERENCE_CONFIG["lead_stat_features"] + INFERENCE_CONFIG["template_features"]]
expected += list(INFERENCE_CONFIG["demographics"]["feature_order"])
assert FEATURE_NAMES == expected, (
    "The model's training columns differ from the expected 140 waveform + 5 demographic feature schema "
    "(was FEATURE_SET changed, or USE_DEMOGRAPHICS=False?). The deployment pipeline cannot reproduce this model.")
assert list(final_model.named_steps) == ["impute", "scale", "model"], "unexpected pipeline structure"
assert final_model.n_features_in_ == len(FEATURE_NAMES) == 145
_est = final_model.named_steps["model"]
assert type(_est).__name__ == "HistGradientBoostingClassifier", f"selected model is {type(_est).__name__}"

THRESHOLD = float(thr)                                   # decision threshold used in the final test (0.5)

# ---- 3. training-feature distribution (lets the app flag unusual feature values)
_q = X_train.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T
TRAIN_SUMMARY = {n: {"mean": float(X_train[n].mean()), "std": float(X_train[n].std()), "p01": float(_q.loc[n, 0.01]),
                     "p05": float(_q.loc[n, 0.05]), "p50": float(_q.loc[n, 0.5]), "p95": float(_q.loc[n, 0.95]),
                     "p99": float(_q.loc[n, 0.99]), "missing_fraction": float(X_train[n].isna().mean())}
                 for n in X_train.columns}

# ---- 4. global explanation: permutation importance on the held-out test set (the model has no built-in importances)
print("Computing permutation importance on the test set (about a minute) ...")
_pi = permutation_importance(final_model, X_test, y_test, scoring="roc_auc", n_repeats=5,
                             random_state=RANDOM_STATE, n_jobs=1)
GLOBAL_IMPORTANCE = {
    "method": "permutation importance (sklearn.inspection.permutation_importance)", "scoring": "roc_auc",
    "n_repeats": 5, "dataset": "test (PTB-XL strat_fold 10)", "n_samples": int(len(X_test)),
    "importances": {n: {"mean": float(m), "std": float(s)} for n, m, s in zip(X_test.columns, _pi.importances_mean, _pi.importances_std)}}

def _versions():
    out = {"python": platform.python_version()}
    for _p in ["scikit-learn", "numpy", "scipy", "pandas", "joblib", "wfdb", "matplotlib"]:
        try:
            out[_p] = _md.version(_p)
        except _md.PackageNotFoundError:
            pass
    return out

# ---- 5. metadata (only values that exist in this notebook session - nothing invented)
_perf = {k.replace("recall (sensitivity)", "sensitivity"): round(float(v), 4) for k, v in test_metrics.items()}
_perf.update(n_test_records=int(len(y_test)), test_set="PTB-XL strat_fold 10")
_params = _est.get_params()
METADATA = {
    "model_name": "PTB-XL binary ECG classifier (Normal vs Abnormal)", "model_version": "1.0.0",
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "is_test_stub": False,
    "task": "Binary classification of 12-lead ECG: 0 = Normal, 1 = Abnormal",
    "algorithm": type(_est).__name__,
    "hyperparameters": {k: _params[k] for k in ["learning_rate", "max_leaf_nodes", "max_iter", "max_depth", "min_samples_leaf",
                                                "l2_regularization", "class_weight", "random_state", "early_stopping"] if k in _params},
    "pipeline_steps": ["impute (median)", "scale (StandardScaler)", "model (HistGradientBoostingClassifier)"],
    "selection": {"procedure": f"{N_FOLDS}-fold patient-grouped stratified cross-validation, model comparison + grid search",
                  "metric": SELECTION_METRIC, "selected_model": best_name, "best_params": {k: v for k, v in grid.best_params_.items()}},
    "decision_threshold": THRESHOLD, "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
    "sampling_frequency_hz": INFERENCE_CONFIG["model_fs_hz"], "n_leads": INFERENCE_CONFIG["n_leads"],
    "lead_names": INFERENCE_CONFIG["lead_names"], "n_features": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES,
    "training_info": {"data": "PTB-XL", "signal_source": f"{REC_DIRS[rec_type]} ({FS} Hz)", "split": SPLIT_MODE,
                      "n_train_records": int(len(y_train)), "n_test_records": int(len(y_test)),
                      "n_train_patients": int(g_train.nunique()), "abnormal_share_train": round(float(y_train.mean()), 4),
                      "random_state": RANDOM_STATE, "cv_folds": N_FOLDS, "cv_max_samples": CV_MAX_SAMPLES,
                      "label_rule": f"Normal = NORM likelihood >= {NORMAL_MIN_LIKELIHOOD} and no other diagnosis; Abnormal = any other diagnosis"},
    "performance_recorded_in_notebook": _perf,
    "library_versions": _versions(),
    "limitations": LIMITATIONS,
    "notes": ["QRS width (qrs_width_ms) is measured at 20 % of the peak height: this is what the cached training features used (verified exactly on 30 test records). Notebook feature cell 12 as written says 10 %, which does NOT reproduce the training features.",
              "Library versions are those of the Colab runtime that ran this cell. If the runtime was restarted after the model "
              "was fitted and the model was re-loaded from disk, they may differ from the training session."],
}

# ---- 6. reference vectors: 30 test records (15 Normal + 15 Abnormal) with the features and probabilities THIS notebook produced
_rng = np.random.default_rng(RANDOM_STATE)
_pick = np.sort(np.concatenate([_rng.choice(np.where(y_test.values == c)[0], 15, replace=False) for c in (0, 1)]))
_ref = X_test.iloc[_pick].reset_index(drop=True).copy()
_ref.insert(0, "p_abnormal", final_model.predict_proba(X_test.iloc[_pick])[:, 1])
_ref.insert(0, "label", y_test.values[_pick])
_ref.insert(0, "record_path", meta.loc[test_mask, FILENAME_COL].values[_pick])
_ref.insert(0, "ecg_id", meta.index[test_mask].values[_pick])

# ---- 7. write everything
BUNDLE = {
    "format_version": 2, "model": final_model, "feature_names": FEATURE_NAMES, "threshold": THRESHOLD,
    "class_mapping": CLASS_MAPPING, "sampling_frequency_hz": INFERENCE_CONFIG["model_fs_hz"],
    "n_leads": INFERENCE_CONFIG["n_leads"], "lead_names": INFERENCE_CONFIG["lead_names"], "waveform_mode": True,
    "demographics_config": INFERENCE_CONFIG["demographics"], "inference_config": INFERENCE_CONFIG,
    "metadata": METADATA, "training_feature_summary": TRAIN_SUMMARY, "global_importance": GLOBAL_IMPORTANCE}
joblib.dump(BUNDLE, os.path.join(EXPORT_DIR, "ptbxl_binary_model.joblib"), compress=3)
with open(os.path.join(EXPORT_DIR, "model_metadata.json"), "w") as fh:
    json.dump({**METADATA, "has_global_importance": True, "has_training_feature_summary": True}, fh, indent=2, default=str)
pd.DataFrame(GLOBAL_IMPORTANCE["importances"]).T.sort_values("mean", ascending=False).to_csv(
    os.path.join(EXPORT_DIR, "permutation_importance.csv"))
_ref.to_csv(os.path.join(EXPORT_DIR, "reference_test_vectors.csv"), index=False)

# ---- 8. self-check: reload the file and confirm it reproduces this notebook's probabilities exactly
_re = joblib.load(os.path.join(EXPORT_DIR, "ptbxl_binary_model.joblib"))
_same = np.allclose(_re["model"].predict_proba(X_test[_re["feature_names"]])[:, 1],
                    final_model.predict_proba(X_test)[:, 1], rtol=0, atol=1e-12)
assert _same, "reloaded model does not reproduce the notebook's test probabilities"
print("Saved to:", EXPORT_DIR)
for _f in sorted(os.listdir(EXPORT_DIR)):
    print(f"  {_f:<32} {os.path.getsize(os.path.join(EXPORT_DIR, _f)) / 1024:9.1f} KB")
print(f"Reload check passed: identical probabilities on all {len(X_test):,} test records.")
print("NEXT: copy the 'models' folder into the app project and run  python scripts/verify_consistency.py")
