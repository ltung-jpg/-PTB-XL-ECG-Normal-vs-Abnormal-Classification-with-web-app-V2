"""Generate (1) scripts/notebook_export_cell.py and (2) a modified copy of the user's executed notebook whose
FINAL cell is the new deployment-export cell (specification 9.3).

    python scripts/build_notebook_export.py <executed_notebook.ipynb> <output_notebook.ipynb>

The inference configuration inside the cell is a snapshot of modules/config.py taken at generation time, so the
exported bundle, the app and the tests all describe the same pipeline.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules import config as C                      # noqa: E402
from modules.bundle_builder import LIMITATIONS      # noqa: E402

CELL_TEMPLATE = r'''
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
INFERENCE_CONFIG = json.loads(r"""__INFERENCE_CONFIG__""")
CLASS_MAPPING = {0: "Normal", 1: "Abnormal"}
LIMITATIONS = json.loads(r"""__LIMITATIONS__""")

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
'''


def make_cell_source() -> str:
    return (CELL_TEMPLATE.replace("__INFERENCE_CONFIG__", json.dumps(C.inference_config(), indent=1))
                         .replace("__LIMITATIONS__", json.dumps(LIMITATIONS, indent=1))).strip() + "\n"


def main():
    src = make_cell_source()
    with open(os.path.join(ROOT, "scripts", "notebook_export_cell.py"), "w") as fh:
        fh.write(src)
    if len(sys.argv) == 3:
        import nbformat
        nb = nbformat.read(sys.argv[1], 4)
        last = max(i for i, c in enumerate(nb.cells) if c.cell_type == "code" and "out_path" in c.source)
        # keep the feature-importance part of the old cell, replace only the export (everything from 'out_path' on)
        old = nb.cells[last].source
        keep = old.split("out_path")[0].rstrip()
        nb.cells[last].source = keep + "\n\n" + src
        nb.cells[last].outputs = []
        nb.cells[last].execution_count = None
        # documentation cell before it
        md = nbformat.v4.new_markdown_cell(
            "## Step 7: Export the deployment bundle\nThis cell packages the **already trained** model (no retraining) together "
            "with the complete inference configuration, feature schema, metadata, permutation importances and reference test "
            "vectors, ready for the Streamlit application (`models/ptbxl_binary_model.joblib`, `models/model_metadata.json`).")
        nb.cells.insert(last, md)
        # correct the misleading sex-coding comment (wording only - no code behaviour changes)
        for c in nb.cells:
            if c.cell_type == "code" and "# 0 = female, 1 = male in PTB-XL" in c.source:
                c.source = c.source.replace("# 0 = female, 1 = male in PTB-XL", "# PTB-XL coding: 0 = male, 1 = female")
        nbformat.write(nb, sys.argv[2])
        print("modified notebook written:", sys.argv[2])
    print("export cell source written:", os.path.join(ROOT, "scripts", "notebook_export_cell.py"))


if __name__ == "__main__":
    main()
