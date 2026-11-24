"""Create a clearly-flagged TEST-STUB model bundle (synthetic ECGs; NOT the real PTB-XL model).

Runs the notebook export cell's source on stand-in variables, so the same code that the user will run in Colab is
exercised.  Output: models/test_stub/ (is_test_stub=True).  Never use for real predictions.
"""
import os, sys, tempfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from modules import config as C
from modules.feature_extraction import waveform_features_from_100hz, build_demographic_features, assemble_feature_frame
from tests.synthetic import make_dataset


def build_stub(out_dir=os.path.join(ROOT, "models", "test_stub"), n=300, seed=0):
    sigs, y, demos = make_dataset(n, 100.0, seed)
    rows = []
    for s, d in zip(sigs, demos):
        wf, _ = waveform_features_from_100hz(s)
        demo = build_demographic_features(d["age"], d["sex"], d["height"], d["weight"])
        rows.append(assemble_feature_frame(wf, demo, C.default_feature_names()).iloc[0])
    X = pd.DataFrame(rows).reset_index(drop=True)
    y = pd.Series(y)
    k = int(0.75 * n)
    X_train, X_test, y_train, y_test = X[:k], X[k:].reset_index(drop=True), y[:k], y[k:].reset_index(drop=True)
    final_model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                            ("model", HistGradientBoostingClassifier(class_weight="balanced", learning_rate=0.1,
                                                                     max_leaf_nodes=31, random_state=42))]).fit(X_train, y_train)
    ns = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, final_model=final_model, thr=0.5,
              test_metrics={"accuracy": 0.0, "recall (sensitivity)": 0.0}, best_name="HistGradientBoosting",
              grid=type("G", (), {"best_params_": {"model__learning_rate": 0.1}})(), N_FOLDS=5, SELECTION_METRIC="balanced_accuracy",
              SPLIT_MODE="stub", g_train=pd.Series(range(len(y_train))), FILENAME_COL="filename_hr",
              REC_DIRS={"x": "stub"}, rec_type="x", FS=100, CV_MAX_SAMPLES=None, NORMAL_MIN_LIKELIHOOD=50,
              RANDOM_STATE=42, DATA_DIR=out_dir.rsplit(os.sep, 1)[0])
    ns["meta"] = pd.DataFrame({"filename_hr": [f"stub/{i}" for i in range(len(y_test))]},
                              index=pd.RangeIndex(len(y_test)))
    ns["test_mask"] = np.ones(len(y_test), bool)
    ns["REC_DIRS"] = {"x": "synthetic"}
    src = open(os.path.join(ROOT, "scripts", "notebook_export_cell.py")).read()
    src = src.replace('os.path.join(DATA_DIR, "models")', repr(out_dir))
    exec(compile(src, "export_cell", "exec"), ns)
    # flag as stub
    import joblib, json
    bp = os.path.join(out_dir, "ptbxl_binary_model.joblib")
    b = joblib.load(bp)
    b["metadata"]["is_test_stub"] = True
    b["metadata"]["model_name"] = "TEST STUB (synthetic ECGs) - NOT the PTB-XL model"
    b["metadata"]["performance_recorded_in_notebook"] = {"note": "none - synthetic stub"}
    joblib.dump(b, bp, compress=3)
    mp = os.path.join(out_dir, "model_metadata.json")
    json.dump({**b["metadata"], "has_global_importance": True, "has_training_feature_summary": True}, open(mp, "w"), indent=2, default=str)
    return out_dir


if __name__ == "__main__":
    print(build_stub())
