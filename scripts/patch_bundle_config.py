"""Correct the SETTINGS stored in an exported bundle (inference_config + notes) to the verified values in modules/config.py.

    python scripts/patch_bundle_config.py models/ptbxl_binary_model.joblib

Why: the bundle exported earlier recorded QRS width fraction 0.1 (what notebook cell 12 says), but the model was trained on
features measured at 0.2 (proved by scripts/diagnose_features.py).  This script changes ONLY the descriptive configuration and
adds an explanatory note.  The fitted pipeline (imputer, scaler, booster), feature names, threshold and metrics are not touched;
a backup copy <name>.before_patch.joblib is written first and the model's predictions are checked to be unchanged.
"""
import os, shutil, sys
import joblib, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules import config as C

NOTE = ("QRS width (qrs_width_ms) is measured at 20 % of the peak height: this is what the cached training features used "
        "(verified exactly on 30 test records). Notebook cell 12 as written says 10 %, which does NOT reproduce the training features.")


def main(path):
    b = joblib.load(path)
    old = b["inference_config"]["template"]["width_fraction"]
    new = C.inference_config()
    if b["inference_config"] == new and NOTE in b["metadata"].get("notes", []):
        print("Bundle already up to date."); return
    shutil.copy2(path, path.replace(".joblib", ".before_patch.joblib"))
    X = np.random.default_rng(0).normal(size=(20, len(b["feature_names"])))
    import pandas as pd, warnings
    Xd = pd.DataFrame(X, columns=b["feature_names"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); before = b["model"].predict_proba(Xd)[:, 1]
    b["inference_config"] = new
    b["metadata"].setdefault("notes", [])
    if NOTE not in b["metadata"]["notes"]:
        b["metadata"]["notes"].append(NOTE)
    joblib.dump(b, path, compress=3)
    after = joblib.load(path)["model"].predict_proba(Xd)[:, 1]
    assert np.array_equal(before, after), "model changed - this must never happen"
    print(f"width_fraction in bundle: {old} -> {new['template']['width_fraction']}; model predictions identical; backup written.")
    import json
    mp = os.path.join(os.path.dirname(path), "model_metadata.json")
    if os.path.exists(mp):
        m = json.load(open(mp)); m.setdefault("notes", [])
        if NOTE not in m["notes"]: m["notes"].append(NOTE)
        json.dump(m, open(mp, "w"), indent=2, default=str); print("model_metadata.json note added.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "ptbxl_binary_model.joblib"))
