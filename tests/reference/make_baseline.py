"""Capture numerical outputs of the analysis backend (stub model + bundled example records) as JSON.
Used by UI-12 to prove that UI changes did not alter any computed value.   python tests/reference/make_baseline.py OUT.json"""
import glob, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def collect():
    from modules.data_ingestion import load_recording
    from modules.ecg_delineation import delineate_consensus
    from modules.prediction import ECGClassifier
    from modules.reference_comparison import comparisons_from_morphology
    clf = ECGClassifier.from_path(os.path.join(ROOT, "models", "test_stub", "ptbxl_binary_model.joblib"))
    out = {}
    for hea in sorted(glob.glob(os.path.join(ROOT, "examples", "ptbxl_records", "0000*_hr.hea"))):
        rec = load_recording(hea)
        res = clf.predict_recording(rec, {"age": 60.0, "sex": 1, "height_cm": None, "weight_kg": None})
        morph = delineate_consensus(rec.signal, rec.fs)
        cmp = comparisons_from_morphology(morph, "female") if morph.ok else []
        out[os.path.basename(hea)] = {
            "status": res.status, "p_abnormal": res.probability_abnormal, "label": res.label_name,
            "features": None if res.feature_frame is None else [None if not np.isfinite(v) else float(v) for v in res.feature_frame.iloc[0].values],
            "morph": {k: [m.value, m.spread, m.status] for k, m in morph.measures.items()} if morph.ok else None,
            "cmp": [[c.label, c.value, c.status] for c in cmp]}
    return out


if __name__ == "__main__":
    json.dump(collect(), open(sys.argv[1], "w"), indent=1, default=float)
    print("written", sys.argv[1])
