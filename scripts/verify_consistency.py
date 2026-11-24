"""Verify that the DEPLOYED inference pipeline reproduces the notebook (run this on YOUR real model).

    python scripts/verify_consistency.py --model models/ptbxl_binary_model.joblib \
        --reference models/reference_test_vectors.csv --records-dir "<Drive>/ses 4/records100" [--tol 1e-9]

Check 1  (always)      bundle loads; schema = 145 features in the notebook order; reloaded model reproduces the notebook's
                       probabilities for the 30 reference test records (feature vectors exported by the notebook).
Check 2  (--records-dir) for each reference record the WFDB signal is re-read, passed through the full deployment pipeline
                       (feature extraction from the waveform) and compared with the notebook's features and probabilities.
Exit status 0 only if every check passes.  No file is modified; nothing is fitted.
"""
import argparse, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules import config as C
from modules.data_ingestion import load_recording
from modules.prediction import ECGClassifier
from modules.feature_extraction import waveform_features_from_100hz
import wfdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "ptbxl_binary_model.joblib"))
    ap.add_argument("--reference", default=None)
    ap.add_argument("--records-dir", default=None, help="folder that contains the PTB-XL 'records100' tree")
    ap.add_argument("--tol", type=float, default=1e-9)
    a = ap.parse_args()
    ref_path = a.reference or os.path.join(os.path.dirname(a.model), "reference_test_vectors.csv")
    clf = ECGClassifier.from_path(a.model)
    print("model:", a.model, "| test stub:", clf.is_test_stub)
    ok = True
    ref = pd.read_csv(ref_path)
    X = ref[clf.feature_names]
    p_pipe = clf.pipeline.predict_proba(X)[:, 1]
    d1 = float(np.max(np.abs(p_pipe - ref["p_abnormal"].values)))
    print(f"[1] saved pipeline vs notebook probabilities on {len(ref)} records: max |diff| = {d1:.2e}  -> {'PASS' if d1 <= a.tol else 'FAIL'}")
    ok &= d1 <= a.tol
    lab = (p_pipe > clf.threshold).astype(int)
    print(f"    labels at threshold {clf.threshold}: {int(lab.sum())} Abnormal of {len(lab)}")
    if a.records_dir:
        fdiff, pdiff, bad = [], [], 0
        for _, row in ref.iterrows():
            base = os.path.join(a.records_dir, str(row["record_path"]).replace("records100/", "").replace("records500/", ""))
            if not os.path.exists(base + ".hea"):
                base = os.path.join(os.path.dirname(a.records_dir.rstrip("/")), str(row["record_path"]))
            sig, f = wfdb.rdsamp(base)
            if f["fs"] != 100:
                print("    skipped (not 100 Hz):", base); bad += 1; continue
            wf, _ = waveform_features_from_100hz(sig)
            mine = np.array([wf[n] for n in C.waveform_feature_names()], float)
            nb = row[C.waveform_feature_names()].to_numpy(float)
            both_nan = np.isnan(mine) & np.isnan(nb)
            fdiff.append(np.nanmax(np.abs(np.where(both_nan, 0, mine - nb))) if not (np.isnan(mine) ^ np.isnan(nb)).any() else np.inf)
            demo = row[["age", "sex", "height", "weight", "bmi"]].to_numpy(float)
            full = pd.DataFrame([np.r_[mine, demo]], columns=clf.feature_names)
            pdiff.append(abs(clf.predict_frame(full) - row["p_abnormal"]))
        if fdiff:
            print(f"[2] deployment feature extraction vs notebook features ({len(fdiff)} records): max |diff| = {max(fdiff):.2e} -> {'PASS' if max(fdiff) <= a.tol else 'FAIL'}")
            print(f"    deployment probability vs notebook probability: max |diff| = {max(pdiff):.2e} -> {'PASS' if max(pdiff) <= a.tol else 'FAIL'}")
            ok &= max(fdiff) <= a.tol and max(pdiff) <= a.tol and bad == 0
        else:
            print("[2] FAIL: no reference records could be read"); ok = False
    else:
        print("[2] skipped (give --records-dir to re-extract features from the original waveform files)")
    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
