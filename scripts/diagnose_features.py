"""Find WHY check [2] of verify_consistency.py fails.  Compares, for each reference record, THREE feature vectors:

  CSV   = what the notebook stored (models/reference_test_vectors.csv, i.e. its cached features)
  ORACLE= the notebook's own feature code (verbatim, tests/reference/notebook_feature_code.py) run NOW on the raw file
  DEPLOY= the deployed modules/feature_extraction.py run on the raw file

    python scripts/diagnose_features.py --records-dir "<...>/records100"

Reading the verdict
  DEPLOY != ORACLE            -> the port in modules/ is wrong (send me the output).
  DEPLOY == ORACLE != CSV     -> the notebook's stored features came from different code/data than the notebook cell shows
                                 (typically a STALE feature cache); the app is faithful to the code, not to the cache.
  all three equal             -> check [2] failed for another reason (file path / row mix-up).
"""
import argparse, os, sys, collections
import numpy as np, pandas as pd, wfdb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules import config as C
from modules.feature_extraction import waveform_features_from_100hz
from tests.reference import notebook_feature_code as NB
from tests.reference.trained_oracle import load_oracle


def locate(records_dir, rp):
    rp = str(rp).replace("\\", "/")
    for cand in (os.path.join(records_dir, rp.replace("records100/", "").replace("records500/", "")),
                 os.path.join(os.path.dirname(records_dir.rstrip("/\\")), rp)):
        if os.path.exists(cand + ".hea"):
            return cand
    return None


def suffix(name):
    return name.split("_", 1)[1] if name.split("_")[0] in C.LEADS else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=os.path.join(ROOT, "models", "reference_test_vectors.csv"))
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--width-fraction", type=float, default=None,
                    help="TEST a hypothesis: measure QRS width at this fraction of the peak (notebook code uses 0.1) in BOTH the "
                         "notebook oracle and the deployed code, e.g. 0.2")
    a = ap.parse_args()
    global NB
    frac = a.width_fraction if a.width_fraction is not None else C.TEMPLATE["width_fraction"]
    NB = load_oracle(frac); C.TEMPLATE["width_fraction"] = frac
    print(f"(QRS width fraction used for oracle and deployed code: {frac:.0%})")
    if a.width_fraction is not None:
        print(f"*** HYPOTHESIS TEST: QRS width measured at {a.width_fraction:.0%} of the peak in oracle AND deployed code ***")
    ref = pd.read_csv(a.reference)
    names = C.waveform_feature_names()
    d_dep_orc, d_orc_csv, per_feature = [], [], collections.Counter()
    first_shown = 0
    for _, row in ref.iterrows():
        base = locate(a.records_dir, row["record_path"])
        if base is None:
            print("NOT FOUND:", row["record_path"]); continue
        sig, f = wfdb.rdsamp(base)
        dep, _ = waveform_features_from_100hz(sig)
        orc = NB.record_features(base)
        if orc is None:
            print("oracle could not read", base, "(fs=%s, leads=%s)" % (f["fs"], sig.shape[1])); continue
        v_dep = np.array([dep[n] for n in names], float); v_orc = np.array([orc[n] for n in names], float)
        v_csv = row[names].to_numpy(float)
        same = lambda x, y: np.isclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True)
        d_dep_orc.append(int((~same(v_dep, v_orc)).sum())); bad = ~same(v_orc, v_csv); d_orc_csv.append(int(bad.sum()))
        for n in np.array(names)[bad]:
            per_feature[suffix(n)] += 1
        if bad.any() and first_shown < 2:
            first_shown += 1
            print(f"\nrecord {row['record_path']}: {int(bad.sum())} of {len(names)} features differ (notebook CSV vs notebook code run now)")
            for n, o, c in list(zip(np.array(names)[bad], v_orc[bad], v_csv[bad]))[:8]:
                print(f"   {n:20s} code-now={o:12.5g}   csv={c:12.5g}")
    if not d_dep_orc:
        print("No record could be processed."); sys.exit(2)
    print(f"\nrecords compared: {len(d_dep_orc)}")
    print(f"DEPLOY vs ORACLE : features differing per record, max = {max(d_dep_orc)}   (0 means the port is exact)")
    print(f"ORACLE vs CSV    : features differing per record, max = {max(d_orc_csv)}  (0 means the stored features match the code)")
    if per_feature:
        print("which feature types differ (count over records x leads):", dict(per_feature.most_common(8)))
    print("\nVERDICT:")
    if max(d_dep_orc) > 0:
        print(" The deployed port differs from the notebook code -> send me this output.")
    elif max(d_orc_csv) > 0:
        print(" The port is exact. The notebook's STORED features differ from what the notebook's own code produces now.")
        print(" Most likely a stale feature cache (computed by an earlier version of the code). See the guidance for what to do.")
    else:
        print(" All three agree; the failure in verify_consistency.py came from something else (paths/row mix-up). Send the output.")


if __name__ == "__main__":
    main()
