"""The notebook's feature code (verbatim, notebook_feature_code.py) with ONE constant changed to what the real model was
trained on: QRS width at 20 % of the peak instead of the 10 % written in the notebook cell.  Verified with
scripts/diagnose_features.py --width-fraction 0.2 on 30 real test records (exact match with the stored features)."""
import types

from . import notebook_feature_code as _verbatim


def load_oracle(width_fraction: float = 0.2):
    src = open(_verbatim.__file__).read()
    assert "0.1 * win.max()" in src
    mod = types.ModuleType("trained_oracle_module")
    exec(compile(src.replace("0.1 * win.max()", f"{width_fraction} * win.max()"), "trained_oracle", "exec"), mod.__dict__)
    return mod


NB = load_oracle()
record_features = NB.record_features
