import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
STUB_DIR = os.path.join(ROOT, "models", "test_stub")
UPLOADS = "/mnt/user-data/uploads"


@pytest.fixture(scope="session")
def stub_path():
    p = os.path.join(STUB_DIR, "ptbxl_binary_model.joblib")
    if not os.path.exists(p):
        from scripts.make_test_stub import build_stub
        build_stub()
    return p


@pytest.fixture(scope="session")
def clf(stub_path):
    from modules.prediction import ECGClassifier
    return ECGClassifier.from_path(stub_path)


@pytest.fixture(scope="session")
def real_records():
    """Real PTB-XL WFDB records (500 Hz) if present in the sandbox."""
    paths = []
    for folder in (UPLOADS, os.path.join(ROOT, "examples", "ptbxl_records")):       # sandbox upload folder or the bundled examples
        paths = sorted(glob.glob(os.path.join(folder, "0000*_hr.hea")))
        if paths:
            break
    if not paths:
        pytest.skip("real PTB-XL example records not available")
    return [p[:-4] for p in paths]
