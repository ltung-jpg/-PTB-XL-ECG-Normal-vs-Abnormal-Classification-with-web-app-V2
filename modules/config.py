"""Single source of truth for every inference parameter.

All values below MIRROR the executed notebook `ptbxl_ecg_binary_classification_4th.ipynb`
(feature-extraction cell 12, demographic cleaning cell 10, model cell 22).  They are also written
into the exported model bundle, and `prediction.load_bundle` refuses a bundle whose stored
parameters differ from these, so the deployed pipeline cannot silently drift from the trained one.

Do NOT change a value here unless the model is retrained with the same change.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------- signal / leads
MODEL_FS = 100                         # Hz - sampling rate of the recorded final experiment (records100)
N_LEADS = 12
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]   # PTB-XL order
WINDOW_S = 10.0                        # PTB-XL records are 10 s long
MIN_DURATION_S = 5.0                   # deployment policy (NOT from the notebook): refuse shorter recordings

# ----------------------------------------------------------------------------- notebook parameters
BANDPASS = {"kind": "butterworth_bandpass", "order": 2, "low_hz": 0.5, "high_hz": 40.0,
            "zero_phase": True, "implementation": "scipy.signal.sosfiltfilt (sos)"}

QRS_DETECTION = {
    "algorithm": "Pan-Tompkins-style (custom, from notebook)",
    "band_hz": [5, 15], "band_order": 2,
    "smoothing_window_s": 0.12,                 # moving average of squared derivative
    "threshold_fraction": 0.35,                 # of the 99th percentile of the energy signal
    "threshold_percentile": 99,
    "min_distance_s": 0.25,                     # find_peaks distance
    "refine_half_window_s": 0.06,               # refine to max |amplitude| of the filtered lead
    "min_peak_spacing_s": 0.20,                 # drop refined peaks closer than this
    "peak_leads": [1, 0, 10, 7],                # II, I, V5, V2 (tried in this order)
    "min_peaks": 3,
}

TEMPLATE = {
    "pre_s": 0.25, "post_s": 0.45,              # median-beat window around the R peak
    "min_beats": 2,
    "baseline_window_s": [-0.08, -0.04],        # PR-segment baseline (relative to R)
    "qrs_window_s": [-0.06, 0.08],
    "width_window_s": [-0.12, 0.12],
    # QRS width is measured at 20 % of the peak height.  VERIFIED against the stored training features of the real model
    # (30 test records, exact match).  Notebook cell 12 as written says 10 %, but the cached features the model was
    # trained on were computed with 20 %; the deployed pipeline must reproduce what the model actually saw.
    "width_fraction": 0.2,
    "st_window_s": [0.08, 0.12],
    "t_window_s": [0.15, 0.40],
    "p_window_s": [-0.20, -0.10],
}

RHYTHM_FEATURES = ["hr_bpm", "mean_rr_s", "sdnn_s", "rmssd_s", "pnn50", "rr_cv", "rr_range_s", "n_beats"]
LEAD_STAT_FEATURES = ["std", "ptp", "skew", "kurt"]
TEMPLATE_FEATURES = ["qrs_max", "qrs_min", "qrs_area", "qrs_width_ms", "st_dev", "t_amp", "p_amp"]
DEMOGRAPHIC_FEATURES = ["age", "sex", "height", "weight", "bmi"]

DEMOGRAPHICS = {
    "feature_order": DEMOGRAPHIC_FEATURES,
    "age_cap_threshold": 120, "age_cap_value": 90,          # PTB-XL stores ages >= 90 as 300
    "height_cm_valid_range": [50, 230],
    "weight_kg_valid_range": [3, 250],
    "bmi_valid_range": [8, 70],
    # IMPORTANT: coding in the PTB-XL CSV the model was trained on (verified from the CSV: code 0 = 52 % of
    # records with mean height 173.5 cm).  The notebook comment "0 = female" was wrong; the DATA is what matters.
    "sex_coding": {"male": 0, "female": 1},
    "median_imputation_of_missing": True,
}

CLASS_MAPPING = {0: "Normal", 1: "Abnormal"}
DEFAULT_THRESHOLD = 0.5                  # probability of class 1 above which the record is "Abnormal"

# Published model hyper-parameters (provided by the user; stored as given, never altered)
MODEL_FAMILY = "HistGradientBoostingClassifier"
MODEL_HYPERPARAMETERS = {"learning_rate": 0.1, "max_leaf_nodes": 31, "class_weight": "balanced", "random_state": 42}
PIPELINE_STEPS = ["impute (median)", "scale (StandardScaler)", "model (HistGradientBoostingClassifier)"]

# Performance recorded in the notebook on the independent test set (fold 10, 2,192 ECGs).
# Provided by the user as existing experimental results.  Never recomputed or modified here.
RECORDED_TEST_PERFORMANCE = {
    "accuracy": 0.8531, "balanced_accuracy": 0.8611, "precision": 0.9257,
    "sensitivity": 0.8171, "specificity": 0.9051, "f1": 0.8680, "roc_auc": 0.9297,
    "n_test_records": 2192, "test_set": "PTB-XL strat_fold 10",
}

# ----------------------------------------------------------------------------- deployment policy
# (NOT part of the trained pipeline; they decide when the app refuses / flags a recording)
POLICY = {
    "min_r_peaks": 3,
    "plausible_hr_bpm": [20, 250],
    "max_missing_fraction_per_lead": 0.20,
    "flat_lead_std_mv": 1e-6,
    "implausible_ptp_mv": [0.05, 20.0],  # peak-to-peak of lead II outside this -> warn (probably wrong units)
    "max_nan_feature_fraction": 0.30,    # of the 140 waveform features
    # deployment policy (not a notebook value): the best lead's beats must resemble their own median beat,
    # otherwise the "R-peaks" are noise and the signal is not an interpretable ECG
    "min_beat_correlation": 0.5,
}

LEAD_ALIASES = {l: l for l in LEADS}
UNIT_TO_MV = {"mv": 1.0, "uv": 1e-3, "v": 1e3, "nv": 1e-6}


def waveform_feature_names() -> list[str]:
    """The 140 waveform features in the exact order produced by the notebook's `record_features`."""
    names = list(RHYTHM_FEATURES)
    for lead in LEADS:
        names += [f"{lead}_{n}" for n in LEAD_STAT_FEATURES + TEMPLATE_FEATURES]
    return names


def default_feature_names() -> list[str]:
    """All 145 model inputs: 140 waveform features followed by 5 demographic features."""
    return waveform_feature_names() + list(DEMOGRAPHIC_FEATURES)


def inference_config() -> dict:
    """Everything the bundle must carry so that inference can be audited against training."""
    return {"model_fs_hz": MODEL_FS, "n_leads": N_LEADS, "lead_names": LEADS, "window_s": WINDOW_S,
            "bandpass": BANDPASS, "qrs_detection": QRS_DETECTION, "template": TEMPLATE,
            "rhythm_features": RHYTHM_FEATURES, "lead_stat_features": LEAD_STAT_FEATURES,
            "template_features": TEMPLATE_FEATURES, "demographics": DEMOGRAPHICS}
