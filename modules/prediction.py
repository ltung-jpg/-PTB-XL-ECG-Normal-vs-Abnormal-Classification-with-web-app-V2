"""Standalone inference pipeline: loads the exported bundle and classifies new ECG recordings.

Workflow (specification 9.4)
    recording -> validate -> resample to 100 Hz -> notebook band-pass -> notebook R-peak detection ->
    notebook rhythm / median-beat features + demographic features -> 145-column frame in the SAVED order ->
    the ORIGINAL fitted sklearn Pipeline (median imputer -> StandardScaler -> HistGradientBoosting).

Hard rules enforced here
    * Nothing is ever fitted on patient data: only `predict_proba` of the loaded pipeline is called.
    * Missing demographics stay NaN and are filled by the ORIGINAL fitted median imputer.
    * A recording that cannot be processed reliably yields status "indeterminate" or "error" - no class label.
"""
from __future__ import annotations

import json
import warnings as _warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .data_ingestion import ECGRecording
from .feature_extraction import (FeatureSchemaError, WaveformInfo, assemble_feature_frame,
                                 build_demographic_features, extract_waveform_features)
from .preprocessing import SignalError, assess_signal_quality, prepare_for_model, preprocess_signal

REQUIRED_BUNDLE_KEYS = ["model", "feature_names", "threshold", "class_mapping", "sampling_frequency_hz",
                        "n_leads", "lead_names", "inference_config", "metadata"]


class BundleError(RuntimeError):
    """The model bundle is missing, malformed or inconsistent with this pipeline."""


def _canonical(obj):
    """JSON round trip so tuples/lists and int/float keys compare equal."""
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def load_bundle(path: str) -> dict:
    """Load and VALIDATE the Joblib bundle.  Raises BundleError with an actionable message."""
    try:
        b = joblib.load(path)
    except FileNotFoundError as e:
        raise BundleError(f"Model file not found: {path}. Run the export cell of the notebook (or "
                          "scripts/export_model_bundle.py) to create models/ptbxl_binary_model.joblib.") from e
    except Exception as e:
        raise BundleError(f"Could not load the model file ({type(e).__name__}: {e}). This usually means the installed "
                          "scikit-learn version differs from the one used for training - see models/model_metadata.json.") from e
    if not isinstance(b, dict):
        raise BundleError("The file is not a model bundle (expected a dictionary).")
    missing = [k for k in REQUIRED_BUNDLE_KEYS if k not in b]
    if missing:
        if {"model", "feature_names", "threshold"} <= set(b):
            raise BundleError("This is the OLD notebook export (model + feature names + threshold only). Convert it with "
                              "`python scripts/export_model_bundle.py --legacy <old.joblib>` (no retraining needed).")
        raise BundleError(f"Bundle is missing required entries: {missing}")

    pipe = b["model"]
    steps = getattr(pipe, "named_steps", None)
    if not steps or list(steps) != ["impute", "scale", "model"]:
        raise BundleError(f"Unexpected pipeline structure: {list(steps) if steps else type(pipe).__name__}; "
                          "expected impute -> scale -> model.")
    names = list(b["feature_names"])
    if getattr(pipe, "n_features_in_", len(names)) != len(names):
        raise BundleError(f"Pipeline expects {pipe.n_features_in_} features but the bundle lists {len(names)}.")
    if set(names) != set(C.default_feature_names()):
        extra = sorted(set(names) - set(C.default_feature_names()))[:5]
        lack = sorted(set(C.default_feature_names()) - set(names))[:5]
        raise BundleError(f"Bundle feature schema differs from this pipeline (unknown: {extra}, absent: {lack}).")
    if _canonical(b["inference_config"]) != _canonical(C.inference_config()):
        raise BundleError("The inference configuration stored in the bundle differs from modules/config.py. The deployed "
                          "feature extraction would no longer match the one the model was trained with.")
    if int(b["sampling_frequency_hz"]) != C.MODEL_FS or list(b["lead_names"]) != C.LEADS:
        raise BundleError("Bundle sampling frequency / lead order does not match the pipeline.")

    import sklearn
    trained_with = (b["metadata"].get("library_versions") or {}).get("scikit-learn")
    b["_load_warnings"] = []
    if trained_with and trained_with != sklearn.__version__:
        b["_load_warnings"].append(f"Bundle was exported with scikit-learn {trained_with}, running {sklearn.__version__}. "
                                   "Re-verify predictions (scripts/verify_consistency.py).")
    return b


def beat_consistency(info: WaveformInfo) -> float:
    """Best (over leads) median correlation between each beat and the median beat - a simple noise detector."""
    pre, post = int(round(C.TEMPLATE["pre_s"] * info.fs)), int(round(C.TEMPLATE["post_s"] * info.fs))
    xf = info.filtered
    valid = info.peaks[(info.peaks - pre >= 0) & (info.peaks + post < len(xf))]
    if len(valid) < 3:
        return float("nan")
    idx = valid[:, None] + np.arange(-pre, post + 1)[None, :]
    best = -1.0
    for k in range(xf.shape[1]):
        w = xf[idx, k]
        med = np.median(w, axis=0)
        if np.std(med) < 1e-12:
            continue
        cors = [np.corrcoef(b, med)[0, 1] for b in w if np.std(b) > 1e-12]
        if cors:
            best = max(best, float(np.median(cors)))
    return best if best > -1 else float("nan")


@dataclass
class PredictionResult:
    status: str                                 # "ok" | "indeterminate" | "error"
    label: int | None = None                    # 0 = Normal, 1 = Abnormal
    label_name: str | None = None
    probability_abnormal: float | None = None
    probability_normal: float | None = None
    threshold: float = C.DEFAULT_THRESHOLD
    reasons: list[str] = field(default_factory=list)       # why indeterminate / error
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    qc: dict = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)
    feature_frame: pd.DataFrame | None = None
    waveform_info: WaveformInfo | None = None
    record_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class ECGClassifier:
    """Loads the bundle once; prediction methods never modify the model."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.pipeline = bundle["model"]
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.threshold: float = float(bundle["threshold"])
        self.class_mapping = {int(k): v for k, v in bundle["class_mapping"].items()}
        self.metadata: dict = bundle["metadata"]
        self.is_test_stub: bool = bool(self.metadata.get("is_test_stub", False))

    @classmethod
    def from_path(cls, path: str) -> "ECGClassifier":
        return cls(load_bundle(path))

    def model_info(self) -> dict:
        m = dict(self.metadata)
        m["threshold"] = self.threshold
        m["n_features"] = len(self.feature_names)
        m["load_warnings"] = list(self.bundle.get("_load_warnings", []))
        return m

    # ------------------------------------------------------------------ features only (no model)
    def compute_features(self, rec: ECGRecording, demographics: dict | None = None, window_start_s: float = 0.0):
        """Steps 1-5 of the workflow.  Returns (PredictionResult-with-status, wf_dict, demo_dict) -
        status != "ok" means the recording must not be classified."""
        res = PredictionResult(status="ok", threshold=self.threshold, record_id=rec.record_id)
        res.warnings += list(rec.warnings)

        q = assess_signal_quality(rec.signal, rec.fs, rec.lead_names)
        res.warnings += q.warnings
        if not q.ok:
            res.status, res.reasons = "error", list(q.errors)
            return res, None, None
        try:
            sig100, notes = prepare_for_model(rec.signal, rec.fs, window_start_s)
        except SignalError as e:
            res.status, res.reasons = "error", [str(e)]
            return res, None, None
        res.notes += notes
        if sig100.shape[0] < C.MIN_DURATION_S * C.MODEL_FS:
            res.status, res.reasons = "error", ["Too few samples after resampling."]
            return res, None, None

        xf = preprocess_signal(sig100, C.MODEL_FS)
        wf, info = extract_waveform_features(xf, C.MODEL_FS)
        res.waveform_info = info
        n_peaks, hr = int(len(info.peaks)), wf["hr_bpm"]
        res.qc = {"n_r_peaks": n_peaks, "hr_bpm": None if not np.isfinite(hr) else float(hr),
                  "n_beats_in_template": info.n_valid_beats, "original_fs_hz": rec.original_fs,
                  "resampled": abs(rec.fs - C.MODEL_FS) > 1e-9, "duration_analysed_s": sig100.shape[0] / C.MODEL_FS,
                  "derived_leads": list(rec.derived_leads)}

        P = C.POLICY
        reasons: list[str] = []
        if n_peaks < P["min_r_peaks"]:
            reasons.append(f"Only {n_peaks} R-peaks detected (at least {P['min_r_peaks']} are needed to compute rhythm features).")
        elif not (P["plausible_hr_bpm"][0] <= hr <= P["plausible_hr_bpm"][1]):
            reasons.append(f"Implausible heart rate ({hr:.0f} bpm) - R-peak detection is probably unreliable.")
        if not reasons:
            bc = beat_consistency(info)
            res.qc["beat_correlation"] = bc
            if not (bc >= P["min_beat_correlation"]):
                reasons.append(f"Beats do not resemble each other (best-lead beat correlation {bc:.2f} < "
                               f"{P['min_beat_correlation']}); the signal is probably too noisy to interpret.")
        nan_frac = float(np.mean([not np.isfinite(v) for v in wf.values()]))
        if nan_frac > P["max_nan_feature_fraction"]:
            reasons.append(f"{nan_frac:.0%} of the waveform features could not be computed.")
        if reasons:
            res.status, res.reasons = "indeterminate", reasons
            return res, None, None

        d = demographics or {}
        demo = build_demographic_features(d.get("age"), d.get("sex"), d.get("height_cm"), d.get("weight_kg"))
        res.missing_inputs = [k for k, v in demo.items() if not np.isfinite(v)]
        if res.missing_inputs:
            res.notes.append("Missing demographic inputs " + ", ".join(res.missing_inputs) +
                             " were filled with the training-set median by the model's original fitted imputer.")
        if "age" in res.missing_inputs or "sex" in res.missing_inputs:
            res.warnings.append("Age and/or sex were not provided. They were present for every training record, so the "
                                "prediction is less reliable without them.")
        try:
            res.feature_frame = assemble_feature_frame(wf, demo, self.feature_names)
        except FeatureSchemaError as e:
            res.status, res.reasons = "error", [str(e)]
            return res, None, None
        return res, wf, demo

    # ------------------------------------------------------------------ prediction
    def predict_frame(self, frame: pd.DataFrame) -> float:
        """P(Abnormal) from the ORIGINAL pipeline.  predict only - never fit."""
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            return float(self.pipeline.predict_proba(frame[self.feature_names])[0, 1])

    def predict_recording(self, rec: ECGRecording, demographics: dict | None = None,
                          window_start_s: float = 0.0) -> PredictionResult:
        res, _, _ = self.compute_features(rec, demographics, window_start_s)
        if res.status != "ok":
            return res
        p1 = self.predict_frame(res.feature_frame)
        res.probability_abnormal, res.probability_normal = p1, 1.0 - p1
        res.label = int(p1 > self.threshold)                   # notebook: (score > thr)
        res.label_name = self.class_mapping[res.label]
        return res

    def predict_many(self, recordings: list[ECGRecording], demographics: "list[dict | None] | dict | None" = None,
                     window_start_s: float = 0.0) -> list[PredictionResult]:
        if isinstance(demographics, dict) or demographics is None:
            demographics = [demographics] * len(recordings)
        if len(demographics) != len(recordings):
            raise ValueError("`demographics` must have one entry per recording.")
        return [self.predict_recording(r, d, window_start_s) for r, d in zip(recordings, demographics)]

    @staticmethod
    def summarize(results: list[PredictionResult]) -> pd.DataFrame:
        return pd.DataFrame([{"record": r.record_id, "status": r.status, "prediction": r.label_name,
                              "P(abnormal)": r.probability_abnormal, "R-peaks": r.qc.get("n_r_peaks"),
                              "HR (bpm)": r.qc.get("hr_bpm"), "issues": "; ".join(r.reasons + r.warnings)[:200]}
                             for r in results])
