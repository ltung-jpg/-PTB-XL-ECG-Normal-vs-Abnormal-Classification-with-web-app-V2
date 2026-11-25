"""Pure presentation logic (no Streamlit): validation badges, field states, model/analysis eligibility and KPI card specs.
Everything here only RE-LABELS facts computed by the backend; it never computes or changes a clinical value.

Evidence vocabulary (scientific integrity):
  MEASURED     value read from the signal by a deterministic detector (R-peaks, RR, HR)
  ESTIMATED    automatic delineation result (NeuroKit2 DWT) - NOT validated against expert annotation
  DERIVED      computed from other values (e.g. QTc from QT and RR) - inherits their uncertainty
  PREDICTED    output of the machine-learning model
  UNAVAILABLE  could not be obtained; shown as 'Not available', never as 0
"""
from __future__ import annotations

import math

from modules import config as C

FIELD_STATES = ("AVAILABLE", "MISSING", "UNKNOWN", "NOT APPLICABLE", "DERIVABLE", "INVALID")
STATE_KIND = {"AVAILABLE": "pass", "MISSING": "needs_review", "UNKNOWN": "neutral", "NOT APPLICABLE": "neutral",
              "DERIVABLE": "info", "INVALID": "fail"}
VALIDATION_KIND = {"PASS": "pass", "NEEDS_REVIEW": "needs_review", "FAIL": "fail"}


def _num(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------------------------------------------- validation
def validation_status(obj) -> tuple[str, list[str]]:
    """PASS / NEEDS_REVIEW / FAIL for one loaded object (an ECGRecording or the exception raised while loading it)."""
    if not hasattr(obj, "signal"):
        return "FAIL", [f"{type(obj).__name__}: {obj}"]
    if obj.original_fs < C.MODEL_FS - 1e-9:
        return "FAIL", [f"Sampling rate {obj.original_fs:g} Hz is below the {C.MODEL_FS} Hz the model requires; "
                        "upsampling would invent signal content, so the recording is not analyzed."]
    reasons = list(obj.warnings)
    if obj.quality is not None:
        reasons += list(obj.quality.warnings)
        if obj.quality.errors:
            return "FAIL", list(obj.quality.errors) + reasons
        if obj.quality.flat_leads:
            reasons.append("Flat (constant) lead(s): " + ", ".join(obj.quality.flat_leads))
    return ("NEEDS_REVIEW" if reasons else "PASS"), reasons


# ---------------------------------------------------------------------------------------------------- field states
def field_states(rec, demographics: dict) -> list[dict]:
    """One row per input field with its availability state."""
    rows = []

    def add(field, state, value="", note=""):
        assert state in FIELD_STATES
        rows.append({"Field": field, "State": (state, STATE_KIND[state]), "Value": value, "Note": note})

    n_leads = len(rec.lead_names)
    if rec.derived_leads:
        add("Lead set", "DERIVABLE", f"{n_leads} leads", "Derived from I and II (exact formulas, flagged): " + ", ".join(rec.derived_leads))
    elif n_leads == 12:
        add("Lead set", "AVAILABLE", "12 leads", "Read from the file; leads are never guessed or duplicated")
    else:
        add("Lead set", "INVALID", f"{n_leads} leads", "12 leads are required")
    if not _num(rec.original_fs):
        add("Sampling frequency", "UNKNOWN", "", "Must be known; it is never guessed")
    elif rec.original_fs < C.MODEL_FS - 1e-9:
        add("Sampling frequency", "INVALID", f"{rec.original_fs:g} Hz", f"Below the {C.MODEL_FS} Hz required by the model")
    else:
        add("Sampling frequency", "AVAILABLE", f"{rec.original_fs:g} Hz",
            "Resampled to 100 Hz for the model" if abs(rec.original_fs - C.MODEL_FS) > 1e-9 else "Native model rate")
    add("Signal units", "AVAILABLE" if rec.original_units else "UNKNOWN", str(rec.original_units or ""), "Converted to mV")
    dur = rec.duration_s
    if dur < C.MIN_DURATION_S:
        add("Duration", "INVALID", f"{dur:.1f} s", f"At least {C.POLICY.get('min_duration_s', 5)} s are required")
    else:
        add("Duration", "AVAILABLE", f"{dur:.1f} s", "Only one 10-s window is analysed" if dur > C.WINDOW_S + 1e-9 else "")
    miss = max((rec.quality.missing_fraction or {}).values(), default=0.0) if rec.quality else 0.0
    add("Missing samples", "AVAILABLE" if miss == 0 else "INVALID", "none" if miss == 0 else f"up to {100 * miss:.1f} % in a lead",
        "" if miss == 0 else "Shaded in the viewer")
    for key, label, unit in (("age", "Age", "years"), ("sex", "Sex", ""), ("height_cm", "Height", "cm"), ("weight_kg", "Weight", "kg")):
        v = demographics.get(key)
        if _num(v):
            shown = ("female" if v == 1 else "male") if key == "sex" else f"{v:g} {unit}"
            add(label, "AVAILABLE", shown, "")
        else:
            add(label, "MISSING", "Not provided", "Model's original median imputer is used (result flagged)")
    if _num(demographics.get("height_cm")) and _num(demographics.get("weight_kg")):
        add("BMI", "DERIVABLE", "from height and weight", "Computed by the feature pipeline")
    else:
        add("BMI", "MISSING", "Not provided", "Needs height and weight")
    return rows


# ---------------------------------------------------------------------------------------------------- eligibility
def eligibility(rec, res, morph, cmp, expl_available: bool) -> list[dict]:
    """Which analyses can run on this recording, with the backend's own reasons."""
    rows = []

    def add(cap, label, kind, reason):
        rows.append({"Analysis": cap, "Status": (label, kind), "Reason": reason})

    add("Waveform viewing", "ELIGIBLE", "pass", "A valid 12-lead recording is loaded")
    if res.status == "ok":
        limited = bool(res.missing_inputs)
        add("Machine-learning classification", "LIMITED" if limited else "ELIGIBLE", "needs_review" if limited else "pass",
            ("Missing inputs filled by the model's imputer: " + ", ".join(res.missing_inputs)) if limited else "All checks passed")
    else:
        add("Machine-learning classification", "BLOCKED", "fail", "; ".join(res.reasons) or f"Model status: {res.status}")
    if morph is not None and morph.ok:
        low = any(m.status == "low_agreement" for m in morph.measures.values())
        add("Waveform measurements (estimates)", "LIMITED" if low else "ELIGIBLE", "needs_review" if low else "pass",
            "Some measures show low agreement between leads" if low else "Delineation succeeded on enough leads")
    else:
        add("Waveform measurements (estimates)", "BLOCKED", "fail", getattr(morph, "message", "") or "Not computed")
    add("Published reference ranges", "ELIGIBLE" if cmp else "BLOCKED", "pass" if cmp else "fail",
        "Compared against sourced ranges" if cmp else "Needs waveform measurements")
    add("Explainability", "ELIGIBLE" if expl_available else "BLOCKED", "pass" if expl_available else "fail",
        "SHAP explanation available" if expl_available else "Needs a successful prediction and a working SHAP explainer")
    add("Dataset reference labels", "NOT SUPPORTED", "neutral", "Not read by this app")
    return rows


# ---------------------------------------------------------------------------------------------------- measurement cards
# name -> (label, unit, decimals, tag, tag_kind)
MEASURE_META = {
    "rr_s": ("RR interval", "s", 3, "MEASURED", "measured"),
    "hr_bpm": ("Heart rate", "bpm", 1, "MEASURED", "measured"),
    "pr_interval_ms": ("PR interval", "ms", 0, "ESTIMATED", "estimated"),
    "qrs_duration_ms": ("QRS duration", "ms", 0, "ESTIMATED", "estimated"),
    "qt_ms": ("QT interval", "ms", 0, "ESTIMATED", "estimated"),
    "qtc_bazett_ms": ("QTc (Bazett)", "ms", 0, "DERIVED", "derived"),
    "qtc_fridericia_ms": ("QTc (Fridericia)", "ms", 0, "DERIVED", "derived"),
    "p_duration_ms": ("P-wave duration", "ms", 0, "ESTIMATED", "estimated"),
    "t_duration_ms": ("T-wave duration", "ms", 0, "ESTIMATED", "estimated"),
    "pr_segment_ms": ("PR segment", "ms", 0, "ESTIMATED", "estimated"),
    "st_segment_ms": ("ST segment", "ms", 0, "ESTIMATED", "estimated"),
    "tp_segment_ms": ("TP segment", "ms", 0, "ESTIMATED", "estimated"),
    "st_level_mv": ("ST level", "mV", 3, "ESTIMATED", "estimated"),
}
EXTRA_ORDER = ["pr_segment_ms", "st_segment_ms", "tp_segment_ms", "st_level_mv"]
KPI_ORDER = ["hr_bpm", "rr_s", "pr_interval_ms", "qrs_duration_ms", "qt_ms", "qtc_bazett_ms", "p_duration_ms", "t_duration_ms"]
VALIDITY_TEXT = {"measured": "Consistent across leads", "low_agreement": "Low agreement between leads - interpret with caution",
                 "not_measurable": "Not measurable in this recording"}


def measurement_card(name: str, m, method: str) -> dict:
    """KPI spec for one backend Measurement. Unavailable values stay None (displayed 'Not available', never 0).
    Only what the reader needs is on the card: value, evidence tag, a LOW AGREEMENT chip when the leads disagree."""
    label, unit, dec, tag, kind = MEASURE_META.get(name, (name, "", 3, "ESTIMATED", "estimated"))
    ok = m is not None and _num(m.value) and m.status != "not_measurable"
    value = None if not ok else (round(float(m.value), dec) if dec else int(round(float(m.value))))
    extra = ("LOW AGREEMENT", "needs_review") if ok and getattr(m, "status", "") == "low_agreement" else None
    return {"label": label, "value": value, "unit": unit, "tag": tag, "tag_kind": kind, "extra_chip": extra, "key": name}


def peaks_card(res) -> dict:
    n = res.qc.get("n_r_peaks") if res is not None and res.qc else None
    return {"label": "Detected R-peaks", "value": None if n is None else int(n), "unit": None, "tag": "MEASURED", "tag_kind": "measured", "key": "rpeaks"}


def add_abnormality(card: dict, c, role: str | None) -> dict:
    """Attach an 'abnormality' block to a card ONLY when its value lies outside the published range (c = Comparison or None).
    Within range, indeterminate, no range or unavailable: nothing is attached, so no warning is shown."""
    from . import triage as TR
    if c is None or card.get("value") is None or c.no_range or c.status not in ("above", "below"):
        return card
    unit = card.get("unit") or ""
    dev = TR.deviation(c, unit)["text"]
    lines = []
    guide = TR.GUIDE.get((c.measure, c.status))
    if guide:
        lines = list(guide[1]) if role == "Doctor" else ([guide[2]] if guide[2] else [])
    high = c.measure == "qtc_bazett_ms" and c.status == "above" and c.value > TR.QTC_HIGH_RISK_MS
    card["abnormality"] = {"title": "ABOVE RANGE" if c.status == "above" else "BELOW RANGE", "range": f"{c.range_text} {unit}".strip(), "deviation": dev,
                           "high": high, "lines_title": "What to verify" if role == "Doctor" else "What it means", "lines": lines}
    return card
