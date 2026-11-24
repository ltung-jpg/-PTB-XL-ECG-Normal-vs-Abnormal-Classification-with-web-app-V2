"""Compare measured ECG properties with published reference ranges.  INDEPENDENT of the classifier and of the
morphology algorithm: it only takes a dict of measurements.

Outcome per measure: "within" / "below" / "above" / "indeterminate".
"indeterminate" is returned when
    * the measurement is not available,
    * no traceable reference is configured for that measure (nothing is invented),
    * or the value lies within the measurement uncertainty of a limit (margin = 2 x timing resolution of the recording,
      an assumption of this application - automatic delineation has not been validated here).

A value outside a reference range is a MEASURED DEVIATION from a population range, not a diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass

REFERENCES = {
    "mason2007": {
        "citation": "Mason JW, Ramseth DJ, Chanter DO, et al. Electrocardiographic reference ranges derived from 79,743 "
                    "ambulatory subjects. J Electrocardiol. 2007;40(3):228-234.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/17276451",
        "basis": "2nd-98th percentile of the reference subgroup reported in the abstract (adults, digital ECGs from clinical trials)",
    },
    "rautaharju2009": {
        "citation": "Rautaharju PM, Surawicz B, Gettes LS. AHA/ACCF/HRS recommendations for the standardization and "
                    "interpretation of the electrocardiogram. Part IV: the ST segment, T and U waves, and the QT interval. "
                    "Circulation. 2009;119(10):e241-e250.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/19228821",
        "basis": "QTc normal limits commonly derived from this statement: up to 450 ms (men) / 460 ms (women)",
    },
    "p_duration_convention": {
        "citation": "Conventional adult upper limit for P-wave duration (< 120 ms), as used in: Magnani JW, Johnson VM, Sullivan LM, et al. "
                    "P-wave indices: derivation of reference values from the Framingham Heart Study. Ann Noninvasive Electrocardiol. "
                    "2010;15(4):344-352. Some sources use < 110 ms (WHO/ISFC task force, cited in Interatrial block literature, PMC6932187).",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3394095",
        "basis": "conventional upper limit 120 ms (adults); a lower limit is not defined",
    },
    "mason2007_rr": {
        "citation": "Derived from Mason et al. 2007 (J Electrocardiol 40(3):228-234): 2nd-98th percentile heart rate 48-98 bpm converted with "
                    "RR (s) = 60 / HR (bpm), i.e. 0.61-1.25 s. The conversion is arithmetic, not a separate measurement study.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/17276451",
        "basis": "Mason 2007 heart-rate range converted to RR interval (60/HR)",
    },
    "textbook_pr": {
        "citation": "Conventional PR interval range 120-200 ms (quoted in: Yang et al., ECG reference ranges in a healthcare "
                    "population, PMC3057498; Thai normal-ECG study, PMC4832775).",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3057498",
        "basis": "textbook range",
    },
}

# measure -> list of rules; the first rule whose `sex` matches is used.  Values in the unit of the measure.
RANGES = {
    "rr_s": [{"lo": 0.61, "hi": 1.25, "ref": "mason2007_rr", "label": "RR interval (s)"}],
    "p_duration_ms": [{"lo": None, "hi": 120, "ref": "p_duration_convention", "label": "P-wave duration (ms)",
                       "caveat": "Upper limit only. Some sources use 110 ms; this P-wave duration is an automatic estimate."}],
    "hr_bpm": [{"lo": 48, "hi": 98, "ref": "mason2007", "label": "Heart rate (bpm)"}],
    "pr_interval_ms": [{"lo": 113, "hi": 212, "ref": "mason2007", "label": "PR interval (ms)"}],
    "qrs_duration_ms": [{"lo": 69, "hi": 109, "ref": "mason2007", "label": "QRS duration (ms)"}],
    "qt_ms": [{"lo": 325, "hi": 452, "ref": "mason2007", "label": "QT interval (ms)",
               "caveat": "Raw QT depends on heart rate; prefer QTc."}],
    "qtc_bazett_ms": [{"sex": "male", "lo": None, "hi": 450, "ref": "rautaharju2009", "label": "QTc Bazett (ms)"},
                      {"sex": "female", "lo": None, "hi": 460, "ref": "rautaharju2009", "label": "QTc Bazett (ms)"},
                      {"sex": None, "lo": 361, "hi": 457, "ref": "mason2007", "label": "QTc Bazett (ms)"}],
    "qtc_fridericia_ms": [{"sex": None, "lo": 359, "hi": 445, "ref": "mason2007", "label": "QTc Fridericia (ms)"}],
}
LABELS = {"rr_s": "RR interval (s)", "p_duration_ms": "P-wave duration (ms)", "t_duration_ms": "T-wave duration (ms)",
          "pr_segment_ms": "PR segment (ms)", "st_segment_ms": "ST segment (ms)", "tp_segment_ms": "TP segment (ms)",
          "st_level_mv": "ST level vs PR baseline (mV)"}
# Measures WITHOUT a reference range. A literature search (Nov 2026) found no population reference range from a reliable
# source for these, so none is applied (nothing is invented). They are shown for information only.
NO_REFERENCE_NOTE = ("No population reference range from a reliable published source was identified for this measure, so none is applied. "
                     "Shown for information only.")
NO_REFERENCE_REASONS = {
    "t_duration_ms": NO_REFERENCE_NOTE + " The T-wave end is also the boundary that automatic methods place least reliably.",
    "pr_segment_ms": NO_REFERENCE_NOTE + " Use the PR interval, which has a published range.",
    "st_segment_ms": NO_REFERENCE_NOTE + " Its length depends on where the T-wave start is placed.",
    "tp_segment_ms": NO_REFERENCE_NOTE + " It depends mainly on heart rate.",
    "st_level_mv": ("No single reference range is applied. Clinical ST-deviation criteria are defined per lead at the J point, whereas this value is "
                    "the median over leads at 60-80 ms after the QRS end, so those criteria do not apply to it. Shown for information only."),
}


@dataclass
class Comparison:
    measure: str
    label: str
    value: float | None
    status: str                         # within | below | above | indeterminate
    lower: float | None = None
    upper: float | None = None
    reference_id: str | None = None
    citation: str | None = None
    url: str | None = None
    reason: str = ""
    no_range: bool = False              # True when no reference range exists for this measure (as opposed to "not measurable")

    @property
    def range_text(self) -> str:
        """Reference range for display: 'lo - hi', '<= hi', '>= lo' or '-' when there is none."""
        f = lambda x: f"{x:g}"
        if self.lower is not None and self.upper is not None:
            return f"{f(self.lower)} - {f(self.upper)}"
        if self.upper is not None:
            return f"<= {f(self.upper)}"
        if self.lower is not None:
            return f">= {f(self.lower)}"
        return "-"

    @property
    def value_text(self) -> str:
        """Value formatted for display: seconds 2 decimals, mV 3 decimals (a small ST level must not read as 0), others whole numbers."""
        if self.value is None:
            return "Not available"
        if self.measure.endswith("_mv"):
            return f"{self.value:.3f}"
        return f"{self.value:.2f}" if self.measure.endswith("_s") else f"{self.value:.0f}"


def _rule(measure: str, sex: str | None):
    rules = RANGES.get(measure)
    if not rules:
        return None
    if sex:
        for r in rules:
            if r.get("sex") == sex:
                return r
    for r in rules:
        if r.get("sex") is None:
            return r
    return rules[0] if all(r.get("sex") is None for r in rules) else None


# Uncertainty floor for measures derived from automatically placed wave boundaries (assumption of this application, NOT a
# validated figure): on synthetic ECGs with known boundaries the wavelet delineation was off by up to ~35 ms for wave
# edges, and on real example records the QRS duration differed by tens of ms between leads and methods.
BOUNDARY_UNCERTAINTY_FLOOR_MS = 30.0


def compare_measurements(values: dict[str, float | None], *, sex: str | None = None, resolution_ms: float = 10.0,
                         measure_status: dict[str, str] | None = None, spreads: dict[str, float | None] | None = None) -> list[Comparison]:
    """values: {measure_name: value or None}.  sex: 'male' | 'female' | None.
    spreads: inter-lead IQR per measure; the margin around a limit is max(2 x resolution, spread, floor)."""
    out = []
    spreads = spreads or {}
    for name, v in values.items():
        rule = _rule(name, sex)
        label = (rule or {}).get("label") or LABELS.get(name, name)
        if measure_status and measure_status.get(name) == "low_agreement":
            out.append(Comparison(name, label, v, "indeterminate", reason="Leads disagree on this measurement; value not reliable enough to compare."))
            continue
        if v is None or (measure_status and measure_status.get(name) == "not_measurable"):
            out.append(Comparison(name, label, None, "indeterminate", reason="Not measurable in this recording."))
            continue
        if rule is None:
            out.append(Comparison(name, label, v, "indeterminate", reason=NO_REFERENCE_REASONS.get(name, NO_REFERENCE_NOTE), no_range=True))
            continue
        ref = REFERENCES[rule["ref"]]
        lo, hi = rule.get("lo"), rule.get("hi")
        # margin applies to time measures in ms only
        m = max(2.0 * resolution_ms, float(spreads.get(name) or 0.0), BOUNDARY_UNCERTAINTY_FLOOR_MS) if name.endswith("_ms") else 0.0
        status, reason = "within", ""
        if hi is not None and v > hi:
            status = "above" if v - hi > m else "indeterminate"
            if status == "indeterminate":
                reason = f"Within measurement uncertainty (+/-{m:.0f} ms) of the upper limit {hi}."
        elif lo is not None and v < lo:
            status = "below" if lo - v > m else "indeterminate"
            if status == "indeterminate":
                reason = f"Within measurement uncertainty (+/-{m:.0f} ms) of the lower limit {lo}."
        elif (hi is not None and hi - v <= m and name.endswith("_ms")) or (lo is not None and v - lo <= m and name.endswith("_ms")):
            status, reason = "indeterminate", f"Within measurement uncertainty (+/-{m:.0f} ms) of a limit; classification not reliable."
        if rule.get("caveat"):
            reason = (reason + " " + rule["caveat"]).strip()
        if name.startswith("qtc") and sex is None and rule["ref"] == "mason2007":
            reason = (reason + " Sex not given: general-population range used instead of sex-specific QTc limits.").strip()
        out.append(Comparison(name, label, v, status, lo, hi, rule["ref"], ref["citation"], ref["url"], reason))
    return out


def comparisons_from_morphology(res, sex: str | None = None) -> list[Comparison]:
    """Convenience: take a `MorphologyResult` from ecg_delineation.delineate()."""
    vals = {k: m.value for k, m in res.measures.items()}
    st = {k: m.status for k, m in res.measures.items()}
    sp = {k: m.spread for k, m in res.measures.items()}
    return compare_measurements(vals, sex=sex, resolution_ms=res.resolution_ms, measure_status=st, spreads=sp)


def reference_table() -> list[dict]:
    """Every reference range the application can use (for the 'traceable references' view)."""
    rows = []
    for m, rules in RANGES.items():
        for r in rules:
            ref = REFERENCES[r["ref"]]
            rows.append({"measure": r["label"], "sex": r.get("sex") or "any", "lower": r.get("lo"), "upper": r.get("hi"),
                         "basis": ref["basis"], "citation": ref["citation"], "url": ref["url"]})
    return rows
