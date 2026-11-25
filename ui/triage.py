"""Triage logic (no Streamlit): which results need attention, how far a value lies from its published range, what a doctor
could verify next and what a student should learn from each flagged value, plus a rule-based commentary.

Scope and honesty rules
  * Nothing here computes a new clinical value. It only RE-READS the three independent analyses (model output, waveform
    measurements, published reference ranges) and lists, side by side, which of them ask for a closer look.
  * The attention level is a reading-priority prompt. It is not a diagnosis and it does not merge the three analyses
    into one conclusion: every trigger is listed with its own source.
  * Values that are unavailable are never treated as normal. A recording with unavailable or low-agreement results is
    INCOMPLETE, not ROUTINE.
  * The commentary is produced by fixed templates from the final numbers. It is NOT a language model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ROLES = ("Doctor", "Student")

# attention levels, most urgent first
LEVEL_ORDER = ("ATTENTION", "REVIEW", "INCOMPLETE", "ROUTINE")
LEVEL_KIND = {"ATTENTION": "fail", "REVIEW": "needs_review", "INCOMPLETE": "neutral", "ROUTINE": "pass"}
LEVEL_HEADLINE = {
    "ATTENTION": "Read this recording carefully",
    "REVIEW": "Check before relying on the numbers",
    "INCOMPLETE": "Some results are missing - read the tracing yourself",
    "ROUTINE": "No flagged findings - routine reading (this is not a statement that the heart is normal)",
}

QTC_HIGH_RISK_MS = 500.0   # QTc > 500 ms is associated with a 2-3 fold higher torsades risk (Drew BJ et al., AHA/ACCF scientific statement 2010, PMC3057430)

# what a doctor could verify and what a student should know, per measure and direction (generic, textbook-level)
GUIDE = {
    ("hr_bpm", "below"): ("Heart rate below the published range.",
                          ["Check symptoms and haemodynamic state.", "Review rate-slowing drugs (beta-blockers, calcium-channel blockers, digoxin).",
                           "Look for AV block or pauses on the tracing; consider electrolytes and thyroid status."],
                          "Slow heart rate (bradycardia range). Athletes and sleep can be normal; look at the P waves and the PR interval to see whether the slowing comes from the sinus node or from AV block."),
    ("hr_bpm", "above"): ("Heart rate above the published range.",
                          ["Check symptoms, pain, fever, volume status, oxygenation and anaemia.", "Check whether the rhythm is regular; look for AF, flutter or SVT on the tracing.",
                           "Review stimulants and other drugs."],
                          "Fast heart rate (tachycardia range). First decide whether it is sinus (a P wave before every QRS) or not; sinus tachycardia is usually a response to something else."),
    ("rr_s", "below"): ("Short RR interval (fast rate).", ["See heart rate above."], "RR is the time between two R waves; a short RR means a fast rate."),
    ("rr_s", "above"): ("Long RR interval (slow rate).", ["See heart rate above."], "RR is the time between two R waves; a long RR means a slow rate."),
    ("pr_interval_ms", "above"): ("Long PR interval.",
                                  ["Check the PR on the tracing; a PR above 200 ms fits first-degree AV block.", "Review AV-nodal drugs and electrolytes.", "Look for dropped beats."],
                                  "PR interval = start of the P wave to the start of the QRS. It is the AV-node delay. Longer than normal suggests slowed AV conduction."),
    ("pr_interval_ms", "below"): ("Short PR interval.",
                                  ["Look for a delta wave (pre-excitation) or a junctional rhythm on the tracing."], "A short PR means the ventricles are activated early, e.g. by an accessory pathway or a low atrial/junctional focus."),
    ("qrs_duration_ms", "above"): ("Wide QRS.",
                                   ["Look for bundle-branch block, ventricular rhythm or pre-excitation on the tracing.", "Compare with a previous ECG.", "Consider hyperkalaemia and sodium-channel-blocking drugs."],
                                   "QRS duration is ventricular depolarisation time. A wide QRS means slow or abnormal spread through the ventricles (bundle-branch block, ventricular beats, drugs, electrolytes)."),
    ("qrs_duration_ms", "below"): ("Narrow QRS.", ["Usually not a concern by itself; check the measurement on the tracing."], "A narrow QRS is normal for supraventricular activation; a value below the range is more often a measurement effect."),
    ("qt_ms", "above"): ("Long raw QT.", ["Raw QT depends on heart rate - look at QTc.", "See QTc."], "QT is the time from QRS start to T-wave end. It shortens at fast rates, so it is corrected (QTc)."),
    ("qt_ms", "below"): ("Short raw QT.", ["Raw QT depends on heart rate - look at QTc."], "QT is the time from QRS start to T-wave end. It shortens at fast rates, so it is corrected (QTc)."),
    ("qtc_bazett_ms", "above"): ("Prolonged QTc.",
                                 ["Check potassium, magnesium and calcium.", "Review QT-prolonging drugs.", "Repeat the ECG and check the T-wave end by eye; Bazett over-corrects at fast rates.",
                                  "A QTc above 500 ms is associated with a 2-3 fold higher torsades risk (AHA/ACCF statement, Drew 2010) - follow local protocol."],
                                 "QTc is the heart-rate-corrected QT. A long QTc raises the risk of torsades de pointes. Bazett over-corrects at high heart rates, so check Fridericia as well."),
    ("qtc_bazett_ms", "below"): ("Short QTc.", ["Check calcium (hypercalcaemia) and drug effects (e.g. digoxin).", "Check the T-wave end on the tracing."],
                                 "A short QTc is uncommon. Hypercalcaemia and digoxin shorten it; congenital short QT is rare."),
    ("qtc_fridericia_ms", "above"): ("Prolonged QTc (Fridericia).", ["See QTc (Bazett)."], "Fridericia correction is less biased at fast or slow rates than Bazett."),
    ("qtc_fridericia_ms", "below"): ("Short QTc (Fridericia).", ["See QTc (Bazett)."], "Fridericia correction is less biased at fast or slow rates than Bazett."),
    ("p_duration_ms", "above"): ("Long P wave.", ["Look at the P wave in II and V1 on the tracing (left atrial abnormality, interatrial conduction delay)."],
                                 "P-wave duration is atrial depolarisation. A long P wave suggests slowed atrial conduction or left atrial enlargement."),
}
MODEL_ABNORMAL = ("The research model labels this recording Abnormal.",
                  ["Read the full 12-lead tracing: rhythm, conduction, ST-T changes, axis.", "Correlate with symptoms and history; compare with a previous ECG.",
                   "The label does not say which condition is present - do not treat it as a diagnosis."],
                  "'Abnormal' here means any PTB-XL diagnosis other than normal sinus rhythm. It is one machine-learning output and says nothing about which condition.")
DISCLAIMER = ("Decision-support text for a qualified reader. It is generic, not a diagnosis, and not an instruction; follow local protocols.")


@dataclass
class Finding:
    source: str                 # "Model" | "Measurement" | "Recording quality"
    title: str
    detail: str
    severity: str               # "high" | "abnormal" | "check"
    measure: str | None = None
    direction: str | None = None
    doctor: list = field(default_factory=list)
    student: str = ""


@dataclass
class Triage:
    level: str
    headline: str
    findings: list
    reasons: list               # one line per trigger, each naming its own source
    n_unavailable: int = 0


def deviation(c, unit: str = "") -> dict:
    """How far a compared value lies outside its published range. Only for 'above'/'below'; no number is invented."""
    if c is None or c.value is None or c.status not in ("above", "below"):
        return {"amount": None, "pct": None, "text": ""}
    limit = c.upper if c.status == "above" else c.lower
    amount = abs(c.value - limit)
    pct = 100.0 * amount / abs(limit) if limit else None
    u = f" {unit}" if unit else ""
    dec = 2 if c.measure.endswith("_s") else (3 if c.measure.endswith("_mv") else 0)
    word = "above the upper" if c.status == "above" else "below the lower"
    txt = f"{amount:.{dec}f}{u} {word} limit ({limit:g}{u})" + (f", {pct:.0f} % beyond it" if pct is not None else "")
    return {"amount": amount, "pct": pct, "text": txt}


def _unit(measure: str) -> str:
    from .inspection import MEASURE_META
    return MEASURE_META.get(measure, ("", "", 0, "", ""))[1]


def _label(measure: str) -> str:
    from .inspection import MEASURE_META
    return MEASURE_META.get(measure, (measure,))[0]


def compute_findings(res, morph, cmp, validation_state: str, validation_reasons: list[str] | None = None) -> Triage:
    """All findings from the three independent analyses, each tagged with its own source."""
    findings: list[Finding] = []
    n_unavailable = 0

    # --- model (reported on its own)
    if res is not None and res.status == "ok" and res.label_name == "Abnormal":
        t, doc, stu = MODEL_ABNORMAL
        findings.append(Finding("Model", "Model output: Abnormal", f"P(Abnormal) = {res.probability_abnormal:.2f}; threshold {res.threshold:g}. Research model, not a diagnosis.",
                                "abnormal", "model", None, doc, stu))
    elif res is not None and res.status != "ok":
        n_unavailable += 1
        findings.append(Finding("Model", "No model prediction", "; ".join(res.reasons)[:200] or f"Status: {res.status}", "check"))

    # --- measurements vs published ranges
    for c in (cmp or []):
        unit = _unit(c.measure)
        if c.status in ("above", "below"):
            d = deviation(c, unit)
            guide = GUIDE.get((c.measure, c.status), (c.label, [], ""))
            sev = "abnormal"
            if c.measure == "qtc_bazett_ms" and c.status == "above" and c.value is not None and c.value > QTC_HIGH_RISK_MS:
                sev = "high"
            findings.append(Finding("Measurement", f"{_label(c.measure)} {c.status} range", f"{c.value_text} {unit}: {d['text']}".strip(), sev, c.measure, c.status, guide[1], guide[2]))
        elif c.status == "indeterminate" and not c.no_range:
            n_unavailable += 1
            ttl = f"{_label(c.measure)} could not be measured" if c.value is None else f"{_label(c.measure)} is too close to a limit to judge"
            findings.append(Finding("Measurement", ttl, c.reason or "Not measurable or too close to a limit.", "check", c.measure))
    if morph is not None and morph.ok:
        for name, m in morph.measures.items():
            if m.status == "low_agreement":
                findings.append(Finding("Measurement", f"{_label(name)}: leads disagree", "Low agreement between leads - interpret with caution.", "check", name))

    # --- recording quality
    if validation_state == "NEEDS_REVIEW":
        findings.append(Finding("Recording quality", "Recording quality needs review", "; ".join((validation_reasons or [])[:2])[:200], "check"))

    sev = {f.severity for f in findings}
    if "high" in sev or "abnormal" in sev:
        level = "ATTENTION"
    elif "check" in sev:
        level = "REVIEW" if validation_state == "NEEDS_REVIEW" and n_unavailable == 0 else "INCOMPLETE"
    else:
        level = "ROUTINE"
    # a missing morphology means nothing could be compared: never ROUTINE
    if level == "ROUTINE" and (morph is None or not morph.ok or not cmp or (res is not None and res.status != "ok")):
        level = "INCOMPLETE"
    reasons = []
    for f in findings:
        if f.severity in ("high", "abnormal"):
            reasons.append(f.title if f.source == "Model" else f"{f.source}: {f.title}" + (f" ({f.detail})" if f.source == "Measurement" else ""))
    checks = [f for f in findings if f.severity == "check"]       # many 'check' items would bury the important ones: one line per source
    for src in ("Measurement", "Model", "Recording quality"):
        grp = [f for f in checks if f.source == src]
        if len(grp) == 1:
            reasons.append(f"{src}: {grp[0].title}")
        elif grp:
            names = ", ".join(f.title.split(" is too close")[0].split(" could not")[0].split(":")[0] for f in grp[:4]) + (", ..." if len(grp) > 4 else "")
            reasons.append(f"{src}: {len(grp)} results to check by eye ({names})")
    order = {"high": 0, "abnormal": 1, "check": 2}
    findings.sort(key=lambda f: order[f.severity])
    return Triage(level, LEVEL_HEADLINE[level], findings, reasons, n_unavailable)


def section_status(triage: Triage, source: str, measures: tuple | None = None) -> tuple[str, str]:
    """(label, badge kind) for a page section: tells the reader whether the section needs reading."""
    fs = [f for f in triage.findings if f.source == source and (measures is None or f.measure in measures or f.measure is None)]
    n_ab = sum(f.severity in ("high", "abnormal") for f in fs)
    n_ck = sum(f.severity == "check" for f in fs)
    if n_ab:
        return f"{n_ab} FLAGGED", "fail"
    if n_ck:
        return f"{n_ck} TO CHECK", "needs_review"
    return "NOTHING FLAGGED", "info"


def commentary(role: str, res, morph, cmp, triage: Triage, setting: str | None = None) -> list[str]:
    """Rule-based commentary written from the final numbers. Not a language model; every sentence maps to a computed value."""
    out = []
    lead = {"ATTENTION": "Overall, this recording has findings that ask for a careful read.",
            "REVIEW": "Overall, the numbers should be checked against the tracing before they are relied on.",
            "INCOMPLETE": "Overall, some results are missing, so nothing can be called normal from the numbers alone.",
            "ROUTINE": "Overall, none of the three analyses flagged anything."}[triage.level]
    out.append(lead)
    if res is not None and res.status == "ok":
        out.append(f"The research model output is {res.label_name} (P(Abnormal) = {res.probability_abnormal:.2f}, decision threshold {res.threshold:g}). "
                   "It is a single machine-learning output and does not say which condition is present.")
    elif res is not None:
        out.append("The research model gave no prediction for this recording.")
    flagged = [c for c in (cmp or []) if c.status in ("above", "below")]
    if flagged:
        parts = []
        for c in flagged:
            d = deviation(c, _unit(c.measure))
            parts.append(f"{_label(c.measure)} {c.value_text} {_unit(c.measure)} ({d['text']})".replace("  ", " "))
        out.append("Waveform measurements outside the published ranges: " + "; ".join(parts) + ". These are automatic estimates and deviations from a population range, not diagnoses.")
    elif cmp:
        n_in = sum(c.status == "within" for c in cmp)
        out.append(f"{n_in} compared waveform measurements lie within their published ranges.")
    if triage.n_unavailable:
        out.append(f"{triage.n_unavailable} result(s) could not be judged (see the list of findings).")
    if setting and setting.lower().startswith("ambulance"):
        out.append("This recording is marked as made in an ambulance: motion and mains artefact are more likely, so check the baseline and the R-peak markers on the tracing.")
    if role == "Doctor":
        checks = []
        for f in triage.findings:
            for s in f.doctor:
                if s not in checks:
                    checks.append(s)
        if checks:
            out.append("Points to verify: " + " ".join(checks[:6]))
    else:
        for f in triage.findings:
            if f.student:
                out.append(f"{f.title}: {f.student}")
    out.append(DISCLAIMER)
    return out
