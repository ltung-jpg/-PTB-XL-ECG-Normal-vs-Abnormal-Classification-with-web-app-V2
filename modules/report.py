"""Unified patient report (self-contained HTML).  Three independent sections, deliberately kept apart:

  A. MACHINE-LEARNING PREDICTION   - what the classifier output (probability, class); research model, not a diagnosis
  B. MEASURED PROPERTIES           - numbers measured from the waveform by the delineation module (no model involved)
  C. COMPARISON WITH REFERENCES    - measured values vs published ranges (within / below / above / indeterminate)
Nothing in B or C uses the classifier output, and nothing in A uses B or C.
"""
from __future__ import annotations

from ui import tokens as T
import base64
import datetime as dt
import html
import io

from . import config as C
from .explainability import DISCLAIMER

_CSS = f"""@page{{size:A4;margin:1.3cm 1.5cm}}body{{font-family:"Segoe UI",Roboto,Arial,Helvetica,sans-serif;max-width:980px;margin:24px auto;color:{T.TEXT};line-height:1.5;background:{T.SURFACE}}}
h1{{font-size:22px;color:{T.ACCENT}}}h2{{font-size:17px;border-bottom:2px solid {T.ACCENT};padding-bottom:4px;margin-top:28px;color:{T.TEXT}}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid {T.BORDER};padding:5px 8px;text-align:left}}
th{{background:{T.SURFACE_2};color:{T.TEXT_2}}}.within{{color:{T.INFO_TEXT};font-weight:bold}}.above,.below{{color:{T.WARNING_TEXT};font-weight:bold}}
.indeterminate{{color:{T.TEXT_MUTED};font-weight:bold}}.box{{border:1px solid {T.BORDER};border-radius:12px;padding:10px 14px;margin:10px 0;background:{T.APP_BG}}}
.warn{{background:{T.WARNING_BG};border-color:{T.WARNING}}}.stub{{background:{T.ERROR_BG};border-color:{T.ERROR};color:{T.ERROR_TEXT};font-weight:bold}}
img{{max-width:100%}}small{{color:{T.TEXT_2}}}"""


def _img(fig) -> str:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt; plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def _e(x) -> str:
    return html.escape(str(x))


def build_report(*, record_id, source_info: dict, demographics: dict, prediction, model_info: dict, morph, comparisons,
                 figures: dict, explanation: dict | None = None, language="en", commentary: list | None = None, triage=None,
                 role: str | None = None, setting: str | None = None) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    h = [f"<html><head><meta charset='utf-8'><title>ECG report {_e(record_id)}</title><style>{_CSS}</style></head><body>",
         f"<h1>ECG analysis report - {_e(record_id)}</h1><small>Generated {now}. Research prototype - not a medical device, "
         "not a diagnosis. Interpretation must be done by a qualified clinician.</small>"]
    if model_info.get("is_test_stub"):
        h.append("<div class='box stub'>TEST-STUB MODEL (trained on synthetic data). The predictions below are NOT from the PTB-XL model.</div>")

    # --- attention summary + rule-based commentary (written from the final numbers; not a language model)
    if triage is not None:
        col = {"ATTENTION": T.ERROR_TEXT, "REVIEW": T.WARNING_TEXT, "INCOMPLETE": T.TEXT_2, "ROUTINE": T.SUCCESS_TEXT}[triage.level]
        h.append(f"<h2>Summary for the reader{' (' + _e(role) + ')' if role else ''}</h2><div class='box'><b style='color:{col};font-size:16px'>"
                 f"{_e(triage.headline)}</b>" + ("<ul>" + "".join(f"<li>{_e(r)}</li>" for r in triage.reasons) + "</ul>" if triage.reasons else "") + "</div>")
    if commentary:
        h.append("<div class='box'><b>Automated commentary</b> <small>(rule-based text built from the numbers in this report; not a language model, not a diagnosis)</small>"
                 + "".join(f"<p>{_e(x)}</p>" for x in commentary) + "</div>")

    # --- recording & input
    h.append("<h2>1. Recording and inputs</h2><table>")
    if setting:
        h.append(f"<tr><th>Recording made in</th><td>{_e(setting)}</td></tr>")
    for k, v in source_info.items():
        h.append(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>")
    demo = ", ".join(f"{k}={'not provided' if v is None else v}" for k, v in demographics.items())
    h.append(f"<tr><th>Demographics given</th><td>{_e(demo)}</td></tr></table>")
    if figures.get("ecg"):
        h.append(figures["ecg"])

    # --- A. ML prediction (not printed for the Doctor reader)
    if role != "Doctor":
        h.append("<h2>A. Machine-learning prediction (classifier only)</h2>")
        if prediction.status == "ok":
            cls = prediction.label_name
            h.append(f"<div class='box'><b>Model output: {cls}</b> &nbsp; P(Abnormal) = {prediction.probability_abnormal:.3f}, "
                     f"P(Normal) = {prediction.probability_normal:.3f} (decision threshold {prediction.threshold:g}).<br>"
                     "'Abnormal' means <i>any</i> recorded diagnosis other than normal sinus rhythm in the PTB-XL labels; it does not say which.</div>")
            if figures.get("probability"):
                h.append(figures["probability"])
        else:
            h.append(f"<div class='box warn'><b>No prediction ({_e(prediction.status)}).</b><ul>" +
                     "".join(f"<li>{_e(r)}</li>" for r in prediction.reasons) + "</ul></div>")
        for w in prediction.warnings:
            h.append(f"<div class='box warn'>{_e(w)}</div>")
        for n in prediction.notes:
            h.append(f"<small>Note: {_e(n)}</small><br>")
        perf = model_info.get("performance_recorded_in_notebook") or {}
        shown = {k: v for k, v in perf.items() if isinstance(v, (int, float))}
        if shown:
            h.append("<h3>Model performance recorded on the held-out test set (from the training notebook)</h3><table><tr>" +
                     "".join(f"<th>{_e(k)}</th>" for k in shown) + "</tr><tr>" + "".join(f"<td>{v:.4f}</td>" for v in shown.values()) + "</tr></table>")
        h.append(f"<small>Model: {_e(model_info.get('algorithm'))}, {model_info.get('n_features')} features, trained on PTB-XL at "
                 f"{C.MODEL_FS} Hz. Limitations: " + _e(" ".join(model_info.get("limitations", [])[:3])) + "</small>")
        if explanation and explanation.get("table") is not None:
            h.append(f"<h3>Why did the model say this? ({_e(explanation['method'])})</h3>")
            if figures.get("explanation"):
                h.append(figures["explanation"])
            h.append(f"<small>{_e(DISCLAIMER)}</small>")

    # --- B. morphology
    h.append("<h2>B. Waveform measurements - automatic estimates (independent of the classifier)</h2>")
    if morph is None or not morph.ok:
        h.append(f"<div class='box warn'>Morphology could not be measured: {_e(getattr(morph, 'message', 'n/a'))}</div>")
    else:
        h.append(f"<small>Method: {_e(morph.method)}; leads: {_e(morph.lead)}; timing resolution {morph.resolution_ms:.0f} ms. "
                 "Automatic estimates - not validated against expert annotations here; wave boundaries can be wrong. "
                 "Check the delineation figure.</small>")
        h.append("<table><tr><th>Measure</th><th>Value (median)</th><th>Spread (IQR)</th><th>Support</th><th>Validity</th></tr>")
        for m in morph.measures.values():
            val = "-" if m.value is None else f"{m.value:.3g}"
            sp = "-" if m.spread is None else f"{m.spread:.2g}"
            h.append(f"<tr><td>{_e(m.name)}</td><td>{val}</td><td>{sp}</td><td>{m.n_beats}</td><td>{_e({'measured': 'estimate, consistent across leads', 'low_agreement': 'estimate, LOW agreement between leads', 'not_measurable': 'not available'}.get(m.status, m.status))} {_e(m.note)}</td></tr>")
        h.append("</table>")
        for w in morph.warnings:
            h.append(f"<div class='box warn'>{_e(w)}</div>")
        if figures.get("delineation"):
            h.append(figures["delineation"])

    # --- C. comparison
    h.append("<h2>C. Comparison with published reference ranges</h2>")
    if comparisons:
        h.append("<table><tr><th>Measure</th><th>Value</th><th>Reference range</th><th>Result</th><th>Source / comment</th></tr>")
        for c in comparisons:
            rng = _e(c.range_text)
            val = "-" if c.value is None else f"{c.value:.3g}"
            src = f"<a href='{_e(c.url)}' target='_blank' rel='noopener noreferrer'>{_e(c.reference_id)}</a> " if c.url else ""
            h.append(f"<tr><td>{_e(c.label)}</td><td>{val}</td><td>{rng}</td><td class='{c.status}'>{'no range' if c.no_range else c.status}</td><td>{src}{_e(c.reason)}</td></tr>")
        h.append("</table>")
        cited = {c.reference_id: (c.citation, c.url) for c in comparisons if c.reference_id}
        h.append("<small><b>References:</b><ul>" + "".join(f"<li>{_e(t)} <a href='{_e(u)}' target='_blank' rel='noopener noreferrer'>{_e(u)}</a></li>" for t, u in cited.values()) + "</ul></small>")
        if figures.get("reference"):
            h.append(figures["reference"])
        h.append("<div style='text-align:left;margin:8px 0;font-size:12px;line-height:1.45'>A value outside a range is a deviation from a published population range, not a diagnosis. "
                 "'indeterminate' = not measurable or within the measurement uncertainty of a limit; 'no range' = no reference range from a reliable published source was identified for that measure (shown for information only).</div>")

    h.append("</body></html>")
    return "\n".join(h)


def image_html(fig) -> str:
    return _img(fig)


def html_to_pdf(html_text: str) -> bytes:
    """PDF of the report, built with reportlab only (see report_pdf.py)."""
    from .report_pdf import html_to_pdf as _pdf
    return _pdf(html_text)
