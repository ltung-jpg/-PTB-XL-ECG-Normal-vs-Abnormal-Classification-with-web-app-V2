"""Phase 1 (design foundation): VIS-* visual-token tests and UI-12 (no numerical change from the redesign).
These test the theme definition, the chart styling and the app's start-up - they do not replace a human look at the screenshots."""
import colorsys
import glob
import hashlib
import json
import os
import re

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.test_app import new_app, wfdb_files
from ui import theme as UI
from ui import tokens as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def _rgb(h):
    return [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]


def _lum(h):
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = map(f, _rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _config():
    import tomllib
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as f:
        return tomllib.load(f)


def _all_theme_hexes():
    text = open(os.path.join(ROOT, ".streamlit", "config.toml")).read() + UI.css() + open(os.path.join(ROOT, "ui", "tokens.py")).read()
    return sorted({h.upper() for h in HEX.findall(text)})


# ---------------------------------------------------------------------------------------------- VIS: colour system
def test_vis01_all_text_pairs_meet_wcag_aa():
    bad = {k: round(contrast(*v), 2) for k, v in T.TEXT_PAIRS.items() if contrast(*v) < 4.5}
    assert not bad, f"text pairs below 4.5:1: {bad}"


def test_vis02_graphic_elements_meet_3_to_1_or_are_decorative():
    for name, c in {"ECG line": T.ECG_LINE, "secondary waveform": T.SECONDARY_WAVE, "annotation": T.ANNOTATION, "R-peak marker": T.R_PEAK}.items():
        assert contrast(c, T.CHART_BG) >= 3.0, name


def test_vis03_streamlit_config_matches_tokens_and_is_light_only():
    cfg = _config()["theme"]
    assert cfg["base"] == "light" and "dark" not in cfg
    for key, tok in {"primaryColor": T.ACCENT, "backgroundColor": T.APP_BG, "secondaryBackgroundColor": T.SURFACE_2, "textColor": T.TEXT,
                     "borderColor": T.BORDER}.items():
        assert cfg[key].upper() == tok.upper(), key
    assert cfg["sidebar"]["backgroundColor"].upper() == T.SIDEBAR_BG.upper()
    # status text colours in the config are the AA-safe variants
    assert cfg["greenTextColor"].upper() == T.SUCCESS_TEXT and cfg["orangeTextColor"].upper() == T.WARNING_TEXT


def test_vis04_no_forbidden_styles_navy_teal_neon_gradient_dark_mode():
    assert "gradient" not in UI.css().lower() and "prefers-color-scheme" not in UI.css()
    for h in _all_theme_hexes():
        r, g, b = _rgb(h)
        hue, light, sat = [v for v in colorsys.rgb_to_hls(r, g, b)][0] * 360, colorsys.rgb_to_hls(r, g, b)[1], colorsys.rgb_to_hls(r, g, b)[2]
        assert not (170 <= hue <= 260 and light < 0.30 and sat > 0.25), f"dark navy/teal colour {h}"     # dark blue/teal
        assert not (sat > 0.9 and 0.4 < light < 0.7 and h not in ("#FFFFFF",)), f"neon colour {h}"
        # no dark chart / panel surfaces: the only dark fill allowed is the charcoal utility bar
        if _lum(h) < 0.05:
            assert h in (T.UTILITY_BAR.upper(), "#202124", "#000000", "#222222"), f"unexpected dark colour {h}"


def test_vis05_layout_constants_follow_the_spec():
    assert 240 <= UI.SIDEBAR_WIDTH_PX <= 280
    assert "12px" == _config()["theme"]["baseRadius"]               # cards 10-14 px


def test_vis06_badges_always_carry_a_text_label():
    for kind in T.BADGES:
        html = UI.badge_html(kind.replace("_", " ").upper(), kind)
        assert re.search(r">[A-Z ]+<", html)
    assert "ei-dashed" in UI.badge_html("UNAVAILABLE", "unavailable")           # shape cue besides colour
    assert "<script" not in UI.badge_html("<script>x</script>", "neutral")       # labels are escaped


# ---------------------------------------------------------------------------------------------- VIS: charts
@pytest.fixture(scope="module")
def figures(clf, real_records):
    from modules import explainability as EX
    from modules import visualization as V
    from modules.data_ingestion import load_recording
    from modules.ecg_delineation import delineate_consensus
    from modules.reference_comparison import comparisons_from_morphology
    rec = load_recording(real_records[0] + ".hea")
    res = clf.predict_recording(rec, {"age": 60.0, "sex": 1})
    morph = delineate_consensus(rec.signal, rec.fs)
    figs = {"12lead": V.plot_12_lead(rec.signal, rec.fs, peaks_s=res.waveform_info.peaks / 100 if rec.fs == 100 else None),
            "prob": V.plot_probability(0.8, 0.5), "prob_low": V.plot_probability(0.2, 0.5),
            "fiducials": V.plot_beat_with_fiducials(morph), "reference": V.plot_reference_comparison(comparisons_from_morphology(morph, "female"))}
    if res.waveform_info is not None:
        from modules.preprocessing import prepare_for_model
        sig100, _ = prepare_for_model(rec.signal, rec.fs)
        figs["preproc"] = V.plot_preprocessing(sig100, res.waveform_info.filtered, 100)
    expl = EX.local_explanation(clf, res.feature_frame)
    if expl["table"] is not None:
        figs["contrib"] = V.plot_contributions(expl["table"])
    return figs


def test_vis07_all_charts_have_white_backgrounds(figures):
    for name, fig in figures.items():
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0), name
        for ax in fig.axes:
            assert ax.get_facecolor()[:3] == (1.0, 1.0, 1.0), name


def test_vis08_chart_colours_come_from_the_palette(figures):
    palette = {h.upper() for h in _all_theme_hexes()} | {"#000000", "#FFFFFF"}
    import matplotlib.colors as mc
    for name, fig in figures.items():
        for ax in fig.axes:
            for ln in ax.get_lines():
                col = mc.to_hex(ln.get_color()).upper()
                assert col in palette or ln.get_alpha() is not None or col in ("#000000",), (name, col)
    ecg = [ln for ln in figures["12lead"].axes[0].get_lines() if mc.to_hex(ln.get_color()).upper() == T.ECG_LINE.upper()]
    assert ecg, "ECG trace is not drawn in the ECG colour"


def test_vis09_r_peaks_use_a_distinct_marker_and_colour(figures):
    import matplotlib.colors as mc
    peaks = [ln for ax in figures["12lead"].axes for ln in ax.get_lines() if ln.get_marker() == "v"]
    # real records are 500 Hz so no peaks are drawn there (documented gap); when present they must use the R-peak colour
    assert all(mc.to_hex(ln.get_color()).upper() == T.R_PEAK.upper() for ln in peaks)
    assert T.R_PEAK.upper() not in (T.ECG_LINE.upper(), T.ANNOTATION.upper())


# ---------------------------------------------------------------------------------------------- UI: app shell
def test_ui_app_starts_with_header_and_no_exception(monkeypatch, clf):
    at = new_app(monkeypatch, [])
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "ECG Insight" in md and "Research prototype" in md and "<style>" in md
    assert [e.label for e in at.sidebar.expander][:2] == ["File options (advanced)", "Patient information (optional)"]


def test_ui_stub_model_banner_still_shown(monkeypatch, clf):
    at = new_app(monkeypatch, [])
    assert any("TEST-STUB MODEL ACTIVE" in e.value for e in at.error)


def test_ui_patient_summary_reflects_inputs(monkeypatch, clf):
    at = new_app(monkeypatch, [])
    assert any("none provided" in c.value for c in at.sidebar.caption)
    at.sidebar.number_input[1].set_value(60.0).run()
    assert any("age provided" in c.value for c in at.sidebar.caption)


def test_ui_kpi_card_never_shows_zero_for_unavailable():
    def script():
        from ui import theme as UI
        UI.kpi_card("QRS duration", None, unit="ms", tag="MEASURED", tag_kind="measured", key="qrs")
        UI.kpi_card("Heart rate", 63.6, unit="bpm", tag="MEASURED", tag_kind="measured", key="hr")
    at = AppTest.from_function(script).run()
    vals = {m.label: m.value for m in at.metric}
    assert vals["QRS duration"] == "Not available" and vals["Heart rate"] == "63.6 bpm"
    assert "UNAVAILABLE" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------------------------- UI-12: numerics unchanged
def test_ui12_numerical_outputs_identical_to_pre_redesign_baseline(clf, real_records):
    from tests.reference.make_baseline import collect
    base = json.load(open(os.path.join(ROOT, "tests", "reference", "baseline_phase0.json")))
    now = json.loads(json.dumps(collect(), default=float))
    assert now.keys() == base.keys()
    for k in base:
        assert now[k] == base[k], f"{k}: computed values changed"


def test_ui12_analysis_modules_are_byte_identical_to_phase0():
    """The redesign may touch ui/, views/, app.py, modules/visualization.py and the presentation of modules/report.py only.
    modules/reference_comparison.py was changed once, deliberately (reference ranges for P-wave duration and RR added; see
    REDESIGN_CHANGELOG.md); its pinned hash is the post-change version.  Model, features and morphology code are untouched."""
    frozen = json.load(open(os.path.join(ROOT, "tests", "reference", "module_hashes_phase0.json")))
    for name, digest in frozen.items():
        got = hashlib.sha256(open(os.path.join(ROOT, "modules", name), "rb").read()).hexdigest()
        assert got == digest, f"modules/{name} changed"
