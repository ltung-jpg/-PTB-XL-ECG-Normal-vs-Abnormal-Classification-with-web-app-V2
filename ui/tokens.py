"""Design tokens - "Clinical Burgundy & White", colour theme sampled from the user's reference site.
Pure Python (no Streamlit import) so charts, CSS, config checks and tests all read ONE definition.
Every foreground/background pair that carries text is checked against WCAG AA (4.5:1) by tests/test_ui_theme.py."""

# surfaces
APP_BG = "#F7F8FA"
SURFACE = "#FFFFFF"
SIDEBAR_BG = "#FAFAFB"
SURFACE_2 = "#EEF1F6"          # light grey panel (sampled from reference)
UTILITY_BAR = "#222222"        # (sampled)
BORDER = "#DFE3E7"
# brand
ACCENT = "#9D3141"             # burgundy (sampled)
ACCENT_HOVER = "#832636"
ACCENT_TINT = "#F8ECEE"
# text
TEXT = "#202124"
TEXT_2 = "#59616B"
TEXT_MUTED = "#68717D"
# functional (fill / tint; *_TEXT is the AA-safe text colour for use on the tint AND on white)
SUCCESS, SUCCESS_BG, SUCCESS_TEXT = "#33845B", "#E8F5EC", "#2E7953"
WARNING, WARNING_BG, WARNING_TEXT = "#A86F26", "#FFF5E6", "#9A6622"
ERROR, ERROR_BG, ERROR_TEXT = "#B84242", "#FCECEC", "#B84242"
INFO, INFO_BG, INFO_TEXT = "#466D95", "#EAF2F8", "#466D95"
# charts
ECG_LINE = "#9D3141"
SECONDARY_WAVE = "#526A7A"
ANNOTATION = "#6675A4"         # reference annotations
R_PEAK = "#B9824F"             # graphic marker only (3.3:1 on white) - never used for text
GRID_MAJOR = "#D8DDE2"
GRID_MINOR = "#EDF0F2"
CHART_BG = "#FFFFFF"

FONT_STACK = '"Source Sans 3", "Source Sans Pro", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

# evidence tags (colour never carries the meaning on its own: the label is always shown)
#   kind -> (text, background)
BADGES = {
    "pass": (SUCCESS_TEXT, SUCCESS_BG), "needs_review": (WARNING_TEXT, WARNING_BG), "fail": (ERROR_TEXT, ERROR_BG),
    "measured": (INFO_TEXT, INFO_BG), "derived": (TEXT_2, SURFACE_2), "estimated": (WARNING_TEXT, WARNING_BG),
    "predicted": (ACCENT, ACCENT_TINT), "unavailable": (TEXT_2, SURFACE_2), "neutral": (TEXT_2, SURFACE_2),
    "info": (INFO_TEXT, INFO_BG),
}

# every (text, background) pair that must satisfy WCAG AA for normal text
TEXT_PAIRS = {
    "body text on app bg": (TEXT, APP_BG), "body text on surface": (TEXT, SURFACE), "body text on grey panel": (TEXT, SURFACE_2),
    "secondary text on surface": (TEXT_2, SURFACE), "secondary text on grey panel": (TEXT_2, SURFACE_2),
    "muted text on surface": (TEXT_MUTED, SURFACE), "muted text on app bg": (TEXT_MUTED, APP_BG),
    "accent text on surface": (ACCENT, SURFACE), "accent text on tint": (ACCENT, ACCENT_TINT), "accent text on grey panel": (ACCENT, SURFACE_2),
    "white on accent button": (SURFACE, ACCENT), "white on accent hover": (SURFACE, ACCENT_HOVER), "white on utility bar": (SURFACE, UTILITY_BAR),
    **{f"badge {k} text": v for k, v in BADGES.items()},
    **{f"{k} text on white": (v[0], SURFACE) for k, v in BADGES.items()},
    "annotation text on surface": (ANNOTATION, SURFACE),
}
