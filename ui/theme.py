"""Streamlit presentation helpers: CSS, utility bar, header, cards, badges, KPI cards and notices.
No analysis code lives here and nothing here changes a computed value."""
from __future__ import annotations

import html
from contextlib import contextmanager
from string import Template

import streamlit as st

from . import tokens as T

SIDEBAR_WIDTH_PX = 272          # spec: 240-280 px

_CSS = Template("""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://api.fontshare.com/v2/css?f[]=clash-display@500,600,700&f[]=clash-grotesk@500,600,700&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button, .stApp input, .stApp textarea,
.stApp summary, .stApp a, .ei-badge, .ei-util span, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] * { font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif !important; }
.ei-brand, .ei-sub { font-family: "Clash Display", "Inter", "Segoe UI", Arial, sans-serif !important; }
html body .stApp [data-testid="stTopNavLink"], html body .stApp [data-testid="stTopNavLink"] *, html body .stApp [data-testid="stTopNav"] a, html body .stApp [data-testid="stTopNav"] a * { font-family: "Clash Display", "Inter", "Segoe UI", Arial, sans-serif !important; }
/* no anchor-link icon next to headings (it has no function here) */
[data-testid="stHeaderActionElements"], [data-testid="stHeadingActionElements"], h1 a[href^="#"], h2 a[href^="#"], h3 a[href^="#"] { display: none !important; }
/* page and section titles use the brand font */
html body .stApp h1, html body .stApp h2, html body .stApp h3, html body .stApp [data-testid="stHeading"] *, html body .stApp .ei-cardtitle { font-family: "Clash Display", "Inter", "Segoe UI", Arial, sans-serif !important; letter-spacing: -.01em; }
[data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 3.25rem !important; padding-bottom: 3rem !important;
  padding-left: 2rem !important; padding-right: 2rem !important; max-width: 1600px !important; }
html { font-size: 19.5px; }
/* top bar (page navigation): solid burgundy, never transparent, so scrolled content cannot show through it */
[data-testid="stHeader"] { background: ${accent} !important; box-shadow: 0 2px 6px rgba(32,33,36,.25); }
[data-testid="stHeader"] a, [data-testid="stHeader"] a *, [data-testid="stHeader"] [data-testid="stToolbar"] button,
[data-testid="stHeader"] [data-testid="stToolbar"] a { color: #FFFFFF !important; }
/* the five page links share the bar evenly and are centred in it */
[data-testid="stTopNavLink"], [data-testid="stTopNavLink"] * { font-size: clamp(0.95rem, calc(0.5rem + 1vw), 1.28rem) !important; }
[data-testid="stTopNav"] { flex: 1 1 auto; justify-content: center; }
[data-testid="stTopNav"] > div { display: flex !important; flex: 1 1 auto; justify-content: space-evenly; max-width: 900px; margin: 0 auto; }
[data-testid="stTopNav"] > div > div { display: flex !important; flex: 1 1 0; justify-content: center; }
[data-testid="stTopNavLinkContainer"] { width: 100%; display: flex; justify-content: center; }
[data-testid="stTopNavLink"] { justify-content: center; min-width: 0; padding-left: .6rem; padding-right: .6rem; }
[data-testid="stHeader"] a { font-weight: 600; border-radius: 8px; }
[data-testid="stHeader"] a:hover { background: ${accent_hover} !important; }
[data-testid="stHeader"] a[aria-current="page"], [data-testid="stHeader"] a[data-active="true"] { background: rgba(255,255,255,.22) !important; }
/* the hint 'Press Enter to apply' sat on top of the typed number in narrow inputs */
[data-testid="InputInstructions"] { display: none !important; }
[data-testid="stSidebar"] label p { font-size: 1rem; }
.stButton button, .stDownloadButton button { min-height: 2.8rem; font-size: 1.05rem; }
/* no sidebar: every control lives on the page where it is needed */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .5rem; }
h1, h2, h3 { color: ${text}; letter-spacing: -0.01em; }

/* utility bar + header */
.ei-util { background: ${util}; color: #FFFFFF; font-size: .8rem; line-height: 1.25; padding: .35rem 1.25rem; border-radius: 12px 12px 0 0;
           display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
.ei-header { background: ${surface}; border: 1px solid ${border}; border-top: 0; border-radius: 0 0 12px 12px; padding: 1.5rem 1.75rem;
             display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
.ei-mark { width: 68px; height: 68px; border-radius: 16px; background: ${accent}; display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto; }
.ei-brand { font-size: 2.6rem; font-weight: 800; letter-spacing: -.02em; color: ${accent}; line-height: 1.1; }
.ei-sub { font-size: 1.25rem; margin-top: .2rem; color: ${text2}; }

/* cards: st.container(border=True, key="card-...") */
[class*="st-key-card"] { background: ${surface}; border-radius: 12px; box-shadow: 0 1px 2px rgba(32,33,36,.06); padding-bottom: .5rem; }
.ei-cardtitle { font-size: 1.05rem; font-weight: 600; color: ${text}; margin: 0 0 .15rem 0; }
.ei-guide-title { font-size: 1.6rem; font-weight: 700; color: ${accent}; line-height: 1.2; margin: 1.25rem 0 .5rem 0; }
.ei-guide-sub { font-size: 1.1rem; font-weight: 700; color: ${text}; margin: .25rem 0 .35rem 0; }
.ei-sectionlabel { font-size: .8125rem; font-weight: 600; color: ${text2}; text-transform: uppercase; letter-spacing: .04em; margin: .25rem 0 .25rem 0; }

/* badges */
.ei-badge { display: inline-block; font-size: .8125rem; font-weight: 600; line-height: 1; padding: .3rem .6rem; border-radius: 999px; border: 1px solid transparent; white-space: nowrap; }
.ei-badge.ei-dashed { border-style: dashed; border-color: ${text2}; }
.ei-kpi-note { font-size: 1rem; color: ${text2}; margin-top: .25rem; }

/* tables with badges */
table.ei-table { width: 100%; border-collapse: collapse; font-size: 1rem; background: ${surface}; border: 1px solid ${border}; border-radius: 12px; overflow: hidden; }
table.ei-table th { background: ${grey}; color: ${text2}; text-align: left; font-weight: 600; padding: .55rem .75rem; border-bottom: 1px solid ${border}; }
table.ei-table td { padding: .55rem .75rem; border-bottom: 1px solid ${border}; vertical-align: top; color: ${text}; }
table.ei-table tr:last-child td { border-bottom: 0; }

/* workflow steps + empty state */
.ei-step { background: ${surface}; border: 1px solid ${border}; border-radius: 12px; padding: 1rem 1.1rem; height: 100%; box-shadow: 0 1px 2px rgba(32,33,36,.06); }
.ei-stepnum { display: inline-flex; width: 28px; height: 28px; border-radius: 50%; background: ${tint}; color: ${accent}; font-weight: 700; align-items: center; justify-content: center; margin-bottom: .5rem; }
.ei-steptitle { font-weight: 600; font-size: 1.05rem; margin-bottom: .2rem; color: ${text}; }
.ei-steptext { color: ${text2}; font-size: 1rem; }
.ei-empty { background: ${surface}; border: 1px dashed ${border}; border-radius: 12px; padding: 1.6rem 1.4rem; text-align: center; color: ${text2}; }
.ei-empty b { color: ${text}; font-size: 1.1rem; display: block; margin-bottom: .3rem; }
.ei-kpi-line { font-size: .95rem; color: ${text2}; margin-top: .2rem; }
.ei-kpi-dev { font-size: 1rem; font-weight: 600; margin-top: .3rem; }
.ei-kpi-dev.ei-bad { color: ${error_text}; } .ei-kpi-dev.ei-ok { color: ${info_text}; }
.ei-kpi-chips { display: flex; gap: .4rem; flex-wrap: wrap; align-items: center; margin: .15rem 0 .3rem 0; }
details.ei-det { margin-top: .4rem; font-size: .95rem; color: ${text2}; }
details.ei-det summary { cursor: pointer; font-weight: 600; color: ${accent}; }
details.ei-det div { margin-top: .25rem; }
[class*="st-key-card-kpi-bad"] { border-left: 5px solid ${error} !important; }
[class*="st-key-card-kpi-chk"] { border-left: 5px solid ${warning} !important; }

/* attention banner (right after a recording is loaded) */
.ei-banner { border-radius: 12px; padding: 1rem 1.25rem; margin: 0 0 1rem 0; border: 1px solid; }
.ei-banner .ei-b-head { font-size: 1.35rem; font-weight: 700; line-height: 1.25; }
.ei-banner .ei-b-sub { font-size: 1rem; margin-top: .2rem; }
.ei-banner ul { margin: .4rem 0 0 1.1rem; padding: 0; font-size: 1rem; }
.ei-banner.ei-lv-ATTENTION { background: ${error_bg}; border-color: ${error}; color: ${error_text}; }
.ei-banner.ei-lv-REVIEW { background: ${warning_bg}; border-color: ${warning}; color: ${warning_text}; }
.ei-banner.ei-lv-INCOMPLETE { background: ${grey}; border-color: ${border}; color: ${text}; }
.ei-banner.ei-lv-ROUTINE { background: ${success_bg}; border-color: ${success}; color: ${success_text}; }
.ei-banner a { color: inherit; font-weight: 600; }
[class*="st-key-card-kpi"] { padding-bottom: 1.75rem !important; }

/* tabs / buttons: keep focus visible */
button:focus-visible, [role="tab"]:focus-visible { outline: 2px solid ${accent}; outline-offset: 2px; }
""")


def css() -> str:
    return _CSS.substitute(sidebar=SIDEBAR_WIDTH_PX, text=T.TEXT, text2=T.TEXT_2, util=T.UTILITY_BAR, surface=T.SURFACE,
                           border=T.BORDER, accent=T.ACCENT, accent_hover=T.ACCENT_HOVER, grey=T.SURFACE_2, tint=T.ACCENT_TINT,
                           error=T.ERROR, error_bg=T.ERROR_BG, error_text=T.ERROR_TEXT, success=T.SUCCESS, success_bg=T.SUCCESS_BG,
                           success_text=T.SUCCESS_TEXT, warning=T.WARNING, warning_bg=T.WARNING_BG, warning_text=T.WARNING_TEXT, info_text=T.INFO_TEXT)


def inject_css() -> None:
    st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)


_MARK_SVG = ('<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" '
             'stroke-linejoin="round" aria-hidden="true"><polyline points="2,13 7,13 9,6 12,19 15,10 17,13 22,13"/></svg>')


def header(app_name: str = "ECG Insight", tagline: str = "Computer-Aided ECG Analysis Platform",
           utility_left: str = "Research prototype - not a medical device", utility_right: str = "") -> None:
    """Charcoal utility bar (real content only) + white header. Navigation links are added when real pages exist."""
    st.markdown(
        f'<div class="ei-util"><span>{html.escape(utility_left)}</span>' + (f'<span>{html.escape(utility_right)}</span>' if utility_right else '') + '</div>'
        f'<div class="ei-header"><span class="ei-mark">{_MARK_SVG}</span>'
        f'<div><div class="ei-brand">{html.escape(app_name)}</div><div class="ei-sub">{html.escape(tagline)}</div></div></div>',
        unsafe_allow_html=True)


def badge_html(label: str, kind: str = "neutral") -> str:
    """A pill whose TEXT always names the state (colour is never the only signal). `unavailable` is dashed-outlined."""
    fg, bg = T.BADGES[kind]
    cls = "ei-badge ei-dashed" if kind == "unavailable" else "ei-badge"
    return f'<span class="{cls}" style="color:{fg};background:{bg}">{html.escape(label)}</span>'


def badge(label: str, kind: str = "neutral") -> None:
    st.markdown(badge_html(label, kind), unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="ei-sectionlabel">{html.escape(text)}</div>', unsafe_allow_html=True)


@contextmanager
def card(title: str | None = None, key: str = "main"):
    """Bordered white card (10-14 px radius, set by the theme). Use as `with card("Title", key="validation"):`."""
    with st.container(border=True, key=f"card-{key}"):
        if title:
            st.markdown(f'<div class="ei-cardtitle">{html.escape(title)}</div>', unsafe_allow_html=True)
        yield


def kpi_card(label: str, value, *, unit: str | None = None, tag: str | None = None, tag_kind: str = "neutral", key: str | None = None,
             extra_chip: tuple | None = None, abnormality: dict | None = None) -> None:
    """KPI card = native st.metric + the evidence tag. An 'Abnormality' fold exists ONLY when the value is outside its published range
    (range, how far outside, and what to verify / what it means). A normal value shows no warning at all.
    A missing value (None) is shown as 'Not available' with an UNAVAILABLE label and never as 0."""
    shown = "Not available" if value is None else (f"{value} {unit}" if unit else f"{value}")
    cls = "kpi" + ("-bad" if abnormality else "")
    with card(key=f"{cls}-{key or label}"):
        st.metric(label, shown)
        chips = [badge_html("UNAVAILABLE", "unavailable")] if value is None else ([badge_html(tag, tag_kind)] if tag else [])
        if extra_chip:
            chips.append(badge_html(*extra_chip))
        if chips:
            st.markdown('<div class="ei-kpi-chips">' + "".join(chips) + "</div>", unsafe_allow_html=True)
        if abnormality:
            ab = abnormality
            parts = [f'<div><b>{html.escape(ab["title"])}</b>' + (f' - published range: <b>{html.escape(ab["range"])}</b>' if ab.get("range") else "") + '</div>',
                     f'<div class="ei-kpi-dev ei-bad">{html.escape(ab["deviation"])}'
                     + ('; above 500 ms (higher torsades risk)' if ab.get("high") else "") + "</div>"]
            if ab.get("lines"):
                parts.append(f'<div><b>{html.escape(ab["lines_title"])}</b></div><ul style="margin:.2rem 0 0 1.1rem;padding:0">'
                             + "".join(f"<li>{html.escape(x)}</li>" for x in ab["lines"]) + "</ul>")
            st.markdown('<details class="ei-det"><summary>Abnormality</summary>' + "".join(parts) + "</details>", unsafe_allow_html=True)


def banner(level: str, headline: str, reasons: list[str], sub: str = "") -> None:
    """Attention banner shown on every page as soon as a recording is analysed. Lists WHICH analysis asks for attention (never merged)."""
    items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons[:6])
    more = f"<li>and {len(reasons) - 6} more (see Assessment)</li>" if len(reasons) > 6 else ""
    st.markdown(f'<div class="ei-banner ei-lv-{level}" role="status"><div class="ei-b-head">{html.escape(headline)}</div>'
                + (f'<div class="ei-b-sub">{html.escape(sub)}</div>' if sub else "")
                + (f"<ul>{items}{more}</ul>" if items else "") + "</div>", unsafe_allow_html=True)


def link_html(url: str, text: str) -> str:
    """External link that opens in a NEW tab (rel keeps the app page from being controlled by the target)."""
    return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(text)}</a>'


def html_table(columns: list[str], rows: list[dict], badge_cols: tuple = (), html_cols: tuple = ()) -> None:
    """Static table; cells in `badge_cols` are (label, kind) tuples rendered as badges; cells in `html_cols` are trusted HTML built by
    this module (e.g. link_html). All other text is escaped."""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(f"<td>{badge_html(*v) if c in badge_cols else (v if c in html_cols else html.escape('' if v is None else str(v)))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(f'<table class="ei-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>', unsafe_allow_html=True)


def workflow(steps: list[tuple[str, str]]) -> None:
    cols = st.columns(len(steps))
    for i, (col, (title, text)) in enumerate(zip(cols, steps), 1):
        col.markdown(f'<div class="ei-step"><div class="ei-stepnum">{i}</div><div class="ei-steptitle">{html.escape(title)}</div>'
                     f'<div class="ei-steptext">{html.escape(text)}</div></div>', unsafe_allow_html=True)


def empty_state(title: str, text: str) -> None:
    st.markdown(f'<div class="ei-empty"><b>{html.escape(title)}</b>{html.escape(text)}</div>', unsafe_allow_html=True)


def notice(kind: str, text: str) -> None:
    """Native Streamlit status box (themed): kind in success | warning | error | info."""
    {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[kind](text)
