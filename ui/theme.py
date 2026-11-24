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
[data-testid="stMainBlockContainer"], .block-container { padding-top: 3.25rem !important; padding-bottom: 3rem !important;
  padding-left: 2rem !important; padding-right: 2rem !important; max-width: 1600px !important; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { min-width: ${sidebar}px; max-width: ${sidebar}px; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .5rem; }
h1, h2, h3 { color: ${text}; letter-spacing: -0.01em; }

/* utility bar + header */
.ei-util { background: ${util}; color: #FFFFFF; font-size: .9375rem; padding: .6rem 1.75rem; border-radius: 12px 12px 0 0;
           display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
.ei-header { background: ${surface}; border: 1px solid ${border}; border-top: 0; border-radius: 0 0 12px 12px; padding: 1.5rem 1.75rem;
             display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
.ei-mark { width: 68px; height: 68px; border-radius: 16px; background: ${accent}; display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto; }
.ei-brand { font-size: 2.25rem; font-weight: 700; color: ${accent}; line-height: 1.1; }
.ei-sub { font-size: 1.125rem; margin-top: .2rem; color: ${text2}; }

/* cards: st.container(border=True, key="card-...") */
[class*="st-key-card"] { background: ${surface}; border-radius: 12px; box-shadow: 0 1px 2px rgba(32,33,36,.06); padding-bottom: .5rem; }
.ei-cardtitle { font-size: 1.05rem; font-weight: 600; color: ${text}; margin: 0 0 .15rem 0; }
.ei-guide-title { font-size: 1.6rem; font-weight: 700; color: ${accent}; line-height: 1.2; margin: 1.25rem 0 .5rem 0; }
.ei-guide-sub { font-size: 1.1rem; font-weight: 700; color: ${text}; margin: .25rem 0 .35rem 0; }
.ei-sectionlabel { font-size: .8125rem; font-weight: 600; color: ${text2}; text-transform: uppercase; letter-spacing: .04em; margin: .25rem 0 .25rem 0; }

/* badges */
.ei-badge { display: inline-block; font-size: .8125rem; font-weight: 600; line-height: 1; padding: .3rem .6rem; border-radius: 999px; border: 1px solid transparent; white-space: nowrap; }
.ei-badge.ei-dashed { border-style: dashed; border-color: ${text2}; }
.ei-kpi-note { font-size: .875rem; color: ${text2}; margin-top: .25rem; }

/* tables with badges */
table.ei-table { width: 100%; border-collapse: collapse; font-size: .9rem; background: ${surface}; border: 1px solid ${border}; border-radius: 12px; overflow: hidden; }
table.ei-table th { background: ${grey}; color: ${text2}; text-align: left; font-weight: 600; padding: .55rem .75rem; border-bottom: 1px solid ${border}; }
table.ei-table td { padding: .55rem .75rem; border-bottom: 1px solid ${border}; vertical-align: top; color: ${text}; }
table.ei-table tr:last-child td { border-bottom: 0; }

/* workflow steps + empty state */
.ei-step { background: ${surface}; border: 1px solid ${border}; border-radius: 12px; padding: 1rem 1.1rem; height: 100%; box-shadow: 0 1px 2px rgba(32,33,36,.06); }
.ei-stepnum { display: inline-flex; width: 28px; height: 28px; border-radius: 50%; background: ${tint}; color: ${accent}; font-weight: 700; align-items: center; justify-content: center; margin-bottom: .5rem; }
.ei-steptitle { font-weight: 600; font-size: 1.05rem; margin-bottom: .2rem; color: ${text}; }
.ei-steptext { color: ${text2}; font-size: .93rem; }
.ei-empty { background: ${surface}; border: 1px dashed ${border}; border-radius: 12px; padding: 1.6rem 1.4rem; text-align: center; color: ${text2}; }
.ei-empty b { color: ${text}; font-size: 1.1rem; display: block; margin-bottom: .3rem; }
.ei-kpi-line { font-size: .82rem; color: ${text2}; margin-top: .15rem; }

/* tabs / buttons: keep focus visible */
button:focus-visible, [role="tab"]:focus-visible { outline: 2px solid ${accent}; outline-offset: 2px; }
""")


def css() -> str:
    return _CSS.substitute(sidebar=SIDEBAR_WIDTH_PX, text=T.TEXT, text2=T.TEXT_2, util=T.UTILITY_BAR, surface=T.SURFACE,
                           border=T.BORDER, accent=T.ACCENT, grey=T.SURFACE_2, tint=T.ACCENT_TINT)


def inject_css() -> None:
    st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)


_MARK_SVG = ('<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" '
             'stroke-linejoin="round" aria-hidden="true"><polyline points="2,13 7,13 9,6 12,19 15,10 17,13 22,13"/></svg>')


def header(app_name: str = "ECG Insight", tagline: str = "Computer-Aided ECG Analysis Platform",
           utility_left: str = "Research prototype - not a medical device", utility_right: str = "PTB-XL Normal vs Abnormal model") -> None:
    """Charcoal utility bar (real content only) + white header. Navigation links are added when real pages exist."""
    st.markdown(
        f'<div class="ei-util"><span>{html.escape(utility_left)}</span><span>{html.escape(utility_right)}</span></div>'
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


def kpi_card(label: str, value, *, unit: str | None = None, tag: str | None = None, tag_kind: str = "neutral", note: str | None = None,
             source: str | None = None, validity: str | None = None, key: str | None = None) -> None:
    """KPI card = native st.metric (keeps AppTest/accessibility) + evidence tag + source + validity + note.
    A missing value (None) is shown as 'Not available' with an UNAVAILABLE label and never as 0."""
    shown = "Not available" if value is None else (f"{value} {unit}" if unit else f"{value}")
    with card(key=f"kpi-{key or label}"):
        st.metric(label, shown)
        if value is None:
            st.markdown(badge_html("UNAVAILABLE", "unavailable"), unsafe_allow_html=True)       # the label itself says so
        elif tag:
            st.markdown(badge_html(tag, tag_kind), unsafe_allow_html=True)
        for k, v in (("Source", source), ("Validity", validity)):
            if v:
                st.markdown(f'<div class="ei-kpi-line"><b>{k}:</b> {html.escape(v)}</div>', unsafe_allow_html=True)
        if note:
            st.markdown(f'<div class="ei-kpi-note">{html.escape(note)}</div>', unsafe_allow_html=True)


def html_table(columns: list[str], rows: list[dict], badge_cols: tuple = ()) -> None:
    """Static table; cells in `badge_cols` are (label, kind) tuples rendered as badges. All text is escaped."""
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(f"<td>{badge_html(*v) if c in badge_cols else html.escape('' if v is None else str(v))}</td>")
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
