"""VIS-01 ... VIS-10: checks on REAL rendered pages (headless Chrome via Playwright + a live Streamlit server).
They measure pixels and computed CSS; they complement, and do not replace, a human look at the screenshots.
Skipped automatically when Playwright / Chrome are not installed.  Screenshots are saved to tests/visual/_shots/."""
import colorsys
import glob
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
from PIL import Image
from playwright.sync_api import sync_playwright

from ui import tokens as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHOTS = os.path.join(ROOT, "tests", "visual", "_shots")
EXAMPLE = [os.path.join(ROOT, "examples", "ptbxl_records", f"00001_hr.{e}") for e in ("hea", "dat")]
CHROME = os.environ.get("CHROME_PATH") or shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chrome")
SIZES = {"1366": (1366, 768), "1920": (1920, 1080), "900": (900, 800)}


def _hex(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])


@pytest.fixture(scope="module")
def pages():
    cache = os.path.join(SHOTS, "dom.json")
    if os.environ.get("VIS_REUSE") and os.path.exists(cache):          # development shortcut: reuse the last capture
        import json
        yield json.load(open(cache)); return
    if not all(os.path.exists(f) for f in EXAMPLE):
        pytest.skip("example record missing")
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(port),
                             "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(url + "/_stcore/health", timeout=1); break
            except Exception:
                time.sleep(1)
        else:
            pytest.skip("streamlit did not start")
        os.makedirs(SHOTS, exist_ok=True)
        out = {"png": {}, "dom": {}}
        with sync_playwright() as p:
            try:
                b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"]) if CHROME else p.chromium.launch()
            except Exception as e:
                pytest.skip(f"no browser: {e}")
            for key, (w, h) in SIZES.items():
                pg = b.new_page(viewport={"width": w, "height": h}); pg.goto(url); pg.wait_for_timeout(8000)
                path = os.path.join(SHOTS, f"overview_{key}.png"); pg.screenshot(path=path); out["png"][f"overview_{key}"] = path
                out["dom"][f"overview_{key}"] = _dom(pg); pg.close()
            for key in ("1366", "1920"):
                w, h = SIZES[key]
                pg = b.new_page(viewport={"width": w, "height": 2000}); pg.goto(url); pg.wait_for_timeout(8000)
                pg.set_input_files("input[type=file]", EXAMPLE); pg.wait_for_timeout(5000)
                pg.get_by_role("button", name="Analyze Recording").click(); pg.wait_for_timeout(20000)
                for label, name in (("Signal Analysis", "signal"), ("Assessment", "assessment"), ("Upload", "upload")):
                    pg.get_by_role("link", name=label, exact=True).first.click(); pg.wait_for_timeout(12000 if name != "upload" else 5000)
                    path = os.path.join(SHOTS, f"{name}_{key}.png"); pg.screenshot(path=path); out["png"][f"{name}_{key}"] = path
                    out["dom"][f"{name}_{key}"] = _dom(pg)
                pg.close()
            b.close()
        import json
        json.dump(out, open(os.path.join(SHOTS, "dom.json"), "w"))
        yield out
    finally:
        proc.terminate()


def _dom(pg):
    return pg.evaluate("""() => {
      const rgb = c => c;
      const vis = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
      const fonts = []; 
      document.querySelectorAll('p, span, label, td, th, li, h1, h2, h3, button, a, small, div').forEach(e => {
        if (!vis(e)) return; const own = [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length > 1) && e.innerText && e.innerText.trim().length > 1;
        if (own && !e.closest('svg') && !e.closest('.js-plotly-plot')) fonts.push(parseFloat(getComputedStyle(e).fontSize)); });
      const bg = sel => [...document.querySelectorAll(sel)].filter(vis).map(e => getComputedStyle(e).backgroundColor);
      const prim = [...document.querySelectorAll('button')].filter(vis).filter(b => b.innerText.trim() === 'Analyze Recording' && !b.disabled).map(b => getComputedStyle(b).backgroundColor);
      const badges = [...document.querySelectorAll('.ei-badge')].filter(vis).map(e => ({t: e.innerText, c: getComputedStyle(e).color, b: getComputedStyle(e).backgroundColor}));
      return {overflow: document.documentElement.scrollWidth - window.innerWidth, minFont: fonts.length ? Math.min(...fonts) : null,
              cardBg: bg('[class*="st-key-card"]'), primary: prim, badges: badges, bodyFont: getComputedStyle(document.body).fontFamily,
              sidebarW: (document.querySelector('[data-testid=stSidebar]') || {getBoundingClientRect: () => ({width: 0})}).getBoundingClientRect().width};
    }""")


def _lab(h):
    c = _hex(h) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]]) @ c / np.array([.95047, 1, 1.08883])
    f = np.where(xyz > .008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _img(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(float)


def _hsv(a):
    mx, mn = a.max(-1), a.min(-1)
    v = mx / 255; s = np.where(mx == 0, 0, (mx - mn) / np.maximum(mx, 1e-9))
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    d = np.maximum(mx - mn, 1e-9)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, s, v


# ------------------------------------------------------------------------------------------------ VIS-01 / 02
def test_vis01_no_large_dark_content_backgrounds(pages):
    for name, path in pages["png"].items():
        a = _img(path); lum = a.mean(-1)
        dark_rows = (lum < 60).mean(axis=1) > 0.25                     # rows where >25 % of the width is dark (text in the bar lowers it)
        bands, start, gap = [], None, 0
        for i, r in enumerate(list(dark_rows) + [False] * 12):
            if r:
                if start is None: start = i
                gap = 0
            elif start is not None:
                gap += 1
                if gap > 10:                                          # a gap > 10 px ends the band (merges the lines of a wrapped bar)
                    bands.append(i - gap - start + 1); start = None; gap = 0
        bands = [b for b in bands if b >= 5]                           # ignore hairlines / underlines
        assert all(b <= 80 for b in bands), f"{name}: dark band(s) {bands} (only the utility bar, 1-2 lines high, is allowed)"
        assert len(bands) <= 1, f"{name}: {bands}"
        assert (lum < 60).mean() < 0.06, f"{name}: {100 * (lum < 60).mean():.1f} % dark pixels"


def test_vis02_white_and_light_grey_dominate(pages):
    for name, path in pages["png"].items():
        a = _img(path)
        light = (a.min(-1) >= 225).mean()                                          # white, light grey or a pale tint
        assert light >= 0.85, f"{name}: only {100 * light:.0f} % white/light-grey"


# ------------------------------------------------------------------------------------------------ VIS-03 / 04
def test_vis03_burgundy_is_the_primary_accent(pages):
    for name in ("signal_1366", "assessment_1366", "overview_1366"):
        h, s, v = _hsv(_img(pages["png"][name]))
        sat = (s > 0.35) & (v > 0.25)
        assert sat.sum() > 500, name
        hh = h[sat]
        burg = ((hh >= 335) | (hh <= 12)).mean()
        # the other saturated pixels are status tints (amber / blue / green badges and notices)
        assert burg >= 0.30, f"{name}: burgundy/red is only {100 * burg:.0f} % of saturated pixels"
        hist = np.histogram(hh, bins=12, range=(0, 360))[0]
        assert hist.argmax() in (0, 11), f"{name}: the most common saturated hue is not burgundy"


def test_vis04_no_neon_teal_or_dark_navy(pages):
    for name, path in pages["png"].items():
        h, s, v = _hsv(_img(path))
        teal = ((h >= 165) & (h <= 195) & (s > 0.5) & (v > 0.3)).mean()
        navy = ((h >= 205) & (h <= 260) & (s > 0.4) & (v < 0.35)).mean()
        neon = ((s > 0.9) & (v > 0.85) & (((h >= 80) & (h <= 190)) | ((h >= 270) & (h <= 330)))).mean()
        assert teal < 0.0005 and navy < 0.0005 and neon < 0.0005, (name, teal, navy, neon)


# ------------------------------------------------------------------------------------------------ VIS-05 / 06 / 07
def test_vis05_buttons_badges_and_tables_follow_the_design_system(pages):
    want = "rgb(%d, %d, %d)" % tuple(_hex(T.ACCENT))
    for name, d in pages["dom"].items():
        assert all(c == want for c in d["primary"]), (name, d["primary"])
        allowed = {tuple(_hex(fg)) + tuple(_hex(bg)) for fg, bg in T.BADGES.values()}
        for b in d["badges"]:
            nums = [int(x) for x in __import__("re").findall(r"\d+", b["c"])[:3] + __import__("re").findall(r"\d+", b["b"])[:3]]
            assert tuple(nums) in allowed, (name, b)


def test_vis06_readable_and_balanced_at_common_resolutions(pages):
    for name, d in pages["dom"].items():
        assert d["overflow"] <= 1, f"{name}: horizontal overflow {d['overflow']} px"
        assert d["minFont"] is None or d["minFont"] >= 12, f"{name}: text smaller than 12 px ({d['minFont']})"
    assert 240 <= pages["dom"]["overview_1366"]["sidebarW"] <= 280
    for key in ("1366", "1920"):
        a = _img(pages["png"][f"signal_{key}"]); h, w, _ = a.shape
        content = a[:, int(0.22 * w):]
        assert (content.mean(-1) < 250).any(axis=0).mean() > 0.5       # content is spread over the width, not squeezed into a strip


def test_vis07_no_unnecessary_full_card_colouring(pages):
    for name, d in pages["dom"].items():
        assert all(c in ("rgb(255, 255, 255)",) for c in d["cardBg"]), (name, set(d["cardBg"]))


# ------------------------------------------------------------------------------------------------ VIS-08 / 09 / 10
def test_vis08_waveform_and_annotations_are_distinguishable(pages):
    cols = {"ecg": T.ECG_LINE, "secondary": T.SECONDARY_WAVE, "annotation": T.ANNOTATION, "r_peak": T.R_PEAK}
    names = list(cols)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            de = np.linalg.norm(_lab(cols[names[i]]) - _lab(cols[names[j]]))                 # CIE76 colour difference (>= 15 = clearly different)
            assert de >= 15, (names[i], names[j], de)
    a = _img(pages["png"]["signal_1366"])
    for key in ("ecg", "r_peak"):                                      # both really appear in the rendered viewer
        near = (np.abs(a - _hex(cols[key])).sum(-1) < 40).sum()
        assert near > 30, f"{key} colour not visible in the viewer"


def test_vis09_no_misleading_colours_for_clinical_conclusions(pages):
    ok = {tuple(_hex(T.INFO_BG)), tuple(_hex(T.WARNING_BG)), tuple(_hex(T.ACCENT_TINT)), tuple(_hex(T.SURFACE_2))}
    d = pages["dom"]["assessment_1366"]
    import re
    for b in d["badges"]:
        bg = tuple(int(x) for x in re.findall(r"\d+", b["b"])[:3])
        assert bg in ok, f"assessment badge {b['t']!r} uses a pass/fail colour {bg}"            # no green 'good' / red 'bad' on clinical results
    assert not any(b["t"].upper() in ("NORMAL", "HEALTHY", "ABNORMAL", "DISEASE") for b in d["badges"])


def test_vis10_matches_the_clinical_burgundy_and_white_direction(pages):
    palette = np.array([_hex(c) for c in (T.APP_BG, T.SURFACE, T.SIDEBAR_BG, T.SURFACE_2, T.UTILITY_BAR, T.ACCENT, T.ACCENT_HOVER, T.ACCENT_TINT,
                                           T.TEXT, T.TEXT_2, T.TEXT_MUTED, T.BORDER, T.ERROR_BG, T.INFO_BG, T.WARNING_BG, T.SUCCESS_BG, T.GRID_MAJOR,
                                           T.GRID_MINOR, T.INFO, T.WARNING, T.SUCCESS, T.ERROR, T.SECONDARY_WAVE, T.ANNOTATION, T.R_PEAK)]).astype(float)
    for name in ("overview_1366", "signal_1366", "assessment_1366"):
        a = _img(pages["png"][name]).reshape(-1, 3)
        q = (a // 16).astype(int); keys, counts = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_counts=True)
        top = keys[counts / len(a) >= 0.01]
        assert len(top) >= 3
        for k in top:
            col = np.array([k // 256, (k // 16) % 16, k % 16]) * 16 + 8
            dist = np.linalg.norm(palette - col, axis=1).min()
            assert dist < 30, f"{name}: dominant colour {col} is not in the approved palette (distance {dist:.0f})"
    a = _img(pages["png"]["overview_1366"])
    assert (np.abs(a - _hex(T.UTILITY_BAR)).sum(-1) < 12).sum() > 1000          # charcoal utility bar present
