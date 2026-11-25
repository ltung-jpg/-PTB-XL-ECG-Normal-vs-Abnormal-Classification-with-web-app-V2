"""PDF export of the report with reportlab only (a wheel on every platform; no HTML engine, no cairo, no system libraries).

The report HTML is produced by modules/report.py from a small, fixed set of tags (h1-h3, div.box, small, p, br, ul/li, table, img, a, b, i).
This module walks exactly that markup and builds the PDF from it, so the PDF and the HTML always contain the same content."""
from __future__ import annotations

import base64
import html as _html
import io
import re
from html.parser import HTMLParser

from . import config as _  # noqa: F401  (keeps the package import order stable)
from ui import tokens as T


def _clean(s: str) -> str:
    """Core PDF fonts cover Windows-1252 only: map the few common symbols that fall outside it, replace anything else."""
    for a, b in (("≥", ">="), ("≤", "<="), ("→", "->"), ("−", "-"), ("≈", "~"), (" ", " "), (" ", " "), ("‑", "-")):
        s = s.replace(a, b)
    return s.encode("cp1252", errors="replace").decode("cp1252")


class _Builder(HTMLParser):
    def __init__(self, styles):
        super().__init__(convert_charrefs=True)
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image, ListFlowable, ListItem  # noqa: F401
        self.st = styles
        self.flow: list = []
        self.buf: list[str] = []
        self.block = "p"            # current paragraph style key
        self.cls = ""
        self.skip = 0
        self.table: list | None = None
        self.row: list | None = None
        self.cell: list | None = None
        self.cell_is_head = False
        self.in_li = False
        self.small = 0

    # -------------------------------------------------------------------------------- helpers
    def _text(self) -> str:
        t = "".join(self.buf)
        self.buf = []
        return re.sub(r"[ \t\r\n]+", " ", t).strip()

    def _flush(self):
        from reportlab.platypus import Paragraph
        t = self._text()
        if not t:
            return
        key = self.block
        if key == "div":
            key = "warn" if "warn" in self.cls else "stub" if "stub" in self.cls else "box"
        if self.small and key == "p":
            key = "small"
        self.flow.append(Paragraph(t, self.st[key]))

    # -------------------------------------------------------------------------------- parser callbacks
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("style", "title", "head"):
            self.skip += 1
            return
        if self.cell is not None:
            if tag in ("b", "i"):
                self.buf.append(f"<{tag}>")
            elif tag == "br":
                self.buf.append("<br/>")
            elif tag == "a" and a.get("href", "").startswith("http"):
                self.buf.append(f'<a href="{_html.escape(a["href"], quote=True)}" color="{T.ACCENT}">')
            return
        if tag in ("h1", "h2", "h3", "p", "div"):
            self._flush(); self.block = tag; self.cls = a.get("class", "")
        elif tag == "small":
            self._flush(); self.small += 1
        elif tag == "li":
            self._flush(); self.block = "li"; self.buf.append("• ".encode("cp1252").decode("cp1252"))
        elif tag in ("ul", "ol"):
            self._flush()
        elif tag == "br":
            self.buf.append("<br/>")
        elif tag in ("b", "i"):
            self.buf.append(f"<{tag}>")
        elif tag == "a" and a.get("href", "").startswith("http"):
            self.buf.append(f'<a href="{_html.escape(a["href"], quote=True)}" color="{T.ACCENT}">')
        elif tag == "table":
            self._flush(); self.table = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []; self.buf_saved = self.buf; self.buf = []; self.cell_is_head = tag == "th"
        elif tag == "img":
            src = a.get("src", "")
            m = re.match(r"data:image/\w+;base64,(.*)", src, re.S)
            if m:
                self._flush(); self._image(base64.b64decode(m.group(1)))

    def handle_endtag(self, tag):
        if tag in ("style", "title", "head"):
            self.skip -= 1
            return
        if self.cell is not None and tag not in ("td", "th"):
            if tag in ("b", "i", "a"):
                self.buf.append(f"</{tag}>")
            return
        if tag in ("h1", "h2", "h3", "p", "div", "li"):
            self._flush(); self.block = "p"; self.cls = ""
        elif tag == "small":
            self._flush(); self.small = max(0, self.small - 1)
        elif tag in ("b", "i", "a"):
            self.buf.append(f"</{tag}>")
        elif tag in ("td", "th"):
            from reportlab.platypus import Paragraph
            txt = self._text()
            self.row.append(Paragraph(txt, self.st["th" if self.cell_is_head else "td"]))
            self.cell = None; self.buf = self.buf_saved
        elif tag == "tr" and self.row is not None:
            if self.table is not None:
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self._table(self.table); self.table = None

    def handle_data(self, data):
        if self.skip:
            return
        self.buf.append(_html.escape(_clean(data), quote=False))

    # -------------------------------------------------------------------------------- flowables
    def _image(self, raw: bytes):
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import Image, Spacer
        w, h = ImageReader(io.BytesIO(raw)).getSize()
        maxw = self.st["maxw"]
        sc = min(1.0, maxw / w)
        self.flow.append(Image(io.BytesIO(raw), width=w * sc, height=h * sc))
        self.flow.append(Spacer(1, 6))

    def _table(self, rows):
        from reportlab.lib import colors
        from reportlab.platypus import Table, TableStyle, Spacer
        rows = [r for r in rows if r]
        if not rows:
            return
        n = max(len(r) for r in rows)
        rows = [r + [""] * (n - len(r)) for r in rows]
        wts = [max(min(len(re.sub(r"<[^>]+>", "", getattr(r[j], "text", str(r[j])))), 45) for r in rows) + 6 for j in range(n)]
        tot = float(sum(wts))
        fr = [max(w / tot, 0.07) for w in wts]
        fr = [f / sum(fr) for f in fr]
        t = Table(rows, colWidths=[self.st["maxw"] * f for f in fr], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(T.BORDER)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(T.SURFACE_2)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        self.flow.append(t); self.flow.append(Spacer(1, 8))


def _reportlab_pdf(html_text: str) -> bytes:
    """Primary engine (reportlab). Raises ImportError when reportlab is missing."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate
    except ImportError as exc:
        raise ImportError("The PDF package is missing. Install it with: pip install reportlab  (then restart the app). "
                          "The HTML report can be downloaded meanwhile.") from exc
    W, _h = A4
    margin = 1.6 * cm
    base = dict(fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor(T.TEXT))
    boxkw = dict(borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=10, leftIndent=6, rightIndent=6)
    styles = {
        "maxw": W - 2 * margin,
        "p": ParagraphStyle("p", **base, spaceAfter=4),
        "small": ParagraphStyle("small", **{**base, "fontSize": 8, "leading": 10.5, "textColor": colors.HexColor(T.TEXT_2)}, spaceAfter=3),
        "li": ParagraphStyle("li", **base, leftIndent=12, spaceAfter=2),
        "h1": ParagraphStyle("h1", **{**base, "fontName": "Helvetica-Bold", "fontSize": 17, "leading": 21, "textColor": colors.HexColor(T.ACCENT)}, spaceAfter=6),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "Helvetica-Bold", "fontSize": 13, "leading": 16}, spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle("h3", **{**base, "fontName": "Helvetica-Bold", "fontSize": 11, "leading": 14}, spaceBefore=8, spaceAfter=4),
        "box": ParagraphStyle("box", **base, **boxkw, borderColor=colors.HexColor(T.BORDER), backColor=colors.HexColor(T.APP_BG)),
        "warn": ParagraphStyle("warn", **base, **boxkw, borderColor=colors.HexColor(T.WARNING), backColor=colors.HexColor(T.WARNING_BG)),
        "stub": ParagraphStyle("stub", **{**base, "fontName": "Helvetica-Bold", "textColor": colors.HexColor(T.ERROR_TEXT)}, **boxkw,
                               borderColor=colors.HexColor(T.ERROR), backColor=colors.HexColor(T.ERROR_BG)),
        "td": ParagraphStyle("td", **{**base, "fontSize": 8.5, "leading": 11}),
        "th": ParagraphStyle("th", **{**base, "fontName": "Helvetica-Bold", "fontSize": 8.5, "leading": 11, "textColor": colors.HexColor(T.TEXT_2)}),
    }
    b = _Builder(styles)
    b.feed(html_text)
    b._flush()
    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin,
                      title="ECG analysis report", author="ECG Insight").build(b.flow)
    return out.getvalue()


# ------------------------------------------------------------------------------------------------ fallback: matplotlib only
class _Plain(HTMLParser):
    """Collects the report as a flat list of ('h'|'p'|'row'|'img', payload) items. No markup can break it."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items, self.buf, self.row, self.skip, self.head = [], [], None, 0, ""

    def _flush(self, kind="p"):
        t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if t:
            self.items.append((kind, t))

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("style", "title"):
            self.skip += 1
        elif tag in ("h1", "h2", "h3"):
            self._flush(); self.head = tag
        elif tag in ("p", "div", "li", "small", "ul", "table", "br"):
            if self.row is None:
                self._flush()
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.buf = []
        elif tag == "img":
            m = re.match(r"data:image/\w+;base64,(.*)", a.get("src", ""), re.S)
            if m:
                self._flush(); self.items.append(("img", base64.b64decode(m.group(1))))

    def handle_endtag(self, tag):
        if tag in ("style", "title"):
            self.skip -= 1
        elif tag in ("h1", "h2", "h3"):
            self._flush("h"); self.head = ""
        elif tag in ("td", "th"):
            t = re.sub(r"\s+", " ", "".join(self.buf)).strip(); self.buf = []
            if self.row is not None:
                self.row.append(t)
        elif tag == "tr" and self.row is not None:
            self.items.append(("row", self.row)); self.row = None
        elif tag in ("p", "div", "li", "small", "ul", "table") and self.row is None:
            self._flush()

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)


def _matplotlib_pdf(html_text: str) -> bytes:
    """Fallback engine: needs nothing but matplotlib (already required by the app). Plain layout, same content."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages

    pl = _Plain(); pl.feed(html_text); pl._flush()
    W, H = 8.27, 11.69
    out = io.BytesIO()
    with PdfPages(out) as pdf:
        state = {"fig": None, "y": 0.0}

        def new_page():
            if state["fig"] is not None:
                pdf.savefig(state["fig"]); plt.close(state["fig"])
            state["fig"] = plt.figure(figsize=(W, H)); state["y"] = H - 0.7

        def ensure(h):
            if state["fig"] is None or state["y"] - h < 0.6:
                new_page()

        def text(t, size=8.5, bold=False, color="#202124", width=118, gap=0.05):
            lines = textwrap.wrap(t, width=int(width * 8.5 / size)) or [""]
            for ln in lines:
                step = size / 72 * 1.45
                ensure(step)
                state["fig"].text(0.7 / W, state["y"] / H, ln, fontsize=size, weight="bold" if bold else "normal", color=color, va="top")
                state["y"] -= step
            state["y"] -= gap

        from matplotlib.patches import Rectangle

        def draw_table(rows):
            rows = [r for r in rows if r]
            n = max(len(r) for r in rows)
            rows = [r + [""] * (n - len(r)) for r in rows]
            avail = W - 1.4
            wts = [max(min(len(r[j]), 45) for r in rows) + 6 for j in range(n)]
            fr = [max(w / sum(wts), 0.07) for w in wts]; fr = [f / sum(fr) for f in fr]
            widths = [avail * f for f in fr]
            size = 7.5; step = size / 72 * 1.4; pad = 0.05
            for ri, r in enumerate(rows):
                cells = [textwrap.wrap(c, width=max(6, int(widths[j] * 72 / (size * 0.62)))) or [""] for j, c in enumerate(r)]
                h = max(len(c) for c in cells) * step + 2 * pad
                ensure(h)
                x = 0.7; top = state["y"]
                for j, c in enumerate(cells):
                    state["fig"].add_artist(Rectangle((x / W, (top - h) / H), widths[j] / W, h / H, transform=state["fig"].transFigure,
                                                      facecolor="#EEF1F6" if ri == 0 else "white", edgecolor="#9AA0A6", linewidth=0.6))
                    for li, ln in enumerate(c):
                        state["fig"].text((x + 0.05) / W, (top - pad - li * step) / H, ln, fontsize=size, weight="bold" if ri == 0 else "normal", va="top", color="#202124")
                    x += widths[j]
                state["y"] -= h
            state["y"] -= 0.15

        i = 0
        items = pl.items
        while i < len(items):
            kind, v = items[i]
            if kind == "h":
                state["y"] -= 0.12; text(v, 12, True, T.ACCENT, 80, 0.08)
            elif kind == "p":
                text(v)
            elif kind == "row":
                grp = []
                while i < len(items) and items[i][0] == "row":
                    grp.append(items[i][1]); i += 1
                draw_table(grp)
                continue
            elif kind == "img":
                try:
                    im = mpimg.imread(io.BytesIO(v), format="png")
                except Exception:
                    i += 1; continue
                ih, iw = im.shape[:2]
                w = W - 1.4; h = w * ih / iw
                if h > H - 1.4:
                    h = H - 1.4; w = h * iw / ih
                ensure(h + 0.1)
                ax = state["fig"].add_axes([0.7 / W, (state["y"] - h) / H, w / W, h / H]); ax.imshow(im, interpolation="none"); ax.axis("off")
                state["y"] -= h + 0.15
            i += 1
        if state["fig"] is None:
            new_page()
        pdf.savefig(state["fig"]); plt.close(state["fig"])
    return out.getvalue()


LAST_ENGINE = ""


def html_to_pdf(html_text: str) -> bytes:
    """PDF of the report. First reportlab (nicer layout); if that is missing or fails for any reason, the matplotlib engine,
    which needs no extra package. Only when both fail is an error raised (with both reasons)."""
    global LAST_ENGINE
    try:
        out = _reportlab_pdf(html_text)
        LAST_ENGINE = "reportlab"
        return out
    except Exception as first:
        try:
            out = _matplotlib_pdf(html_text)
            LAST_ENGINE = f"basic layout (reportlab not used: {type(first).__name__}: {str(first)[:150]})"
            return out
        except Exception as second:
            raise RuntimeError(f"reportlab: {type(first).__name__}: {first}; fallback: {type(second).__name__}: {second}") from second
