"""Analysis history (no Streamlit): one small JSON file next to the app, one line of facts per analysed recording.

What is stored: time, file name, recording setting, role, the patient fields that were typed in, the three results in short form
and the attention level. What is NOT stored: the ECG samples, the uploaded files or any free text.
The file lives in `history/` (listed in .gitignore) on the computer that runs the app. Anyone with access to that computer can read it;
the Clear button deletes it. On a hosted service the file may disappear when the app restarts.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.environ.get("ECG_HISTORY_PATH", os.path.join(HERE, "history", "analysis_history.json"))
MAX_ENTRIES = 500


def load(path: str | None = None) -> list[dict]:
    path = path or PATH
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(entries: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(entries[-MAX_ENTRIES:], fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def make_entry(*, name: str, signature: str | None, role: str, setting: str, demographics: dict, sex_txt, triage, res, cmp) -> dict:
    """Short, factual summary of one analysis. Missing values stay None (never 0)."""
    from .triage import _label, _unit
    vals = {}
    for c in (cmp or []):
        if c.value is not None:
            vals[c.measure] = {"label": _label(c.measure), "value": round(float(c.value), 3), "unit": _unit(c.measure), "status": c.status}
    return {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "record": name,
        "signature": (signature or "")[:12],
        "role": role,
        "setting": setting,
        "age": demographics.get("age"),
        "sex": sex_txt,
        "level": triage.level,
        "model": None if res is None or res.status != "ok" else res.label_name,
        "p_abnormal": None if res is None or res.status != "ok" else round(float(res.probability_abnormal), 3),
        "flagged": [f.title for f in triage.findings if f.severity in ("high", "abnormal")],
        "to_check": [f.title for f in triage.findings if f.severity == "check"],
        "measures": vals,
    }


def add(entry: dict, path: str | None = None) -> bool:
    """Append unless the same recording and inputs (signature + record + role + setting) were saved last. Returns True when added."""
    path = path or PATH
    entries = load(path)
    for e in reversed(entries):
        if e.get("record") == entry["record"]:
            if e.get("signature") == entry["signature"] and e.get("role") == entry["role"] and e.get("setting") == entry["setting"]:
                return False
            break
    entries.append(entry)
    try:
        _save(entries, path)
    except OSError:
        return False
    return True


def clear(path: str | None = None) -> None:
    path = path or PATH
    try:
        os.remove(path)
    except OSError:
        pass


def set_notes(notes: dict[int, str], path: str | None = None) -> bool:
    """Free-text note typed by the user per entry (position in the saved list). Only the note is changed."""
    path = path or PATH
    entries = load(path)
    changed = False
    for i, v in notes.items():
        v = (v or "").strip()
        if 0 <= i < len(entries) and (entries[i].get("note") or "") != v:
            entries[i]["note"] = v
            changed = True
    if changed:
        try:
            _save(entries, path)
        except OSError:
            return False
    return changed


def set_flags(flags: dict[int, bool], path: str | None = None) -> bool:
    """User flag ('needs attention') per entry, by position in the saved list. Only the flag is changed."""
    path = path or PATH
    entries = load(path)
    changed = False
    for i, v in flags.items():
        if 0 <= i < len(entries) and bool(entries[i].get("starred")) != bool(v):
            entries[i]["starred"] = bool(v)
            changed = True
    if changed:
        try:
            _save(entries, path)
        except OSError:
            return False
    return changed
