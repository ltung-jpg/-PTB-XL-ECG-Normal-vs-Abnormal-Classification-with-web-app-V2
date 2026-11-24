"""Data ingestion: format-specific adapters -> one standardized `ECGRecording`.

Standardized representation (what every adapter delivers)
    signal        (n_samples, 12) float64, PHYSICAL units = millivolt, columns in PTB-XL lead order
    fs            sampling frequency of that signal, Hz
    missing_mask  bool array, True where a sample was missing / invalid in the source
    lead identity verified against the 12 standard leads - never assumed

Policies required by the specification
    * Lead identity comes from the file.  If the file carries no lead names, the standard order is used ONLY when
      the caller explicitly confirms it (`assume_standard_order=True`).
    * Recordings with missing leads are rejected with an explanatory error.  Leads are never duplicated or
      invented.  The single exception is the explicit, opt-in `derive_limb_leads=True`, which computes
      III, aVR, aVL, aVF from I and II with the exact Einthoven/Goldberger identities and says so in the warnings.
    * Units must be known (file header or caller); they are converted to mV, never guessed.
    * Sampling frequency must be known (file header, time column or caller); it is never guessed.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .preprocessing import QualityReport, assess_signal_quality, resample_signal


# ============================================================================ errors
class IngestionError(ValueError):
    """The file could not be read or is inconsistent."""


class UnsupportedFormatError(IngestionError):
    pass


class IncompatibleLeadsError(IngestionError):
    pass


# ============================================================================ standardized representation
@dataclass
class ECGRecording:
    signal: np.ndarray                      # (n, 12) mV, NaN where invalid
    fs: float
    source_format: str
    record_id: str
    original_fs: float
    original_units: str
    original_lead_names: list[str]
    missing_mask: np.ndarray
    lead_names: list[str] = field(default_factory=lambda: list(C.LEADS))
    warnings: list[str] = field(default_factory=list)
    derived_leads: list[str] = field(default_factory=list)
    quality: QualityReport | None = None

    @property
    def duration_s(self) -> float:
        return self.signal.shape[0] / self.fs

    @property
    def time_s(self) -> np.ndarray:
        return np.arange(self.signal.shape[0]) / self.fs


# ============================================================================ helpers
_LEAD_PREFIXES = ("ECG", "LEAD", "EKG")


def normalize_lead_name(name: str) -> str:
    """'ECG II', 'lead_aVR', 'V1 ', 'aVF' -> 'II', 'AVR', 'V1', 'AVF'."""
    s = re.sub(r"[^A-Za-z0-9]", "", str(name)).upper()
    for p in _LEAD_PREFIXES:
        if s.startswith(p) and len(s) > len(p):
            s = s[len(p):]
            break
    return s


def unit_factor_to_mv(unit: str) -> float:
    """Multiplier converting a unit string to millivolt; unknown units raise (never guessed)."""
    u = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    u = {"millivolt": "mv", "millivolts": "mv", "microvolt": "uv", "microvolts": "uv", "volt": "v", "volts": "v"}.get(u, u)
    if u in C.UNIT_TO_MV:
        return C.UNIT_TO_MV[u]
    raise IngestionError(f"Unknown signal unit {unit!r}. Supported: mV, uV, V. Please specify the units explicitly.")


def standardize_leads(data: np.ndarray, names: list[str], *, derive_limb_leads: bool = False):
    """Reorder/validate channels into the 12 standard leads.

    Returns (array (n,12), warnings, derived_lead_names).  Raises IncompatibleLeadsError otherwise.
    """
    warnings: list[str] = []
    norm = [normalize_lead_name(n) for n in names]
    if len(set(norm)) != len(norm):
        dup = sorted({n for n in norm if norm.count(n) > 1})
        raise IncompatibleLeadsError(f"Duplicate lead names in the recording: {dup}. Cannot decide which channel to use.")
    col = {n: i for i, n in enumerate(norm) if n in C.LEAD_ALIASES}
    unrecognised = [names[i] for i, n in enumerate(norm) if n not in C.LEAD_ALIASES]
    missing = [l for l in C.LEADS if l not in col]
    derived: list[str] = []
    out = np.full((data.shape[0], C.N_LEADS), np.nan)
    for j, lead in enumerate(C.LEADS):
        if lead in col:
            out[:, j] = data[:, col[lead]]
    if missing:
        limb_only = set(missing) <= {"III", "AVR", "AVL", "AVF"}
        if not (derive_limb_leads and {"I", "II"} <= set(col) and limb_only):
            hint = (" Leads III, aVR, aVL and aVF can be derived exactly from I and II; enable 'derive limb leads' "
                    "if you accept that.") if ({"I", "II"} <= set(col) and limb_only) else ""
            raise IncompatibleLeadsError(
                f"The model needs all 12 standard leads. Found: {[l for l in C.LEADS if l in col]}. "
                f"Missing: {missing}." + (f" Unrecognised channels: {unrecognised}." if unrecognised else "") + hint)
        I, II = out[:, 0], out[:, 1]
        eq = {"III": II - I, "AVR": -(I + II) / 2.0, "AVL": I - II / 2.0, "AVF": II - I / 2.0}
        for lead in missing:
            out[:, C.LEADS.index(lead)] = eq[lead]
            derived.append(lead)
        warnings.append(f"Leads {derived} were DERIVED from leads I and II (Einthoven/Goldberger identities); "
                        "they were not recorded.")
    if unrecognised:
        warnings.append(f"Ignored channels that are not standard ECG leads: {unrecognised}.")
    return out, warnings, derived


def _finish(signal_mv, fs, fmt, rid, orig_fs, orig_units, orig_names, warnings, derived) -> ECGRecording:
    signal_mv = np.asarray(signal_mv, dtype=float)
    if signal_mv.ndim != 2 or signal_mv.shape[0] < 2:
        raise IngestionError("The recording contains no usable samples.")
    if not np.isfinite(fs) or fs <= 0:
        raise IngestionError(f"Invalid sampling frequency: {fs}")
    mask = ~np.isfinite(signal_mv)
    if mask.all():
        raise IngestionError("All samples are missing/invalid.")
    rec = ECGRecording(signal=signal_mv, fs=float(fs), source_format=fmt, record_id=rid, original_fs=float(orig_fs),
                       original_units=orig_units, original_lead_names=list(orig_names), missing_mask=mask,
                       warnings=list(warnings), derived_leads=list(derived))
    rec.quality = assess_signal_quality(signal_mv, fs)
    return rec


# ============================================================================ WFDB adapter
def read_wfdb(base_path: str, *, units: str | None = None, derive_limb_leads: bool = False) -> ECGRecording:
    """PhysioNet WFDB record: `<base>.hea` + `<base>.dat`.  `base_path` has no extension."""
    import wfdb
    for ext in (".hea", ".dat"):
        if not os.path.exists(base_path + ext):
            raise IngestionError(f"WFDB record is incomplete: {os.path.basename(base_path)}{ext} is missing "
                                 "(a WFDB recording needs both the .hea header and the .dat signal file).")
    try:
        rec = wfdb.rdrecord(base_path)
    except Exception as e:
        raise IngestionError(f"Could not read the WFDB record: {e}") from e
    if not isinstance(rec, wfdb.Record) or rec.p_signal is None:
        raise IngestionError("Multi-segment or non-physical WFDB records are not supported.")
    sig = np.asarray(rec.p_signal, dtype=float)
    hdr_units = list(rec.units) if rec.units else [None] * sig.shape[1]
    if units is not None:
        hdr_units = [units] * sig.shape[1]
    if any(u in (None, "") for u in hdr_units):
        raise IngestionError("The WFDB header does not state the signal units; specify them explicitly.")
    factors = np.array([unit_factor_to_mv(u) for u in hdr_units])
    std, warns, derived = standardize_leads(sig * factors[None, :], list(rec.sig_name), derive_limb_leads=derive_limb_leads)
    warns = list(warns)
    if len(set(hdr_units)) > 1:
        warns.append(f"Channels had different units {sorted(set(hdr_units))}; each was converted to mV.")
    return _finish(std, rec.fs, "WFDB", os.path.basename(base_path), rec.fs, "/".join(sorted(set(map(str, hdr_units)))),
                   rec.sig_name, warns, derived)


# ============================================================================ CSV / TXT adapter
_TIME_NAMES = {"time", "t", "times", "timestamp", "seconds", "sec", "times", "elapsed", "timems"}
_INDEX_NAMES = {"sample", "samples", "index", "idx", "n", "no", "sampleno", "samplenumber"}
_HEADER_UNIT = re.compile(r"^(.*?)[\s_]*[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*$")
_NUM = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?|nan|NaN|NA|")


def _sniff_delimiter(lines: list[str]) -> str | None:
    for d in (",", ";", "\t", "|"):
        counts = {len(l.split(d)) for l in lines[:10]}
        if len(counts) == 1 and counts.pop() > 1:
            return d
    return None                                                      # whitespace-separated


def read_delimited(path: str, *, fs: float | None = None, units: str | None = None,
                   lead_names: list[str] | None = None, assume_standard_order: bool = False,
                   derive_limb_leads: bool = False, fmt: str = "CSV") -> ECGRecording:
    """CSV / TXT / TSV with one column per lead (samples x leads), or one row per lead (leads x samples)."""
    try:
        text = open(path, "r", encoding="utf-8", errors="replace").read()
    except OSError as e:
        raise IngestionError(f"Cannot open file: {e}") from e
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith(("#", "%"))]
    if len(lines) < 3:
        raise IngestionError("The file contains fewer than 3 data lines - not an ECG recording.")
    delim = _sniff_delimiter(lines)
    tokens0 = [t.strip() for t in (lines[0].split(delim) if delim else lines[0].split())]
    has_header = sum(bool(_NUM.fullmatch(t)) for t in tokens0) < 0.5 * len(tokens0)
    try:
        # no delimiter + header text (e.g. a single column "II (mV)") -> one column per line; whitespace tables need no header
        sep = delim if delim else (r"\s+" if not has_header else ",")
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, header=0 if has_header else None,
                         engine="python", na_values=["", "NA", "NaN", "nan", "NAN", "null", "-"])
    except Exception as e:
        raise IngestionError(f"Could not parse the file as a delimited table: {e}") from e
    if df.shape[1] < 2 and df.shape[0] < 2:
        raise IngestionError("The file does not look like a multi-channel ECG table.")

    warnings: list[str] = []
    raw_names = [str(c).strip() for c in df.columns] if has_header else [None] * df.shape[1]
    col_units: list[str | None] = []
    base_names: list[str | None] = []
    for n in raw_names:
        if n is None:
            col_units.append(None); base_names.append(None); continue
        m = _HEADER_UNIT.match(n)
        base_names.append(m.group(1) if m else n)
        col_units.append(m.group(2) if m else None)

    time_idx, time_scale, drop = None, 1.0, []
    if has_header:
        for j, n in enumerate(base_names):
            key = re.sub(r"[^a-z0-9]", "", (n or "").lower())
            if key in _TIME_NAMES or key.startswith("time"):
                time_idx = j
                u = ((col_units[j] or "") + " " + (raw_names[j] or "")).lower()
                time_scale = 1e-3 if (re.search(r"\bms\b|msec|millis", u) or key.endswith("ms")) else 1.0
                drop.append(j)
            elif key in _INDEX_NAMES:
                drop.append(j)
    t_vec = None
    if time_idx is not None:
        t_vec = pd.to_numeric(df.iloc[:, time_idx], errors="coerce").to_numpy(float) * time_scale
    keep = [j for j in range(df.shape[1]) if j not in drop]
    names_kept = [base_names[j] for j in keep] if has_header else [None] * len(keep)
    units_kept = [col_units[j] for j in keep] if has_header else [None] * len(keep)
    arr = df.iloc[:, keep].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    bad_cols = [names_kept[i] or i for i in range(arr.shape[1]) if np.isnan(arr[:, i]).all()]
    if bad_cols:
        raise IngestionError(f"Columns {bad_cols} contain no numeric data. Remove non-signal columns or check the delimiter.")

    n_rows, n_cols = arr.shape
    if has_header:
        names = names_kept
        if not any(normalize_lead_name(n) in C.LEAD_ALIASES for n in names):
            if lead_names is not None:
                if len(lead_names) != n_cols:
                    raise IngestionError(f"{len(lead_names)} lead names were given but the file has {n_cols} signal columns.")
                names = list(lead_names)
            elif assume_standard_order and n_cols == C.N_LEADS:
                names = list(C.LEADS)
                warnings.append("Column names were not recognisable as leads; the standard order "
                                "(I, II, III, aVR, aVL, aVF, V1-V6) was used as you requested.")
            else:
                raise IncompatibleLeadsError(f"Column names {names[:12]} are not recognisable as ECG leads. Provide the lead "
                                             "names, or confirm that the columns follow the standard 12-lead order.")
    else:
        if n_rows == C.N_LEADS and n_cols > C.N_LEADS:               # leads x samples
            arr = arr.T
            n_rows, n_cols = arr.shape
            warnings.append("The file was laid out as 12 rows (leads) x samples; it was transposed.")
        if lead_names is not None:
            if len(lead_names) != n_cols:
                raise IngestionError(f"{len(lead_names)} lead names were given but the file has {n_cols} signal columns.")
            names = list(lead_names)
        elif assume_standard_order and n_cols == C.N_LEADS:
            names = list(C.LEADS)
            warnings.append("The file has no lead names; the standard order (I, II, III, aVR, aVL, aVF, V1-V6) "
                            "was used as you requested.")
        else:
            raise IncompatibleLeadsError(
                f"The file has {n_cols} signal columns and no lead names. Different devices order leads differently, "
                "so the lead order is not guessed: provide the lead names or explicitly confirm the standard 12-lead order.")

    chosen_units = units
    if chosen_units is None:
        found = {u for u in units_kept if u}
        if len(found) == 1 and all(units_kept):
            chosen_units = found.pop()
        elif found:
            raise IngestionError(f"Mixed or partial units in the header: {sorted(found)}. Specify the units explicitly.")
        else:
            raise IngestionError("Signal units are unknown (not in the file header). Specify them (mV, uV or V); they are "
                                 "not guessed because a wrong unit changes every amplitude feature.")
    arr = arr * unit_factor_to_mv(chosen_units)

    fs_time = None
    if t_vec is not None and np.isfinite(t_vec).sum() > 2:
        dt = np.diff(t_vec[np.isfinite(t_vec)])
        med = np.median(dt)
        if med <= 0 or np.max(np.abs(dt / med - 1)) > 0.05:
            raise IngestionError("The time column is not uniformly sampled (irregular or non-monotonic time stamps).")
        fs_time = 1.0 / med
    if fs is not None and fs_time is not None and abs(fs - fs_time) / fs_time > 0.02:
        raise IngestionError(f"Sampling frequency {fs:g} Hz contradicts the time column ({fs_time:.1f} Hz).")
    fs_final = fs if fs is not None else fs_time
    if fs_final is None:
        raise IngestionError("Sampling frequency is unknown (no time column in the file). Please specify it in Hz.")
    if fs is None:
        warnings.append(f"Sampling frequency {fs_final:.2f} Hz inferred from the time column.")

    std, w2, derived = standardize_leads(arr, names, derive_limb_leads=derive_limb_leads)
    return _finish(std, fs_final, fmt, os.path.basename(path), fs_final, chosen_units, names, warnings + w2, derived)


# ============================================================================ EDF adapter
def read_edf(path: str, *, derive_limb_leads: bool = False) -> ECGRecording:
    """European Data Format (EDF / EDF+ / BDF) via pyedflib."""
    try:
        import pyedflib
    except ImportError as e:
        raise IngestionError("EDF support needs the 'pyedflib' package (pip install pyedflib).") from e
    try:
        f = pyedflib.EdfReader(path)
    except Exception as e:
        raise IngestionError(f"Could not read the EDF file: {e}") from e
    try:
        labels = f.getSignalLabels()
        fss = [float(f.getSampleFrequency(i)) for i in range(f.signals_in_file)]
        dims = [f.getPhysicalDimension(i) for i in range(f.signals_in_file)]
        sigs = [f.readSignal(i) for i in range(f.signals_in_file)]
    finally:
        f.close()
    ecg_idx = [i for i, l in enumerate(labels) if normalize_lead_name(l) in C.LEAD_ALIASES]
    if not ecg_idx:
        raise IncompatibleLeadsError(f"No channel in the EDF file is a recognisable ECG lead. Channels: {labels}.")
    warnings: list[str] = []
    fs_target = max(fss[i] for i in ecg_idx)
    cols = []
    for i in ecg_idx:
        x = np.asarray(sigs[i], dtype=float) * unit_factor_to_mv(dims[i] or "")
        if abs(fss[i] - fs_target) > 1e-9:
            x = resample_signal(x[:, None], fss[i], fs_target)[:, 0]
            warnings.append(f"Channel {labels[i]} ({fss[i]:g} Hz) was resampled to {fs_target:g} Hz to align channels.")
        cols.append(x)
    n = min(len(c) for c in cols)
    data = np.stack([c[:n] for c in cols], axis=1)
    std, w2, derived = standardize_leads(data, [labels[i] for i in ecg_idx], derive_limb_leads=derive_limb_leads)
    return _finish(std, fs_target, "EDF", os.path.basename(path), fs_target, "/".join(sorted({d for d in dims if d})),
                   [labels[i] for i in ecg_idx], warnings + w2, derived)


# ============================================================================ dispatch
_DELIMITED = {".csv": "CSV", ".txt": "TXT", ".tsv": "TSV", ".asc": "TXT"}


def load_recording(paths: list[str] | str, *, fs: float | None = None, units: str | None = None,
                   lead_names: list[str] | None = None, assume_standard_order: bool = False,
                   derive_limb_leads: bool = False) -> ECGRecording:
    """Load ONE recording from one or more file paths (WFDB needs both .hea and .dat)."""
    paths = [paths] if isinstance(paths, str) else list(paths)
    if not paths:
        raise IngestionError("No file given.")
    exts = {os.path.splitext(p)[1].lower() for p in paths}
    if ".hea" in exts or (".dat" in exts and len(paths) > 1):
        stems = {os.path.splitext(p)[0] for p in paths if os.path.splitext(p)[1].lower() in (".hea", ".dat")}
        if len(stems) != 1:
            raise IngestionError("A WFDB upload must contain exactly one .hea/.dat pair per recording.")
        return read_wfdb(stems.pop(), units=units, derive_limb_leads=derive_limb_leads)
    p = paths[0]
    ext = os.path.splitext(p)[1].lower()
    if ext == ".dat":
        raise IngestionError("A .dat file alone cannot be read: WFDB signal files need the matching .hea header file.")
    if ext in _DELIMITED:
        return read_delimited(p, fs=fs, units=units, lead_names=lead_names, assume_standard_order=assume_standard_order,
                              derive_limb_leads=derive_limb_leads, fmt=_DELIMITED[ext])
    if ext in (".edf", ".bdf"):
        return read_edf(p, derive_limb_leads=derive_limb_leads)
    raise UnsupportedFormatError(f"Unsupported file type {ext or '(none)'!r}. Supported: WFDB (.hea + .dat), CSV, TXT, EDF.")


def load_uploaded(files: list[tuple[str, bytes]], **opts) -> list[tuple[str, "ECGRecording | Exception"]]:
    """Load every recording in a set of uploaded (filename, bytes) pairs.

    A WFDB pair (.hea + .dat with the same stem) forms one recording; every other file is a recording on its own.
    Returns [(display_name, ECGRecording | exception)] so one bad file never hides the others.
    """
    tmp = tempfile.mkdtemp(prefix="ecg_upload_")
    try:
        groups: dict[str, list[str]] = {}
        for name, data in files:
            dest = os.path.join(tmp, os.path.basename(name))
            with open(dest, "wb") as fh:
                fh.write(data)
            stem, ext = os.path.splitext(dest)
            groups.setdefault(stem if ext.lower() in (".hea", ".dat") else dest, []).append(dest)
        results = []
        for key, group in sorted(groups.items()):
            try:
                results.append((os.path.basename(key), load_recording(group, **opts)))
            except Exception as e:
                results.append((os.path.basename(key), e))
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
