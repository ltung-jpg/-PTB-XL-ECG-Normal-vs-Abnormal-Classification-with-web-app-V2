"""Extra input formats, added on top of the frozen `data_ingestion` module (which is not modified).

  * MATLAB .mat (PhysioNet/CinC style: a 12 x N or N x 12 array, optional WFDB .hea of the same name for gain / units / lead names)
  * XML: HL7 aECG (FDA / many carts) and GE MUSE
  * CSV / TXT written with decimal commas (';' or tab separated), UTF-8 with BOM, UTF-16 or Latin-1 text

Policies are the same as everywhere else: lead names, units and the sampling frequency come from the file or from the user, never from
a guess; leads are never duplicated or invented (III, aVR, aVL and aVF can only be derived when the user opts in).
These readers were tested with synthetic files that follow each format's published layout. They have NOT been tested with files from
every manufacturer: a file that fails is reported with the reason instead of being read wrongly.
"""
from __future__ import annotations

import base64
import io
import os
import re

import numpy as np

from . import config as C
from .data_ingestion import (IncompatibleLeadsError, IngestionError, _finish, load_uploaded, standardize_leads, unit_factor_to_mv)

_FS_KEYS = ("fs", "Fs", "FS", "sampling_rate", "samplingrate", "sample_rate", "samplerate", "SamplingRate", "freq", "Freq")
_SIG_KEYS = ("val", "ecg", "ECG", "data", "Data", "signal", "Signal", "sig", "x", "X", "ecg_data")
_NAME_KEYS = ("lead_names", "leadnames", "leads", "Leads", "LeadNames", "channels", "sig_name")


# ============================================================================ text normalisation
def decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def normalize_delimited(data: bytes) -> bytes:
    """Re-encode a CSV/TXT as UTF-8 and, when the file is ';' or tab separated with decimal COMMAS, turn them into points.
    Comma-separated files are never touched (a comma there is the delimiter)."""
    text = decode_text(data)
    lines = [l for l in text.splitlines() if l.strip()][:20]
    if lines and all(";" in l or "\t" in l for l in lines[:10]) and not any("," in l and "." in l.split(",")[0] for l in lines[:3]):
        sep = ";" if all(";" in l for l in lines[:10]) else "\t"
        cell = re.compile(r"^\s*[-+]?\d+,\d+([eE][-+]?\d+)?\s*$")
        cells = [c for l in lines[1:10] for c in l.split(sep)]
        if cells and sum(bool(cell.match(c)) for c in cells) >= 0.5 * len(cells):
            text = "\n".join(re.sub(r"(?<=\d),(?=\d)", ".", l) for l in text.splitlines())
    return text.encode("utf-8")


# ============================================================================ MATLAB
def _parse_hea(text: str) -> dict:
    """Minimal WFDB header parse: sampling rate, and per signal gain, unit, baseline and name."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise IngestionError("The .hea header is empty.")
    head = lines[0].split()
    if len(head) < 3:
        raise IngestionError("The .hea header line is malformed.")
    nsig, fs = int(head[1]), float(head[2].split("/")[0])
    sigs = []
    for l in lines[1:1 + nsig]:
        f = l.split()
        gain, unit, base = 200.0, "mV", None
        if len(f) > 2:
            m = re.match(r"^([-+\d.eE]+)(?:\(([-+\d]+)\))?(?:/(.*))?$", f[2])
            if m:
                gain = float(m.group(1)) or 200.0
                base = None if m.group(2) is None else float(m.group(2))
                unit = m.group(3) or "mV"
        if base is None:
            base = float(f[4]) if len(f) > 4 and re.fullmatch(r"[-+]?\d+", f[4]) else 0.0
        sigs.append({"gain": gain, "unit": unit, "baseline": base, "name": f[-1]})
    return {"fs": fs, "sigs": sigs}


def read_mat(data: bytes, name: str, *, hea_text: str | None = None, fs: float | None = None, units: str | None = None,
             lead_names: list[str] | None = None, assume_standard_order: bool = False, derive_limb_leads: bool = False):
    try:
        from scipy.io import loadmat
        mat = loadmat(io.BytesIO(data), squeeze_me=False)
    except NotImplementedError as e:
        raise IngestionError("This is a MATLAB v7.3 (HDF5) file, which is not supported. Save it as v7 (-v7) or export CSV.") from e
    except Exception as e:
        raise IngestionError(f"Could not read the MATLAB file: {e}") from e
    arrs = {k: np.asarray(v) for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.dtype.kind in "iuf" and v.ndim == 2 and min(v.shape) >= 2}
    if not arrs:
        raise IngestionError("The MATLAB file contains no 2-D numeric array (expected leads x samples).")
    key = next((k for k in _SIG_KEYS if k in arrs), None) or max(arrs, key=lambda k: arrs[k].size)
    arr = arrs[key].astype(float)
    if arr.shape[0] == C.N_LEADS and arr.shape[1] != C.N_LEADS:      # leads x samples
        arr = arr.T
    elif arr.shape[1] != C.N_LEADS and arr.shape[0] < arr.shape[1]:  # other channel counts: the short side holds the channels
        arr = arr.T
    warnings = [f"MATLAB array '{key}' was used ({arr.shape[0]} samples x {arr.shape[1]} channels)."]

    hdr = _parse_hea(hea_text) if hea_text else None
    names = None
    if hdr:
        if len(hdr["sigs"]) != arr.shape[1]:
            raise IngestionError(f"The .hea header lists {len(hdr['sigs'])} signals but the MATLAB array has {arr.shape[1]} channels.")
        names = [s["name"] for s in hdr["sigs"]]
        phys = np.empty_like(arr)
        for j, s in enumerate(hdr["sigs"]):
            phys[:, j] = (arr[:, j] - s["baseline"]) / s["gain"] * (unit_factor_to_mv(units or s["unit"]))
        arr, chosen_units = phys, units or "/".join(sorted({s["unit"] for s in hdr["sigs"]}))
        fs_final = fs or hdr["fs"]
    else:
        key_units = units
        if key_units is None:
            raise IngestionError("Signal units are unknown: a .mat file does not state them. Upload the matching .hea header, or choose the units "
                                 "(mV, uV or V) under 'File options'.")
        arr = arr * unit_factor_to_mv(key_units)
        chosen_units = key_units
        fs_final = fs or next((float(np.ravel(mat[k])[0]) for k in _FS_KEYS if k in mat and np.size(mat[k]) == 1), None)
        if fs_final is None:
            raise IngestionError("Sampling frequency is unknown: not in the .mat file. Enter it under 'File options'.")
    if names is None:
        for k in _NAME_KEYS:
            if k in mat:
                try:
                    cand = [str(np.ravel(x)[0]) if isinstance(x, np.ndarray) else str(x) for x in np.ravel(mat[k])]
                    if len(cand) == arr.shape[1]:
                        names = cand
                        break
                except Exception:
                    pass
    if names is None:
        if lead_names is not None and len(lead_names) == arr.shape[1]:
            names = list(lead_names)
        elif assume_standard_order and arr.shape[1] == C.N_LEADS:
            names = list(C.LEADS)
            warnings.append("The file has no lead names; the standard 12-lead order was used as you requested.")
        else:
            raise IncompatibleLeadsError("The .mat file has no lead names. Upload the .hea header with it, or confirm that the 12 channels follow the "
                                         "standard order (I, II, III, aVR, aVL, aVF, V1-V6) under 'File options'.")
    std, w2, derived = standardize_leads(arr, names, derive_limb_leads=derive_limb_leads)
    return _finish(std, fs_final, "MAT", os.path.splitext(name)[0], fs_final, chosen_units, names, warnings + w2, derived)


# ============================================================================ XML
def _safe_xml(data: bytes):
    import xml.etree.ElementTree as ET
    head = data[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in head:
        raise IngestionError("XML with a DOCTYPE or entity declaration is refused for safety.")
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise IngestionError(f"The XML file could not be parsed: {e}") from e


def _tag(e) -> str:
    return e.tag.rsplit("}", 1)[-1]


def _find_all(root, name):
    return [e for e in root.iter() if _tag(e) == name]


def read_xml(data: bytes, name: str, *, units: str | None = None, derive_limb_leads: bool = False):
    root = _safe_xml(data)
    warnings: list[str] = []
    # ---- GE MUSE
    lead_data = _find_all(root, "LeadData")
    if lead_data:
        wf = None
        for w in _find_all(root, "Waveform"):
            kind = next((e.text for e in w if _tag(e) == "WaveformType"), "")
            if kind and kind.strip().lower() == "rhythm":
                wf = w
                break
        if wf is None:
            raise IngestionError("MUSE XML: no 'Rhythm' waveform found (a median-beat waveform alone is not a 10-s recording).")
        g = lambda e, t: next((x.text for x in e.iter() if _tag(x) == t and x.text), None)
        fs = float(g(wf, "SampleBase") or 0)
        exp = int(float(g(wf, "SampleExponent") or 0))
        fs = fs * (10 ** exp) if fs else 0
        names, cols = [], []
        for ld in [e for e in wf.iter() if _tag(e) == "LeadData"]:
            lid = g(ld, "LeadID")
            wfd = g(ld, "WaveFormData")
            if not lid or not wfd:
                continue
            per_bit = float(g(ld, "LeadAmplitudeUnitsPerBit") or 0)
            u = (g(ld, "LeadAmplitudeUnits") or "").upper()
            if not per_bit or u not in ("MICROVOLTS", "MILLIVOLTS", "VOLTS"):
                raise IngestionError(f"MUSE XML: lead {lid} has no usable amplitude scale ({u or 'no unit'}).")
            samples = np.frombuffer(base64.b64decode(wfd), dtype="<i2").astype(float)
            cols.append(samples * per_bit * {"MICROVOLTS": 1e-3, "MILLIVOLTS": 1.0, "VOLTS": 1e3}[u]); names.append(lid)
        if not cols or not fs:
            raise IngestionError("MUSE XML: no lead data or no sampling base found.")
        n = min(len(c) for c in cols)
        arr = np.stack([c[:n] for c in cols], axis=1)
        warnings.append("GE MUSE XML: the 'Rhythm' waveform was read. MUSE stores 8 leads; III, aVR, aVL and aVF are only available if you allow deriving them.")
        std, w2, derived = standardize_leads(arr, names, derive_limb_leads=derive_limb_leads)
        return _finish(std, fs, "XML (GE MUSE)", os.path.splitext(name)[0], fs, "mV", names, warnings + w2, derived)
    # ---- HL7 aECG
    seqs = [e for e in root.iter() if _tag(e) == "sequence" and any(_tag(x) == "value" and any(_tag(y) == "digits" for y in x) for x in e.iter())]
    if seqs:
        names, cols, fs_list, unit_list = [], [], [], []
        for s in seqs:
            code = next((e.get("code") for e in s.iter() if _tag(e) == "code" and e.get("code")), "")
            m = re.search(r"LEAD_(?:[A-Z]+_)?(I{1,3}|AVR|AVL|AVF|V[1-6])$", (code or "").upper())
            if not m:
                continue
            val = next(e for e in s.iter() if _tag(e) == "value" and any(_tag(y) == "digits" for y in e))
            digits = np.array(next(y.text for y in val if _tag(y) == "digits").split(), dtype=float)
            sub = {_tag(y): y for y in val}
            inc = sub.get("increment")
            if inc is None:
                raise IngestionError(f"HL7 aECG: lead {code} has no sampling increment.")
            step = float(inc.get("value")); iu = (inc.get("unit") or "s").lower()
            step_s = step * {"s": 1.0, "ms": 1e-3, "us": 1e-6}.get(iu, 1.0)
            scale = sub.get("scale"); origin = sub.get("origin")
            sc = float(scale.get("value")) if scale is not None else 1.0
            og = float(origin.get("value")) if origin is not None else 0.0
            u = (origin.get("unit") if origin is not None else None) or (scale.get("unit") if scale is not None else None) or units
            if not u:
                raise IngestionError(f"HL7 aECG: lead {code} states no unit; choose the units under 'File options'.")
            cols.append((og + sc * digits) * unit_factor_to_mv(units or u))
            names.append(m.group(1)); fs_list.append(1.0 / step_s); unit_list.append(u)
        if not cols:
            raise IngestionError("HL7 aECG: no standard ECG lead sequences (MDC_ECG_LEAD_*) were found.")
        if max(fs_list) - min(fs_list) > 1e-6 * max(fs_list):
            raise IngestionError("HL7 aECG: leads have different sampling rates.")
        n = min(len(c) for c in cols)
        arr = np.stack([c[:n] for c in cols], axis=1)
        warnings.append("HL7 aECG XML was read (lead sequences 'SLIST_PQ').")
        std, w2, derived = standardize_leads(arr, names, derive_limb_leads=derive_limb_leads)
        return _finish(std, fs_list[0], "XML (HL7 aECG)", os.path.splitext(name)[0], fs_list[0], "/".join(sorted(set(unit_list))), names, warnings + w2, derived)
    raise IngestionError("This XML file is not a supported ECG dialect. Supported: HL7 aECG and GE MUSE. (Philips and other XML layouts are not read.)")


# ============================================================================ entry point used by the app
def load_uploaded_ext(files: list[tuple[str, bytes]], **opts) -> list[tuple[str, object]]:
    """Like data_ingestion.load_uploaded, plus .mat and .xml, and decimal-comma / non-UTF-8 text tables."""
    plain, extra, heas = [], [], {}
    for name, data in files:
        stem, ext = os.path.splitext(os.path.basename(name))
        ext = ext.lower()
        if ext in (".mat", ".xml"):
            extra.append((name, data, ext))
        elif ext in (".csv", ".txt", ".tsv", ".asc"):
            plain.append((name, normalize_delimited(data)))
        else:
            plain.append((name, data))
            if ext == ".hea":
                heas[stem] = data
    mat_stems = {os.path.splitext(os.path.basename(n))[0] for n, _, e in extra if e == ".mat"}
    plain = [(n, d) for n, d in plain if not (os.path.splitext(os.path.basename(n))[1].lower() == ".hea" and os.path.splitext(os.path.basename(n))[0] in mat_stems)]
    results = list(load_uploaded(plain, **opts)) if plain else []
    for name, data, ext in extra:
        stem = os.path.splitext(os.path.basename(name))[0]
        try:
            if ext == ".mat":
                hea = heas.get(stem)
                rec = read_mat(data, name, hea_text=decode_text(hea) if hea else None, fs=opts.get("fs"), units=opts.get("units"),
                               lead_names=opts.get("lead_names"), assume_standard_order=opts.get("assume_standard_order", False),
                               derive_limb_leads=opts.get("derive_limb_leads", False))
            else:
                rec = read_xml(data, name, units=opts.get("units"), derive_limb_leads=opts.get("derive_limb_leads", False))
            results.append((stem, rec))
        except Exception as e:
            results.append((stem, e))
    return sorted(results, key=lambda r: r[0])
