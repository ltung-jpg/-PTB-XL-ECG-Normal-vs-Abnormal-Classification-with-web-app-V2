"""Overview: what the platform does, the 3-step workflow and the honest scope."""
import streamlit as st

from ui import context as X
from ui import theme as UI

st.markdown("## Computer-aided ECG analysis")

with st.expander("User's guidelines: how to view the web and what to do", expanded=False):
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="ei-guide-sub">How to view the web</div>', unsafe_allow_html=True)
        st.markdown(
            "1. Use the tabs at the top to move between pages: **Overview**, **Upload**, **Signal Analysis**, **Reports** and **History**.\n"
            "2. **Upload** is where you add files, set file options, enter optional patient information, choose who is reading and see the checks on your file.\n"
            "3. **Signal Analysis** shows the ECG (choose leads, view mode, time range, R-peaks) and the key measurements, including the model's Normal / Abnormal output.\n"
            "4. A card shows an **Abnormality** fold only when its value is outside the published range or the model says Abnormal. No fold means nothing to flag.\n"
            "5. Read the tag next to every value: MEASURED, ESTIMATED, DERIVED or PREDICTED. 'Not available' means the value could not be obtained.\n"
            "6. **Reports** creates a report you can download. **History** lists the recordings analysed before.")
    with g2:
        st.markdown('<div class="ei-guide-sub">What to do</div>', unsafe_allow_html=True)
        st.markdown(
            "1. Open **Upload** and add a 12-lead ECG: WFDB (both the .hea and .dat files), CSV, TXT, EDF, MAT or XML.\n"
            "2. If a CSV or TXT file has no sampling rate or units, open **File options (advanced)** and fill them in.\n"
            "3. Optionally enter age, sex, height and weight. Leave a field empty if you do not know it.\n"
            "4. Check that the recording is accepted. Fix or replace it if it is marked FAIL.\n"
            "5. Choose **Doctor** or **Student** under *Who is reading*. This is required, and it sets how the results and the report are written.\n"
            "6. The recording is analysed automatically. Review **Signal Analysis**, then open **Reports**, press **Generate report** and download it.\n"
            "7. Remember that every result is a research output, not a diagnosis.")

# ---- glossary: the words and abbreviations used in the app, hidden until opened
with st.expander("Glossary: terms and abbreviations"):
    st.markdown("**Evidence tags** (shown under every value)")
    st.markdown(
        "| Term | Meaning |\n|---|---|\n"
        "| MEASURED | Read directly from the signal. |\n"
        "| ESTIMATED | Found by automatic wave delineation; not validated against expert annotation. |\n"
        "| DERIVED | Computed from other values (for example QTc from QT and heart rate). |\n"
        "| PREDICTED | Output of the machine-learning model. |\n"
        "| UNAVAILABLE / Not available | The value could not be obtained. It is never shown as 0. |")
    st.markdown("**Result labels**")
    st.markdown(
        "| Term | Meaning |\n|---|---|\n"
        "| ABOVE RANGE / BELOW RANGE | The value lies outside the published population range. This is a deviation, not a diagnosis. |\n"
        "| WITHIN RANGE | The value lies inside the published range. |\n"
        "| INDETERMINATE | Not measurable, or within the measurement uncertainty of a limit. |\n"
        "| NO RANGE | No reference range from a reliable published source was found; the value is shown for information only. |\n"
        "| LOW AGREEMENT | The leads disagree about this value, so read it with caution. |\n"
        "| MODEL: NORMAL / MODEL: ABNORMAL | What the classifier output. 'Abnormal' means any PTB-XL diagnosis other than normal sinus rhythm; it does not name a condition. |\n"
        "| PASS / NEEDS REVIEW / FAIL | Result of the checks on the uploaded file. |\n"
        "| AVAILABLE / MISSING / DERIVABLE / INVALID | State of an input field on the Upload page. |")
    st.markdown("**Abbreviations**")
    st.markdown(
        "| Term | Meaning |\n|---|---|\n"
        "| ECG | Electrocardiogram. |\n"
        "| PTB-XL | The public 12-lead ECG dataset the model was trained on. |\n"
        "| WFDB | PhysioNet file format: a .hea header plus a .dat signal file. |\n"
        "| R-peak | The highest point of the QRS complex; one per heartbeat. |\n"
        "| HR, RR | Heart rate (beats per minute) and the time between two R-peaks (seconds). |\n"
        "| P wave, PR interval | Atrial activation, and the time from its start to the start of the QRS. |\n"
        "| QRS | Ventricular depolarisation; the duration is measured from its start to its end. |\n"
        "| QT, QTc | Time from QRS start to T-wave end, and the same corrected for heart rate (Bazett or Fridericia formula). |\n"
        "| T wave, ST segment, ST level | Ventricular recovery; the flat part after the QRS and its height against the baseline. |\n"
        "| PR segment, TP segment | Flat parts between P and QRS, and between the T wave and the next P. |\n"
        "| Flagged, Notes | History columns you fill in yourself to mark and describe a recording. |")

# ---- technical notes: last on the page, hidden until opened
with st.expander("Technical notes"):
    st.markdown("**What is accepted.** 12 leads are required (I, II, III, aVR, aVL, aVF, V1-V6); leads are never duplicated or invented. Any sampling rate of "
                "100 Hz or more is resampled to 100 Hz (the training rate). Units mV / uV / V are converted. 10 s are analyzed (at least 5 s are required). "
                "Not supported: other lead sets (for example MIT-BIH), dataset reference labels and PDF export.")
    st.markdown("**Training data.** The model was trained on PTB-XL (resting 12-lead ECGs recorded in hospital settings). Accuracy on ambulance recordings, "
                "which carry more motion and mains noise, has not been measured.")
    st.markdown("**R-peak markers.** The detector works on a 100 Hz grid on one lead. In the viewer each marker is placed on the highest point of the QRS in the lead shown; "
                "markers with no clear complex are not drawn. This is display only: the measurements and model features are unchanged.")
    a_ = X.current()
    if a_ is not None:
        st.markdown(f"**Current recording ({a_.name}).**")
        m_ = a_.morph()
        if m_.ok:
            st.write(f"- Waveform method: {m_.method}; timing resolution {m_.resolution_ms:.0f} ms. Values are medians over beats; some may show low agreement between leads.")
            for w in m_.warnings:
                st.write(f"- {w}")
        else:
            st.write(f"- Waveform measurements not available: {m_.message}")
        for w in a_.rec.warnings + (a_.rec.quality.warnings if a_.rec.quality else []) + list(a_.res.warnings) + list(a_.res.notes):
            st.write(f"- {w}")
