"""Overview: what the platform does, the 3-step workflow and the honest scope."""
import streamlit as st

from ui import context as X
from ui import theme as UI

st.markdown("## Computer-aided ECG analysis")
st.caption("Research prototype - not a medical device. The machine-learning class, the waveform measurements and the published reference "
           "ranges are three independent analyses and are never merged into one conclusion.")

st.markdown('<div class="ei-guide-title">User\'s guidelines</div>', unsafe_allow_html=True)
g1, g2 = st.columns(2)
with g1:
    st.markdown('<div class="ei-guide-sub">How to view the web</div>', unsafe_allow_html=True)
    st.markdown(
        "1. Use the tabs at the top to move between pages: **Overview**, **Upload**, **Signal Analysis**, **Assessment** and **Reports**.\n"
        "2. Use the sidebar on the left to add files, choose a recording, set file options and enter optional patient information.\n"
        "3. **Upload** shows the checks on your file (format, sampling rate, units, leads and missing fields).\n"
        "4. **Signal Analysis** shows the ECG: choose leads, switch the view mode, move the time slider and show R-peaks.\n"
        "5. **Assessment** shows three separate results: A signal measurements, B model prediction and C reference ranges.\n"
        "6. Read the tag next to every value: MEASURED, ESTIMATED, DERIVED or PREDICTED. 'Not available' means the value could not be obtained.\n"
        "7. **Reports** creates a report you can download.")
with g2:
    st.markdown('<div class="ei-guide-sub">What to do</div>', unsafe_allow_html=True)
    st.markdown(
        "1. In the sidebar, upload a 12-lead ECG: WFDB (both the .hea and .dat files), CSV, TXT or EDF.\n"
        "2. If a CSV or TXT file has no sampling rate or units, open **File options (advanced)** and fill them in.\n"
        "3. Optionally open **Patient information** and enter age, sex, height and weight. Leave a field empty if you do not know it.\n"
        "4. Open the **Upload** page and check that the recording is accepted. Fix or replace it if it is marked FAIL.\n"
        "5. Press **Analyze Recording** in the sidebar.\n"
        "6. Review **Signal Analysis**, then **Assessment**.\n"
        "7. Open **Reports**, press **Generate report**, then **Download report (HTML)**.\n"
        "8. Remember that every result is a research output, not a diagnosis.")
st.write("")

UI.workflow([("Upload a recording", "Add a 12-lead ECG in the sidebar: WFDB (.hea + .dat), CSV, TXT or EDF."),
             ("Inspect the data", "Check format, sampling rate, units, leads and missing fields before anything is analyzed."),
             ("Analyze and review", "Press Analyze Recording, then review the signal, the assessment and the report.")])
st.write("")

c1, c2, c3 = st.columns(3)
with c1:
    with UI.card("A. Signal measurements", key="ov-a"):
        st.write("Heart rate and wave timings measured or estimated directly from the waveform. Each value says how it was obtained "
                 "and is shown as 'Not available' when it cannot be obtained.")
with c2:
    with UI.card("B. Model prediction", key="ov-b"):
        st.write("A Normal / Abnormal output of a gradient-boosting model trained on PTB-XL. It is a research output, not a diagnosis.")
with c3:
    with UI.card("C. Reference ranges", key="ov-c"):
        st.write("Measured values compared with published population ranges, with the source of every range.")

if not st.session_state.get("ei_good"):
    st.write("")
    UI.empty_state("No recording loaded yet", "Use the uploader in the sidebar to add an ECG. Nothing is analyzed until you press Analyze Recording.")
with st.expander("What is accepted, and what is not"):
    st.markdown("- **12 leads are required** (I, II, III, aVR, aVL, aVF, V1-V6). Recordings with fewer or unrecognised leads are rejected with an "
                "explanation - leads are never duplicated or invented.\n"
                "- Any sampling rate of 100 Hz or more is resampled to 100 Hz (the training rate). Units mV / uV / V are converted.\n"
                "- 10 s are analyzed (at least 5 s are required).\n"
                "- Not supported: recordings with other lead sets (for example MIT-BIH), dataset reference labels and PDF export.")
