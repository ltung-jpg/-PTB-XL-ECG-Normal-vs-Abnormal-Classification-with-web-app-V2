# Redesign changelog (Phases 0-5)

**Phase 0** audit (UI_AUDIT.md). **Phase 1** theme, tokens, header, sidebar, cards, charts. **Phase 2** Overview and Upload pages, workflow, empty states, "Not provided" patient fields.
**Phase 3** data inspection (validation, field states, eligibility), interactive viewer, KPI cards with evidence tags. **Phase 4** Assessment (A measurements, B model, C reference ranges, D dataset labels, explainability), restyled report. **Phase 5** QA, docs.

## Changed
* `app.py` is now a shell; pages are in `views/`; presentation helpers in `ui/`.
* `modules/visualization.py`: colours/grid only. `modules/report.py`: palette and wording only ("B. Waveform measurements - automatic estimates"; validity column).
* Reference-range charts no longer use green/red.
* Patient fields start empty ("Not provided"); an unavailable value is "Not available", never 0.
* Recordings below 100 Hz, or failing the quality gate, are marked FAIL and not analyzed (previously a 50 Hz file was silently up-sampled).
* (Superseded in the test round below) Analysis was run only on "Analyze Recording".

## Reference ranges (after Phase 5, deliberate change to `modules/reference_comparison.py`)
* Added P-wave duration (<= 120 ms, conventional adult upper limit, cited via Magnani 2010 / PMC3394095; some sources use 110 ms) and RR interval (0.61-1.25 s = Mason 2007 heart-rate range 48-98 bpm converted by 60/HR).
* Measures with no reliable published range (T-wave duration, PR/ST/TP segments, ST level) are labelled NO RANGE with a specific reason; no range was invented.
* Display: RR in seconds (2 decimals), ST level 3 decimals, one-sided ranges shown as "<= x".
* Only the `cmp` part of the numerical baseline changed (RR and P duration now within/above/below instead of indeterminate); probabilities, features and morphology are identical. The module's pinned hash was updated.

## Unchanged (verified)
Every analysis module (`config`, `data_ingestion`, `preprocessing`, `feature_extraction`, `prediction`, `ecg_delineation`, `explainability`, `bundle_builder`) is byte-identical to Phase 0 (hash test) and the probabilities, 145 features, morphology and comparisons on the example records are identical to the pre-redesign baseline (UI-12). The model files were never touched.

## Known gaps
Dataset labels, non-12-lead records, PDF export, Inter font - see docs/LIMITATIONS.md. `st.switch_page` was not used after "Analyze" because it dropped the sidebar upload state in a real browser; page links are shown instead.

## Test round (22 feedback items)
Analysis modules are unchanged (hash test passes). Probabilities, the 145 features, morphology and comparisons are identical.
* **Roles** (item 1): sidebar "I am a Doctor / Medical student" changes only the text next to flagged values (what to verify vs what it means) and how much is expanded.
* **Attention banner + toast right after loading** (20, 21): a valid recording is now analyzed automatically (the button re-runs it). Banner levels: ATTENTION (red), REVIEW (amber), INCOMPLETE (grey), ROUTINE (green). Every trigger is listed with its own source; missing or low-agreement results can never give ROUTINE. It is a reading-priority prompt, not a diagnosis, and does not merge the three analyses.
* **Abnormal values marked, with the size of the deviation** (10, 16): KPI cards and the Assessment table show the published range, ABOVE/BELOW RANGE and how far beyond the nearest limit. QTc above 500 ms is additionally marked HIGH RISK (AHA/ACCF statement, Drew 2010: 2-3 fold higher torsades risk).
* **Less noise, larger text** (8, 4): Key findings first; measures without a range, source/validity, the 145 features and SHAP are collapsed; base font 17.5 px; "Details" fold on each card.
* **Commentary in the report** (3): rule-based text written from the final numbers, per role. It is NOT a language model.
* **R-peak markers** (6): display-only snapping onto the highest extremum of each QRS in the lead shown (fixes markers on the first bump of a notched QRS and on small/inverted leads such as III). The detector, the "Detected R-peaks" count and the model features are unchanged.
* **Chart-only PNG** (5): "Download chart only (PNG)" renders just the selected leads and time range.
* **Inputs** (2): new `modules/ingestion_extra.py` (the frozen `data_ingestion.py` is untouched): MATLAB .mat (+ optional .hea), HL7 aECG XML, GE MUSE XML (8 leads; III/aVR/aVL/aVF only if the user allows deriving), decimal-comma and non-UTF-8 CSV/TXT. Tested with synthetic files only.
* **UI fixes**: solid burgundy top bar with white navigation (9, 18); KPI card text no longer leaves the card (7); the "Press Enter to apply" hint no longer overlaps typed numbers (13); source links open in a new tab (11); Overview step cards, the "All configured references" table removed (14, 15); "Time window" renamed "Time range" (19); age/height/weight step = 1 (17).
* **History** (22): `history/analysis_history.json` (git-ignored) holds one factual line per analysis; no ECG samples. History page with CSV download and delete.
* **Recording setting** (12): "Recording made in" (ambulance / ED / ward). Notes and history only; the model was trained on PTB-XL hospital recordings, not ambulance recordings (stated on the Overview).
* Colour rule changed on request: values outside a published range and a model "Abnormal" are red (always with text); within-range chips stay blue; nothing is green on a clinical value.

## Second test round (21 items)
- No sidebar. The uploader, file options, patient information, "Recording made in" and the mandatory "Who is reading" choice (Doctor / Medical student) are on the Upload page; Signal Analysis and Reports stay locked until a role is chosen. Loaded files and choices survive page changes (kept in session state).
- Overview: guidelines in a collapsed expander, "Intended use" removed, a collapsed "Technical notes" expander at the end. Black bar: "Research prototype - not a medical device" only.
- Upload: the 3 guidance cards, the Analysis eligibility table and the rows "Dataset reference labels" and "Lead placement / polarity" removed.
- Assessment page removed. Signal Analysis keeps the viewer and Key measurements and adds T-wave duration, PR/ST/TP segments, ST level and the model's Normal/Abnormal output. Each card has an "Abnormality" fold only when its value is outside the published range (or the model says Abnormal). No attention banner, no technical notes on the cards.
- Viewer: R-peak markers with no clear complex are not drawn and double markers on one spike are merged (display only); estimated wave boundaries have diamond markers and hover text with the time in seconds. `ui/viewer.py` now also contains `export_png` (the missing function behind the AttributeError).
- Reports: no "Automated commentary"; the report text follows the chosen role. History: no caption, no attention filter, no Level or To-check column.
- Model, features and recorded metrics unchanged.
