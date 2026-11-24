# Redesign changelog (Phases 0-5)

**Phase 0** audit (UI_AUDIT.md). **Phase 1** theme, tokens, header, sidebar, cards, charts. **Phase 2** Overview and Upload pages, workflow, empty states, "Not provided" patient fields.
**Phase 3** data inspection (validation, field states, eligibility), interactive viewer, KPI cards with evidence tags. **Phase 4** Assessment (A measurements, B model, C reference ranges, D dataset labels, explainability), restyled report. **Phase 5** QA, docs.

## Changed
* `app.py` is now a shell; pages are in `views/`; presentation helpers in `ui/`.
* `modules/visualization.py`: colours/grid only. `modules/report.py`: palette and wording only ("B. Waveform measurements - automatic estimates"; validity column).
* Reference-range charts no longer use green/red.
* Patient fields start empty ("Not provided"); an unavailable value is "Not available", never 0.
* Recordings below 100 Hz, or failing the quality gate, are marked FAIL and not analyzed (previously a 50 Hz file was silently up-sampled).
* Analysis runs only on "Analyze Recording" and is cached; results go stale visibly when inputs change.

## Reference ranges (after Phase 5, deliberate change to `modules/reference_comparison.py`)
* Added P-wave duration (<= 120 ms, conventional adult upper limit, cited via Magnani 2010 / PMC3394095; some sources use 110 ms) and RR interval (0.61-1.25 s = Mason 2007 heart-rate range 48-98 bpm converted by 60/HR).
* Measures with no reliable published range (T-wave duration, PR/ST/TP segments, ST level) are labelled NO RANGE with a specific reason; no range was invented.
* Display: RR in seconds (2 decimals), ST level 3 decimals, one-sided ranges shown as "<= x".
* Only the `cmp` part of the numerical baseline changed (RR and P duration now within/above/below instead of indeterminate); probabilities, features and morphology are identical. The module's pinned hash was updated.

## Unchanged (verified)
Every analysis module (`config`, `data_ingestion`, `preprocessing`, `feature_extraction`, `prediction`, `ecg_delineation`, `explainability`, `bundle_builder`) is byte-identical to Phase 0 (hash test) and the probabilities, 145 features, morphology and comparisons on the example records are identical to the pre-redesign baseline (UI-12). The model files were never touched.

## Known gaps
Dataset labels, non-12-lead records, PDF export, Inter font - see docs/LIMITATIONS.md. `st.switch_page` was not used after "Analyze" because it dropped the sidebar upload state in a real browser; page links are shown instead.
