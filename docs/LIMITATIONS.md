# What is and is not verified

Verified by automated tests (48): feature extraction identical to the notebook code (bit-for-bit on 3 real records, synthetic ECGs);
bundle schema/config validation; export -> load round trip; prediction with a stub model; missing demographics; invalid files; incompatible
leads; 100-1000 Hz recordings; WFDB/CSV/TXT/EDF; single and multiple recordings; app end-to-end via AppTest.

NOT verified here (needs your Colab run): that your real exported model reproduces the notebook's probabilities and the recorded metrics
(`scripts/verify_consistency.py`). The sandbox never had your trained model or the PTB-XL cache. Until you run it, integration is NOT confirmed.

Morphology: NeuroKit2 wavelet delineation is not validated against expert annotations. On synthetic ECGs boundaries were off by up to ~35 ms;
on 3 real normal records QRS duration differed by tens of ms between leads and methods. Values are medians over leads, flagged
`low_agreement` when leads disagree, and the reference comparison widens its uncertainty margin (>= 30 ms) and returns `indeterminate`
instead of `above`/`below` near a limit. Reference ranges: P-wave duration (upper limit 120 ms, conventional) and RR interval (Mason 2007 heart-rate range converted, 0.61-1.25 s) were added. T-wave duration, PR/ST/TP segments and ST level have NO range: no population range from a reliable source was identified, so none is applied (shown as NO RANGE, for information only).

Open items: library versions at original training time are unknown (the export records the Colab runtime's); window start for recordings > 10 s,
the 5 s minimum, the beat-correlation gate (0.5) and the uncertainty floor (30 ms) are deployment policies of this application, not notebook values.

## Interface (redesign)
* Waveform values other than RR / heart rate are **estimates** and are labelled ESTIMATED; QTc is DERIVED from an estimate. Neither is validated.
* The viewer's "filtered" trace is a display filter at the file's own sampling rate; the model uses its separate 100 Hz pipeline. R-peaks come from the model's detector (100 Hz grid, 10 ms resolution).
* Wave boundaries are drawn only on the lead the consensus delineation reports (usually II).
* Not implemented: dataset reference labels (PTB-XL `scp_codes`), non-12-lead records (e.g. MIT-BIH), PDF export, a custom Inter font (Streamlit's bundled Source Sans is used so the app works offline).
* Recordings below 100 Hz and recordings that fail the quality checks (for example shorter than 5 s) are marked FAIL and not analyzed. The backend itself would have up-sampled a 50 Hz file; this is now refused at validation.
* Reference-range results use neutral blue (within) and amber (outside) - never green / red - because a range comparison is not a clinical verdict.
