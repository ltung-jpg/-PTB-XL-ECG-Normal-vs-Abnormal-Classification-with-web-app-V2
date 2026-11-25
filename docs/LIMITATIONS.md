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
* Reference-range results: blue (within) and red (outside), always with a text label. Red marks a deviation from a published population range (requested after testing); it is not a clinical verdict. Nothing is green on a clinical value.

## After the test round
* The attention banner and the "what to verify / what it means" texts are generic, rule-based and not validated clinically. They never claim a diagnosis. QTc > 500 ms (higher torsades risk) is the only threshold added beyond the published ranges (AHA/ACCF statement, Drew 2010).
* The commentary in the report is built from templates, not by a language model. Using a language model would need an API key and would send patient data to an outside service; it is not done here.
* The model was trained on PTB-XL (resting hospital ECGs). Its accuracy on ambulance recordings, with more motion and mains noise, has not been measured.
* R-peak markers in the viewer are placed on the highest QRS extremum of the shown lead (display only). The count of "Detected R-peaks" and the model features still come from the original detector; a QRS with no clear extremum (very small lead) can still get a marker on noise.
* New input readers (.mat, HL7 aECG, GE MUSE, decimal-comma CSV) were tested with synthetic files that follow each format's layout. Philips XML, SCP-ECG, DICOM and other vendor formats are not read. A file that fails shows the reason.
* History is stored in a local file (`history/`); on a hosted service it can disappear on restart.

## After the second test round
- Switching the lead selection or turning on the boundary markers only changes the chart; the measurements do not depend on them.
- R-peak marker clean-up is display only. A marker may be hidden where the QRS is very small in the lead shown (for example lead III); the detector and the model features are unchanged.
- History is saved when a recording is analyzed, before a role may have been chosen; the role is not part of the history table.
- Tested here with the test-stub model only; loading the real model in the Python 3.13 environment is still to be confirmed (the TEST-STUB banner must disappear).
