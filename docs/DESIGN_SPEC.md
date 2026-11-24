# Design specification - Clinical Burgundy & White

Colour theme sampled from the reference site supplied by the user (only colours were adopted). Tokens live in `ui/tokens.py` and are mirrored in `.streamlit/config.toml`; `tests/test_ui_theme.py` fails if they drift apart or if any text pair drops below WCAG AA (4.5:1).

| Role | Value | | Role | Value |
|---|---|---|---|---|
| App background | #F7F8FA | | Burgundy accent | #9D3141 |
| Surface / cards | #FFFFFF | | Burgundy hover | #832636 |
| Sidebar | #FAFAFB | | Burgundy tint | #F8ECEE |
| Light grey panel | #EEF1F6 | | Utility bar | #222222 |
| Text / secondary / muted | #202124 / #59616B / #68717D | | Border | #DFE3E7 |
| Success / Warning / Error / Info (fill) | #33845B / #A86F26 / #B84242 / #466D95 | | AA text variants | #2E7953 / #9A6622 / #B84242 / #466D95 |
| ECG line | #9D3141 | | Secondary waveform | #526A7A |
| Reference annotations | #6675A4 | | R-peak marker | #B9824F |
| Grid major / minor | #D8DDE2 / #EDF0F2 | | Chart background | #FFFFFF |

Rules: light theme only; no dark panels (only the thin charcoal utility bar), no gradients, no navy/teal/neon; cards 12 px radius with a 1 px border; sidebar 272 px; body text >= 12 px; every badge carries a text label (colour never the only signal).

Evidence vocabulary: MEASURED / ESTIMATED / DERIVED / PREDICTED / UNAVAILABLE. Validation: PASS / NEEDS REVIEW / FAIL. Field states: AVAILABLE / MISSING / UNKNOWN / NOT APPLICABLE / DERIVABLE / INVALID. Eligibility: ELIGIBLE / LIMITED / BLOCKED / NOT SUPPORTED.

Colour never states a clinical conclusion: model output uses the brand "PREDICTED" tag, reference-range results use neutral blue (within) and amber (outside), nothing is green "good" or red "bad".

Chart styling: white background, burgundy ECG trace, slate raw trace when both are shown, triangles (amber-brown) for R-peaks, dashed indigo lines for estimated wave boundaries. Static matplotlib figures (report, probability bar, explanations) use the same tokens.

Not adopted from the reference: layout, photography, logo, wording. Known deviation: font is Streamlit's bundled Source Sans (offline-safe) instead of Inter.
