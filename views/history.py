"""History of analysed recordings (facts only: no ECG samples are stored)."""
import pandas as pd
import streamlit as st

from ui import history as HIST
from ui import theme as UI

st.markdown("## History")
entries = HIST.load()
if not entries:
    UI.empty_state("No history yet", "Load a recording; its summary is saved here after it is analysed.")
    st.stop()

only_flagged = st.checkbox("Show flagged recordings only", value=False, key="hist_only")
rows = []
for i in range(len(entries) - 1, -1, -1):
    e = entries[i]
    if only_flagged and not e.get("starred"):
        continue
    rows.append({"_i": i, "Flagged": bool(e.get("starred")), "Time": e["time"].replace("T", " "), "Recording": e["record"],
                 "Setting": e.get("setting") or "", "Age": "" if e.get("age") is None else f"{e['age']:g}", "Sex": e.get("sex") or "",
                 "Model": e.get("model") or "none", "P(Abnormal)": "" if e.get("p_abnormal") is None else f"{e['p_abnormal']:.2f}",
                 "Notes": e.get("note") or ""})
if not rows:
    UI.empty_state("No flagged recordings", "Tick the Flagged box on a row to mark a recording you want to come back to.")
else:
    df = pd.DataFrame(rows)
    edited = st.data_editor(df.drop(columns="_i"), hide_index=True, width="stretch", key="hist_table",
                            disabled=[c for c in df.columns if c not in ("Flagged", "Notes")],
                            column_config={"Notes": st.column_config.TextColumn("Notes", help="Type a note for this recording. Saved automatically.", max_chars=500, width="large"), "Flagged": st.column_config.CheckboxColumn("Flagged", help="Tick to mark a recording that needs attention. Saved automatically.", default=False)})
    new = {int(df["_i"].iloc[k]): bool(edited["Flagged"].iloc[k]) for k in range(len(df))}
    HIST.set_notes({int(df["_i"].iloc[k]): str(edited["Notes"].iloc[k] or "") for k in range(len(df))})
    if HIST.set_flags(new):
        st.rerun()
flat = pd.DataFrame([{k: (v if not isinstance(v, (list, dict)) else "; ".join(map(str, v)) if isinstance(v, list) else "; ".join(f"{m['label']}={m['value']} {m['unit']}" for m in v.values()))
                      for k, v in e.items()} for e in entries])
c1, c2 = st.columns(2)
c1.download_button("Download history (CSV)", flat.to_csv(index=False), file_name="ecg_history.csv", mime="text/csv", key="hist_csv")
with c2:
    if st.button("Delete the whole history", key="hist_clear"):
        HIST.clear(); st.rerun()
