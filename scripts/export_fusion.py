import json
from pathlib import Path

import pandas as pd

# Note: Streamlit looks up p_video_only and p_physio_only, which are not columns in this CSV.
# The React page will map video_only -> p_video and physio_only -> p_physio instead.
SRC = Path("results/fusion_lstm/aligned_oof_predictions.csv")
SUMMARY = Path("results/fusion_lstm/fusion_summary.csv")
OUT = Path("frontend/src/content/fusion")

df = pd.read_csv(SRC)
cols = list(df.columns)
subjects = {}
for sid, g in df.groupby("subject_id", sort=True):
    subjects[sid] = json.loads(g.to_json(orient="values", double_precision=10))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "aligned.json").write_text(
    json.dumps({"columns": cols, "subjects": subjects}, separators=(",", ":")),
    encoding="utf-8",
)

summary = pd.read_csv(SUMMARY) if SUMMARY.exists() else pd.DataFrame()
(OUT / "summary.json").write_text(
    summary.to_json(orient="records", double_precision=6), encoding="utf-8"
)

print("columns:", cols)
print("subjects:", len(subjects), "windows:", len(df))
print("summary rows:", len(summary), "summary columns:", list(summary.columns))