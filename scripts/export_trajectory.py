import json
from pathlib import Path

import pandas as pd

SRC = Path("results/fusion_lstm/aligned_oof_predictions.csv")
OUT = Path("frontend/src/content/trajectory.json")
SUBJECT = "s3"
FUSED_COL = "p_blend_0.3_uf"
TASKS = {1: "Rest (T1)", 2: "Speech (T2)", 3: "Math (T3)"}
WINDOW = 10.0

df = pd.read_csv(SRC)
sub = df[df["subject_id"] == SUBJECT]
if sub.empty:
    raise SystemExit(f"No rows found for subject {SUBJECT}")

points, tasks = [], []
offset = 0.0
for tid, name in TASKS.items():
    part = sub[sub["task"] == tid].sort_values("window_start_sec")
    if part.empty:
        continue
    base = float(part["window_start_sec"].iloc[0])
    for r in part.to_dict("records"):
        points.append(
            {
                "t": round(offset + float(r["window_start_sec"]) - base, 2),
                "physio": round(float(r["p_physio"]), 4),
                "video": round(float(r["p_video"]), 4),
                "fused": round(float(r[FUSED_COL]), 4),
            }
        )
    length = float(part["window_start_sec"].iloc[-1]) - base + WINDOW
    tasks.append({"id": tid, "name": name, "start": offset, "end": offset + length})
    offset += length

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps({"subject": SUBJECT, "points": points, "tasks": tasks}, indent=2),
    encoding="utf-8",
)
print(f"Wrote {len(points)} windows for {SUBJECT} to {OUT}")