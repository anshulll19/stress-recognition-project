import json
from pathlib import Path

import pandas as pd

SRC = Path("results/fusion_lstm/aligned_oof_predictions.csv")
OUT = Path("frontend/src/content/trajectory.json")
SUBJECT = "s3"
FUSED_COL = "p_blend_0.3_uf"
TASKS = {1: "Rest (T1)", 2: "Speech (T2)", 3: "Math (T3)"}

df = pd.read_csv(SRC)
sub = df[df["subject_id"] == SUBJECT].sort_values("window_start_sec")
if sub.empty:
    raise SystemExit(f"No rows found for subject {SUBJECT}")

points = [
    {
        "t": float(r["window_start_sec"]),
        "physio": round(float(r["p_physio"]), 4),
        "video": round(float(r["p_video"]), 4),
        "fused": round(float(r[FUSED_COL]), 4),
    }
    for r in sub.to_dict("records")
]

tasks = []
for tid, name in TASKS.items():
    m = sub[sub["task"] == tid]
    if not m.empty:
        tasks.append(
            {
                "id": tid,
                "name": name,
                "start": float(m["window_start_sec"].iloc[0]),
                "end": float(m["window_start_sec"].iloc[-1]) + 10,
            }
        )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps({"subject": SUBJECT, "points": points, "tasks": tasks}, indent=2),
    encoding="utf-8",
)
print(f"Wrote {len(points)} windows for {SUBJECT} to {OUT}")