from pathlib import Path

import pandas as pd

SRC = Path("results/fusion_lstm/fusion_confound_analysis.csv")
OUT = Path("frontend/src/content/fusion/confounds.json")

df = pd.read_csv(SRC)
OUT.write_text(df.to_json(orient="records", double_precision=6), encoding="utf-8")
print("rows:", len(df), "columns:", list(df.columns))