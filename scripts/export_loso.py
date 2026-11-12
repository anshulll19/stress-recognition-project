import json
from pathlib import Path

import pandas as pd

SOURCES = {
    "wesad": Path("results/loso_corrected_valloss/loso_corrected_per_subject.csv"),
    "ubfc": Path("models/ubfc_deployment/loso_per_subject.csv"),
}
OUT = Path("frontend/src/content/loso")
OUT.mkdir(parents=True, exist_ok=True)

for name, src in SOURCES.items():
    if not src.exists():
        print(f"{name}: MISSING {src}")
        continue
    df = pd.read_csv(src)
    payload = {
        "columns": list(df.columns),
        "rows": json.loads(df.to_json(orient="values", double_precision=6)),
    }
    (OUT / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"{name}: {len(df)} rows, columns = {list(df.columns)}")
    print(df.head(3).to_string())
