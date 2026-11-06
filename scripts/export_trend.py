"""Export causal trend results for the Trends tab using the same adapter as Streamlit."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.adapters.trend_adapter import TemporalTrendAdapter  # noqa: E402

NPZ = ROOT / "results/physiology/raw_signal_windows.npz"
META = ROOT / "results/physiology/raw_signal_window_metadata.csv"
OOF = ROOT / "results/temporal_models/tcn_lstm_oof_predictions_full.csv"
OUT = ROOT / "frontend/src/content/trend"

groups = np.load(NPZ)["groups"]
meta = pd.read_csv(META)
if len(meta) != len(groups) or not (meta["participant"].to_numpy() == groups).all():
    raise SystemExit(
        "Metadata rows do not line up with the signal arrays. "
        f"meta rows={len(meta)}, windows={len(groups)}. Paste this message to Claude."
    )

oof = pd.read_csv(OOF)
models = sorted(oof["model"].unique().tolist())
participants = sorted(int(p) for p in oof["participant"].unique())

# first row wins, matching the Streamlit adapter
lookup = {}
for r in oof.itertuples(index=False):
    lookup.setdefault((int(r.window_index), r.model), float(r.probability))

adapter = TemporalTrendAdapter()
K = adapter.cfg.K


def num(v, digits):
    v = float(v)
    return round(v, digits) if math.isfinite(v) else None


OUT.mkdir(parents=True, exist_ok=True)
check = None

for p in participants:
    idx = np.where(groups == p)[0]
    task = meta.loc[idx, "task"].astype(int).tolist()
    start = [num(v, 1) for v in meta.loc[idx, "start_s"]]
    # start_s restarts at 0 in each task, so only compare windows inside the same task,
    # and make sure the task sequence never goes backwards (1, 2, 3).
    bad_order = any(
        tb < ta or (tb == ta and sb < sa)
        for ta, tb, sa, sb in zip(task, task[1:], start, start[1:])
    )
    if bad_order:
        raise SystemExit(f"participant {p}: windows are not in time order. Paste this to Claude.")

    # True when the last K windows cover more than one task (slope straddles a task change)
    spans = [len(set(task[max(0, t - K + 1): t + 1])) > 1 for t in range(len(task))]

    out_models = {}
    for m in models:
        probs = np.array([lookup.get((int(gi), m), np.nan) for gi in idx])
        if np.isnan(probs).all():
            continue
        res = adapter.evaluate_series(probs)
        out_models[m] = {
            "p": [num(v, 4) for v in probs],
            "slope": [num(v, 4) for v in res["slope"]],
            "lo": [num(v, 4) for v in res["lo"]],
            "hi": [num(v, 4) for v in res["hi"]],
            "nValid": res["n_valid"].tolist(),
            "sufficient": [bool(v) for v in res["is_sufficient"]],
            "raw": [int(v) for v in res["raw"]],
            "displayed": [int(v) for v in res["displayed"]],
        }
        if p == 1 and m == "LSTM":
            check = {k: out_models[m][k][15] for k in ("slope", "lo", "hi", "nValid", "displayed")}

    path = OUT / f"p{p}.json"
    path.write_text(
        json.dumps(
            {"participant": p, "task": task, "start": start, "spansTasks": spans, "models": out_models},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

(OUT / "index.json").write_text(
    json.dumps({"participants": participants, "models": models, "K": K}), encoding="utf-8"
)
total = sum(f.stat().st_size for f in OUT.glob("p*.json")) / 1024
print(f"{len(participants)} participants, models: {models}, total {total:.0f} KB")
print("CHECK participant 1, LSTM, window 15:", check)