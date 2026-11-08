import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

NPZ = Path("results/physiology/raw_signal_windows.npz")
META = Path("results/physiology/raw_signal_window_metadata.csv")
OOF = Path("results/temporal_models/tcn_lstm_oof_predictions_full.csv")
OUT = Path("frontend/src/content/physio")
PARTICIPANTS = [1, 2, 3, 4, 5]

z = np.load(NPZ)
bvp = z["X_bvp"][:, :, 0]
eda = z["X_eda"][:, :, 0]
y = z["y"]
groups = z["groups"]
meta = pd.read_csv(META)

if len(meta) != len(bvp) or not (meta["participant"].to_numpy() == groups).all():
    raise SystemExit(
        "Metadata rows do not line up with the signal arrays. "
        f"meta rows={len(meta)}, windows={len(bvp)}. Paste this message to Claude."
    )

oof = pd.read_csv(OOF)
oof = oof[oof["participant"].isin(PARTICIPANTS)]
models = sorted(oof["model"].unique().tolist())

# first row wins, matching what the Streamlit adapter does
lookup = {}
for r in oof.itertuples(index=False):
    lookup.setdefault((int(r.window_index), r.model), (float(r.probability), int(r.prediction), int(r.true)))


def num(v, digits):
    v = float(v)
    return round(v, digits) if math.isfinite(v) else None


OUT.mkdir(parents=True, exist_ok=True)
mismatches = 0
check = None

for p in PARTICIPANTS:
    idx = np.where(groups == p)[0]
    windows = []
    for gi in idx:
        b, e = bvp[gi], eda[gi]
        pred = {}
        for m in models:
            hit = lookup.get((int(gi), m))
            if hit:
                pred[m] = {"p": num(hit[0], 4), "c": hit[1]}
                if hit[2] != int(y[gi]):
                    mismatches += 1
        win = {
            "index": int(gi),
            "task": int(meta.at[gi, "task"]),
            "start": num(meta.at[gi, "start_s"], 1),
            "end": num(meta.at[gi, "end_s"], 1),
            "label": int(y[gi]),
            "pred": pred,
            "bvp": [num(v, 2) for v in b[::2]],  # 64 Hz shown at 32 Hz
            "eda": [num(v, 3) for v in e],  # 4 Hz, full resolution
            "nBvp": int(len(b)),
            "nEda": int(len(e)),
            "bvpStd": num(np.std(b), 4),  # quality stats use the full 64 Hz signal
            "edaStd": num(np.std(e), 4),
        }
        if p == 1 and len(windows) == 0:
            check = win
        windows.append(win)
    path = OUT / f"p{p}.json"
    path.write_text(
        json.dumps({"participant": p, "windows": windows}, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"participant {p}: {len(windows)} windows, {path.stat().st_size / 1024:.0f} KB")

(OUT / "index.json").write_text(
    json.dumps({"participants": PARTICIPANTS, "models": models}), encoding="utf-8"
)
print("models:", models)
print("label mismatches between signal file and predictions file:", mismatches)
print(
    "CHECK participant 1, window 0:",
    {k: check[k] for k in ("index", "task", "start", "end", "label", "pred", "bvpStd", "edaStd")},
)