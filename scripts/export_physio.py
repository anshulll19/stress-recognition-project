import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

NPZ = Path("results/physiology/raw_signal_windows.npz")
META = Path("results/physiology/raw_signal_window_metadata.csv")
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


def num(v, digits):
    v = float(v)
    return round(v, digits) if math.isfinite(v) else None


OUT.mkdir(parents=True, exist_ok=True)

for p in PARTICIPANTS:
    idx = np.where(groups == p)[0]
    windows = []
    for gi in idx:
        b, e = bvp[gi], eda[gi]
        windows.append(
            {
                "index": int(gi),
                "task": int(meta.at[gi, "task"]),
                "start": num(meta.at[gi, "start_s"], 1),
                "end": num(meta.at[gi, "end_s"], 1),
                "label": int(y[gi]),
                "bvp": [num(v, 2) for v in b[::2]],  # 64 Hz shown at 32 Hz
                "eda": [num(v, 3) for v in e],  # 4 Hz, full resolution
                "nBvp": int(len(b)),
                "nEda": int(len(e)),
                "bvpStd": num(np.std(b), 4),  # quality stats use the full 64 Hz signal
                "edaStd": num(np.std(e), 4),
            }
        )
    path = OUT / f"p{p}.json"
    path.write_text(
        json.dumps({"participant": p, "windows": windows}, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"participant {p}: {len(windows)} windows, {path.stat().st_size / 1024:.0f} KB")