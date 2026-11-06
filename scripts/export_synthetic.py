"""Export the synthetic verification patterns from the Streamlit Trends tab, using the same adapter."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.adapters.trend_adapter import TemporalTrendAdapter  # noqa: E402

OUT = ROOT / "frontend/src/content/synthetic.json"
N = 35
t = np.arange(N)


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


# Same generators as demo/ui/tab_trend.py
flat = np.clip(np.full(N, 0.15) + np.random.default_rng(42).normal(0, 0.02, N), 0.01, 0.99)

PATTERNS = [
    ("ramp_up", "Strong increasing ramp (distress onset)", "increasing", logistic(-2.0 + 4.0 * (t / N))),
    ("ramp_down", "Strong decreasing ramp (recovery)", "decreasing", logistic(2.0 - 4.0 * (t / N))),
    ("flat", "Flat stationary series (non-stress baseline)", "stable", flat),
]

adapter = TemporalTrendAdapter()


def num(v, digits):
    v = float(v)
    return round(v, digits) if math.isfinite(v) else None


out = []
for pid, label, planted, probs in PATTERNS:
    res = adapter.evaluate_series(probs)
    out.append(
        {
            "id": pid,
            "label": label,
            "planted": planted,  # what was planted, to compare against what the detector shows
            "p": [num(v, 4) for v in probs],
            "slope": [num(v, 4) for v in res["slope"]],
            "lo": [num(v, 4) for v in res["lo"]],
            "hi": [num(v, 4) for v in res["hi"]],
            "nValid": res["n_valid"].tolist(),
            "sufficient": [bool(v) for v in res["is_sufficient"]],
            "raw": [int(v) for v in res["raw"]],
            "displayed": [int(v) for v in res["displayed"]],
        }
    )
    print(f"{pid}: planted={planted}, displayed label at window 20 = {res['displayed_label'][20]}")

OUT.write_text(
    json.dumps({"K": adapter.cfg.K, "patterns": out}, separators=(",", ":")), encoding="utf-8"
)
print("wrote", OUT)