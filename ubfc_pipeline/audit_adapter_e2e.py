import argparse, inspect, sys
from dataclasses import asdict, is_dataclass
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from demo.adapters.video_adapter import VideoStressAdapter

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--subject", default="s16")
ap.add_argument("--task", default="T3")
ap.add_argument("--method", default=None)
a = ap.parse_args()

adapter = VideoStressAdapter()
methods = [n for n, f in inspect.getmembers(adapter, inspect.ismethod)
           if not n.startswith("_")]
print("public methods:", methods)
name = a.method or next((m for m in methods if any(k in m for k in
                         ("predict", "infer", "analy", "process", "run"))), None)
if name is None:
    sys.exit("No method found; rerun with --method NAME")
print("calling:", name, inspect.signature(getattr(adapter, name)))
result = getattr(adapter, name)(a.video)
print("result type:", type(result).__name__)

def collect(obj, key, out):
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if isinstance(obj, pd.DataFrame):
        if key in obj.columns:
            out.extend(obj[key].tolist())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and np.isscalar(v):
                out.append(v)
            else:
                collect(v, key, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect(v, key, out)
    elif hasattr(obj, "__dict__"):
        collect(vars(obj), key, out)

live = []
collect(result, "stress_probability_score", live)
live = np.array(live, dtype=float)
print("adapter windows:", len(live))
if len(live) == 0:
    sys.exit("No stress_probability_score found; inspect the result structure above.")

df = pd.read_csv("results/video/video_windows.csv")
b = joblib.load("models/video_distress_upperface.joblib")
sub = df[(df.subject_id == a.subject) & (df.session_id == a.task)]
sub = sub.sort_values("window_start_sec")
idx = list(b["model"].classes_).index(1)
direct = b["model"].predict_proba(b["scaler"].transform(
    sub[list(b["features"])].to_numpy(float)))[:, idx]
print("direct (CSV features) windows:", len(direct))

n = min(len(live), len(direct))
l, d = live[:n], direct[:n]
print(f"\ncompared first {n} windows")
print("mean live:", l.mean().round(4), "| mean direct:", d.mean().round(4))
print("mean abs diff:", np.abs(l - d).mean().round(4))
print("max abs diff:", np.abs(l - d).max().round(4))
print("correlation:", np.corrcoef(l, d)[0, 1].round(4))