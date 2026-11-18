import argparse, hashlib, json, platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--n-estimators", type=int, default=100)
ap.add_argument("--out", default="models/video_distress_upperface_reproduced.joblib")
a = ap.parse_args()

CSV = "results/video/video_windows.csv"
feats = json.load(open("results/fusion_upperface/upperface_features_used.json"))
df = pd.read_csv(CSV)

medians = df[feats].median()
X = df[feats].fillna(medians)
y = df["label"].to_numpy()

scaler = StandardScaler().fit(X)
model = RandomForestClassifier(n_estimators=a.n_estimators, random_state=42)
model.fit(scaler.transform(X), y)

assert list(model.classes_) == [0, 1]
joblib.dump({"model": model, "scaler": scaler, "features": feats}, a.out)

meta = {
    "script": "ubfc_pipeline/train_video_upperface.py",
    "n_estimators": a.n_estimators,
    "random_state": 42,
    "imputation": "median of all training rows, applied before scaling",
    "participants": sorted(df["subject_id"].unique().tolist()),
    "n_windows": int(len(df)),
    "source_csv_sha256": hashlib.sha256(open(CSV, "rb").read()).hexdigest(),
    "feature_count": len(feats),
    "sklearn": sklearn.__version__,
    "numpy": np.__version__,
    "python": platform.python_version(),
    "note": "Trained on ALL participants; predictions on these participants are in-sample.",
}
json.dump(meta, open(a.out.replace(".joblib", "_manifest.json"), "w"), indent=2)
print("saved", a.out, "| participants:", len(meta["participants"]))