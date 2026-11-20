import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             confusion_matrix)

df = pd.read_csv("results/video/video_windows.csv")
feats = json.load(open("results/fusion_upperface/upperface_features_used.json"))
subjects = np.sort(df["subject_id"].unique())
print("rows:", len(df), "subjects:", len(subjects))
print(df.groupby("session_id")["label"].agg(["count", "mean"]))

df["p"] = np.nan
for s in subjects:
    tr, te = df["subject_id"] != s, df["subject_id"] == s
    med = df.loc[tr, feats].median()
    clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                             class_weight="balanced", random_state=42)
    clf.fit(df.loc[tr, feats].fillna(med), df.loc[tr, "label"])
    df.loc[te, "p"] = clf.predict_proba(df.loc[te, feats].fillna(med))[:, 1]

def metrics(y, p):
    yh = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p), bacc=balanced_accuracy_score(y, yh),
                sens=tp / (tp + fn), spec=tn / (tn + fp))

print("\nPOOLED:", metrics(df["label"], df["p"]))

rows = [dict(subject_id=s, **metrics(g["label"], g["p"]))
        for s, g in df.groupby("subject_id")]
pp = pd.DataFrame(rows)
print("\nPER PARTICIPANT:\n", pp.round(3).to_string(index=False))

rng = np.random.default_rng(42)
boot = [pp.loc[rng.integers(0, len(pp), len(pp)), ["bacc", "spec", "sens"]].mean()
        for _ in range(2000)]
boot = pd.DataFrame(boot)
print("\nMean per-participant (95% bootstrap CI over participants):")
for c in ["bacc", "spec", "sens"]:
    print(c, round(pp[c].mean(), 3),
          np.percentile(boot[c], [2.5, 97.5]).round(3))

# task confound checks
for a, b in [("T1", "T2"), ("T1", "T3")]:
    m = df["session_id"].isin([a, b])
    print(f"\n{a} vs {b}:", metrics(df.loc[m, "label"], df.loc[m, "p"]))
t23 = df[df["session_id"].isin(["T2", "T3"])]
print("\nMean p by task:\n", df.groupby("session_id")["p"].mean().round(3))

df[["subject_id", "session_id", "window_start_sec", "label", "p"]].to_csv(
    "results/video/audit_loso_predictions.csv", index=False)