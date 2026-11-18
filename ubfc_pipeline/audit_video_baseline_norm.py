import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

df = pd.read_csv("results/video/video_windows.csv")
feats = json.load(open("results/fusion_upperface/upperface_features_used.json"))
df = df.sort_values(["subject_id", "session_id", "window_start_sec"]).reset_index(drop=True)

N_BASE = 9
base_rows = (df[df.session_id == "T1"].groupby("subject_id").head(N_BASE)).index
base_mean = df.loc[base_rows].groupby("subject_id")[feats].mean()

norm = df.copy()
for s, mu in base_mean.iterrows():
    m = norm.subject_id == s
    norm.loc[m, feats] = norm.loc[m, feats] - mu.values

keep = ~df.index.isin(base_rows)
raw, nrm = df[keep].reset_index(drop=True), norm[keep].reset_index(drop=True)
print("evaluation rows:", len(raw), raw.groupby("session_id").size().to_dict())

def loso(d):
    p = np.full(len(d), np.nan)
    for s in d.subject_id.unique():
        tr, te = (d.subject_id != s).to_numpy(), (d.subject_id == s).to_numpy()
        med = d.loc[tr, feats].median()
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(d.loc[tr, feats].fillna(med), d.loc[tr, "label"])
        p[te] = clf.predict_proba(d.loc[te, feats].fillna(med))[:, 1]
    return p

def m(y, p):
    yh = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p), bacc=balanced_accuracy_score(y, yh),
                sens=tp / (tp + fn), spec=tn / (tn + fp))

out = {}
for name, d in [("raw", raw), ("baseline_subtracted", nrm)]:
    d["p"] = loso(d)
    pooled = m(d.label, d.p)
    pp = pd.DataFrame([dict(subject_id=s, **m(g.label, g.p)) for s, g in d.groupby("subject_id")])
    out[name] = pp.set_index("subject_id")
    print(f"\n{name} POOLED:", {k: round(float(v), 3) for k, v in pooled.items()})
    print(f"{name} mean per participant:", pp[["bacc", "spec", "sens"]].mean().round(3).to_dict())

cmp = pd.DataFrame({"spec_raw": out["raw"].spec, "spec_norm": out["baseline_subtracted"].spec,
                    "bacc_raw": out["raw"].bacc, "bacc_norm": out["baseline_subtracted"].bacc})
print("\nPER PARTICIPANT:\n", cmp.round(3).to_string())

rng = np.random.default_rng(42)
diff = (out["baseline_subtracted"].bacc - out["raw"].bacc).to_numpy()
bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
print("\nbacc change (norm - raw): mean", diff.mean().round(3),
      "95% CI", np.percentile(bs, [2.5, 97.5]).round(3))