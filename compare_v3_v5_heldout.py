import numpy as np, pandas as pd

a = pd.read_csv("results/neurokit2_v3_full_loso/participant_metrics.csv")
a = a[a.model == "RandomForest"].set_index("participant")
b = pd.read_csv("physiological_pipeline/03_NeuroKit_V2/results_loso_v5/v5_per_participant.csv")
b = b[(b.modality == "EDA_only") & (b.model == "RBF_SVM")].set_index("participant")

common = a.index.intersection(b.index)
v3, v5 = a.loc[common, "balanced_accuracy"], b.loc[common, "balanced_accuracy"]
print("participants:", len(common))

rng = np.random.default_rng(42)
def ci(x):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
    return np.percentile(bs, [2.5, 97.5]).round(3)

for name, x in [("V3 RF", v3), ("V5 EDA SVM", v5), ("V5 - V3", v5 - v3)]:
    x = x.to_numpy()
    print(name, "mean", x.mean().round(3), "95% CI", ci(x))

s3 = a.loc[common, "specificity"].to_numpy()
s5 = b.loc[common, "rest_specificity"].to_numpy()
print("specificity V3", s3.mean().round(3), "V5", s5.mean().round(3), "diff CI", ci(s5 - s3))
print("V5 > V3:", int((v5 > v3).sum()), "| V5 < V3:", int((v5 < v3).sum()))
