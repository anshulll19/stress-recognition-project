from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path("physiological_pipeline/03_NeuroKit_V2")
A = BASE / "results_loso_v4_matched"
B = BASE / "results_loso_v5"
OUT = BASE / "results_matched_v5_comparison"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
N_BOOT = 10000
metrics = ["balanced_accuracy", "macro_f1", "accuracy"]
keys = ["modality", "model", "participant"]

a = pd.read_csv(A / "matched_per_participant.csv")
b = pd.read_csv(B / "v5_per_participant.csv")

merged = a.merge(
    b, on=keys, suffixes=("_control", "_v5"),
    how="outer", indicator=True
)

if not (merged["_merge"] == "both").all():
    raise ValueError("Participant/model records do not match.")

merged.drop(columns="_merge", inplace=True)

for col in ["test_rest", "test_distress"]:
    if not (
        merged[f"{col}_control"] == merged[f"{col}_v5"]
    ).all():
        raise ValueError(f"Mismatch in {col}.")

print("Matched records:", len(merged))
print("Participants:", merged["participant"].nunique())
print("Test splits identical: YES")

# Verify prediction-level true labels
p1 = pd.read_csv(A / "matched_predictions.csv")
p2 = pd.read_csv(B / "v5_predictions.csv")

pkeys = ["modality", "model", "participant"]

def label_groups(df):
    return {
        key: group["true_label"].to_numpy()
        for key, group in df.groupby(pkeys, sort=True)
    }

g1, g2 = label_groups(p1), label_groups(p2)

if g1.keys() != g2.keys():
    raise ValueError("Prediction groups differ.")

if not all(np.array_equal(g1[k], g2[k]) for k in g1):
    raise ValueError("True-label sequences differ.")

print("Prediction groups:", len(g1))
print("True-label sequences identical: YES")

def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0

    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)

    return adjusted

rows = []

for metric in metrics:
    for (modality, model), group in merged.groupby(
        ["modality", "model"]
    ):
        control = group[f"{metric}_control"].to_numpy(float)
        v5 = group[f"{metric}_v5"].to_numpy(float)
        diff = v5 - control

        test = wilcoxon(
            v5, control,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        )

        boot = np.array([
            rng.choice(diff, len(diff), replace=True).mean()
            for _ in range(N_BOOT)
        ])

        nonzero = diff[~np.isclose(diff, 0)]
        if len(nonzero):
            ranks = pd.Series(np.abs(nonzero)).rank().to_numpy()
            pos = ranks[nonzero > 0].sum()
            neg = ranks[nonzero < 0].sum()
            effect = (pos - neg) / (pos + neg)
        else:
            effect = 0.0

        rows.append({
            "modality": modality,
            "model": model,
            "metric": metric,
            "participants": len(diff),
            "control_mean": control.mean(),
            "v5_mean": v5.mean(),
            "mean_change": diff.mean(),
            "median_change": np.median(diff),
            "ci_low": np.percentile(boot, 2.5),
            "ci_high": np.percentile(boot, 97.5),
            "improved": int((diff > 0).sum()),
            "unchanged": int(np.isclose(diff, 0).sum()),
            "worsened": int((diff < 0).sum()),
            "p_value": test.pvalue,
            "rank_biserial": effect,
        })

results = pd.DataFrame(rows)
results["holm_adjusted_p"] = np.nan

for metric in metrics:
    mask = results["metric"] == metric
    results.loc[mask, "holm_adjusted_p"] = holm(
        results.loc[mask, "p_value"].to_numpy()
    )

results["significant_after_holm"] = (
    results["holm_adjusted_p"] < 0.05
)

results.to_csv(OUT / "matched_v5_statistical_tests.csv", index=False)
merged.to_csv(OUT / "matched_v5_participant_data.csv", index=False)

for metric in ["balanced_accuracy", "macro_f1"]:
    print("\n" + "=" * 100)
    print("MATCHED CONTROL vs V5:", metric.upper())
    print("=" * 100)

    cols = [
        "modality", "model", "participants",
        "control_mean", "v5_mean", "mean_change",
        "ci_low", "ci_high", "improved", "worsened",
        "p_value", "holm_adjusted_p",
        "rank_biserial", "significant_after_holm"
    ]

    print(
        results[results["metric"] == metric][cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

print("\nSaved to:", OUT.resolve())
