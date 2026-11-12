from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path("physiological_pipeline/03_NeuroKit_V2")
V4 = BASE / "results_loso_v4"
V5 = BASE / "results_loso_v5"
OUT = BASE / "results_v4_v5_comparison"

OUT.mkdir(exist_ok=True)

METRICS = ["balanced_accuracy", "macro_f1", "accuracy"]
PRIMARY = "balanced_accuracy"
N_BOOT = 10000
RNG = np.random.default_rng(42)

# --------------------------------------------------
# Load results
# --------------------------------------------------

v4 = pd.read_csv(V4 / "v4_per_participant.csv")
v5 = pd.read_csv(V5 / "v5_per_participant.csv")

keys = ["modality", "model", "participant"]

required = keys + METRICS + ["test_rest", "test_distress"]

for name, df in [("V4", v4), ("V5", v5)]:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    if df.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicate participant records.")

# --------------------------------------------------
# Verify matching experiments and participants
# --------------------------------------------------

experiments_v4 = set(map(tuple, v4[["modality", "model"]].drop_duplicates().values))
experiments_v5 = set(map(tuple, v5[["modality", "model"]].drop_duplicates().values))

if experiments_v4 != experiments_v5:
    raise ValueError("V4 and V5 experiment combinations do not match.")

merged = v4.merge(
    v5,
    on=keys,
    suffixes=("_v4", "_v5"),
    how="outer",
    indicator=True
)

if not (merged["_merge"] == "both").all():
    raise ValueError("V4 and V5 participant records do not match.")

merged.drop(columns="_merge", inplace=True)

for col in ["test_rest", "test_distress"]:
    if not (merged[f"{col}_v4"] == merged[f"{col}_v5"]).all():
        raise ValueError(f"Mismatch in {col} between V4 and V5.")

print("\nPARTICIPANT AND TEST-SPLIT VALIDATION")
print("------------------------------------")
print("Matched participant/model records:", len(merged))
print("Unique participants:", merged["participant"].nunique())
print("Experiment combinations:", len(experiments_v4))
print("Test counts identical: YES")

# --------------------------------------------------
# Verify prediction-level true labels
# --------------------------------------------------

p4 = pd.read_csv(V4 / "v4_predictions.csv")
p5 = pd.read_csv(V5 / "v5_predictions.csv")

prediction_keys = ["modality", "model", "participant"]

for name, df in [("V4", p4), ("V5", p5)]:
    missing = set(prediction_keys + ["true_label", "predicted_label"]) - set(df.columns)
    if missing:
        raise ValueError(f"{name} predictions missing: {missing}")

    if df.duplicated(prediction_keys + ["true_label", "predicted_label"]).all():
        print(f"Warning: {name} has duplicate prediction rows.")

def check_true_labels(a, b):
    groups_a = {
        key: group["true_label"].to_numpy()
        for key, group in a.groupby(prediction_keys, sort=True)
    }
    groups_b = {
        key: group["true_label"].to_numpy()
        for key, group in b.groupby(prediction_keys, sort=True)
    }

    if groups_a.keys() != groups_b.keys():
        return False

    return all(
        np.array_equal(groups_a[k], groups_b[k])
        for k in groups_a
    )

labels_match = check_true_labels(p4, p5)

print("Prediction groups:", p4.groupby(prediction_keys).ngroups)
print("True-label sequences match:", "YES" if labels_match else "NO")

if not labels_match:
    raise ValueError(
        "Prediction-level true labels differ. "
        "Do not proceed with a direct V4-V5 comparison."
    )

# --------------------------------------------------
# Paired statistical analysis
# --------------------------------------------------

def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)

    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (n - rank) * pvalues[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted

def analyze_pair(group, metric):
    x = group[f"{metric}_v4"].to_numpy(dtype=float)
    y = group[f"{metric}_v5"].to_numpy(dtype=float)

    diff = y - x
    n = len(diff)

    if n < 2:
        raise ValueError("Insufficient paired participants.")

    if np.allclose(diff, 0):
        statistic, pvalue = 0.0, 1.0
    else:
        result = wilcoxon(
            y, x,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        )
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)

    nonzero = diff[~np.isclose(diff, 0)]

    if len(nonzero):
        ranks = pd.Series(np.abs(nonzero)).rank().to_numpy()
        positive = ranks[nonzero > 0].sum()
        negative = ranks[nonzero < 0].sum()
        rank_biserial = (positive - negative) / (positive + negative)
    else:
        rank_biserial = 0.0

    # Participant-level paired bootstrap
    boot = np.empty(N_BOOT)

    for i in range(N_BOOT):
        sample = RNG.choice(diff, size=n, replace=True)
        boot[i] = sample.mean()

    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    return {
        "participants": n,
        "v4_mean": x.mean(),
        "v5_mean": y.mean(),
        "mean_change": diff.mean(),
        "median_change": np.median(diff),
        "v4_std": x.std(ddof=1),
        "v5_std": y.std(ddof=1),
        "improved": int((diff > 0).sum()),
        "unchanged": int(np.isclose(diff, 0).sum()),
        "worsened": int((diff < 0).sum()),
        "wilcoxon_statistic": statistic,
        "p_value": pvalue,
        "rank_biserial_effect": rank_biserial,
        "bootstrap_95ci_low": ci_low,
        "bootstrap_95ci_high": ci_high,
    }

all_results = []

for metric in METRICS:
    for (modality, model), group in merged.groupby(
        ["modality", "model"], sort=True
    ):
        row = {
            "modality": modality,
            "model": model,
            "metric": metric,
        }
        row.update(analyze_pair(group, metric))
        all_results.append(row)

results = pd.DataFrame(all_results)

# Holm correction separately for each metric
results["holm_adjusted_p"] = np.nan

for metric in METRICS:
    mask = results["metric"] == metric
    results.loc[mask, "holm_adjusted_p"] = holm_adjust(
        results.loc[mask, "p_value"].to_numpy()
    )

results["significant_after_holm"] = (
    results["holm_adjusted_p"] < 0.05
)

# --------------------------------------------------
# Save outputs
# --------------------------------------------------

results.to_csv(OUT / "paired_statistical_tests.csv", index=False)
merged.to_csv(OUT / "matched_participant_comparison.csv", index=False)

# Compact primary metric table
primary = results[results["metric"] == PRIMARY].copy()

primary.to_csv(
    OUT / "balanced_accuracy_comparison.csv",
    index=False
)

print("\n" + "=" * 100)
print("PAIRED V4 vs V5 RESULTS — BALANCED ACCURACY")
print("=" * 100)

columns = [
    "modality", "model", "participants",
    "v4_mean", "v5_mean", "mean_change",
    "bootstrap_95ci_low", "bootstrap_95ci_high",
    "improved", "unchanged", "worsened",
    "p_value", "holm_adjusted_p",
    "rank_biserial_effect", "significant_after_holm"
]

print(
    primary[columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\n" + "=" * 100)
print("PAIRED V4 vs V5 RESULTS — MACRO-F1")
print("=" * 100)

macro = results[results["metric"] == "macro_f1"]

print(
    macro[columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.5f}"
    )
)

print("\nSaved comparison files to:", OUT.resolve())
