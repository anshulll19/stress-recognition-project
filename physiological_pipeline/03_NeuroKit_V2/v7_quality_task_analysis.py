
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent
INPUT = Path("../../results/neurokit2_v2/fused_features_v2.csv")
OUT = BASE / "results_v7_quality_analysis"
OUT.mkdir(exist_ok=True)

N_BOOT = 10_000
RANDOM_STATE = 42

FEATURES = [
    "eda_raw_std",
    "eda_unique_fraction",
    "eda_flat_step_fraction",
    "eda_median_abs_difference",
    "eda_max_abs_jump",
    "eda_jump_ratio",
]

COMPARISONS = [
    (1, 2, "T1_vs_T2"),
    (1, 3, "T1_vs_T3"),
    (2, 3, "T2_vs_T3"),
]


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running_max = 0.0

    for rank, idx in enumerate(order):
        running_max = max(running_max, (n - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def bootstrap_mean_ci(differences, rng):
    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    samples = rng.choice(
        differences,
        size=(N_BOOT, n),
        replace=True,
    )
    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


df = pd.read_csv(INPUT)

required = {"participant", "task", *FEATURES}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["task"] = pd.to_numeric(df["task"], errors="coerce")
df = df[df["task"].isin([1, 2, 3])].copy()

for feature in FEATURES:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

# One row per participant per task.
participant_task = (
    df.groupby(["participant", "task"])[FEATURES]
    .median()
    .reset_index()
)

participant_task.to_csv(
    OUT / "v7_participant_task_medians.csv",
    index=False,
)

print("Participant-task table shape:", participant_task.shape)
print("Participants:", participant_task["participant"].nunique())

print("\nParticipant counts by task:")
print(participant_task.groupby("task")["participant"].nunique())

print("\nParticipant-level task medians:")
print(
    participant_task.groupby("task")[FEATURES]
    .median()
    .round(6)
)

rng = np.random.default_rng(RANDOM_STATE)
records = []

for feature in FEATURES:
    wide = participant_task.pivot(
        index="participant",
        columns="task",
        values=feature,
    )

    for task_a, task_b, comparison in COMPARISONS:
        pair = wide[[task_a, task_b]].dropna()
        x = pair[task_a].to_numpy(dtype=float)
        y = pair[task_b].to_numpy(dtype=float)
        diff = x - y

        if len(diff) < 10:
            raise ValueError(
                f"Too few paired participants: {feature}, {comparison}"
            )

        if np.allclose(diff, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(
                x,
                y,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)

        ci_low, ci_high = bootstrap_mean_ci(diff, rng)

        records.append({
            "feature": feature,
            "comparison": comparison,
            "task_a": task_a,
            "task_b": task_b,
            "n_participants": len(diff),
            "task_a_median": float(np.median(x)),
            "task_b_median": float(np.median(y)),
            "mean_paired_difference_a_minus_b": float(diff.mean()),
            "median_paired_difference_a_minus_b": float(np.median(diff)),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "participants_a_greater": int((diff > 0).sum()),
            "participants_b_greater": int((diff < 0).sum()),
            "participants_equal": int((diff == 0).sum()),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        })

results = pd.DataFrame(records)

# Holm correction across all 18 planned feature/task comparisons.
results["holm_adjusted_p"] = holm_adjust(results["p_value"])
results["significant_holm_005"] = (
    results["holm_adjusted_p"] < 0.05
)

results.to_csv(
    OUT / "v7_quality_task_statistical_tests.csv",
    index=False,
)

print("\n" + "=" * 75)
print("V7 PARTICIPANT-LEVEL QUALITY ANALYSIS")
print("=" * 75)
print("Bootstrap resamples:", N_BOOT)
print("Holm correction: 18 comparisons")
print("Difference is task A minus task B.")

for _, row in results.iterrows():
    print(
        f"\n{row['feature']} | {row['comparison']}\n"
        f"  Participants: {row['n_participants']}\n"
        f"  Task A median: {row['task_a_median']:.6f}\n"
        f"  Task B median: {row['task_b_median']:.6f}\n"
        f"  Mean paired difference: "
        f"{row['mean_paired_difference_a_minus_b']:.6f}\n"
        f"  95% bootstrap CI: "
        f"[{row['ci_95_low']:.6f}, {row['ci_95_high']:.6f}]\n"
        f"  A greater / B greater / equal: "
        f"{row['participants_a_greater']} / "
        f"{row['participants_b_greater']} / "
        f"{row['participants_equal']}\n"
        f"  Wilcoxon p: {row['p_value']:.6g}\n"
        f"  Holm p: {row['holm_adjusted_p']:.6g}\n"
        f"  Significant: {row['significant_holm_005']}"
    )

print("\nSaved results to:", OUT)

