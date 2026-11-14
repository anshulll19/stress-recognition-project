
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent
INPUT = BASE / "results_loso_v6" / "v6_per_participant.csv"
OUT = BASE / "results_v6_ablation_comparison"
OUT.mkdir(exist_ok=True)

N_BOOT = 10_000
RANDOM_STATE = 42

# Positive difference means the first feature group performs better.
COMPARISONS = [
    ("EDA_full_19", "EDA_physiological_SCR"),
    ("EDA_physiological_SCR", "EDA_quality_diagnostics"),
    ("EDA_physiological_SCR", "BVP_EDA_no_quality"),
    ("EDA_full_19", "EDA_quality_diagnostics"),
]

METRICS = [
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "rest_specificity",
    "distress_sensitivity",
]

MODELS = ["RBF_SVM", "RandomForest"]


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def bootstrap_ci(differences, rng):
    """Participant-level percentile bootstrap CI for mean paired difference."""
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

required = {
    "participant", "model", "modality", *METRICS
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df.duplicated(["participant", "model", "modality"]).any():
    raise ValueError("Duplicate participant/model/modality rows found.")

expected_modalities = {
    "EDA_full_19",
    "EDA_physiological_SCR",
    "EDA_quality_diagnostics",
    "BVP_EDA_no_quality",
}

if set(df["modality"].unique()) != expected_modalities:
    raise ValueError("Unexpected V6 modality names.")

rng = np.random.default_rng(RANDOM_STATE)
records = []
paired_records = []

for model in MODELS:
    model_df = df[df["model"] == model]

    for first, second in COMPARISONS:
        a = model_df[model_df["modality"] == first].set_index("participant")
        b = model_df[model_df["modality"] == second].set_index("participant")

        common = sorted(set(a.index) & set(b.index))

        if len(common) != 56:
            raise ValueError(
                f"Expected 56 matched participants for "
                f"{model}: {first} vs {second}; got {len(common)}"
            )

        a = a.loc[common]
        b = b.loc[common]

        for metric in METRICS:
            x = a[metric].to_numpy(dtype=float)
            y = b[metric].to_numpy(dtype=float)
            diff = x - y

            if not np.isfinite(diff).all():
                raise ValueError(
                    f"Non-finite values: {model}, {metric}, {first} vs {second}"
                )

            if np.allclose(diff, 0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    x, y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            ci_low, ci_high = bootstrap_ci(diff, rng)

            records.append({
                "model": model,
                "comparison": f"{first} vs {second}",
                "first_group": first,
                "second_group": second,
                "metric": metric,
                "n_participants": len(common),
                "first_mean": float(x.mean()),
                "second_mean": float(y.mean()),
                "mean_difference": float(diff.mean()),
                "difference_pp": float(diff.mean() * 100),
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "ci_95_low_pp": ci_low * 100,
                "ci_95_high_pp": ci_high * 100,
                "improved": int((diff > 0).sum()),
                "worsened": int((diff < 0).sum()),
                "unchanged": int((diff == 0).sum()),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            })

            for participant, av, bv, dv in zip(common, x, y, diff):
                paired_records.append({
                    "model": model,
                    "comparison": f"{first} vs {second}",
                    "metric": metric,
                    "participant": participant,
                    "first_value": av,
                    "second_value": bv,
                    "difference": dv,
                })

results = pd.DataFrame(records)

# Holm correction across all 8 planned comparisons within each metric.
results["holm_adjusted_p"] = np.nan

for metric in METRICS:
    mask = results["metric"] == metric
    results.loc[mask, "holm_adjusted_p"] = holm_adjust(
        results.loc[mask, "p_value"].to_numpy()
    )

results["significant_holm_005"] = results["holm_adjusted_p"] < 0.05

results.to_csv(OUT / "v6_paired_statistical_tests.csv", index=False)
pd.DataFrame(paired_records).to_csv(
    OUT / "v6_paired_participant_differences.csv",
    index=False,
)

print("\n" + "=" * 75)
print("V6 PAIRED STATISTICAL COMPARISON")
print("=" * 75)
print("Participants per comparison: 56")
print("Bootstrap resamples:", N_BOOT)
print("Holm correction: 8 comparisons per metric")
print("Positive difference favours the first feature group.")

for metric in ["balanced_accuracy", "macro_f1"]:
    print(f"\n{'-' * 75}")
    print(metric.upper())
    print("-" * 75)

    subset = results[results["metric"] == metric]

    for _, row in subset.iterrows():
        print(
            f"\n{row['model']} | {row['comparison']}\n"
            f"  Difference: {row['difference_pp']:+.2f} pp\n"
            f"  95% bootstrap CI: "
            f"[{row['ci_95_low_pp']:+.2f}, "
            f"{row['ci_95_high_pp']:+.2f}] pp\n"
            f"  Improved / worsened / unchanged: "
            f"{row['improved']} / {row['worsened']} / "
            f"{row['unchanged']}\n"
            f"  Wilcoxon p: {row['p_value']:.6f}\n"
            f"  Holm-adjusted p: {row['holm_adjusted_p']:.6f}\n"
            f"  Significant: {row['significant_holm_005']}"
        )

print("\nSaved results to:", OUT)


