
"""
EDA Quality Analysis - NeuroKit2 Version 2

Analyses:
1. T1/T2/T3 EDA feature distributions.
2. Recording-level signal quality.
3. SCR event detection behaviour.
4. Tonic/phasic feature behaviour.
5. Missing and non-finite values.
6. Participant-level variability.

No recordings are automatically rejected.

Important:
30-second windows with 5-second stride overlap heavily.
Window-level observations are not independent.
This script performs descriptive analysis, not inferential testing.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "results" / "neurokit2_v2" / "fused_features_v2.csv"

OUT = ROOT / "results" / "neurokit2_v2"

OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

required = [
    "participant",
    "task",
    "label",
    "start_s",
    "end_s",
    "eda_mean",
    "eda_std",
    "eda_tonic_mean",
    "eda_tonic_std",
    "eda_tonic_slope",
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_scr_event_amplitude_mean",
    "eda_scr_valid_event_count",
    "eda_scr_onsets_diagnostic",
    "eda_scr_peaks_diagnostic",
    "eda_raw_std",
    "eda_unique_fraction",
    "eda_flat_step_fraction",
    "eda_median_abs_difference",
    "eda_max_abs_jump",
    "eda_jump_ratio",
]

missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")


# --------------------------------------------------
# NORMALIZE TASK LABELS
# --------------------------------------------------

df["task"] = df["task"].astype(str).str.strip()

df["task"] = df["task"].replace({
    "1": "T1",
    "2": "T2",
    "3": "T3",
})

if set(df["task"].unique()) != {"T1", "T2", "T3"}:
    raise ValueError(
        f"Unexpected tasks: {df['task'].unique()}"
    )

df["participant"] = df["participant"].astype(int)

df["label"] = df["label"].astype(int)

assert (df.loc[df["task"] == "T1", "label"] == 0).all()
assert (df.loc[df["task"].isin(["T2", "T3"]), "label"] == 1).all()

print("Loaded windows:", len(df))
print("Participants:", df["participant"].nunique())


# --------------------------------------------------
# FEATURE GROUPS
# --------------------------------------------------

eda_features = [
    "eda_mean",
    "eda_std",
    "eda_median",
    "eda_iqr",
    "eda_range",
    "eda_tonic_mean",
    "eda_tonic_std",
    "eda_tonic_slope",
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_scr_event_amplitude_mean",
    "eda_scr_valid_event_count",
    "eda_scr_onsets_diagnostic",
    "eda_scr_peaks_diagnostic",
]

quality_features = [
    "eda_raw_std",
    "eda_unique_fraction",
    "eda_flat_step_fraction",
    "eda_median_abs_difference",
    "eda_max_abs_jump",
    "eda_jump_ratio",
]

# Replace infinite values for descriptive analysis.
numeric_columns = eda_features + quality_features

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)


# --------------------------------------------------
# 1. DATASET VALIDATION
# --------------------------------------------------

validation = {
    "total_windows": int(len(df)),
    "participants": int(df["participant"].nunique()),
    "tasks": sorted(df["task"].unique().tolist()),
    "windows_per_task": {
        str(k): int(v)
        for k, v in df.groupby("task").size().items()
    },
    "duplicate_windows": int(
        df.duplicated([
            "participant",
            "task",
            "start_s",
            "end_s",
            "label",
        ]).sum()
    ),
    "recordings": int(
        df[["participant", "task"]].drop_duplicates().shape[0]
    ),
}

print("\nDATASET VALIDATION")
print(json.dumps(validation, indent=4))

if validation["total_windows"] != 5208:
    print("WARNING: Unexpected window count.")

if validation["participants"] != 56:
    print("WARNING: Expected 56 participants.")

if validation["duplicate_windows"] != 0:
    raise ValueError("Duplicate windows detected.")


# --------------------------------------------------
# 2. TASK-WISE FEATURE SUMMARY
# --------------------------------------------------

task_summary = (
    df.groupby("task")[eda_features + quality_features]
    .agg(["mean", "median", "std", "min", "max"])
)

task_summary.to_csv(
    OUT / "eda_task_summary_v2.csv"
)

print("\nTASK-WISE FEATURE SUMMARY")
print(
    df.groupby("task")[
        [
            "eda_tonic_mean",
            "eda_tonic_std",
            "eda_phasic_std",
            "eda_scr_event_amplitude_mean",
            "eda_scr_valid_event_count",
            "eda_unique_fraction",
            "eda_flat_step_fraction",
            "eda_jump_ratio",
        ]
    ].median().round(5)
)


# --------------------------------------------------
# 3. RECORDING-LEVEL QUALITY SUMMARY
# --------------------------------------------------

recording_summary = (
    df.groupby(["participant", "task"])
    .agg(
        windows=("start_s", "count"),

        tonic_mean=("eda_tonic_mean", "mean"),
        tonic_std=("eda_tonic_std", "mean"),
        tonic_slope=("eda_tonic_slope", "mean"),

        phasic_mean=("eda_phasic_mean", "mean"),
        phasic_std=("eda_phasic_std", "mean"),

        scr_event_amplitude=("eda_scr_event_amplitude_mean", "mean"),
        scr_valid_event_count=("eda_scr_valid_event_count", "mean"),

        scr_onsets=("eda_scr_onsets_diagnostic", "mean"),
        scr_peaks=("eda_scr_peaks_diagnostic", "mean"),

        raw_std=("eda_raw_std", "mean"),
        unique_fraction=("eda_unique_fraction", "mean"),
        flat_step_fraction=("eda_flat_step_fraction", "mean"),

        median_abs_difference=("eda_median_abs_difference", "mean"),
        max_abs_jump=("eda_max_abs_jump", "max"),
        jump_ratio=("eda_jump_ratio", "median"),
    )
    .reset_index()
)

recording_summary.to_csv(
    OUT / "eda_recording_quality_v2.csv",
    index=False
)


# --------------------------------------------------
# 4. SCR DETECTION ANALYSIS
# --------------------------------------------------

scr_summary = (
    df.groupby("task")
    .agg(
        windows=("start_s", "count"),

        mean_onsets=("eda_scr_onsets_diagnostic", "mean"),
        median_onsets=("eda_scr_onsets_diagnostic", "median"),

        mean_peaks=("eda_scr_peaks_diagnostic", "mean"),
        median_peaks=("eda_scr_peaks_diagnostic", "median"),

        mean_valid_amplitude_count=(
            "eda_scr_valid_event_count", "mean"
        ),

        median_event_amplitude=(
            "eda_scr_event_amplitude_mean", "median"
        ),

        missing_event_amplitude=(
            "eda_scr_event_amplitude_mean",
            lambda x: int(x.isna().sum())
        ),
    )
)

scr_summary.to_csv(
    OUT / "eda_scr_analysis_v2.csv"
)

print("\nSCR TASK SUMMARY")
print(scr_summary.round(5))


# --------------------------------------------------
# 5. MISSING VALUE ANALYSIS
# --------------------------------------------------

missing_summary = pd.DataFrame({
    "feature": numeric_columns,
    "missing_count": [
        int(df[c].isna().sum())
        for c in numeric_columns
    ],
    "missing_percentage": [
        float(df[c].isna().mean() * 100)
        for c in numeric_columns
    ],
})

missing_summary = missing_summary.sort_values(
    "missing_percentage",
    ascending=False
)

missing_summary.to_csv(
    OUT / "eda_missing_values_v2.csv",
    index=False
)


# --------------------------------------------------
# 6. PARTICIPANT-LEVEL TASK COMPARISON
# --------------------------------------------------

participant_task = (
    df.groupby(["participant", "task"])
    .agg(
        tonic_mean=("eda_tonic_mean", "mean"),
        tonic_std=("eda_tonic_std", "mean"),
        phasic_std=("eda_phasic_std", "mean"),
        event_amplitude=(
            "eda_scr_event_amplitude_mean", "mean"
        ),
        event_count=(
            "eda_scr_valid_event_count", "mean"
        ),
        unique_fraction=("eda_unique_fraction", "mean"),
        flat_step_fraction=("eda_flat_step_fraction", "mean"),
        jump_ratio=("eda_jump_ratio", "median"),
    )
    .reset_index()
)

participant_task.to_csv(
    OUT / "eda_participant_task_summary_v2.csv",
    index=False
)


# --------------------------------------------------
# 7. FEATURE CORRELATION
# --------------------------------------------------

correlation_features = [
    "eda_mean",
    "eda_std",
    "eda_tonic_mean",
    "eda_tonic_std",
    "eda_tonic_slope",
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_scr_event_amplitude_mean",
    "eda_scr_valid_event_count",
    "eda_unique_fraction",
    "eda_flat_step_fraction",
    "eda_jump_ratio",
]

correlation = df[correlation_features].corr(
    method="spearman"
)

correlation.to_csv(
    OUT / "eda_feature_correlation_v2.csv"
)


# --------------------------------------------------
# 8. QUALITY RANKING FOR INSPECTION
# --------------------------------------------------

# This is a ranking for manual inspection only.
# It does not label recordings as good or bad.

inspection = recording_summary.copy()

inspection["relative_jump_indicator"] = (
    inspection["jump_ratio"]
)

inspection["low_unique_fraction_indicator"] = (
    1.0 - inspection["unique_fraction"]
)

inspection["high_flat_fraction_indicator"] = (
    inspection["flat_step_fraction"]
)

inspection.sort_values(
    [
        "relative_jump_indicator",
        "low_unique_fraction_indicator",
        "high_flat_fraction_indicator",
    ],
    ascending=False,
    inplace=True
)

inspection.to_csv(
    OUT / "eda_recordings_for_manual_inspection_v2.csv",
    index=False
)


# --------------------------------------------------
# 9. SAVE SUMMARY JSON
# --------------------------------------------------

summary = {
    **validation,

    "eda_features": len(eda_features),

    "quality_features": len(quality_features),

    "missing_values": {
        str(row["feature"]): int(row["missing_count"])
        for _, row in missing_summary.iterrows()
    },

    "task_median_tonic_mean": {
        str(k): float(v)
        for k, v in df.groupby("task")[
            "eda_tonic_mean"
        ].median().items()
    },

    "task_median_phasic_std": {
        str(k): float(v)
        for k, v in df.groupby("task")[
            "eda_phasic_std"
        ].median().items()
    },

    "interpretation": (
        "Descriptive quality analysis only. "
        "No recordings automatically rejected. "
        "Overlapping windows are not independent observations."
    ),
}

with open(
    OUT / "eda_quality_summary_v2.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 65)
print("NEUROKIT2 V2 EDA QUALITY ANALYSIS COMPLETE")
print("=" * 65)

print("\nMissing values:")
print(missing_summary.to_string(index=False))

print("\nTask-wise median features:")
print(
    df.groupby("task")[
        [
            "eda_tonic_mean",
            "eda_tonic_std",
            "eda_phasic_std",
            "eda_scr_event_amplitude_mean",
            "eda_unique_fraction",
            "eda_flat_step_fraction",
        ]
    ].median().round(5)
)

print("\nSaved analysis files:")
for name in [
    "eda_task_summary_v2.csv",
    "eda_recording_quality_v2.csv",
    "eda_scr_analysis_v2.csv",
    "eda_missing_values_v2.csv",
    "eda_participant_task_summary_v2.csv",
    "eda_feature_correlation_v2.csv",
    "eda_recordings_for_manual_inspection_v2.csv",
    "eda_quality_summary_v2.json",
]:
    print(" -", name)

print("\nOutput directory:", OUT)

