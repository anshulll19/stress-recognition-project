"""Build a V5-compatible training reference for V6 domain monitoring."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "results/neurokit2_v2/fused_features_v2.csv"
OUTPUT_PATH = (
    ROOT / "physiological_pipeline/06_Defensive_Inference/"
    "v5_training_reference.json"
)

CALIBRATION_END = 85.0

FEATURES = [
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


def calculate_baseline(subject):
    calibration = subject[
        (subject["task"] == 1)
        & (subject["end_s"] <= CALIBRATION_END)
    ]

    if calibration.empty:
        raise ValueError("Participant has no calibration windows.")

    x = calibration[FEATURES].astype(float)

    median = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    std = x.std(ddof=0)

    scale = iqr.where(
        np.isfinite(iqr) & (iqr > 1e-10), std
    )
    scale = scale.where(
        np.isfinite(scale) & (scale > 1e-10), 1.0
    )

    if not np.isfinite(median.to_numpy()).all():
        raise ValueError("Invalid participant calibration median.")

    return median, scale, len(calibration)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df = df[df["task"].isin([1, 2, 3])].copy()

    required = ["participant", "task", "end_s", *FEATURES]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    participants = sorted(df["participant"].unique())

    if len(participants) != 56:
        raise ValueError(f"Expected 56 participants, found {len(participants)}")

    normalized_parts = []
    calibration_counts = []

    for participant in participants:
        subject = df[df["participant"] == participant].copy()

        median, scale, count = calculate_baseline(subject)

        subject[FEATURES] = (
            subject[FEATURES].astype(float) - median
        ) / scale

        normalized_parts.append(subject)
        calibration_counts.append(count)

    normalized = pd.concat(normalized_parts, ignore_index=True)

    rest = normalized[
        (normalized["task"] == 1)
        & (normalized["end_s"] <= CALIBRATION_END)
    ]

    distress = normalized[normalized["task"].isin([2, 3])]

    training = pd.concat([rest, distress], ignore_index=True)
    x = training[FEATURES].astype(float)

    if len(rest) != 672 or len(distress) != 3472:
        raise ValueError(
            f"Unexpected training counts: rest={len(rest)}, "
            f"distress={len(distress)}"
        )

    feature_reference = {}

    for feature in FEATURES:
        values = x[feature].replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()

        feature_reference[feature] = {
            "count_finite": int(len(finite)),
            "count_missing_or_nonfinite": int(len(values) - len(finite)),
            "median": float(finite.median()) if len(finite) else None,
            "std": float(finite.std(ddof=0)) if len(finite) else None,
            "quantiles": {
                str(q): float(finite.quantile(q))
                for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            } if len(finite) else None,
        }

    reference = {
        "reference_name": "UBFC V5 normalized training feature reference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(DATA_PATH.relative_to(ROOT)),
        "source_model": "UBFC Phys V5 EDA-only RBF-SVM",
        "reference_type": "training_distribution_not_independent_validation",
        "participants": len(participants),
        "rest_training_windows": len(rest),
        "distress_training_windows": len(distress),
        "total_training_windows": len(training),
        "normalization": "participant-specific T1 median/IQR",
        "calibration_end_seconds": CALIBRATION_END,
        "features": FEATURES,
        "calibration_window_counts": calibration_counts,
        "feature_reference": feature_reference,
        "limitations": [
            "Reference is derived from V5 training participants.",
            "Reference quantiles are not validated rejection thresholds.",
            "Distribution shift does not automatically imply invalid signal.",
            "Missing values are summarized separately.",
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(reference, indent=2))

    print("V6 TRAINING REFERENCE CREATED")
    print("Participants:", len(participants))
    print("Rest windows:", len(rest))
    print("Distress windows:", len(distress))
    print("Total windows:", len(training))
    print("Features:", len(FEATURES))
    print("Output:", OUTPUT_PATH)
    print("Size (KB):", round(OUTPUT_PATH.stat().st_size / 1024, 2))


if __name__ == "__main__":
    main()
