#!/usr/bin/env python3

"""
UBFC Phys V5 EDA-only deployment candidate.

Training:
- 56 participants
- Participant-specific T1 median/IQR normalization
- T1 windows ending <= 85 seconds: rest (0)
- T2/T3 windows: distress (1)
- 19 EDA features
- Median imputation + StandardScaler + RBF-SVM

This is a calibrated deployment candidate, not a new
independent generalization evaluation.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = Path("results/neurokit2_v2/fused_features_v2.csv")

MODEL_DIR = Path("models/ubfc_deployment")
MODEL_PATH = MODEL_DIR / "ubfc_phys_eda_svm_v5.joblib"
MANIFEST_PATH = MODEL_DIR / "eda_svm_v5_manifest.json"

CALIBRATION_END = 85.0
BOUNDARY_START = 85.0
BOUNDARY_END = 100.0
RANDOM_STATE = 42

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


def calculate_baseline(subject_df):
    calibration = subject_df[
        (subject_df["task"] == 1)
        & (subject_df["end_s"] <= CALIBRATION_END)
    ]

    if calibration.empty:
        raise ValueError("No valid T1 calibration windows.")

    X = calibration[FEATURES].astype(float)

    median = X.median()
    iqr = X.quantile(0.75) - X.quantile(0.25)
    std = X.std(ddof=0)

    scale = iqr.where(
        np.isfinite(iqr) & (iqr > 1e-10),
        std
    )

    scale = scale.where(
        np.isfinite(scale) & (scale > 1e-10),
        1.0
    )

    if median.isna().any():
        invalid = median[median.isna()].index.tolist()
        raise ValueError(
            f"Invalid calibration medians: {invalid}"
        )

    return median, scale, len(calibration)


def main():
    print("=" * 65)
    print("UBFC PHYS — V5 EDA SVM DEPLOYMENT TRAINING")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df = df[df["task"].isin([1, 2, 3])].copy()

    required = [
        "participant", "task", "start_s", "end_s", *FEATURES
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    participants = sorted(df["participant"].unique())

    if len(participants) != 56:
        raise ValueError(
            f"Expected 56 participants, found {len(participants)}"
        )

    normalized_parts = []
    calibration_audit = []

    print("\nNormalizing participants...")

    for participant in participants:
        subject = df[
            df["participant"] == participant
        ].copy()

        median, scale, count = calculate_baseline(subject)

        subject[FEATURES] = (
            subject[FEATURES].astype(float) - median
        ) / scale

        normalized_parts.append(subject)

        calibration_audit.append({
            "participant": str(participant),
            "calibration_windows": count,
        })

    normalized = pd.concat(
        normalized_parts,
        ignore_index=True
    )

    # Match V5 training selection:
    # calibration rest + all distress task windows.
    rest = normalized[
        (normalized["task"] == 1)
        & (normalized["end_s"] <= CALIBRATION_END)
    ]

    distress = normalized[
        normalized["task"].isin([2, 3])
    ]

    train = pd.concat([rest, distress], ignore_index=True)

    X_train = train[FEATURES].astype(float)
    y_train = np.concatenate([
        np.zeros(len(rest), dtype=int),
        np.ones(len(distress), dtype=int),
    ])

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training data must contain both classes.")

    print("\nTRAINING DATA")
    print("-" * 65)
    print("Participants:", len(participants))
    print("Features:", len(FEATURES))
    print("Rest windows:", len(rest))
    print("Distress windows:", len(distress))
    print("Total training windows:", len(train))
    print("Calibration windows per participant:")
    print(pd.Series(
        [x["calibration_windows"] for x in calibration_audit]
    ).describe())

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    print("\nFitting SVM...")
    model.fit(X_train, y_train)

    # Basic training sanity check only; this is not validation.
    predictions = model.predict(X_train)

    print("Training predictions:", len(predictions))
    print("Predicted rest:", int((predictions == 0).sum()))
    print("Predicted distress:", int((predictions == 1).sum()))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "features": FEATURES,
        "model_name": "V5_EDA_RBF_SVM",
        "calibration_end_seconds": CALIBRATION_END,
        "boundary_exclusion_seconds": [
            BOUNDARY_START, BOUNDARY_END
        ],
        "normalization": "participant-specific T1 median/IQR",
        "calibration_audit": calibration_audit,
    }

    joblib.dump(artifact, MODEL_PATH)

    manifest = {
        "model_name": "UBFC Phys V5 EDA-only RBF-SVM",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATA_PATH),
        "participants": len(participants),
        "dataset_windows": len(df),
        "training_windows": len(train),
        "rest_training_windows": len(rest),
        "distress_training_windows": len(distress),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "classifier": {
            "type": "sklearn.svm.SVC",
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "pipeline": [
            "SimpleImputer(strategy=median)",
            "StandardScaler",
            "RBF-SVM",
        ],
        "calibration": {
            "modality": "EDA",
            "task": 1,
            "window_end_max_seconds": CALIBRATION_END,
            "statistics": ["median", "IQR"],
            "fallback": "standard deviation, then 1.0",
            "participant_specific": True,
        },
        "label_mapping": {
            "0": "rest",
            "1": "distress",
        },
        "training_protocol": (
            "All participants' own T1 calibration windows "
            "and T2/T3 distress windows"
        ),
        "evaluation_note": (
            "Final model fitted on all 56 participants. "
            "Training metrics are not independent validation. "
            "V5 LOSO results are reported separately."
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2)
    )

    print("\nSAVED")
    print("Model:", MODEL_PATH)
    print("Manifest:", MANIFEST_PATH)
    print("\nExisting Random Forest models were not modified.")


if __name__ == "__main__":
    main()
