import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

INPUT = Path("results/neurokit2_v2/fused_features_v2.csv")
MODEL_DIR = Path("models/ubfc_deployment")

MODEL_PATH = MODEL_DIR / "ubfc_phys_bvp_eda_rf_v3.joblib"
MANIFEST_PATH = MODEL_DIR / "bvp_eda_rf_v3_manifest.json"

BVP_FEATURES = [
    "bvp_mean", "bvp_std", "bvp_median", "bvp_iqr",
    "bvp_slope_per_sample", "bvp_range", "bvp_peak_count",
    "bvp_heart_rate_mean", "bvp_heart_rate_std",
    "bvp_ibi_mean", "bvp_ibi_std", "bvp_peak_quality",
]

EDA_FEATURES = [
    "eda_mean", "eda_std", "eda_median", "eda_iqr",
    "eda_range", "eda_tonic_mean", "eda_tonic_std",
    "eda_tonic_slope", "eda_phasic_mean", "eda_phasic_std",
]

FEATURES = BVP_FEATURES + EDA_FEATURES
CALIBRATION_END = 85
REST_TEST_START = 100


def normalize_participant(frame):
    frame = frame.copy()
    frame[FEATURES] = frame[FEATURES].astype(float)

    baseline = frame[
        (frame["task"] == 1) &
        (frame["end_s"] <= CALIBRATION_END)
    ]

    if len(baseline) < 2:
        raise ValueError("Insufficient T1 calibration windows.")

    medians = baseline[FEATURES].mean()
    scales = baseline[FEATURES].std().fillna(0)
    scales = scales.mask(scales < 1e-8, 1.0)

    frame[FEATURES] = (frame[FEATURES] - medians) / scales

    return frame, medians, scales


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "participant", "task", "label", "start_s", "end_s", *FEATURES
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["participant"] = df["participant"].astype(str)
    df["task"] = pd.to_numeric(df["task"]).astype(int)
    df["label"] = pd.to_numeric(df["label"]).astype(int)
    df["start_s"] = pd.to_numeric(df["start_s"])
    df["end_s"] = pd.to_numeric(df["end_s"])

    normalized_parts = []
    calibration_audit = {}

    for participant, group in df.groupby("participant"):
        normalized, medians, scales = normalize_participant(group)
        normalized_parts.append(normalized)

        calibration_audit[participant] = {
            "calibration_windows": int(
                (
                    (group["task"] == 1) &
                    (group["end_s"] <= CALIBRATION_END)
                ).sum()
            ),
            "feature_count": len(FEATURES),
            "scale_fallback_count": int((scales == 1.0).sum()),
        }

    normalized_df = pd.concat(normalized_parts, ignore_index=True)

    # Match corrected LOSO training:
    # use all normalized windows from training participants.
    # Each participant's T1 baseline is used for normalization.
    train_df = normalized_df.copy()

    X = train_df[FEATURES]
    y = train_df["label"]

    if set(y.unique()) != {0, 1}:
        raise ValueError("Training data must contain both classes.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "features": FEATURES,
        "model_name": "BVP_EDA_RandomForest_V3",
        "calibration_end_seconds": CALIBRATION_END,
        "rest_training_start_seconds": REST_TEST_START,
        "normalization": "participant T1 baseline mean/std",
        "training_participants": sorted(calibration_audit),
        "calibration_audit": calibration_audit,
        "training_windows": int(len(train_df)),
        "training_class_counts": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "training_note": (
            "Refit on all 56 UBFC-Phys participants after participant-specific "
            "T1 baseline normalization. LOSO metrics are reported separately; "
            "these training samples are not an independent test set."
        ),
    }

    joblib.dump(artifact, MODEL_PATH)

    manifest = {
        "model_name": artifact["model_name"],
        "dataset": "UBFC-Phys",
        "model_file": str(MODEL_PATH),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "n_participants": len(calibration_audit),
        "calibration_end_seconds": CALIBRATION_END,
        "rest_training_start_seconds": REST_TEST_START,
        "normalization": artifact["normalization"],
        "training_windows": int(len(train_df)),
        "training_class_counts": artifact["training_class_counts"],
        "loso_evaluation": (
            "results/neurokit2_v3_full_loso/summary.json"
        ),
        "metrics_warning": (
            "LOSO metrics are held-out participant evaluation results. "
            "The final artifact was refit on all participants."
        ),
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print("\nMODEL TRAINING COMPLETE")
    print("Participants:", len(calibration_audit))
    print("Training windows:", len(train_df))
    print("Class counts:", artifact["training_class_counts"])
    print("Model:", MODEL_PATH)
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
