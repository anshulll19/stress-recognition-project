#!/usr/bin/env python3

"""
NeuroKit2 V5 — Personalized Rest Calibration LOSO

Protocol:
- 56-participant Leave-One-Subject-Out evaluation.
- Each participant has an independent T1 resting baseline.
- Baseline statistics: median and IQR.
- Training participants are normalized using their own T1 calibration.
- Held-out participant is normalized using their own T1 calibration.
- Held-out calibration data is never used to fit the classifier.

Features:
1. BVP only
2. EDA only
3. BVP + EDA

Models:
1. RBF-SVM
2. Random Forest
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

CALIBRATION_END = 85.0
BOUNDARY_END = 100.0

RANDOM_STATE = 42

OUTPUT = Path(
    "physiological_pipeline/03_NeuroKit_V2/results_loso_v5"
)


# ============================================================
# MODELS
# ============================================================

def make_models():

    return {
        "RBF_SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path):

    df = pd.read_csv(path)

    required = {
        "participant", "task", "start_s", "end_s"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    df["participant"] = (
        df["participant"].astype(str).str.strip()
    )

    for col in ["task", "start_s", "end_s"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=[
            "participant", "task", "start_s", "end_s"
        ]
    )

    df = df[df["task"].isin([1, 2, 3])].copy()

    excluded = {
        "participant", "task", "start_s", "end_s",
        "label", "target", "class", "ground_truth",
    }

    def valid_features(prefix):

        return [
            col for col in df.columns
            if col.startswith(prefix)
            and col not in excluded
            and pd.api.types.is_numeric_dtype(df[col])
            and df[col].notna().any()
            and df[col].nunique(dropna=True) > 1
        ]

    bvp_cols = valid_features("bvp_")
    eda_cols = valid_features("eda_")

    feature_sets = {
        "BVP_only": bvp_cols,
        "EDA_only": eda_cols,
        "BVP_EDA_fusion": bvp_cols + eda_cols,
    }

    # Keep only features with a valid T1 calibration
    # median for EVERY participant. This guarantees that
    # all LOSO folds use the same feature definitions.
    participants = sorted(df["participant"].unique())

    print("\\nCHECKING BASELINE FEATURE COVERAGE")

    for modality, cols in feature_sets.items():

        valid_cols = []
        excluded_cols = []

        for col in cols:
            valid_for_all = True

            for participant in participants:
                baseline_rows = df[
                    (df["participant"] == participant)
                    & (df["task"] == 1)
                    & (df["end_s"] <= CALIBRATION_END)
                ]

                if baseline_rows.empty:
                    valid_for_all = False
                    break

                values = pd.to_numeric(
                    baseline_rows[col], errors="coerce"
                )

                if not np.isfinite(values.to_numpy(dtype=float)).any():
                    valid_for_all = False
                    break

            if valid_for_all:
                valid_cols.append(col)
            else:
                excluded_cols.append(col)

        feature_sets[modality] = valid_cols

        print(
            f"{modality}: retained {len(valid_cols)}, "
            f"excluded {len(excluded_cols)}"
        )

        if excluded_cols:
            print("  Excluded:", excluded_cols)

    for name, cols in feature_sets.items():

        if not cols:
            raise ValueError(
                f"No usable features found for {name}"
            )

    participants = sorted(df["participant"].unique())

    print("\nDATASET INFORMATION")
    print("=" * 65)
    print("Shape:", df.shape)
    print("Participants:", len(participants))
    print("BVP features:", len(bvp_cols))
    print("EDA features:", len(eda_cols))
    print("Task distribution:")
    print(df["task"].value_counts().sort_index())

    return df, feature_sets, participants


# ============================================================
# PERSONAL BASELINE CALIBRATION
# ============================================================

def calculate_baseline(df, participant, features):

    subject = df[
        df["participant"] == participant
    ]

    calibration = subject[
        (subject["task"] == 1)
        & (subject["end_s"] <= CALIBRATION_END)
    ]

    if calibration.empty:
        raise ValueError(
            f"No T1 calibration windows for {participant}"
        )

    X = calibration[features]

    median = X.median()
    q75 = X.quantile(0.75)
    q25 = X.quantile(0.25)

    iqr = q75 - q25

    # Fallback for zero or invalid IQR.
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
            f"Missing baseline medians for {participant}: "
            f"{invalid}"
        )

    return {
        "median": median,
        "scale": scale,
        "calibration_windows": len(calibration),
    }


def normalize_subject(df, participant, features, baseline):

    subject = df[
        df["participant"] == participant
    ].copy()

    # Convert feature columns to float before normalization.
    # Robust normalization produces floating-point values,
    # even when original features are integer-valued.
    X = subject[features].astype(float)

    normalized = (
        X - baseline["median"]
    ) / baseline["scale"]

    subject[features] = normalized.astype(float)

    return subject


# ============================================================
# LOSO SPLIT
# ============================================================

def prepare_fold(df, held_out, features):

    normalized_subjects = []

    calibration_counts = {}

    # Every participant is normalized using their own
    # T1 calibration windows, never another participant's.

    for participant in sorted(df["participant"].unique()):

        baseline = calculate_baseline(
            df, participant, features
        )

        normalized = normalize_subject(
            df, participant, features, baseline
        )

        normalized_subjects.append(normalized)

        calibration_counts[participant] = (
            baseline["calibration_windows"]
        )

    normalized_df = pd.concat(
        normalized_subjects,
        ignore_index=True
    )

    train_subjects = normalized_df[
        normalized_df["participant"] != held_out
    ]

    test_subject = normalized_df[
        normalized_df["participant"] == held_out
    ]

    # Training rest: each training participant's own T1 baseline.
    train_rest = train_subjects[
        (train_subjects["task"] == 1)
        & (train_subjects["end_s"] <= CALIBRATION_END)
    ]

    train_distress = train_subjects[
        train_subjects["task"].isin([2, 3])
    ]

    # Held-out rest: strictly after the boundary gap.
    test_rest = test_subject[
        (test_subject["task"] == 1)
        & (test_subject["start_s"] >= BOUNDARY_END)
    ]

    test_distress = test_subject[
        test_subject["task"].isin([2, 3])
    ]

    train = pd.concat(
        [train_rest, train_distress],
        ignore_index=True
    )

    test = pd.concat(
        [test_rest, test_distress],
        ignore_index=True
    )

    X_train = train[features]
    y_train = np.concatenate([
        np.zeros(len(train_rest), dtype=int),
        np.ones(len(train_distress), dtype=int),
    ])

    X_test = test[features]
    y_test = np.concatenate([
        np.zeros(len(test_rest), dtype=int),
        np.ones(len(test_distress), dtype=int),
    ])

    return (
        X_train, y_train,
        X_test, y_test,
        len(train_rest), len(train_distress),
        len(test_rest), len(test_distress),
        calibration_counts[held_out],
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "rest_specificity": (
            tn / (tn + fp) if tn + fp else 0
        ),
        "distress_sensitivity": (
            tp / (tp + fn) if tp + fn else 0
        ),
    }


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(df, feature_sets, participants):

    results = []
    predictions = []
    audit = []

    total = len(participants)

    for modality, features in feature_sets.items():

        print("\n" + "=" * 75)
        print("MODALITY:", modality)
        print("FEATURE COUNT:", len(features))
        print("=" * 75)

        for model_name in make_models():

            print(f"\nMODEL: {model_name}")

            for fold, held_out in enumerate(
                participants, start=1
            ):

                (
                    X_train, y_train,
                    X_test, y_test,
                    train_rest, train_distress,
                    test_rest, test_distress,
                    heldout_calibration,
                ) = prepare_fold(
                    df, held_out, features
                )

                if len(np.unique(y_train)) != 2:
                    print(
                        f"Skipping {held_out}: "
                        "one training class"
                    )
                    continue

                if len(np.unique(y_test)) != 2:
                    print(
                        f"Skipping {held_out}: "
                        "one test class"
                    )
                    continue

                model = make_models()[model_name]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                score = calculate_metrics(
                    y_test, y_pred
                )

                results.append({
                    "modality": modality,
                    "model": model_name,
                    "participant": held_out,
                    "feature_count": len(features),
                    "heldout_calibration_windows": (
                        heldout_calibration
                    ),
                    "train_rest": train_rest,
                    "train_distress": train_distress,
                    "test_rest": test_rest,
                    "test_distress": test_distress,
                    **score,
                })

                for true, pred in zip(y_test, y_pred):

                    predictions.append({
                        "modality": modality,
                        "model": model_name,
                        "participant": held_out,
                        "true_label": int(true),
                        "predicted_label": int(pred),
                    })

                audit.append({
                    "participant": held_out,
                    "modality": modality,
                    "calibration_windows": heldout_calibration,
                })

                print(
                    f"[{fold:02d}/{total}] {held_out} | "
                    f"BalAcc={score['balanced_accuracy']:.4f} | "
                    f"MacroF1={score['macro_f1']:.4f}"
                )

    return (
        pd.DataFrame(results),
        pd.DataFrame(predictions),
        pd.DataFrame(audit),
    )


# ============================================================
# REPORTING
# ============================================================

def save_results(results, predictions, audit):

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if results.empty:
        raise RuntimeError("No successful experiments.")

    results.to_csv(
        OUTPUT / "v5_per_participant.csv",
        index=False
    )

    predictions.to_csv(
        OUTPUT / "v5_predictions.csv",
        index=False
    )

    audit.drop_duplicates().to_csv(
        OUTPUT / "v5_calibration_audit.csv",
        index=False
    )

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "macro_f1",
        "weighted_f1",
        "rest_specificity",
        "distress_sensitivity",
    ]

    summary = results.groupby(
        ["modality", "model"]
    )[metric_cols].agg(["mean", "std"])

    summary.to_csv(
        OUTPUT / "v5_summary.csv"
    )

    pooled = []

    for (modality, model), group in predictions.groupby(
        ["modality", "model"]
    ):

        score = calculate_metrics(
            group["true_label"],
            group["predicted_label"],
        )

        pooled.append({
            "modality": modality,
            "model": model,
            **score,
            "total_windows": len(group),
        })

    pooled_df = pd.DataFrame(pooled)

    pooled_df.to_csv(
        OUTPUT / "v5_pooled_metrics.csv",
        index=False
    )

    report = {
        "experiment": "NeuroKit2 V5 Personalized Calibration",
        "normalization": "Per-participant T1 median/IQR",
        "calibration_end_seconds": CALIBRATION_END,
        "boundary_exclusion": [85, 100],
        "rest_test_start_seconds": BOUNDARY_END,
        "distress_tasks": [2, 3],
        "participants_evaluated": int(
            results["participant"].nunique()
        ),
        "mean_metrics": (
            results.groupby(["modality", "model"])[metric_cols]
            .mean()
            .to_dict(orient="index")
        ),
        "pooled_metrics": pooled_df.to_dict(
            orient="records"
        ),
    }

    with open(
        OUTPUT / "v5_report.json", "w"
    ) as f:
        def make_json_safe(obj):
            if isinstance(obj, dict):
                return {
                    str(k) if isinstance(k, tuple) else k: make_json_safe(v)
                    for k, v in obj.items()
                }

            if isinstance(obj, (list, tuple)):
                return [make_json_safe(v) for v in obj]

            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, np.generic):
                return obj.item()

            return obj

        json.dump(make_json_safe(report), f, indent=4)

    print("\n" + "=" * 75)
    print("V5 FINAL RESULTS — MEAN AND STANDARD DEVIATION")
    print("=" * 75)

    print(summary.round(4).to_string())

    print("\nPOOLED RESULTS:")
    print(pooled_df.round(4).to_string(index=False))

    print("\nResults saved to:", OUTPUT.resolve())


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NeuroKit2 V5 Personalized Calibration LOSO"
    )

    parser.add_argument(
        "--input",
        default="results/neurokit2_v2/fused_features_v2.csv",
    )

    args = parser.parse_args()

    df, feature_sets, participants = load_dataset(
        args.input
    )

    results, predictions, audit = run_experiment(
        df, feature_sets, participants
    )

    save_results(results, predictions, audit)


if __name__ == "__main__":
    main()
