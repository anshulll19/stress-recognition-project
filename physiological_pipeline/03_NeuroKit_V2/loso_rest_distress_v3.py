#!/usr/bin/env python3

"""
NeuroKit2 V2 — LOSO Evaluation V3
Rest vs Distress Classification

Experimental protocol:
- Training participants: T1 calibration (rest) + T2/T3 (distress)
- Held-out participant:
    * T1 windows ending <= 85s: calibration, excluded from testing
    * T1 windows from 85s to 100s: excluded boundary
    * T1 windows starting >= 100s: rest testing
    * T2/T3: distress testing

Models:
1. Random Forest
2. RBF-SVM
3. ExtraTrees

One row represents one physiological feature window.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

CALIBRATION_END = 85.0
BOUNDARY_END = 100.0
WINDOW_DURATION = 30.0

RANDOM_STATE = 42

OUTPUT_DIR = Path(
    "physiological_pipeline/03_NeuroKit_V2/results_loso_v3"
)


# ============================================================
# COLUMN DETECTION
# ============================================================

COLUMN_ALIASES = {
    "subject": [
        "subject_id", "participant_id", "participant",
        "subject", "user_id", "sid"
    ],
    "task": [
        "task", "task_id", "condition", "protocol"
    ],
    "start": [
        "window_start", "start_time", "window_start_sec", "start_s",
        "start_seconds", "start"
    ],
    "end": [
        "window_end", "end_time", "window_end_sec", "end_s",
        "end_seconds", "end"
    ],
}


def detect_column(df, column_type):
    for candidate in COLUMN_ALIASES[column_type]:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not detect {column_type} column. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_task(value):
    value = str(value).strip().upper()

    mapping = {
        "1": "T1",
        "2": "T2",
        "3": "T3",
        "1.0": "T1",
        "2.0": "T2",
        "3.0": "T3",
    }

    if value in mapping:
        return mapping[value]

    if value.startswith("T1"):
        return "T1"
    if value.startswith("T2"):
        return "T2"
    if value.startswith("T3"):
        return "T3"

    return value


# ============================================================
# DATA LOADING
# ============================================================

def load_data(csv_path):
    print("\nLoading dataset...")

    df = pd.read_csv(csv_path)

    subject_col = detect_column(df, "subject")
    task_col = detect_column(df, "task")
    start_col = detect_column(df, "start")

    try:
        end_col = detect_column(df, "end")
    except ValueError:
        end_col = None

    df = df.copy()

    df["_subject"] = df[subject_col].astype(str).str.strip()
    df["_task"] = df[task_col].apply(normalize_task)

    df["_start"] = pd.to_numeric(
        df[start_col], errors="coerce"
    )

    if end_col is not None:
        df["_end"] = pd.to_numeric(
            df[end_col], errors="coerce"
        )
    else:
        df["_end"] = df["_start"] + WINDOW_DURATION

    df = df.dropna(
        subset=["_subject", "_task", "_start", "_end"]
    )

    # Restrict to the three experimental tasks.
    df = df[df["_task"].isin(["T1", "T2", "T3"])].copy()

    # Identify numeric physiological features.
    metadata = {
        subject_col, task_col, start_col,
        "_subject", "_task", "_start", "_end",
        "label", "target", "class", "ground_truth",
    }

    if end_col is not None:
        metadata.add(end_col)

    feature_cols = [
        col for col in df.columns
        if col not in metadata
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric physiological features detected.")

    # Remove features that contain no usable information.
    feature_cols = [
        col for col in feature_cols
        if df[col].notna().any()
        and df[col].nunique(dropna=True) > 1
    ]

    if not feature_cols:
        raise ValueError("No usable physiological features remain.")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Exclude rows where every physiological feature is missing.
    df = df.dropna(subset=feature_cols, how="all")

    print(f"Dataset shape: {df.shape}")
    print(f"Participants: {df['_subject'].nunique()}")
    print(f"Tasks: {sorted(df['_task'].unique())}")
    print(f"Features: {len(feature_cols)}")

    print("\nTask distribution:")
    print(df["_task"].value_counts())

    return df, feature_cols


# ============================================================
# WINDOW SELECTION
# ============================================================

def get_training_data(df, held_out_subject, feature_cols):

    train = df[df["_subject"] != held_out_subject].copy()

    # Each training participant contributes:
    # T1 calibration windows = REST (0)
    # T2/T3 windows = DISTRESS (1)

    calibration = train[
        (train["_task"] == "T1")
        & (train["_end"] <= CALIBRATION_END)
    ].copy()

    distress = train[
        train["_task"].isin(["T2", "T3"])
    ].copy()

    calibration["_label"] = 0
    distress["_label"] = 1

    training = pd.concat(
        [calibration, distress],
        ignore_index=True
    )

    X = training[feature_cols].copy()
    y = training["_label"].astype(int)

    return X, y, calibration, distress


def get_test_data(df, held_out_subject, feature_cols):

    test = df[
        df["_subject"] == held_out_subject
    ].copy()

    # Early T1 calibration is NOT used for testing.
    # The boundary interval is also excluded.

    rest = test[
        (test["_task"] == "T1")
        & (test["_start"] >= BOUNDARY_END)
    ].copy()

    distress = test[
        test["_task"].isin(["T2", "T3"])
    ].copy()

    rest["_label"] = 0
    distress["_label"] = 1

    testing = pd.concat(
        [rest, distress],
        ignore_index=True
    )

    if testing.empty:
        return None, None, rest, distress

    X = testing[feature_cols].copy()
    y = testing["_label"].astype(int)

    return X, y, rest, distress


# ============================================================
# MODELS
# ============================================================

def get_models():

    return {
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

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

        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

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
    }


# ============================================================
# LOSO EVALUATION
# ============================================================

def run_loso(df, feature_cols):

    participants = sorted(df["_subject"].unique())

    all_results = []
    all_predictions = []

    models = get_models()

    print("\n" + "=" * 70)
    print("NEUROKIT2 V2 — LOSO REST VS DISTRESS V3")
    print("=" * 70)

    print(f"Participants: {len(participants)}")
    print(f"Models: {list(models.keys())}")

    for fold, held_out in enumerate(participants, start=1):

        print(
            f"\n[{fold}/{len(participants)}] "
            f"Held-out participant: {held_out}"
        )

        X_train, y_train, train_cal, train_dist = (
            get_training_data(df, held_out, feature_cols)
        )

        X_test, y_test, test_rest, test_dist = (
            get_test_data(df, held_out, feature_cols)
        )

        print(
            f"Training: {len(X_train)} windows "
            f"(rest={len(train_cal)}, distress={len(train_dist)})"
        )

        print(
            f"Testing: {0 if y_test is None else len(y_test)} windows "
            f"(rest={len(test_rest)}, distress={len(test_dist)})"
        )

        if X_test is None or y_test is None:
            print("Skipping: no valid test windows.")
            continue

        if y_train.nunique() < 2:
            print("Skipping: training contains only one class.")
            continue

        if y_test.nunique() < 2:
            print("Skipping: test contains only one class.")
            continue

        for model_name, model_template in models.items():

            model = clone(model_template)

            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                metrics = calculate_metrics(
                    y_test, predictions
                )

                cm = confusion_matrix(
                    y_test,
                    predictions,
                    labels=[0, 1]
                )

                result = {
                    "participant": held_out,
                    "model": model_name,
                    "train_windows": len(X_train),
                    "train_rest": len(train_cal),
                    "train_distress": len(train_dist),
                    "test_windows": len(X_test),
                    "test_rest": len(test_rest),
                    "test_distress": len(test_dist),
                    **metrics,
                    "TN": int(cm[0, 0]),
                    "FP": int(cm[0, 1]),
                    "FN": int(cm[1, 0]),
                    "TP": int(cm[1, 1]),
                }

                all_results.append(result)

                for index, (true_label, pred_label) in enumerate(
                    zip(y_test, predictions)
                ):
                    all_predictions.append({
                        "participant": held_out,
                        "model": model_name,
                        "true_label": int(true_label),
                        "predicted_label": int(pred_label),
                        "task": (
                            "REST"
                            if true_label == 0
                            else "DISTRESS"
                        ),
                    })

                print(
                    f"  {model_name:15s} | "
                    f"Acc={metrics['accuracy']:.4f} | "
                    f"BalAcc={metrics['balanced_accuracy']:.4f} | "
                    f"Macro-F1={metrics['macro_f1']:.4f}"
                )

            except Exception as error:
                print(f"  ERROR in {model_name}: {error}")

    return pd.DataFrame(all_results), pd.DataFrame(all_predictions)


# ============================================================
# AGGREGATION AND REPORTING
# ============================================================

def save_results(results, predictions):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if results.empty:
        raise RuntimeError(
            "No successful LOSO evaluations. Check the data and splits."
        )

    results.to_csv(
        OUTPUT_DIR / "loso_v3_per_participant.csv",
        index=False
    )

    predictions.to_csv(
        OUTPUT_DIR / "loso_v3_predictions.csv",
        index=False
    )

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "macro_f1",
        "weighted_f1",
    ]

    summary = results.groupby("model")[metric_cols].agg(
        ["mean", "std"]
    )

    summary.to_csv(
        OUTPUT_DIR / "loso_v3_summary.csv"
    )

    # Pooled metrics across all held-out participants.
    pooled = []

    for model_name in results["model"].unique():

        subset = predictions[
            predictions["model"] == model_name
        ]

        if subset.empty:
            continue

        y_true = subset["true_label"]
        y_pred = subset["predicted_label"]

        pooled.append({
            "model": model_name,
            **calculate_metrics(y_true, y_pred),
            "total_windows": len(subset),
        })

    pooled_df = pd.DataFrame(pooled)

    pooled_df.to_csv(
        OUTPUT_DIR / "loso_v3_pooled_metrics.csv",
        index=False
    )

    # Save a machine-readable report.
    report = {
        "experiment": "NeuroKit2 V2 LOSO Rest vs Distress V3",
        "calibration_end_seconds": CALIBRATION_END,
        "boundary_exclusion": [
            CALIBRATION_END, BOUNDARY_END
        ],
        "rest_test_start_seconds": BOUNDARY_END,
        "window_duration_seconds": WINDOW_DURATION,
        "participants_evaluated": int(
            results["participant"].nunique()
        ),
        "models": list(results["model"].unique()),
        "mean_metrics": (
            results.groupby("model")[metric_cols]
            .mean()
            .to_dict(orient="index")
        ),
        "pooled_metrics": pooled_df.to_dict(orient="records"),
    }

    with open(
        OUTPUT_DIR / "loso_v3_report.json",
        "w"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 70)
    print("FINAL LOSO RESULTS — MEAN ACROSS PARTICIPANTS")
    print("=" * 70)

    print(
        results.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    print("\nPOOLED METRICS:")
    print(pooled_df.round(4).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="NeuroKit2 V2 LOSO Rest vs Distress V3"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the NeuroKit2 extracted feature CSV"
    )

    args = parser.parse_args()

    df, feature_cols = load_data(args.input)

    results, predictions = run_loso(
        df, feature_cols
    )

    save_results(results, predictions)


if __name__ == "__main__":
    main()
