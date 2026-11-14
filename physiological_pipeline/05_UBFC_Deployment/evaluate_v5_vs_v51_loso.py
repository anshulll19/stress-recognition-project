#!/usr/bin/env python3

"""Paired 56-participant LOSO comparison of V5 and V5.1."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_PATH = Path("results/neurokit2_v2/fused_features_v2.csv")
OUTPUT = Path("results/ubfc_deployment/v5_vs_v51_loso")

CALIBRATION_END = 85.0
BOUNDARY_END = 100.0
RANDOM_STATE = 42

V5_FEATURES = [
    "eda_mean", "eda_std", "eda_median", "eda_iqr",
    "eda_range", "eda_tonic_mean", "eda_tonic_std",
    "eda_tonic_slope", "eda_phasic_mean", "eda_phasic_std",
    "eda_scr_valid_event_count",
    "eda_scr_onsets_diagnostic",
    "eda_scr_peaks_diagnostic",
    "eda_raw_std", "eda_unique_fraction",
    "eda_flat_step_fraction",
    "eda_median_abs_difference",
    "eda_max_abs_jump", "eda_jump_ratio",
]

SCR_FEATURES = [
    "eda_scr_valid_event_count",
    "eda_scr_onsets_diagnostic",
    "eda_scr_peaks_diagnostic",
]

VARIANTS = {
    "V5_19_features": V5_FEATURES,
    "V5_1_16_features": [
        feature for feature in V5_FEATURES
        if feature not in SCR_FEATURES
    ],
}


def make_model():
    return Pipeline([
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


def calculate_baseline(subject, features):
    calibration = subject[
        (subject["task"] == 1)
        & (subject["end_s"] <= CALIBRATION_END)
    ]

    if len(calibration) != 12:
        raise ValueError(
            f"{subject['participant'].iloc[0]}: "
            f"expected 12 calibration windows, got {len(calibration)}"
        )

    X = calibration[features].astype(float)

    median = X.median()
    iqr = X.quantile(0.75) - X.quantile(0.25)
    std = X.std(ddof=0)

    scale = iqr.where(
        np.isfinite(iqr) & (iqr > 1e-10),
        std,
    )
    scale = scale.where(
        np.isfinite(scale) & (scale > 1e-10),
        1.0,
    )

    if not np.isfinite(median.to_numpy()).all():
        raise ValueError("Invalid calibration median.")

    if not np.isfinite(scale.to_numpy()).all():
        raise ValueError("Invalid calibration scale.")

    return median, scale


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "distress_sensitivity": (
            tp / (tp + fn) if tp + fn else 0.0
        ),
        "rest_specificity": (
            tn / (tn + fp) if tn + fp else 0.0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    print("=" * 75)
    print("V5 vs V5.1 — PAIRED 56-PARTICIPANT LOSO")
    print("=" * 75)

    df = pd.read_csv(DATA_PATH)

    df["participant"] = df["participant"].astype(str).str.strip()

    for col in ["task", "start_s", "end_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["task"].isin([1, 2, 3])].copy()

    participants = sorted(df["participant"].unique())

    if len(participants) != 56:
        raise ValueError(
            f"Expected 56 participants, found {len(participants)}"
        )

    for variant, features in VARIANTS.items():
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"{variant}: missing features {missing}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_predictions = []

    for variant, features in VARIANTS.items():
        print(f"\n{'=' * 75}")
        print(f"VARIANT: {variant}")
        print(f"FEATURE COUNT: {len(features)}")
        print("=" * 75)

        for fold, held_out in enumerate(participants, start=1):
            normalized_subjects = []

            for participant in participants:
                subject = df[
                    df["participant"] == participant
                ].copy()

                median, scale = calculate_baseline(
                    subject, features
                )

                subject[features] = (
                    subject[features].astype(float) - median
                ) / scale

                normalized_subjects.append(subject)

            normalized = pd.concat(
                normalized_subjects, ignore_index=True
            )

            train_subjects = normalized[
                normalized["participant"] != held_out
            ]

            test_subject = normalized[
                normalized["participant"] == held_out
            ]

            train_rest = train_subjects[
                (train_subjects["task"] == 1)
                & (train_subjects["end_s"] <= CALIBRATION_END)
            ]

            train_distress = train_subjects[
                train_subjects["task"].isin([2, 3])
            ]

            test_rest = test_subject[
                (test_subject["task"] == 1)
                & (test_subject["start_s"] >= BOUNDARY_END)
            ]

            test_distress = test_subject[
                test_subject["task"].isin([2, 3])
            ]

            train = pd.concat(
                [train_rest, train_distress], ignore_index=True
            )

            test = pd.concat(
                [test_rest, test_distress], ignore_index=True
            )

            if test_rest.empty or test_distress.empty:
                raise ValueError(
                    f"{held_out}: missing a test class."
                )

            X_train = train[features].astype(float)
            y_train = np.concatenate([
                np.zeros(len(train_rest), dtype=int),
                np.ones(len(train_distress), dtype=int),
            ])

            X_test = test[features].astype(float)
            y_test = np.concatenate([
                np.zeros(len(test_rest), dtype=int),
                np.ones(len(test_distress), dtype=int),
            ])

            model = make_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = calculate_metrics(y_test, y_pred)

            all_results.append({
                "variant": variant,
                "participant": held_out,
                "feature_count": len(features),
                "train_rest": len(train_rest),
                "train_distress": len(train_distress),
                "test_rest": len(test_rest),
                "test_distress": len(test_distress),
                **metrics,
            })

            for index, (_, row) in enumerate(test.iterrows()):
                all_predictions.append({
                    "variant": variant,
                    "participant": held_out,
                    "task": int(row["task"]),
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                    "true_label": int(y_test[index]),
                    "predicted_label": int(y_pred[index]),
                })

            print(
                f"[{fold:02d}/56] {held_out:>3} | "
                f"BalAcc={metrics['balanced_accuracy']:.4f} | "
                f"MacroF1={metrics['macro_f1']:.4f} | "
                f"Rest FP={metrics['FP']}/{len(test_rest)}"
            )

    results = pd.DataFrame(all_results)
    predictions = pd.DataFrame(all_predictions)

    results.to_csv(
        OUTPUT / "per_participant.csv", index=False
    )
    predictions.to_csv(
        OUTPUT / "predictions.csv", index=False
    )

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "distress_sensitivity",
        "rest_specificity",
        "macro_f1",
        "weighted_f1",
    ]

    pooled = []

    for variant, group in predictions.groupby("variant"):
        pooled_metrics = calculate_metrics(
            group["true_label"].to_numpy(),
            group["predicted_label"].to_numpy(),
        )

        pooled.append({
            "variant": variant,
            **pooled_metrics,
            "total_windows": len(group),
        })

    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(
        OUTPUT / "pooled_metrics.csv", index=False
    )

    summary = results.groupby("variant")[metric_columns].agg(
        ["mean", "std"]
    )
    summary.to_csv(OUTPUT / "participant_summary.csv")

    # Paired per-participant differences.
    wide = results.pivot(
        index="participant",
        columns="variant",
        values=metric_columns,
    )

    differences = []

    for metric in metric_columns:
        v5 = wide[(metric, "V5_19_features")]
        v51 = wide[(metric, "V5_1_16_features")]

        delta = v51 - v5

        differences.append({
            "metric": metric,
            "V5_mean": v5.mean(),
            "V5_1_mean": v51.mean(),
            "mean_paired_delta": delta.mean(),
            "participants_improved": int((delta > 0).sum()),
            "participants_worsened": int((delta < 0).sum()),
            "participants_tied": int((delta == 0).sum()),
        })

    difference_df = pd.DataFrame(differences)
    difference_df.to_csv(
        OUTPUT / "paired_differences.csv", index=False
    )

    manifest = {
        "experiment": "Paired V5 vs V5.1 LOSO",
        "participants": len(participants),
        "folds": len(participants),
        "calibration_end_seconds": CALIBRATION_END,
        "rest_test_start_seconds": BOUNDARY_END,
        "normalization": "Per-participant T1 median/IQR",
        "classifier": "RBF-SVM, C=1.0, gamma=scale, balanced",
        "variants": {
            name: features for name, features in VARIANTS.items()
        },
        "missing_value_handling": (
            "Median imputation fitted on each training fold only."
        ),
        "note": (
            "Held-out participant calibration is used only for "
            "participant-specific normalization. It is excluded "
            "from classifier fitting."
        ),
    }

    with open(OUTPUT / "manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)

    print("\n" + "=" * 75)
    print("POOLED RESULTS")
    print("=" * 75)
    print(pooled_df.round(4).to_string(index=False))

    print("\nPAIRED PARTICIPANT DIFFERENCES (V5.1 minus V5)")
    print("=" * 75)
    print(difference_df.round(4).to_string(index=False))

    print("\nParticipant mean ± standard deviation")
    print(summary.round(4).to_string())

    print("\nSaved results to:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
