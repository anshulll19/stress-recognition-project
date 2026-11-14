#!/usr/bin/env python3

import argparse
import json
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

CALIBRATION_END = 85.0
BOUNDARY_END = 100.0
RANDOM_STATE = 42

OUTPUT = Path(
    "physiological_pipeline/03_NeuroKit_V2/results_loso_v4"
)


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


def metrics(y_true, y_pred):

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
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def load_dataset(path):

    df = pd.read_csv(path)

    required = {"participant", "task", "start_s", "end_s"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df["participant"] = df["participant"].astype(str)
    df["task"] = pd.to_numeric(df["task"], errors="coerce")
    df["start_s"] = pd.to_numeric(df["start_s"], errors="coerce")
    df["end_s"] = pd.to_numeric(df["end_s"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=["participant", "task", "start_s", "end_s"]
    )

    df = df[df["task"].isin([1, 2, 3])].copy()

    # Never include identifiers, timestamps or target labels
    # as model features.
    excluded = {
        "participant", "task", "start_s", "end_s",
        "label", "target", "class", "ground_truth",
    }

    bvp_cols = [
        c for c in df.columns
        if c.startswith("bvp_")
        and c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
        and df[c].nunique(dropna=True) > 1
    ]

    eda_cols = [
        c for c in df.columns
        if c.startswith("eda_")
        and c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
        and df[c].nunique(dropna=True) > 1
    ]

    feature_sets = {
        "BVP_only": bvp_cols,
        "EDA_only": eda_cols,
        "BVP_EDA_fusion": bvp_cols + eda_cols,
    }

    for name, cols in feature_sets.items():
        if not cols:
            raise ValueError(f"No usable features found for {name}")

        print(f"{name}: {len(cols)} features")

    print("\nDataset:", df.shape)
    print("Participants:", df["participant"].nunique())
    print("Task counts:")
    print(df["task"].value_counts().sort_index())

    return df, feature_sets


def make_split(df, held_out, features):

    training = df[df["participant"] != held_out]

    train_rest = training[
        (training["task"] == 1)
        & (training["end_s"] <= CALIBRATION_END)
    ]

    train_distress = training[
        training["task"].isin([2, 3])
    ]

    train = pd.concat(
        [train_rest, train_distress],
        ignore_index=True,
    )

    test_subject = df[df["participant"] == held_out]

    test_rest = test_subject[
        (test_subject["task"] == 1)
        & (test_subject["start_s"] >= BOUNDARY_END)
    ]

    test_distress = test_subject[
        test_subject["task"].isin([2, 3])
    ]

    test = pd.concat(
        [test_rest, test_distress],
        ignore_index=True,
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
    )


def run_experiment(df, feature_sets):

    participants = sorted(df["participant"].unique())

    results = []
    predictions = []

    total = len(participants)

    for modality, features in feature_sets.items():

        print("\n" + "=" * 75)
        print("MODALITY:", modality)
        print("FEATURES:", len(features))
        print("=" * 75)

        for model_name, model_template in make_models().items():

            print(f"\nMODEL: {model_name}")

            for fold, held_out in enumerate(participants, 1):

                (
                    X_train, y_train,
                    X_test, y_test,
                    train_rest, train_distress,
                    test_rest, test_distress,
                ) = make_split(df, held_out, features)

                if len(X_test) == 0:
                    print(f"Skipping {held_out}: empty test set")
                    continue

                if len(np.unique(y_train)) != 2:
                    print(f"Skipping {held_out}: one training class")
                    continue

                if len(np.unique(y_test)) != 2:
                    print(f"Skipping {held_out}: one test class")
                    continue

                model = make_models()[model_name]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                score = metrics(y_test, y_pred)

                tn, fp, fn, tp = confusion_matrix(
                    y_test, y_pred, labels=[0, 1]
                ).ravel()

                results.append({
                    "modality": modality,
                    "model": model_name,
                    "participant": held_out,
                    "features": len(features),
                    "train_rest": train_rest,
                    "train_distress": train_distress,
                    "test_rest": test_rest,
                    "test_distress": test_distress,
                    **score,
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                })

                for true, pred in zip(y_test, y_pred):
                    predictions.append({
                        "modality": modality,
                        "model": model_name,
                        "participant": held_out,
                        "true_label": int(true),
                        "predicted_label": int(pred),
                    })

                print(
                    f"[{fold:02d}/{total}] {held_out} | "
                    f"BalAcc={score['balanced_accuracy']:.4f} | "
                    f"MacroF1={score['macro_f1']:.4f}"
                )

    return pd.DataFrame(results), pd.DataFrame(predictions)


def save_results(results, predictions):

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if results.empty:
        raise RuntimeError("No successful experiments.")

    results.to_csv(
        OUTPUT / "v4_per_participant.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT / "v4_predictions.csv",
        index=False,
    )

    metric_cols = [
        "accuracy", "balanced_accuracy",
        "precision", "recall",
        "macro_f1", "weighted_f1",
    ]

    summary = results.groupby(
        ["modality", "model"]
    )[metric_cols].agg(["mean", "std"])

    summary.to_csv(OUTPUT / "v4_summary.csv")

    pooled = []

    for (modality, model), group in predictions.groupby(
        ["modality", "model"]
    ):

        y_true = group["true_label"]
        y_pred = group["predicted_label"]

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        pooled.append({
            "modality": modality,
            "model": model,
            **metrics(y_true, y_pred),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "total_windows": len(group),
            "rest_specificity": (
                tn / (tn + fp) if tn + fp else 0
            ),
            "distress_sensitivity": (
                tp / (tp + fn) if tp + fn else 0
            ),
        })

    pooled_df = pd.DataFrame(pooled)

    pooled_df.to_csv(
        OUTPUT / "v4_pooled_metrics.csv",
        index=False,
    )

    report = {
        "experiment": "NeuroKit2 V4 Modality Ablation",
        "protocol": {
            "calibration_end": CALIBRATION_END,
            "boundary_exclusion": [85, 100],
            "rest_test_start": BOUNDARY_END,
            "distress_tasks": [2, 3],
            "evaluation": "LOSO",
        },
        "participants_evaluated": int(
            results["participant"].nunique()
        ),
        "summary": summary.round(6).to_string(),
        "pooled": pooled_df.to_dict(orient="records"),
    }

    with open(OUTPUT / "v4_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 75)
    print("V4 FINAL RESULTS — MEAN AND STANDARD DEVIATION")
    print("=" * 75)
    print(summary.round(4).to_string())

    print("\nPOOLED RESULTS:")
    print(pooled_df.round(4).to_string(index=False))

    print("\nSaved to:", OUTPUT.resolve())


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/neurokit2_v2/fused_features_v2.csv",
    )

    args = parser.parse_args()

    df, feature_sets = load_dataset(args.input)

    results, predictions = run_experiment(
        df, feature_sets
    )

    save_results(results, predictions)


if __name__ == "__main__":
    main()
