
from pathlib import Path
import sys
import pickle
import json
import gc

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(ROOT / "physiological_pipeline/05_UBFC_Deployment")
)

from eda_svm_v5_inference import EDASVMV5Inference

WESAD_ROOT = ROOT / "data/wesad/WESAD"
OUTPUT_DIR = ROOT / "results/wesad_cross_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTICIPANTS = ["S10", "S11", "S13", "S14", "S15", "S16", "S17"]

EDA_FS = 4
LABEL_FS = 700
CALIBRATION_SECONDS = 85


def longest_segment(labels, target):
    indices = np.flatnonzero(labels == target)

    if len(indices) == 0:
        raise ValueError(f"Label {target} not found")

    splits = np.split(
        indices,
        np.where(np.diff(indices) != 1)[0] + 1
    )

    segment = max(splits, key=len)

    return int(segment[0]), int(segment[-1] + 1)


def evaluate_participant(participant):
    pkl_path = WESAD_ROOT / participant / f"{participant}.pkl"

    print(f"\n{'=' * 60}")
    print(f"PROCESSING {participant}")
    print("=" * 60)

    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)

    with pkl_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    eda = np.asarray(
        data["signal"]["wrist"]["EDA"],
        dtype=float
    ).reshape(-1)

    labels = np.asarray(data["label"]).reshape(-1)

    del data
    gc.collect()

    # Align 700 Hz labels with 4 Hz EDA.
    times = np.arange(len(eda)) / EDA_FS
    label_indices = np.rint(times * LABEL_FS).astype(int)
    label_indices = np.clip(label_indices, 0, len(labels) - 1)
    eda_labels = labels[label_indices]

    baseline_start, baseline_end = longest_segment(eda_labels, 1)
    stress_start, stress_end = longest_segment(eda_labels, 2)

    baseline = eda[baseline_start:baseline_end]
    stress = eda[stress_start:stress_end]

    calibration_samples = CALIBRATION_SECONDS * EDA_FS

    if len(baseline) <= calibration_samples:
        raise ValueError("Baseline too short for calibration and testing")

    calibration_signal = baseline[:calibration_samples]
    rest_test_signal = baseline[calibration_samples:]

    print("Baseline duration:", round(len(baseline) / EDA_FS, 2))
    print("Stress duration:", round(len(stress) / EDA_FS, 2))

    adapter = EDASVMV5Inference()

    print("Calibrating...")
    adapter.calibrate(calibration_signal)

    print("Predicting baseline...")
    rest_predictions = adapter.predict_task(rest_test_signal)

    rest_df = pd.DataFrame(rest_predictions)
    rest_df["true_label"] = 0
    rest_df["segment"] = "baseline"
    rest_df["participant"] = participant

    print("Predicting stress...")
    stress_predictions = adapter.predict_task(stress)

    stress_df = pd.DataFrame(stress_predictions)
    stress_df["true_label"] = 1
    stress_df["segment"] = "stress"
    stress_df["participant"] = participant

    results = pd.concat(
        [rest_df, stress_df],
        ignore_index=True
    )

    y_true = results["true_label"].to_numpy()
    y_pred = results["prediction"].to_numpy()
    scores = results["decision_score"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "participant": participant,
        "rest_windows": len(rest_df),
        "stress_windows": len(stress_df),
        "total_windows": len(results),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "sensitivity": float(cm[1, 1] / max(cm[1].sum(), 1)),
        "specificity": float(cm[0, 0] / max(cm[0].sum(), 1)),
        "confusion_matrix": cm.tolist(),
        "baseline_mean_score": float(rest_df["decision_score"].mean()),
        "stress_mean_score": float(stress_df["decision_score"].mean()),
        "score_type": "raw_svm_decision_function",
    }

    print("\nParticipant results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    del eda, labels, eda_labels
    del baseline, stress, calibration_signal, rest_test_signal
    del adapter
    gc.collect()

    return results, metrics


def main():
    all_predictions = []
    all_metrics = []
    failures = []

    for participant in PARTICIPANTS:
        try:
            predictions, metrics = evaluate_participant(participant)

            all_predictions.append(predictions)
            all_metrics.append(metrics)

            # Save incrementally, so completed participants are preserved.
            pd.concat(all_predictions, ignore_index=True).to_csv(
                OUTPUT_DIR / "wesad_v5_7participants_predictions.csv",
                index=False
            )

            (OUTPUT_DIR / "wesad_v5_7participants_metrics.json").write_text(
                json.dumps(all_metrics, indent=2)
            )

        except Exception as exc:
            failures.append({
                "participant": participant,
                "error": str(exc),
            })
            print(f"\nFAILED: {participant}: {exc}")

    if not all_predictions:
        print("No participants completed successfully.")
        return

    combined = pd.concat(all_predictions, ignore_index=True)

    y_true = combined["true_label"].to_numpy()
    y_pred = combined["prediction"].to_numpy()
    scores = combined["decision_score"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    pooled = {
        "participants_completed": len(all_predictions),
        "participants_failed": failures,
        "total_windows": len(combined),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "sensitivity": float(cm[1, 1] / max(cm[1].sum(), 1)),
        "specificity": float(cm[0, 0] / max(cm[0].sum(), 1)),
        "confusion_matrix": cm.tolist(),
        "score_type": "raw_svm_decision_function",
        "evaluation": "Frozen UBFC V5 cross-dataset exploratory pilot",
    }

    (OUTPUT_DIR / "wesad_v5_7participants_pooled_metrics.json").write_text(
        json.dumps(pooled, indent=2)
    )

    print("\n" + "=" * 60)
    print("FINAL POOLED RESULTS")
    print("=" * 60)

    for key, value in pooled.items():
        print(f"{key}: {value}")

    print("\nPer-participant summary:")
    print(pd.DataFrame(all_metrics).to_string(index=False))

    print("\nSaved results:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
