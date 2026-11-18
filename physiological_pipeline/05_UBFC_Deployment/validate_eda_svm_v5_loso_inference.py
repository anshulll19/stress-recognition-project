from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/ubfc_test_upload"
OUTPUT_DIR = ROOT / "results/ubfc_deployment/v5_inference_validation"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda_svm_v5_inference import EDASVMV5Inference, V5InferenceError


FS = 4
REST_LABEL = 0
DISTRESS_LABEL = 1
REST_EVALUATION_START = 100.0


def load_signal(participant, task):
    filename = f"eda_s{participant}_{task}.csv"
    matches = list(DATA_ROOT.rglob(filename))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename}; found {len(matches)}"
        )

    signal = np.asarray(
        np.loadtxt(matches[0], delimiter=","),
        dtype=float,
    ).ravel()

    return signal


def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {}

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "n_windows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1], average="macro")
        ),
        "sensitivity": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(
            tn / (tn + fp) if (tn + fp) else 0.0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_windows = []
    failures = []
    participant_rows = []

    print("=" * 65)
    print("V5 EDA-SVM INFERENCE VALIDATION")
    print("=" * 65)
    print("Calibration: T1 first 85 seconds")
    print("Rest evaluation: T1 windows starting at >= 100 seconds")
    print("Distress evaluation: T2 and T3")
    print("Participants: 1–56")
    print()

    for participant in range(1, 57):
        participant_windows = []
        participant_errors = []

        try:
            t1 = load_signal(participant, "T1")
            t2 = load_signal(participant, "T2")
            t3 = load_signal(participant, "T3")

            engine = EDASVMV5Inference()

            calibration_info = engine.calibrate(t1)

            # Evaluate T1 using the same whole-recording preprocessing
            # as the feature extractor, but retain only later rest windows.
            t1_predictions = engine.predict_task(t1)

            for result in t1_predictions:
                if result["start_s"] < REST_EVALUATION_START:
                    continue

                participant_windows.append({
                    "participant": participant,
                    "task": "T1",
                    "true_label": REST_LABEL,
                    "start_s": result["start_s"],
                    "end_s": result["end_s"],
                    "prediction": result["prediction"],
                    "decision_score": result["decision_score"],
                    "score_type": result["score_type"],
                })

            for task_name, signal in [("T2", t2), ("T3", t3)]:
                predictions = engine.predict_task(signal)

                for result in predictions:
                    participant_windows.append({
                        "participant": participant,
                        "task": task_name,
                        "true_label": DISTRESS_LABEL,
                        "start_s": result["start_s"],
                        "end_s": result["end_s"],
                        "prediction": result["prediction"],
                        "decision_score": result["decision_score"],
                        "score_type": result["score_type"],
                    })

            if not any(
                row["task"] == "T1" for row in participant_windows
            ):
                raise V5InferenceError(
                    "No later T1 rest windows available for evaluation."
                )

            if not any(
                row["task"] == "T2" for row in participant_windows
            ):
                raise V5InferenceError("No T2 windows available.")

            if not any(
                row["task"] == "T3" for row in participant_windows
            ):
                raise V5InferenceError("No T3 windows available.")

            all_windows.extend(participant_windows)

            y_true = [r["true_label"] for r in participant_windows]
            y_pred = [r["prediction"] for r in participant_windows]
            metrics = calculate_metrics(y_true, y_pred)

            participant_rows.append({
                "participant": participant,
                "status": "complete",
                "calibration_windows": calibration_info[
                    "calibration_windows"
                ],
                **metrics,
            })

            print(
                f"Participant {participant:02d}/56 | "
                f"windows={len(participant_windows):3d} | "
                f"BA={metrics['balanced_accuracy']:.3f} | "
                f"F1={metrics['macro_f1']:.3f}"
            )

        except Exception as exc:
            failures.append({
                "participant": participant,
                "error": str(exc),
            })

            participant_rows.append({
                "participant": participant,
                "status": "failed",
                "error": str(exc),
            })

            print(
                f"Participant {participant:02d}/56 | FAILED: {exc}"
            )

    windows_df = pd.DataFrame(all_windows)
    participants_df = pd.DataFrame(participant_rows)
    failures_df = pd.DataFrame(failures)

    windows_df.to_csv(
        OUTPUT_DIR / "v5_inference_window_predictions.csv",
        index=False,
    )
    participants_df.to_csv(
        OUTPUT_DIR / "v5_inference_participant_metrics.csv",
        index=False,
    )
    failures_df.to_csv(
        OUTPUT_DIR / "v5_inference_failures.csv",
        index=False,
    )

    if len(windows_df):
        pooled = calculate_metrics(
            windows_df["true_label"].tolist(),
            windows_df["prediction"].tolist(),
        )

        complete = participants_df[
            participants_df["status"] == "complete"
        ]

        participant_average = {
            metric: float(complete[metric].mean())
            for metric in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "sensitivity",
                "specificity",
            ]
        }

        task_metrics = {}

        for task in ["T1", "T2", "T3"]:
            subset = windows_df[windows_df["task"] == task]
            task_metrics[task] = {
                "n_windows": int(len(subset)),
                "n_participants": int(
                    subset["participant"].nunique()
                ),
                "predicted_rest": int(
                    (subset["prediction"] == 0).sum()
                ),
                "predicted_distress": int(
                    (subset["prediction"] == 1).sum()
                ),
            }

        summary = {
            "protocol": {
                "model": "UBFC-Phys V5 EDA-SVM",
                "calibration": "Participant-specific T1 first 85 seconds",
                "rest_evaluation": "T1 windows starting at >= 100 seconds",
                "distress_evaluation": "T2 and T3",
                "participant_count": 56,
                "window_duration_seconds": 30,
                "window_stride_seconds": 5,
                "sampling_rate_hz": FS,
                "note": (
                    "Overlapping windows are correlated. These results "
                    "are inference-pipeline validation, not independent "
                    "external validation."
                ),
            },
            "completed_participants": int(
                (participants_df["status"] == "complete").sum()
            ),
            "failed_participants": int(len(failures)),
            "total_windows": int(len(windows_df)),
            "pooled_window_metrics": pooled,
            "mean_participant_metrics": participant_average,
            "task_summary": task_metrics,
        }

        with open(
            OUTPUT_DIR / "v5_inference_validation_summary.json",
            "w",
        ) as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 65)
        print("FINAL VALIDATION SUMMARY")
        print("=" * 65)
        print(
            f"Completed participants: "
            f"{summary['completed_participants']}/56"
        )
        print(f"Failed participants: {len(failures)}")
        print(f"Total evaluated windows: {len(windows_df)}")

        print("\nPOOLED WINDOW METRICS")
        for key, value in pooled.items():
            print(f"{key}: {value}")

        print("\nMEAN PARTICIPANT METRICS")
        for key, value in participant_average.items():
            print(f"{key}: {value:.4f}")

        print("\nTASK SUMMARY")
        for task, values in task_metrics.items():
            print(task, values)

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
