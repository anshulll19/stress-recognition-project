"""
evaluation.py
=============

Dataset-agnostic evaluation: metrics computation + a standard experiment
log row, so every experiment (WESAD LOSO, WESAD->CLAS, pooled training, ...)
reports results in the exact same format, into the exact same log schema
specified in the handoff doc's Section 14.
"""

import csv
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

EXPERIMENT_LOG_FIELDS = [
    "experiment_id", "date", "source_dataset", "target_dataset",
    "training_subjects", "testing_subjects", "signal",
    "original_sampling_rate", "resampling_rate", "filter",
    "window_size", "window_stride", "normalization", "label_mapping",
    "model_version", "learning_rate", "batch_size", "max_epochs",
    "early_stopping", "class_weighting",
    "accuracy", "macro_f1", "weighted_f1", "class0_f1", "class1_f1", "auc",
    "notes",
]


def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """
    y_true: (N,) binary ground truth
    y_pred_probs: (N,) predicted probability of class 1
    Returns a dict matching EXPERIMENT_LOG_FIELDS' metric columns.
    """
    y_pred_classes = (y_pred_probs >= 0.5).astype(int)

    auc = None
    if len(np.unique(y_true)) >= 2:
        auc = roc_auc_score(y_true, y_pred_probs)

    return {
        "accuracy": accuracy_score(y_true, y_pred_classes),
        "macro_f1": f1_score(y_true, y_pred_classes, average='macro', zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred_classes, average='weighted', zero_division=0),
        "class0_f1": f1_score(y_true, y_pred_classes, pos_label=0, zero_division=0),
        "class1_f1": f1_score(y_true, y_pred_classes, pos_label=1, zero_division=0),
        "auc": auc,
    }


def log_experiment(log_path: str, **kwargs):
    """
    Appends one row to the experiment log CSV (creating it with a header
    if it doesn't exist yet). Any EXPERIMENT_LOG_FIELDS not passed in
    kwargs are left blank -- fill in what's relevant to each experiment.

    Example:
        log_experiment(
            "docs/experiment_log.csv",
            experiment_id="E1_wesad_loso",
            source_dataset="wesad", target_dataset="wesad (LOSO)",
            signal="wrist BVP", original_sampling_rate=64, resampling_rate=64,
            filter="butterworth 0.7-3.7Hz", window_size="60s", window_stride="5s",
            normalization="z-score per window", label_mapping="2->1, 1/3/4->0, 0/5/6/7 dropped",
            model_version="CNN-TCN-LSTM v1", learning_rate=0.01, batch_size=512,
            max_epochs=350, early_stopping="patience 80", class_weighting="yes",
            accuracy=0.9704, macro_f1=..., weighted_f1=0.9703,
            class0_f1=0.9811, class1_f1=0.9326, auc=0.9667,
            notes="First real Track 1 result"
        )
    """
    row = {field: kwargs.get(field, "") for field in EXPERIMENT_LOG_FIELDS}
    row["date"] = row["date"] or datetime.now().strftime("%Y-%m-%d %H:%M")

    file_exists = os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged experiment '{row['experiment_id']}' to {log_path}")


def print_metrics_report(metrics: dict, title: str = "Results"):
    print(f"\n--- {title} ---")
    for k, v in metrics.items():
        if v is None:
            print(f"{k}: N/A")
        else:
            print(f"{k}: {v:.4f}")


if __name__ == '__main__':
    # Self-test
    print("--- evaluation.py self-test ---\n")
    rng = np.random.default_rng(0)
    y_true = rng.choice([0, 1], size=200, p=[0.78, 0.22])
    y_pred_probs = np.clip(y_true * 0.6 + rng.normal(0.3, 0.2, size=200), 0, 1)

    metrics = compute_metrics(y_true, y_pred_probs)
    print_metrics_report(metrics, "Synthetic test metrics")

    log_experiment(
        "test_experiment_log.csv",
        experiment_id="TEST_E0",
        source_dataset="synthetic", target_dataset="synthetic",
        notes="evaluation.py self-test row",
        **metrics
    )

    with open("test_experiment_log.csv") as f:
        print("\n--- Log file contents ---")
        print(f.read())

    print("Self-test passed.")
