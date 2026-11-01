"""
run_ablation.py
===============

Experiment runner for the CNN-TCN-LSTM architecture ablation study (RQ-A).

Architectures:
    A1: CNN only
    A2: CNN + LSTM
    A3: CNN + TCN
    A4: CNN + TCN + LSTM (Full reference model)

Protocol:
    - Data: WESAD (data/wesad/wesad_binary.npz)
    - Split: Subject-level train / validation (identical for all 4 models)
    - Optimization: Adam (lr=0.01), categorical_crossentropy
    - Class Weighting: Computed strictly from training split
    - Metrics: Accuracy, Macro F1, Weighted F1, Class 0 F1, Class 1 F1, ROC-AUC
    - Best epoch, training time, and parameter count recorded
    - Output: docs/ablation_results.csv (docs/experiment_log.csv remains untouched)
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf

from ablation_models import (
    build_cnn_only_model,
    build_cnn_lstm_model,
    build_cnn_tcn_model,
    build_full_cnn_tcn_lstm_model,
)
from evaluation import compute_metrics
from seed import set_seed

ABLATION_CSV_FIELDS = [
    "experiment_id",
    "architecture",
    "seed",
    "train_subjects",
    "validation_subjects",
    "train_windows",
    "validation_windows",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "class0_f1",
    "class1_f1",
    "auc",
    "best_epoch",
    "training_time_seconds",
    "parameter_count",
]


def load_wesad_data(data_path: str):
    """Loads preprocessed WESAD data and validates contract."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"WESAD dataset file not found at: {data_path}")

    d = np.load(data_path, allow_pickle=True)
    windows = d["windows"]
    labels = d["labels"]
    subject_ids = d["subject_ids"]

    assert windows.ndim == 3 and windows.shape[1] == 3840 and windows.shape[2] == 1, (
        f"Invalid windows shape: {windows.shape}"
    )
    assert len(labels) == len(windows), "Mismatch between labels and windows count"
    assert len(subject_ids) == len(windows), "Mismatch between subject_ids and windows count"

    return windows, labels, subject_ids


def create_fixed_subject_split(
    windows: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    test_size: float = 0.20,
    random_seed: int = 42,
):
    """
    Creates a fixed subject-level train/validation split.
    Guarantees that train and validation sets contain completely disjoint subjects.
    """
    unique_subjects = np.unique(subject_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)

    train_idx, val_idx = next(
        splitter.split(windows, labels, groups=subject_ids)
    )

    X_train, y_train = windows[train_idx], labels[train_idx]
    X_val, y_val = windows[val_idx], labels[val_idx]

    train_subs = [str(s) for s in sorted(list(set(subject_ids[train_idx])))]
    val_subs = [str(s) for s in sorted(list(set(subject_ids[val_idx])))]

    # Strict leakage assertion
    overlap = set(train_subs).intersection(set(val_subs))
    assert not overlap, f"Subject leakage detected between train and val: {overlap}"

    return (
        X_train, y_train, train_subs,
        X_val, y_val, val_subs,
    )


def compute_class_weights(y_train: np.ndarray) -> dict:
    """Computes inverse-frequency class weights strictly on the training partition."""
    counts = np.bincount(y_train.astype(int))
    total = len(y_train)
    weights = {i: total / (2.0 * c) for i, c in enumerate(counts) if c > 0}
    return weights


def log_ablation_row(output_csv: str, row_dict: dict):
    """Appends one result row to the dedicated ablation CSV."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    file_exists = os.path.exists(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ABLATION_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def run_ablation_experiment(
    data_path: str = "data/wesad/wesad_binary.npz",
    output_csv: str = "docs/ablation_results.csv",
    quick: bool = False,
    random_seed: int = 42,
):
    """Runs the controlled architecture ablation study across A1, A2, A3, and A4."""
    print("=" * 80)
    print("PHYSIOLOGICAL ARCHITECTURE ABLATION STUDY (RQ-A)")
    print(f"Mode: {'QUICK PIPELINE TEST (Sanity check)' if quick else 'FULL EXPERIMENT PROTOCOL'}")
    print(f"Data: {data_path} | Seed: {random_seed}")
    print("=" * 80)

    # 1. Load Data
    windows, labels, subject_ids = load_wesad_data(data_path)
    total_subjects = [str(s) for s in sorted(list(np.unique(subject_ids)))]
    print(f"Loaded WESAD: {len(windows)} windows across {len(total_subjects)} subjects: {total_subjects}")

    # 2. Fixed Subject Split (Identical across all 4 architectures)
    (
        X_train, y_train, train_subs,
        X_val, y_val, val_subs,
    ) = create_fixed_subject_split(
        windows, labels, subject_ids, test_size=0.20, random_seed=random_seed
    )

    print("\n--- Subject Partition ---")
    print(f"Training Subjects   ({len(train_subs)}): {train_subs}")
    print(f"Validation Subjects ({len(val_subs)}): {val_subs}")
    print(f"Train Windows: {len(y_train)} ({100*y_train.mean():.1f}% stress)")
    print(f"Validation Windows: {len(y_val)} ({100*y_val.mean():.1f}% stress)")

    # 3. Class Weights (Computed strictly from training partition)
    class_weights = compute_class_weights(y_train)
    print(f"Training Class Weights: {class_weights}")

    # One-hot encode targets
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)

    # Training Hyperparameters
    if quick:
        epochs = 5
        batch_size = 32
        patience = 2
        verbose = 1
    else:
        epochs = 350
        batch_size = 512
        patience = 80
        verbose = 0

    print(f"\nTraining Protocol: Epochs={epochs}, BatchSize={batch_size}, Patience={patience}, Optimizer=Adam(lr=0.01)")

    # Architecture Registry
    architectures = [
        ("A1_CNN_only", "CNN only", build_cnn_only_model),
        ("A2_CNN_LSTM", "CNN + LSTM", build_cnn_lstm_model),
        ("A3_CNN_TCN", "CNN + TCN", build_cnn_tcn_model),
        ("A4_CNN_TCN_LSTM", "CNN + TCN + LSTM", build_full_cnn_tcn_lstm_model),
    ]

    results_table = []

    for exp_id, arch_name, builder_fn in architectures:
        print("\n" + "-" * 70)
        print(f"Running [{exp_id}]: {arch_name}")
        print("-" * 70)

        # Set seed before building and fitting to enforce identical initial state
        set_seed(random_seed)

        model = builder_fn()
        params = model.count_params()
        print(f"Model parameters: {params:,}")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,
            )
        ]

        t0 = time.perf_counter()
        history = model.fit(
            X_train,
            y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose,
        )
        elapsed_sec = round(time.perf_counter() - t0, 2)

        # Determine best epoch based on monitored metric
        val_acc_history = history.history.get("val_accuracy", [])
        best_epoch = int(np.argmax(val_acc_history)) + 1 if val_acc_history else len(val_acc_history)

        # Evaluation on the exact same validation split
        pred_probs = model.predict(X_val, verbose=0)[:, 1]
        metrics = compute_metrics(y_val, pred_probs)

        auc_str = f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A"
        print(f"Completed in {elapsed_sec:.1f}s | Best Epoch: {best_epoch}")
        print(f"Val Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | AUC: {auc_str}")

        # Assemble record
        row = {
            "experiment_id": exp_id,
            "architecture": arch_name,
            "seed": random_seed,
            "train_subjects": "+".join(train_subs),
            "validation_subjects": "+".join(val_subs),
            "train_windows": len(y_train),
            "validation_windows": len(y_val),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "class0_f1": metrics["class0_f1"],
            "class1_f1": metrics["class1_f1"],
            "auc": metrics["auc"],
            "best_epoch": best_epoch,
            "training_time_seconds": elapsed_sec,
            "parameter_count": params,
        }

        log_ablation_row(output_csv, row)

        results_table.append({
            "Architecture": arch_name,
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "Weighted F1": f"{metrics['weighted_f1']:.4f}",
            "Class 0 F1": f"{metrics['class0_f1']:.4f}",
            "Class 1 F1": f"{metrics['class1_f1']:.4f}",
            "AUC": f"{metrics['auc']:.4f}" if metrics["auc"] is not None else "N/A",
            "Params": f"{params:,}",
            "Best Epoch": best_epoch,
        })

    # Summary Display
    print("\n" + "=" * 80)
    print("ARCHITECTURE ABLATION COMPARISON SUMMARY")
    print(f"(Logged to: {output_csv})")
    print("=" * 80)

    # Print Table
    header = ["Architecture", "Accuracy", "Macro F1", "Weighted F1", "Class 0 F1", "Class 1 F1", "AUC", "Params", "Best Epoch"]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in results_table)) for h in header}

    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in header)
    sep_line = "-+-".join("-" * col_widths[h] for h in header)
    print(header_line)
    print(sep_line)
    for r in results_table:
        line = " | ".join(f"{str(r[h]):<{col_widths[h]}}" for h in header)
        print(line)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run physiological CNN-TCN-LSTM architecture ablation study (RQ-A).")
    parser.add_argument("--quick", action="store_true", help="Run fast sanity check (5 epochs, batch_size 32)")
    parser.add_argument("--data-path", default="data/wesad/wesad_binary.npz", help="Path to WESAD binary .npz")
    parser.add_argument("--output-csv", default="docs/ablation_results.csv", help="Path to output ablation CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_ablation_experiment(
        data_path=args.data_path,
        output_csv=args.output_csv,
        quick=args.quick,
        random_seed=args.seed,
    )
