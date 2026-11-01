"""
cross_dataset.py
=================

Runs cross-dataset generalization experiments (Track 2 / RQ2):
train on one dataset entirely, evaluate on a different dataset entirely.

Usage:
    python cross_dataset.py --train wesad --test clas
    python cross_dataset.py --train clas --test wesad
    python cross_dataset.py --train wesad,clas --test empathicschool   # pooled training (E4)

Enforces the leakage rules from the handoff doc Section 13:
    - TRAIN = source dataset(s) only, TEST = target dataset only
    - no participant appears in both train and test (guaranteed here
      since train/test come from ENTIRELY DIFFERENT datasets --
      the one case needing care is pooled training + evaluating on a
      held-out split of one of the pooled datasets, which is handled
      separately by loso_harness.py, not this script)
    - preprocessing stats (z-score) are already computed per-window by
      each dataset's own preprocessing script, never using target data
"""

import argparse

import numpy as np
import tensorflow as tf

from model import build_cnn_tcn_lstm_model
from dataset_loader import load_dataset, register_dataset
from evaluation import compute_metrics, print_metrics_report, log_experiment
from seed import set_seed


def register_known_datasets():
    """Central place to register dataset name -> file path. Update paths
    as teammates' preprocessing outputs land."""
    register_dataset("wesad", "data/wesad/wesad_binary.npz")
    register_dataset("clas", "data/clas/clas_binary_clean.npz")           # not yet produced
    register_dataset("ppge", "ppge_binary.npz")            # not yet produced
    register_dataset("empathicschool", "empathicschool_binary.npz")  # not yet produced


def run_cross_dataset_experiment(train_names, test_name, epochs=350, batch_size=512,
                                  early_stopping_patience=80, verbose=0, random_seed=42):
    """
    Cross-dataset experiment.

    Source dataset(s):
        - Split by SUBJECT into source-train and source-validation.
        - Validation is used only for early stopping.

    Target dataset:
        - Never used for training or validation.
        - Entire target dataset is used once as the final test set.
    """
    set_seed(random_seed)

    train_windows = []
    train_labels = []
    train_subjects = []

    for name in train_names:
        d = load_dataset(name)

        train_windows.append(d["windows"])
        train_labels.append(d["labels"])

        # Prefix subject IDs with dataset name so pooled datasets
        # cannot accidentally share subject identifiers.
        subject_ids = np.asarray(d["subject_ids"]).astype(str)
        subject_ids = np.array(
            [f"{name}_{sid}" for sid in subject_ids]
        )
        train_subjects.append(subject_ids)

    X_source = np.concatenate(train_windows, axis=0)
    y_source = np.concatenate(train_labels, axis=0)
    subjects_source = np.concatenate(train_subjects, axis=0)

    # Target dataset is NEVER used for training or validation.
    test_data = load_dataset(test_name)
    X_test = test_data["windows"]
    y_test = test_data["labels"]

    # ---------------------------------------------------------
    # SUBJECT-LEVEL SOURCE TRAIN / VALIDATION SPLIT
    # ---------------------------------------------------------
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, val_idx = next(
        splitter.split(
            X_source,
            y_source,
            groups=subjects_source
        )
    )

    X_train = X_source[train_idx]
    y_train = y_source[train_idx]

    X_val = X_source[val_idx]
    y_val = y_source[val_idx]

    train_subject_ids = subjects_source[train_idx]
    val_subject_ids = subjects_source[val_idx]

    print(
        f"Source train: {'+'.join(train_names)} "
        f"({len(y_train)} windows, "
        f"{len(np.unique(train_subject_ids))} subjects, "
        f"{100*y_train.mean():.1f}% class-1)"
    )

    print(
        f"Source validation: "
        f"({len(y_val)} windows, "
        f"{len(np.unique(val_subject_ids))} subjects, "
        f"{100*y_val.mean():.1f}% class-1)"
    )

    print(
        f"Target test: {test_name} "
        f"({len(y_test)} windows, "
        f"{len(np.unique(test_data['subject_ids']))} subjects, "
        f"{100*y_test.mean():.1f}% class-1)"
    )

    # ---------------------------------------------------------
    # CLASS WEIGHTS — SOURCE TRAIN ONLY
    # ---------------------------------------------------------
    class_counts = np.bincount(y_train.astype(int))

    class_weight = {
        i: len(y_train) / (2 * count)
        for i, count in enumerate(class_counts)
        if count > 0
    }

    y_train_cat = tf.keras.utils.to_categorical(
        y_train, num_classes=2
    )

    y_val_cat = tf.keras.utils.to_categorical(
        y_val, num_classes=2
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------
    model = build_cnn_tcn_lstm_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose
    )

    # ---------------------------------------------------------
    # FINAL TARGET EVALUATION
    # ---------------------------------------------------------
    pred_probs = model.predict(
        X_test,
        verbose=0
    )[:, 1]

    metrics = compute_metrics(
        y_test,
        pred_probs
    )

    return metrics, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a cross-dataset generalization experiment")
    parser.add_argument("--train", required=True, help="Comma-separated dataset name(s), e.g. 'wesad' or 'wesad,clas' for pooled")
    parser.add_argument("--test", required=True, help="Target dataset name, e.g. 'clas'")
    parser.add_argument("--quick", action="store_true", help="Fast sanity check (few epochs) instead of the full run")
    parser.add_argument("--experiment_id", default=None, help="ID for the experiment log, e.g. E2_wesad_to_clas")
    parser.add_argument("--log_path", default="docs/experiment_log.csv")
    args = parser.parse_args()

    register_known_datasets()

    train_names = [n.strip() for n in args.train.split(",")]
    test_name = args.test.strip()

    if args.quick:
        print("--- QUICK sanity check (not the real result) ---")
        metrics, _ = run_cross_dataset_experiment(
            train_names, test_name, epochs=5, batch_size=32, early_stopping_patience=3, verbose=1
        )
    else:
        print(f"--- FULL cross-dataset run: {'+'.join(train_names)} -> {test_name} ---")
        metrics, _ = run_cross_dataset_experiment(train_names, test_name, verbose=0)

    print_metrics_report(metrics, f"{'+'.join(train_names)} -> {test_name}")

    if not args.quick:
        exp_id = args.experiment_id or f"{'+'.join(train_names)}_to_{test_name}"
        log_experiment(
            args.log_path,
            experiment_id=exp_id,
            source_dataset="+".join(train_names), target_dataset=test_name,
            model_version="CNN-TCN-LSTM v1", learning_rate=0.01, batch_size=512,
            max_epochs=350, early_stopping="patience 80", class_weighting="yes",
            **metrics,
            notes="Generated by cross_dataset.py"
        )
