"""
Dataset-agnostic Leave-One-Subject-Out (LOSO) training/eval harness.

Design goal: this file should not need to change when swapping between
WESAD, PPGE, or a backup dataset like CLAS. It only assumes the data has
already been preprocessed into the common format below:

    windows: np.ndarray of shape (n_windows, window_length, 1)
    labels:  np.ndarray of shape (n_windows,) -- integer class labels (0/1)
    subject_ids: np.ndarray of shape (n_windows,) -- which subject each window belongs to

Whoever owns preprocessing for a given dataset (e.g. Person 1 for WESAD)
is responsible for producing exactly these three arrays; everything below
is generic on top of that contract.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import tensorflow as tf

from model import build_cnn_tcn_lstm_model


def run_loso_cv(windows, labels, subject_ids, model_builder_fn=build_cnn_tcn_lstm_model,
                 epochs=350, batch_size=512, early_stopping_patience=80, verbose=0):
    """
    Runs Leave-One-Subject-Out cross-validation.

    For each unique subject: train on everyone else, test on that subject.
    Returns per-subject metrics plus overall aggregated metrics, matching
    the evaluation approach used in Alghoul et al. (2025).

    Args:
        windows: (n_windows, window_length, 1) float32 array
        labels: (n_windows,) int array, values in {0, 1}
        subject_ids: (n_windows,) array identifying each window's subject
        model_builder_fn: callable that returns a fresh, compiled model
            (defaults to the CNN-TCN-LSTM model in model.py)
        epochs, batch_size, early_stopping_patience: training config,
            matching the paper's stated training details
        verbose: passed through to model.fit

    Returns:
        dict with 'per_subject' (list of per-subject result dicts) and
        'overall' (aggregated AUC / F1 / accuracy across all held-out predictions)
    """
    unique_subjects = np.unique(subject_ids)
    per_subject_results = []

    all_true, all_pred_probs = [], []

    for held_out_subject in unique_subjects:
        test_mask = subject_ids == held_out_subject
        train_mask = ~test_mask

        X_train, y_train = windows[train_mask], labels[train_mask]
        X_test, y_test = windows[test_mask], labels[test_mask]

        if len(np.unique(y_test)) < 2:
            # Can't compute AUC for a held-out subject with only one class present;
            # still record accuracy, flag AUC as unavailable.
            auc_computable = False
        else:
            auc_computable = True

        # Class weights to address imbalance, per the paper's methodology
        class_counts = np.bincount(y_train.astype(int))
        class_weight = {i: len(y_train) / (2 * count) for i, count in enumerate(class_counts) if count > 0}

        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=2)

        model = model_builder_fn()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=early_stopping_patience,
                restore_best_weights=True
            )
        ]

        model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs, batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose
        )

        pred_probs = model.predict(X_test, verbose=0)
        pred_classes = np.argmax(pred_probs, axis=1)

        result = {
            'subject': held_out_subject,
            'n_test_windows': len(y_test),
            'accuracy': accuracy_score(y_test, pred_classes),
            'f1_weighted': f1_score(y_test, pred_classes, average='weighted', zero_division=0),
            'f1_class0': f1_score(y_test, pred_classes, pos_label=0, zero_division=0),
            'f1_class1': f1_score(y_test, pred_classes, pos_label=1, zero_division=0),
            'auc': roc_auc_score(y_test, pred_probs[:, 1]) if auc_computable else None,
        }
        per_subject_results.append(result)

        all_true.extend(y_test.tolist())
        all_pred_probs.extend(pred_probs[:, 1].tolist())

        auc_str = 'N/A' if result['auc'] is None else f"{result['auc']:.3f}"
        print(f"[Subject {held_out_subject}] "
              f"acc={result['accuracy']:.3f} "
              f"f1_weighted={result['f1_weighted']:.3f} "
              f"auc={auc_str}")

    all_true = np.array(all_true)
    all_pred_probs = np.array(all_pred_probs)
    all_pred_classes = (all_pred_probs >= 0.5).astype(int)

    overall = {
        'accuracy': accuracy_score(all_true, all_pred_classes),
        'f1_weighted': f1_score(all_true, all_pred_classes, average='weighted', zero_division=0),
        'f1_class0': f1_score(all_true, all_pred_classes, pos_label=0, zero_division=0),
        'f1_class1': f1_score(all_true, all_pred_classes, pos_label=1, zero_division=0),
        'auc': roc_auc_score(all_true, all_pred_probs),
    }

    return {'per_subject': per_subject_results, 'overall': overall}


if __name__ == '__main__':
    # Sanity check with synthetic data standing in for real WESAD/PPGE windows,
    # so the harness itself is validated before any real dataset is plugged in.
    print("--- LOSO harness sanity check (synthetic data, 4 fake subjects) ---\n")

    n_subjects = 4
    windows_per_subject = 20
    window_length = 3840

    rng = np.random.default_rng(42)
    windows = rng.standard_normal((n_subjects * windows_per_subject, window_length, 1)).astype('float32')
    labels = rng.integers(0, 2, size=(n_subjects * windows_per_subject,))
    subject_ids = np.repeat(np.arange(n_subjects), windows_per_subject)

    # Use a tiny epoch count here purely to keep the sanity check fast --
    # real runs will use the full 350/patience-80 config above.
    results = run_loso_cv(windows, labels, subject_ids, epochs=2, batch_size=16, early_stopping_patience=2)

    print("\n--- Overall (aggregated across all held-out subjects) ---")
    for k, v in results['overall'].items():
        print(f"{k}: {v:.4f}")

    print("\nHarness runs end-to-end. Ready to plug in real preprocessed windows.")
