"""Train a separate deployment candidate with subject-level validation.

This does not modify or replace the corrected LOSO evaluation.
The resulting checkpoint is a deployment candidate, not a new
independent performance estimate.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from model import build_cnn_tcn_lstm_model


DATA_PATH = Path("data/wesad/wesad_binary.npz")
OUTPUT_DIR = Path("models/deployment_candidate")
SEED = 42
MAX_EPOCHS = 350
BATCH_SIZE = 512
PATIENCE = 80


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with np.load(DATA_PATH, allow_pickle=True) as data:
        X = data["windows"].astype(np.float32)
        y = data["labels"].astype(np.int64)
        subjects = data["subject_ids"].astype(str)

    if X.ndim != 3 or X.shape[1:] != (3840, 1):
        raise ValueError(f"Expected windows shaped (N, 3840, 1), got {X.shape}")
    if y.ndim != 1 or subjects.ndim != 1:
        raise ValueError("Labels and subject IDs must be one-dimensional.")
    if len(X) != len(y) or len(X) != len(subjects):
        raise ValueError("Windows, labels and subject IDs have different lengths.")
    if not np.isfinite(X).all():
        raise ValueError("Input windows contain NaN or infinite values.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, got {np.unique(y)}")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=SEED
    )
    train_idx, val_idx = next(splitter.split(X, y, groups=subjects))

    train_subjects = sorted(np.unique(subjects[train_idx]).tolist())
    val_subjects = sorted(np.unique(subjects[val_idx]).tolist())

    if set(train_subjects) & set(val_subjects):
        raise RuntimeError("Subject leakage detected between train and validation.")
    if len(np.unique(y[train_idx])) != 2:
        raise ValueError("Training partition does not contain both classes.")
    if len(np.unique(y[val_idx])) != 2:
        raise ValueError("Validation partition does not contain both classes.")

    counts = np.bincount(y[train_idx], minlength=2)
    class_weights = {
        i: len(train_idx) / (2 * counts[i])
        for i in range(2)
        if counts[i] > 0
    }

    y_train = tf.keras.utils.to_categorical(y[train_idx], num_classes=2)
    y_val = tf.keras.utils.to_categorical(y[val_idx], num_classes=2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "classifier.keras"
    manifest_path = OUTPUT_DIR / "manifest.json"

    model = build_cnn_tcn_lstm_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
    ]

    print("Training deployment candidate")
    print("Dataset:", DATA_PATH)
    print("Windows:", len(X), "| subjects:", len(np.unique(subjects)))
    print("Training windows/subjects:", len(train_idx), len(train_subjects))
    print("Validation windows/subjects:", len(val_idx), len(val_subjects))
    print("Training class counts:", counts.tolist())
    print("Checkpoint:", checkpoint_path)

    history = model.fit(
        X[train_idx],
        y_train,
        validation_data=(X[val_idx], y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    # Reload the saved artifact to verify that it is loadable.
    saved_model = tf.keras.models.load_model(checkpoint_path)
    sample = saved_model.predict(X[val_idx[:min(4, len(val_idx))]], verbose=0)

    if sample.shape != (min(4, len(val_idx)), 2):
        raise RuntimeError(f"Unexpected prediction shape: {sample.shape}")
    if not np.isfinite(sample).all():
        raise RuntimeError("Checkpoint produced non-finite predictions.")
    if not np.allclose(sample.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Classifier outputs do not sum to 1.")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATA_PATH),
        "checkpoint": str(checkpoint_path),
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping_patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_validation_loss": float(min(history.history["val_loss"])),
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "train_windows": int(len(train_idx)),
        "validation_windows": int(len(val_idx)),
        "train_class_counts": counts.tolist(),
        "class_weights": class_weights,
        "input_shape": [3840, 1],
        "output_classes": {"0": "non-stress", "1": "stress"},
        "validation_note": (
            "Subject-level validation used for checkpoint selection. "
            "This score is not an independent test result and must not "
            "be reported as the corrected LOSO performance."
        ),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("\nCheckpoint reload and prediction smoke test passed.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", manifest["best_validation_loss"])
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
