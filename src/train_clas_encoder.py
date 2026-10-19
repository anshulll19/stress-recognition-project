"""
train_clas_encoder.py
=====================

Self-supervised masked autoencoder pretraining using CLAS windows ONLY.
Guarantees 100% isolation from WESAD:
    - Zero WESAD subjects or waveforms are included in pretraining.
    - Zero labels are used (neither CLAS nor WESAD). Raw PPG waveforms only.
    - Validation split is subject-aware across CLAS subjects (no subject overlap).
    - Checkpoint metadata (epochs, batch size, seed, loss curves, git commit)
      is saved alongside model weights as a structured JSON artifact.
"""

import argparse
import datetime
import json
import os
import subprocess
import numpy as np
import tensorflow as tf

from model import build_cnn_branch, build_tcn_branch
from pretrain_encoder import build_pretrain_autoencoder, apply_random_mask, MASK_FRACTION, WINDOW_LENGTH
from seed import set_seed


def get_git_commit() -> str:
    """Returns the current Git commit hash if available, otherwise 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def load_clas_data(data_path: str = "data/clas/clas_binary.npz"):
    """
    Loads CLAS windows and subject IDs.
    Labels are explicitly ignored / not loaded to enforce self-supervised integrity.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CLAS data file not found at: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    windows = data["windows"].astype("float32")
    subject_ids = data["subject_ids"]

    assert windows.ndim == 3 and windows.shape[1] == WINDOW_LENGTH and windows.shape[2] == 1, (
        f"Expected windows of shape (N, {WINDOW_LENGTH}, 1), got {windows.shape}"
    )
    assert len(subject_ids) == len(windows), "Mismatch between subject_ids and windows count"

    return windows, subject_ids


def subject_aware_pretrain_split(windows: np.ndarray, subject_ids: np.ndarray,
                                 val_fraction: float = 0.10, seed: int = 42):
    """
    Partitions CLAS windows into train and validation sets based on subject IDs.
    Guarantees that train and validation sets contain completely disjoint subjects.
    """
    unique_subjects = np.unique(subject_ids)
    n_val_subs = max(1, int(round(len(unique_subjects) * val_fraction)))

    rng = np.random.default_rng(seed)
    shuffled_subs = rng.permutation(unique_subjects)
    val_subs = set(shuffled_subs[:n_val_subs])
    train_subs = set(shuffled_subs[n_val_subs:])

    val_mask = np.isin(subject_ids, list(val_subs))
    train_mask = ~val_mask

    # Strict leakage check
    overlap = train_subs.intersection(val_subs)
    assert not overlap, f"Subject leakage in pretraining split: {overlap}"

    return (
        windows[train_mask], subject_ids[train_mask], sorted(list(train_subs)),
        windows[val_mask], subject_ids[val_mask], sorted(list(val_subs))
    )


def train_clas_encoder(data_path: str = "data/clas/clas_binary.npz",
                       out_weights: str = "models/pretrained_clas_mae_encoder.weights.h5",
                       out_meta: str = "models/pretrained_clas_mae_encoder.json",
                       epochs: int = 100,
                       batch_size: int = 64,
                       learning_rate: float = 0.001,
                       mask_fraction: float = MASK_FRACTION,
                       early_stopping_patience: int = 15,
                       val_fraction: float = 0.10,
                       seed: int = 42,
                       verbose: int = 1):
    """
    Trains the masked autoencoder on CLAS PPG windows and saves weights and metadata.
    """
    set_seed(seed)
    os.makedirs(os.path.dirname(out_weights) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_meta) or ".", exist_ok=True)

    print("=" * 80)
    print("CLAS-ONLY MASKED AUTOENCODER PRETRAINING (RQ3 / RQ5)")
    print(f"Data source: {data_path} (No WESAD data used)")
    print(f"Seed: {seed} | Epochs: {epochs} | Batch size: {batch_size} | LR: {learning_rate}")
    print("=" * 80)

    # 1. Load data
    windows, subject_ids = load_clas_data(data_path)
    print(f"Loaded {len(windows)} raw CLAS windows across {len(np.unique(subject_ids))} subjects.")

    # 2. Subject-aware train/val split
    (
        X_train_target, s_train, train_subs,
        X_val_target, s_val, val_subs
    ) = subject_aware_pretrain_split(windows, subject_ids, val_fraction=val_fraction, seed=seed)

    print(f"Split: {len(X_train_target)} train windows ({len(train_subs)} subjects), "
          f"{len(X_val_target)} val windows ({len(val_subs)} subjects).")

    # 3. Apply contiguous masking (targets are original unmasked windows)
    X_train_input = apply_random_mask(X_train_target, mask_fraction=mask_fraction, seed=seed + 1)
    X_val_input = apply_random_mask(X_val_target, mask_fraction=mask_fraction, seed=seed + 2)

    # 4. Build models
    autoencoder, encoder = build_pretrain_autoencoder(learning_rate=learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True
        )
    ]

    # 5. Train autoencoder
    start_time = datetime.datetime.now()
    history = autoencoder.fit(
        X_train_input, X_train_target,
        validation_data=(X_val_input, X_val_target),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    end_time = datetime.datetime.now()

    # 6. Save encoder weights
    encoder.save_weights(out_weights)
    print(f"\nSaved trained encoder weights to: {out_weights}")

    # 7. Collect and save structured provenance metadata
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_val_loss = float(np.min(history.history["val_loss"]))
    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])

    metadata = {
        "experiment": "CLAS_only_MAE_Pretraining",
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "git_commit": get_git_commit(),
        "data_source": data_path,
        "n_total_windows": len(windows),
        "n_train_windows": len(X_train_target),
        "n_val_windows": len(X_val_target),
        "n_train_subjects": len(train_subs),
        "n_val_subjects": len(val_subs),
        "train_subjects": [str(s) for s in train_subs],
        "val_subjects": [str(s) for s in val_subs],
        "wesad_waveforms_used": False,
        "labels_used": False,
        "model_architecture": "CNN_TCN_Encoder_Deconv_Decoder",
        "hyperparameters": {
            "epochs_requested": epochs,
            "epochs_trained": len(history.history["loss"]),
            "best_epoch": best_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "mask_fraction": mask_fraction,
            "early_stopping_patience": early_stopping_patience,
            "random_seed": seed,
            "val_fraction": val_fraction
        },
        "preprocessing_contract": {
            "sampling_rate_hz": 64,
            "window_seconds": 60,
            "window_samples": WINDOW_LENGTH,
            "normalization": "per_window_zscore",
            "bandpass_hz": [0.7, 3.7],
            "bandpass_order": 3
        },
        "results": {
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "loss_history": [float(x) for x in history.history["loss"]],
            "val_loss_history": [float(x) for x in history.history["val_loss"]]
        }
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved complete provenance metadata to: {out_meta}")

    return autoencoder, encoder, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain masked autoencoder on CLAS windows only")
    parser.add_argument("--data", default="data/clas/clas_binary.npz", help="Path to CLAS .npz")
    parser.add_argument("--out_weights", default="models/pretrained_clas_mae_encoder.weights.h5")
    parser.add_argument("--out_meta", default="models/pretrained_clas_mae_encoder.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Fast smoke test on synthetic data")
    args = parser.parse_args()

    if args.quick:
        print("--- Running fast CLAS-pretrain smoke test (synthetic data) ---")
        rng = np.random.default_rng(0)
        dummy_windows = rng.standard_normal((60, WINDOW_LENGTH, 1)).astype("float32")
        dummy_subs = np.array([f"Part{i % 6 + 1}" for i in range(60)])

        # Save temporary synthetic test npz
        temp_npz = "scratch/temp_quick_clas.npz"
        os.makedirs("scratch", exist_ok=True)
        np.savez_compressed(temp_npz, windows=dummy_windows, subject_ids=dummy_subs)

        train_clas_encoder(
            data_path=temp_npz,
            out_weights="scratch/quick_clas_encoder.weights.h5",
            out_meta="scratch/quick_clas_encoder.json",
            epochs=2,
            batch_size=16,
            early_stopping_patience=2,
            seed=42,
            verbose=0
        )
        print("Smoke test completed successfully.")
    else:
        train_clas_encoder(
            data_path=args.data,
            out_weights=args.out_weights,
            out_meta=args.out_meta,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            early_stopping_patience=args.patience,
            seed=args.seed
        )
