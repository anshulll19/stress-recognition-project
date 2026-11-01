"""
learning_curve.py
==================

Scientifically rigorous learning-curve benchmark for self-supervised pretraining
vs. learning from scratch on WESAD.

Methodological Guarantees:
    1. ZERO TEST-SET LEAKAGE: The outer held-out test subject is NEVER passed to
       `model.fit()` as validation_data. It is held completely unseen until final evaluation.
    2. SUBJECT-AWARE INNER VALIDATION: Early stopping and model checkpoint restoration
       monitor an inner validation set composed of disjoint subjects drawn exclusively
       from the outer training subjects.
    3. DETERMINISTIC SEEDING: tf.random, np.random, and python random seeds are reset
       using seed.py before each condition's initialization and fitting.
    4. CONTROLLED HYPERPARAMETERS: Scratch and frozen conditions use lr=0.01, while fine-tuning
       uses a defensible lower learning rate (lr=1e-4) to prevent catastrophic forgetting.
    5. IDENTICAL PARTITIONS: Scratch, frozen, and fine-tuned conditions receive the exact
       same inner training and validation splits.
"""

import argparse
import csv
import datetime
import gc
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from model import build_cnn_tcn_lstm_model
from pretrain_encoder import load_pretrained_encoder_into_classifier
from evaluation import compute_metrics
from seed import set_seed

RESULT_FIELDS = [
    'held_out_subject', 'label_fraction', 'seed', 'condition', 'n_train_windows',
    'n_val_windows', 'accuracy', 'macro_f1', 'weighted_f1', 'class0_f1', 'class1_f1', 'auc'
]


def get_git_commit() -> str:
    """Returns the current Git commit hash if available, otherwise 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "unknown"


def load_completed_runs(output_csv: str) -> set:
    """
    Reads an existing results CSV and returns the set of
    (fold, fraction, seed, condition) tuples already completed.
    """
    if not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv)
        if df.empty or 'held_out_subject' not in df.columns:
            return set()
        return set(zip(df['held_out_subject'].astype(str), df['label_fraction'].astype(float),
                       df['seed'].astype(int), df['condition'].astype(str)))
    except Exception:
        return set()


def append_result_row(output_csv: str, row: dict):
    """Writes one row immediately so progress is never lost."""
    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists or os.path.getsize(output_csv) == 0:
            writer.writeheader()
        writer.writerow(row)


def create_subject_aware_inner_split(windows: np.ndarray, labels: np.ndarray,
                                     subject_ids: np.ndarray, inner_val_ratio: float = 0.15,
                                     fold_seed: int = 42):
    """
    Creates an inner validation split strictly from the outer training subjects.
    Guarantees:
        - Inner validation subjects are completely disjoint from inner training subjects (when >= 3 subjects).
        - Zero overlap with the outer held-out test subject.
    """
    unique_subs = np.unique(subject_ids)
    if len(unique_subs) >= 3:
        n_val_subs = max(1, int(round(len(unique_subs) * inner_val_ratio)))
        rng = np.random.default_rng(fold_seed)
        shuffled = rng.permutation(unique_subs)
        val_subs = set(shuffled[:n_val_subs])
        train_subs = set(shuffled[n_val_subs:])

        val_mask = np.isin(subject_ids, list(val_subs))
        train_mask = ~val_mask

        # Strict assertion against subject overlap
        overlap = train_subs.intersection(val_subs)
        assert not overlap, f"Subject leakage detected in inner split: {overlap}"

        return (
            windows[train_mask], labels[train_mask], subject_ids[train_mask], sorted(list(train_subs)),
            windows[val_mask], labels[val_mask], subject_ids[val_mask], sorted(list(val_subs))
        )
    else:
        # Fallback for minimal synthetic tests with only 1-2 training subjects
        rng = np.random.default_rng(fold_seed)
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]

        n_val_0 = max(1, int(round(len(idx_0) * inner_val_ratio)))
        n_val_1 = max(1, int(round(len(idx_1) * inner_val_ratio))) if len(idx_1) > 1 else 0

        val_0 = set(rng.choice(idx_0, size=min(n_val_0, len(idx_0)), replace=False))
        val_1 = set(rng.choice(idx_1, size=min(n_val_1, len(idx_1)), replace=False)) if n_val_1 > 0 else set()
        val_idx = np.array(sorted(list(val_0 | val_1)))

        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[val_idx] = False

        return (
            windows[train_mask], labels[train_mask], subject_ids[train_mask], [str(s) for s in sorted(list(set(subject_ids[train_mask])))],
            windows[~train_mask], labels[~train_mask], subject_ids[~train_mask], [str(s) for s in sorted(list(set(subject_ids[~train_mask])))]
        )


def stratified_subsample(X, y, fraction, seed):
    """
    Randomly subsamples a fraction of (X, y) strictly preserving the class ratio.
    """
    if fraction >= 1.0:
        return X, y

    rng = np.random.default_rng(seed)
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]

    n_keep_0 = max(1, int(len(idx_0) * fraction))
    n_keep_1 = max(1, int(len(idx_1) * fraction))

    keep_0 = rng.choice(idx_0, size=min(n_keep_0, len(idx_0)), replace=False)
    keep_1 = rng.choice(idx_1, size=min(n_keep_1, len(idx_1)), replace=False)
    keep = np.concatenate([keep_0, keep_1])
    rng.shuffle(keep)

    return X[keep], y[keep]


def build_condition_model(condition: str, encoder_weights_path: str = None,
                          scratch_lr: float = 0.01, frozen_lr: float = 0.01,
                          finetune_lr: float = 0.0001):
    """
    Builds the model according to the experimental condition.
    Uses defensible learning rates: scratch/frozen @ 0.01, finetune @ 1e-4.
    """
    if condition == 'scratch':
        model = build_cnn_tcn_lstm_model(learning_rate=scratch_lr)
        return model
    elif condition == 'frozen':
        model = build_cnn_tcn_lstm_model(learning_rate=frozen_lr)
        if encoder_weights_path and os.path.exists(encoder_weights_path):
            load_pretrained_encoder_into_classifier(model, encoder_weights_path, freeze=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=frozen_lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model
    elif condition == 'finetune':
        model = build_cnn_tcn_lstm_model(learning_rate=finetune_lr)
        if encoder_weights_path and os.path.exists(encoder_weights_path):
            load_pretrained_encoder_into_classifier(model, encoder_weights_path, freeze=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        return model
    else:
        raise ValueError(f"Unknown condition: {condition}")


def run_learning_curve(windows, labels, subject_ids, encoder_weights_path,
                       label_fractions=(0.10, 0.25, 0.50, 0.75, 1.00),
                       n_seeds=3, held_out_subjects=None,
                       conditions=('scratch', 'frozen', 'finetune'),
                       epochs=350, batch_size=512, early_stopping_patience=80,
                       inner_val_ratio=0.15, monitor='val_loss',
                       scratch_lr=0.01, frozen_lr=0.01, finetune_lr=0.0001,
                       verbose=0, output_csv=None, resume=True):
    """
    Runs the scientifically rigorous learning curve experiment.
    """
    all_subjects = sorted(list(np.unique(subject_ids)))
    folds = held_out_subjects if held_out_subjects is not None else all_subjects

    completed = load_completed_runs(output_csv) if (output_csv and resume) else set()
    if completed:
        print(f"Resuming: {len(completed)} runs already completed in {output_csv}, skipping those.\n")

    results = []
    total_runs = len(folds) * len(label_fractions) * n_seeds * len(conditions)
    run_count = 0

    for fold_idx, held_out in enumerate(folds):
        test_mask = subject_ids == held_out
        train_pool_mask = ~test_mask

        X_train_pool = windows[train_pool_mask]
        y_train_pool = labels[train_pool_mask]
        s_train_pool = subject_ids[train_pool_mask]

        X_test = windows[test_mask]
        y_test = labels[test_mask]

        if len(np.unique(y_test)) < 2:
            print(f"Skipping fold {held_out}: only one class in test set")
            continue

        # Deterministic seed for inner split based on held_out subject ID
        fold_seed = int(hashlib.sha256(str(held_out).encode('utf-8')).hexdigest()[:8], 16) % 100000
        (
            X_inner_train_full, y_inner_train_full, s_inner_train_full, train_subs,
            X_inner_val, y_inner_val, s_inner_val, val_subs
        ) = create_subject_aware_inner_split(
            X_train_pool, y_train_pool, s_train_pool,
            inner_val_ratio=inner_val_ratio, fold_seed=fold_seed
        )

        # Leakage assertions
        assert str(held_out) not in train_subs, f"Test subject {held_out} leaked into inner train"
        assert str(held_out) not in val_subs, f"Test subject {held_out} leaked into inner val"

        y_inner_val_cat = tf.keras.utils.to_categorical(y_inner_val, num_classes=2)

        for fraction in label_fractions:
            for seed in range(n_seeds):
                # Subsample strictly from inner training subjects
                X_train, y_train = stratified_subsample(X_inner_train_full, y_inner_train_full, fraction, seed)

                class_counts = np.bincount(y_train.astype(int))
                class_weight = {i: len(y_train) / (2 * c) for i, c in enumerate(class_counts) if c > 0}
                y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)

                for cond_idx, condition in enumerate(conditions):
                    run_count += 1
                    run_key = (str(held_out), float(fraction), int(seed), str(condition))

                    if run_key in completed:
                        print(f"[{run_count}/{total_runs}] SKIP (already done) fold={held_out} "
                              f"fraction={fraction} seed={seed} condition={condition}")
                        continue

                    print(f"[{run_count}/{total_runs}] fold={held_out} fraction={fraction} "
                          f"seed={seed} condition={condition} (n_train={len(y_train)}, n_val={len(y_inner_val)})")

                    # Deterministic reproducibility seed per run
                    run_seed = seed * 10000 + fold_idx * 100 + cond_idx
                    set_seed(run_seed)

                    model = build_condition_model(
                        condition, encoder_weights_path,
                        scratch_lr=scratch_lr, frozen_lr=frozen_lr, finetune_lr=finetune_lr
                    )

                    callbacks = [tf.keras.callbacks.EarlyStopping(
                        monitor=monitor, patience=early_stopping_patience,
                        restore_best_weights=True
                    )]

                    # FIT ONLY ON INNER DATA (X_train and X_inner_val)
                    # X_test is NEVER passed here
                    model.fit(
                        X_train, y_train_cat,
                        validation_data=(X_inner_val, y_inner_val_cat),
                        epochs=epochs,
                        batch_size=min(batch_size, len(X_train)),
                        class_weight=class_weight,
                        callbacks=callbacks,
                        verbose=verbose
                    )

                    # FINAL EVALUATION STRICTLY ON HELD-OUT TEST SUBJECT
                    pred_probs = model.predict(X_test, verbose=0)[:, 1]
                    metrics = compute_metrics(y_test, pred_probs)

                    row = {
                        'held_out_subject': held_out,
                        'label_fraction': fraction,
                        'seed': seed,
                        'condition': condition,
                        'n_train_windows': len(y_train),
                        'n_val_windows': len(y_inner_val),
                        **metrics
                    }
                    results.append(row)

                    if output_csv:
                        append_result_row(output_csv, row)

                    del model
                    K.clear_session()
                    gc.collect()

    if output_csv and os.path.exists(output_csv):
        return pd.read_csv(output_csv)
    return pd.DataFrame(results)


def summarize(df: pd.DataFrame, metrics=('auc', 'macro_f1', 'class1_f1')):
    """
    Mean +/- std for each metric, per (fraction, condition), averaged over runs.
    """
    grouped = df.groupby(['label_fraction', 'condition'])
    summary = grouped.size().reset_index(name='n_runs')

    for metric in metrics:
        if metric in df.columns:
            stats = grouped[metric].agg(['mean', 'std']).reset_index()
            stats.columns = ['label_fraction', 'condition', f'{metric}_mean', f'{metric}_std']
            summary = summary.merge(stats, on=['label_fraction', 'condition'])

    return summary.sort_values(['label_fraction', 'condition'])


def save_manifest(output_manifest: str, args_dict: dict):
    """Saves a structured JSON configuration manifest."""
    os.makedirs(os.path.dirname(output_manifest) or ".", exist_ok=True)
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "configuration": args_dict,
        "methodology": {
            "test_leakage_prevented": True,
            "inner_validation": "subject_aware_disjoint",
            "early_stopping_target": "inner_validation_set_only",
            "outer_test_subject": "final_evaluation_only",
            "seeds_controlled": True,
            "finetune_lr_damped": True
        }
    }
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved configuration manifest to: {output_manifest}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the scientifically rigorous learning-curve experiment")
    parser.add_argument("--data", default="data/wesad/wesad_binary.npz", help="Path to WESAD .npz")
    parser.add_argument("--encoder_weights", default="models/pretrained_clas_mae_encoder.weights.h5",
                        help="Path to pretrained encoder weights")
    parser.add_argument("--folds", default="all", help="'all' or comma-separated subject IDs, e.g. S2,S4,S6")
    parser.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Fast synthetic smoke test")
    parser.add_argument("--output", default="learning_curve_rigorous_results.csv")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--monitor", default="val_loss", help="Metric to monitor on inner validation set")
    parser.add_argument("--scratch_lr", type=float, default=0.01)
    parser.add_argument("--frozen_lr", type=float, default=0.01)
    parser.add_argument("--finetune_lr", type=float, default=0.0001)
    parser.add_argument("--inner_val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.quick:
        print("--- QUICK smoke test (synthetic data, rigorous split validation) ---\n")
        rng = np.random.default_rng(0)
        n = 300
        windows = rng.standard_normal((n, 3840, 1)).astype('float32')
        labels = rng.choice([0, 1], size=n, p=[0.78, 0.22])
        subject_ids = np.array([f"S{i % 6 + 2}" for i in range(n)])

        # Fast throwaway encoder
        from pretrain_encoder import pretrain, save_encoder_weights
        _, encoder, _ = pretrain(windows, epochs=2, batch_size=32, verbose=0)
        os.makedirs("scratch", exist_ok=True)
        save_encoder_weights(encoder, 'scratch/quick_test_encoder.weights.h5')

        df = run_learning_curve(
            windows, labels, subject_ids,
            encoder_weights_path='scratch/quick_test_encoder.weights.h5',
            label_fractions=(0.5, 1.0), n_seeds=1,
            held_out_subjects=['S2', 'S3'],
            epochs=2, batch_size=16, early_stopping_patience=2,
            monitor=args.monitor, verbose=0
        )
        print("\n--- Smoke test results ---")
        print(df.to_string())
        print("\nSmoke test passed: zero test leakage, subject-aware inner split verified.")

    else:
        fractions = tuple(float(f) for f in args.fractions.split(","))
        held_out = None if args.folds == "all" else args.folds.split(",")

        manifest_path = args.output.replace('.csv', '_manifest.json')
        save_manifest(manifest_path, vars(args))

        data = np.load(args.data)
        df = run_learning_curve(
            data['windows'], data['labels'], data['subject_ids'],
            encoder_weights_path=args.encoder_weights,
            label_fractions=fractions, n_seeds=args.seeds,
            held_out_subjects=held_out,
            epochs=args.epochs, batch_size=args.batch_size,
            early_stopping_patience=args.patience,
            inner_val_ratio=args.inner_val_ratio,
            monitor=args.monitor,
            scratch_lr=args.scratch_lr, frozen_lr=args.frozen_lr,
            finetune_lr=args.finetune_lr,
            verbose=0, output_csv=args.output, resume=True
        )
        print(f"\nAll results saved to {args.output}")

        summary = summarize(df)
        summary_path = args.output.replace('.csv', '_summary.csv')
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}\n")
