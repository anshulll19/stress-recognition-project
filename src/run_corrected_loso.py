import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from loso_harness import run_loso_cv


def main():
    parser = argparse.ArgumentParser(
        description="Run corrected subject-wise LOSO evaluation."
    )
    parser.add_argument(
        "--data",
        default="data/wesad/wesad_binary.npz",
    )
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--output_dir", default="results/loso_corrected")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(args.data, allow_pickle=True)
    windows = data["windows"]
    labels = data["labels"].astype(int)
    subject_ids = data["subject_ids"]

    print(f"Dataset: {args.data}")
    print(f"Windows: {windows.shape}")
    print(f"Subjects: {len(np.unique(subject_ids))}")
    print(f"Label counts: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"Epochs: {args.epochs}, batch size: {args.batch_size}")
    print("Validation: subject-disjoint inner split; outer test held out.")
    print("Starting LOSO evaluation...\n")

    results = run_loso_cv(
        windows,
        labels,
        subject_ids,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        verbose=args.verbose,
    )

    per_subject_path = os.path.join(
        args.output_dir, "loso_corrected_per_subject.csv"
    )
    summary_path = os.path.join(
        args.output_dir, "loso_corrected_summary.json"
    )
    manifest_path = os.path.join(
        args.output_dir, "loso_corrected_manifest.json"
    )

    pd.DataFrame(results["per_subject"]).to_csv(
        per_subject_path, index=False
    )

    summary = {
        "overall": results["overall"],
        "n_subjects": len(results["per_subject"]),
        "per_subject_results_file": per_subject_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": args.data,
        "windows_shape": list(windows.shape),
        "n_subjects": len(np.unique(subject_ids)),
        "epochs_max": args.epochs,
        "batch_size": args.batch_size,
        "early_stopping_patience": args.patience,
        "early_stopping_monitor": "val_loss",
        "validation": "GroupShuffleSplit on outer-training subjects only",
        "outer_test_subject_used_for_early_stopping": False,
        "class_weights_computed_from": "inner training partition only",
        "per_subject_file": per_subject_path,
        "summary_file": summary_path,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n--- Overall corrected LOSO metrics ---")
    for key, value in results["overall"].items():
        print(f"{key}: {value:.4f}")

    print("\nSaved:")
    print(per_subject_path)
    print(summary_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
