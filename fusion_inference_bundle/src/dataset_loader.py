"""
dataset_loader.py
==================

Thin, dataset-agnostic loader. Every dataset's preprocessing pipeline
(WESAD, CLAS, PPGE, ...) outputs a single .npz with the common contract:

    windows:     (N, 3840, 1)  float32
    labels:      (N,)          int, values in {0, 1}
    subject_ids: (N,)          string or int, one per window

This module ONLY validates and loads that contract -- it does not know
anything about any specific dataset's raw format. That keeps
cross_dataset.py and evaluation.py fully dataset-agnostic.
"""

import numpy as np

from preprocessing import WINDOW_LENGTH_SAMPLES

REGISTRY = {}  # populated via register_dataset(), e.g. {"wesad": "wesad_binary.npz"}


def register_dataset(name: str, npz_path: str):
    """Register a dataset name -> file path, so cross_dataset.py can be
    called with --train wesad --test clas instead of raw file paths."""
    REGISTRY[name.lower()] = npz_path


def load_dataset(name_or_path: str, validate: bool = True):
    """
    Loads a dataset either by registered name (see register_dataset)
    or by direct .npz file path.

    Returns: dict with 'windows', 'labels', 'subject_ids'
    """
    path = REGISTRY.get(name_or_path.lower(), name_or_path)
    data = np.load(path, allow_pickle=True)

    windows = data['windows']
    labels = data['labels']
    subject_ids = data['subject_ids']

    if validate:
        _validate_contract(windows, labels, subject_ids, source=path)

    return {'windows': windows, 'labels': labels, 'subject_ids': subject_ids, 'source': path}


def _validate_contract(windows, labels, subject_ids, source="<unknown>"):
    errors = []

    if windows.ndim != 3:
        errors.append(f"windows should be 3D (N, window_length, channels), got shape {windows.shape}")
    elif windows.shape[1] != WINDOW_LENGTH_SAMPLES:
        errors.append(f"windows.shape[1] should be {WINDOW_LENGTH_SAMPLES} (60s @ 64Hz), "
                       f"got {windows.shape[1]} -- did this dataset get resampled to 64Hz "
                       f"before windowing?")

    if labels.ndim != 1:
        errors.append(f"labels should be 1D, got shape {labels.shape}")

    if subject_ids.ndim != 1:
        errors.append(f"subject_ids should be 1D, got shape {subject_ids.shape}")

    n_windows = windows.shape[0] if windows.ndim >= 1 else None
    if n_windows is not None:
        if len(labels) != n_windows:
            errors.append(f"labels length ({len(labels)}) != windows count ({n_windows})")
        if len(subject_ids) != n_windows:
            errors.append(f"subject_ids length ({len(subject_ids)}) != windows count ({n_windows})")

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        errors.append(f"labels should only contain {{0, 1}}, found {unique_labels} -- "
                       f"has this dataset been through its label-mapping step yet "
                       f"(e.g. map_wesad_labels.py)?")

    if errors:
        raise ValueError(
            f"Dataset contract validation FAILED for '{source}':\n  - " + "\n  - ".join(errors)
        )


def dataset_summary(name_or_path: str) -> str:
    """Human-readable summary -- subject count, window count, class balance."""
    d = load_dataset(name_or_path)
    n_subjects = len(np.unique(d['subject_ids']))
    n_windows = len(d['labels'])
    n_stress = int((d['labels'] == 1).sum())
    pct_stress = 100 * n_stress / n_windows if n_windows else 0
    return (f"{d['source']}: {n_subjects} subjects, {n_windows} windows "
            f"({n_stress} stress / {n_windows - n_stress} non-stress, {pct_stress:.1f}% stress)")


if __name__ == '__main__':
    # Self-test with synthetic data matching the real contract
    print("--- dataset_loader self-test ---\n")
    rng = np.random.default_rng(0)
    n = 100
    windows = rng.standard_normal((n, WINDOW_LENGTH_SAMPLES, 1)).astype('float32')
    labels = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    subject_ids = np.array([f"S{i % 5 + 2}" for i in range(n)])
    np.savez('test_dataset.npz', windows=windows, labels=labels, subject_ids=subject_ids)

    register_dataset("test", "test_dataset.npz")
    print(dataset_summary("test"))

    # Also confirm validation catches a bad contract
    print("\n--- Testing validation catches bad data ---")
    try:
        bad_windows = rng.standard_normal((n, 1000, 1)).astype('float32')  # wrong length
        np.savez('test_bad.npz', windows=bad_windows, labels=labels, subject_ids=subject_ids)
        load_dataset('test_bad.npz')
        print("ERROR: validation should have failed but didn't!")
    except ValueError as e:
        print(f"Correctly caught bad contract:\n{e}")

    print("\nSelf-test passed.")
