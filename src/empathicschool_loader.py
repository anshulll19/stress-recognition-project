"""
empathicschool_loader.py
========================

Loader skeleton and contract validator for the EmpathicSchool dataset.

STATUS:
- Access request has been submitted on Zenodo (STATUS: PENDING).
- Physical dataset files are not yet present in this repository.
- NO raw-file parsing logic is implemented here to avoid fabricating directory
  structures, column headers, sampling rates, or video formats.
- Once physical files are downloaded and inspected with `scripts/inspect_empathicschool.py`,
  the concrete parsing and windowing logic will be implemented.

CONTRACT SPECIFICATION:
Unimodal Physiology (.npz):
    - 'windows':     (N, 3840, 1) float32 (60s @ 64Hz, z-score normalized)
    - 'labels':      (N,) int in {0, 1}
    - 'subject_ids': (N,) string / object

Multimodal (.npz):
    - 'physio_windows': (N, 3840, 1) float32
    - 'video_features': (N, ...) float32 aligned to the identical 60s windows
    - 'labels':         (N,) int in {0, 1}
    - 'subject_ids':    (N,) string / object
    - 'session_ids':    (N,) string / object (optional metadata)
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from preprocessing import WINDOW_LENGTH_SAMPLES


def validate_physio_contract(
    windows: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    source: str = "<empathicschool_physio>",
):
    """
    Validates that preprocessed physiological arrays strictly adhere to the project contract.
    """
    errors = []

    if windows.ndim != 3 or windows.shape[1] != WINDOW_LENGTH_SAMPLES or windows.shape[2] != 1:
        errors.append(
            f"windows must have shape (N, {WINDOW_LENGTH_SAMPLES}, 1), got {windows.shape}"
        )
    if labels.ndim != 1:
        errors.append(f"labels must be 1D, got shape {labels.shape}")
    if subject_ids.ndim != 1:
        errors.append(f"subject_ids must be 1D, got shape {subject_ids.shape}")

    n_windows = len(windows) if windows.ndim >= 1 else 0
    if len(labels) != n_windows:
        errors.append(f"len(labels) ({len(labels)}) != len(windows) ({n_windows})")
    if len(subject_ids) != n_windows:
        errors.append(f"len(subject_ids) ({len(subject_ids)}) != len(windows) ({n_windows})")

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        errors.append(f"labels must only contain {{0, 1}}, got {unique_labels}")

    if errors:
        raise ValueError(
            f"EmpathicSchool physio contract violation for '{source}':\n  - "
            + "\n  - ".join(errors)
        )


def validate_multimodal_contract(
    physio_windows: np.ndarray,
    video_features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    source: str = "<empathicschool_multimodal>",
):
    """
    Validates paired physiological windows and video representations.
    Ensures strict sample-count and temporal-alignment consistency.
    """
    errors = []
    # Check physio basics
    try:
        validate_physio_contract(physio_windows, labels, subject_ids, source=source)
    except ValueError as e:
        errors.append(str(e))

    # Check video alignment
    n_windows = len(physio_windows)
    if video_features.ndim < 2:
        errors.append(f"video_features must have at least 2 dimensions (N, ...), got shape {video_features.shape}")
    elif len(video_features) != n_windows:
        errors.append(
            f"len(video_features) ({len(video_features)}) != len(physio_windows) ({n_windows}). "
            "Physiology and video must be aligned 1-to-1 per window."
        )

    if errors:
        raise ValueError(
            f"EmpathicSchool multimodal contract violation for '{source}':\n  - "
            + "\n  - ".join(errors)
        )


def preprocess_empathicschool_raw(raw_dir: str, output_path: str):
    """
    PLACEHOLDER: Raw file extraction and preprocessing.

    TODO (Upon Zenodo access approval and inspection):
    1. Parse subject folders and metadata.
    2. Extract physiological signals (PPG/BVP) and determine native sampling rate.
    3. Filter: Butterworth bandpass 0.7 - 3.7 Hz.
    4. Resample to locked 64 Hz.
    5. Segment into 60s windows with 5s stride (55s step).
    6. Z-score normalize each window individually.
    7. Synchronize with facial video segments / extracted video features.
    8. Export to standardized .npz format.
    """
    raise NotImplementedError(
        "EmpathicSchool access is currently pending on Zenodo. "
        "Raw dataset files are not yet available. "
        "Run `scripts/inspect_empathicschool.py <DATASET_PATH>` once files are downloaded "
        "to determine exact layout before implementing parser."
    )


def load_empathicschool_preprocessed(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Loads preprocessed EmpathicSchool data from .npz, checking contract adherence.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed EmpathicSchool file not found at: {npz_path}. "
            "Data must be downloaded and preprocessed first."
        )

    data = np.load(path, allow_pickle=True)
    if "physio_windows" in data and "video_features" in data:
        # Multimodal contract
        physio = data["physio_windows"]
        video = data["video_features"]
        labels = data["labels"]
        subjects = data["subject_ids"]
        validate_multimodal_contract(physio, video, labels, subjects, source=str(path))
        return {
            "physio_windows": physio,
            "video_features": video,
            "labels": labels,
            "subject_ids": subjects,
            "session_ids": data.get("session_ids", None),
        }
    elif "windows" in data:
        # Unimodal physio contract
        windows = data["windows"]
        labels = data["labels"]
        subjects = data["subject_ids"]
        validate_physio_contract(windows, labels, subjects, source=str(path))
        return {
            "windows": windows,
            "labels": labels,
            "subject_ids": subjects,
        }
    else:
        raise KeyError(
            f"Unrecognized keys in {npz_path}. Expected ('windows', ...) or ('physio_windows', 'video_features', ...)"
        )


if __name__ == "__main__":
    print("--- empathicschool_loader self-test (Contract Validation) ---")

    n = 20
    dummy_physio = np.zeros((n, 3840, 1), dtype=np.float32)
    dummy_labels = np.array([0, 1] * (n // 2), dtype=int)
    dummy_subjects = np.array([f"P{i // 4}" for i in range(n)])
    dummy_video = np.zeros((n, 128), dtype=np.float32)  # e.g. 128-d video feature per window

    # Test valid contracts
    validate_physio_contract(dummy_physio, dummy_labels, dummy_subjects)
    validate_multimodal_contract(dummy_physio, dummy_video, dummy_labels, dummy_subjects)
    print("Contract validator passed on compliant dummy structures.")

    # Test that invalid shape triggers error
    try:
        bad_physio = np.zeros((n, 1000, 1), dtype=np.float32)
        validate_physio_contract(bad_physio, dummy_labels, dummy_subjects)
        raise AssertionError("Validation should have failed for bad window length!")
    except ValueError as e:
        print("Contract validator correctly rejected non-compliant array.")

    # Test that NotImplementedError is cleanly raised for raw parser
    try:
        preprocess_empathicschool_raw("dummy_raw_path", "dummy_out.npz")
    except NotImplementedError:
        print("Raw parser placeholder correctly raises NotImplementedError.")

    print("\nSelf-test passed.")
