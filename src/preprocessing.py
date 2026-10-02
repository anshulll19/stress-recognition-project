"""
preprocessing.py
=================

Shared, dataset-agnostic preprocessing primitives. Every dataset-specific
preprocessing script (wesad_preprocess.py, a future clas_preprocess.py,
ppge_preprocess.py, etc.) should import from here rather than
re-implementing filtering/windowing/normalization, so the pipeline is
guaranteed identical across datasets.

Locked configuration (per the research proposal / handoff doc —
do not casually change when comparing experiments):
    - Butterworth bandpass: 0.7-3.7 Hz
    - Window length: 60s
    - Window stride: 5s (55s step between window starts)
    - Target sampling rate: 64 Hz (so window length = 3840 samples)
    - Per-window z-score normalization
"""

import numpy as np
from scipy.signal import butter, filtfilt, resample

# ---- Locked constants -------------------------------------------------
TARGET_SAMPLING_RATE_HZ = 64
WINDOW_SECONDS = 60
STRIDE_SECONDS = 5
WINDOW_LENGTH_SAMPLES = TARGET_SAMPLING_RATE_HZ * WINDOW_SECONDS      # 3840
STRIDE_SAMPLES = TARGET_SAMPLING_RATE_HZ * STRIDE_SECONDS            # 320... 
# NOTE: stride in the original WESAD script was computed as
# (window - overlap) = 60 - 5 = 55s step between window STARTS, i.e.
# STEP_SECONDS = WINDOW_SECONDS - STRIDE_SECONDS = 55s -> 3520 samples.
# Keeping both named explicitly to avoid the classic off-by-definition bug.
STEP_SECONDS = WINDOW_SECONDS - STRIDE_SECONDS                        # 55
STEP_SAMPLES = TARGET_SAMPLING_RATE_HZ * STEP_SECONDS                 # 3520

BANDPASS_LOW_HZ = 0.7
BANDPASS_HIGH_HZ = 3.7
BANDPASS_ORDER = 3


def resample_to_target_rate(signal: np.ndarray, original_rate_hz: float,
                             target_rate_hz: int = TARGET_SAMPLING_RATE_HZ) -> np.ndarray:
    """
    Resample a 1D signal from its native rate to the common target rate.
    Needed whenever a dataset's native sampling rate != 64Hz
    (e.g. PPGE at 100Hz, CLAS at 256Hz).
    """
    if original_rate_hz == target_rate_hz:
        return signal
    n_target_samples = int(round(len(signal) * target_rate_hz / original_rate_hz))
    return resample(signal, n_target_samples)


def bandpass_filter(signal: np.ndarray, fs: int = TARGET_SAMPLING_RATE_HZ,
                     low_hz: float = BANDPASS_LOW_HZ, high_hz: float = BANDPASS_HIGH_HZ,
                     order: int = BANDPASS_ORDER) -> np.ndarray:
    """
    Butterworth bandpass filter, applied to the FULL continuous signal
    BEFORE windowing (not per-window) to avoid edge artifacts at every
    window boundary.
    """
    nyq = fs / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def sliding_windows(signal: np.ndarray, window_length: int = WINDOW_LENGTH_SAMPLES,
                     step: int = STEP_SAMPLES):
    """
    Yields (start_idx, end_idx) pairs for each window over a 1D signal.
    Does not slice the array itself -- caller slices, so this works
    identically whether applied to the signal or an aligned label array.
    """
    n = len(signal)
    start = 0
    while start + window_length <= n:
        yield start, start + window_length
        start += step


def zscore_normalize(window: np.ndarray) -> np.ndarray:
    """Per-window z-score normalization. Returns None if the window is
    flat (zero variance) -- caller should skip these (likely a sensor dropout)."""
    std = window.std()
    if std == 0:
        return None
    return (window - window.mean()) / std


def majority_label(label_window: np.ndarray):
    """
    Returns the majority-vote label for a window of per-sample labels.
    Used when a window may straddle a label transition.
    (Alternative: 'discard' rule -- see dataset-specific scripts for that variant.)
    """
    values, counts = np.unique(label_window, return_counts=True)
    return values[np.argmax(counts)]
