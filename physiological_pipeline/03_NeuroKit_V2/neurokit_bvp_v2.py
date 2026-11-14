
"""
NeuroKit2 BVP processing for the UBFC-Phys experiment.

Input: raw BVP signal at 64 Hz.
Output: cleaned signal, detected peaks, and window-level features.

Uses 30-second windows with a 5-second stride.
Does not modify the existing baseline pipeline.
"""

import numpy as np
import neurokit2 as nk


BVP_SAMPLING_RATE = 64
WINDOW_SECONDS = 30
STEP_SECONDS = 5


def process_bvp(signal, sampling_rate=BVP_SAMPLING_RATE):
    """Clean BVP and detect pulse peaks."""
    signal = np.asarray(signal, dtype=float).flatten()

    if len(signal) < sampling_rate * WINDOW_SECONDS:
        raise ValueError(
            "BVP signal is shorter than one 30-second window."
        )

    if not np.isfinite(signal).all():
        raise ValueError("BVP contains NaN or infinite values.")

    if np.std(signal) == 0:
        raise ValueError("BVP signal has zero variance.")

    cleaned = nk.ppg_clean(
        signal,
        sampling_rate=sampling_rate
    )

    _, peak_info = nk.ppg_peaks(
        cleaned,
        sampling_rate=sampling_rate
    )

    peaks = np.asarray(
        peak_info["PPG_Peaks"],
        dtype=int
    )

    return cleaned, peaks


def extract_bvp_features(signal, sampling_rate=BVP_SAMPLING_RATE):
    """
    Extract features from every 30-second window.

    Windows with insufficient pulse peaks are retained.
    Their unavailable HR/IBI features are represented by NaN.
    """

    cleaned, peaks = process_bvp(signal, sampling_rate)

    window_samples = WINDOW_SECONDS * sampling_rate
    step_samples = STEP_SECONDS * sampling_rate

    rows = []

    for start in range(
        0,
        len(cleaned) - window_samples + 1,
        step_samples
    ):

        end = start + window_samples
        window = cleaned[start:end]

        local_peaks = peaks[
            (peaks >= start) & (peaks < end)
        ] - start

        intervals = np.diff(local_peaks) / sampling_rate

        valid_intervals = intervals[
            (intervals >= 0.3) & (intervals <= 2.0)
        ]

        # Always calculate signal-level features.
        row = {
            "bvp_mean": float(np.mean(window)),
            "bvp_std": float(np.std(window)),
            "bvp_median": float(np.median(window)),
            "bvp_iqr": float(
                np.percentile(window, 75)
                - np.percentile(window, 25)
            ),
            "bvp_slope_per_sample": float(
                np.polyfit(
                    np.arange(len(window)),
                    window,
                    1
                )[0]
            ),
            "bvp_range": float(np.ptp(window)),
            "bvp_peak_count": int(len(local_peaks)),
        }

        # Calculate pulse-derived features only when reliable
        # intervals are available.
        if len(local_peaks) >= 3 and len(valid_intervals) >= 2:

            heart_rates = 60.0 / valid_intervals

            row.update({
                "bvp_heart_rate_mean": float(
                    np.mean(heart_rates)
                ),
                "bvp_heart_rate_std": float(
                    np.std(heart_rates)
                ),
                "bvp_ibi_mean": float(
                    np.mean(valid_intervals)
                ),
                "bvp_ibi_std": float(
                    np.std(valid_intervals)
                ),
                "bvp_peak_quality": 1,
            })

        else:

            row.update({
                "bvp_heart_rate_mean": np.nan,
                "bvp_heart_rate_std": np.nan,
                "bvp_ibi_mean": np.nan,
                "bvp_ibi_std": np.nan,
                "bvp_peak_quality": 0,
            })

        rows.append(row)

    return rows
