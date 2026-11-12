
"""
NeuroKit2 EDA processing for UBFC-Phys Version 2.

Input:
    Raw EDA signal sampled at 4 Hz.

Windowing:
    30-second windows with 5-second stride.

Output:
    EDA statistical features, tonic/phasic features,
    event-only SCR amplitude and diagnostic signal-quality metrics.

Important:
    SCR event detection at 4 Hz requires further validation.
    SCR-derived features are retained for diagnostics, not
    automatically treated as reliable physiological measurements.
"""

import numpy as np
import neurokit2 as nk


EDA_SAMPLING_RATE = 4
WINDOW_SECONDS = 30
STEP_SECONDS = 5


def process_eda(signal, sampling_rate=EDA_SAMPLING_RATE):
    """Validate and process one complete EDA recording."""

    signal = np.asarray(signal, dtype=float).flatten()

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    if len(signal) < sampling_rate * WINDOW_SECONDS:
        raise ValueError(
            "EDA signal is shorter than one 30-second window."
        )

    if not np.isfinite(signal).all():
        raise ValueError("EDA contains NaN or infinite values.")

    if np.std(signal) == 0:
        raise ValueError("EDA signal has zero variance.")

    signals, info = nk.eda_process(
        signal,
        sampling_rate=sampling_rate
    )

    return signals, info


def extract_eda_features(signal, sampling_rate=EDA_SAMPLING_RATE):
    """Extract 30-second EDA features with a 5-second stride."""

    signal = np.asarray(signal, dtype=float).flatten()

    signals, info = process_eda(signal, sampling_rate)

    window_samples = WINDOW_SECONDS * sampling_rate
    step_samples = STEP_SECONDS * sampling_rate

    rows = []

    for start in range(
        0,
        len(signals) - window_samples + 1,
        step_samples
    ):

        end = start + window_samples

        raw = signal[start:end]
        window = signals.iloc[start:end]

        clean = window["EDA_Clean"].to_numpy(dtype=float)
        tonic = window["EDA_Tonic"].to_numpy(dtype=float)
        phasic = window["EDA_Phasic"].to_numpy(dtype=float)

        if not all(
            np.isfinite(values).all()
            for values in (raw, clean, tonic, phasic)
        ):
            continue

        # --------------------------------------------------
        # RAW SIGNAL QUALITY
        # --------------------------------------------------

        differences = np.diff(raw)
        absolute_differences = np.abs(differences)

        median_abs_difference = float(
            np.median(absolute_differences)
        )

        max_abs_jump = float(
            np.max(absolute_differences)
        )

        jump_ratio = (
            max_abs_jump / median_abs_difference
            if median_abs_difference > 0
            else np.nan
        )

        unique_fraction = (
            len(np.unique(raw)) / len(raw)
        )

        flat_step_fraction = float(
            np.mean(absolute_differences == 0)
        )

        # --------------------------------------------------
        # SCR EVENT DIAGNOSTICS
        # --------------------------------------------------

        amplitudes = window[
            "SCR_Amplitude"
        ].to_numpy(dtype=float)

        valid_amplitudes = amplitudes[
            np.isfinite(amplitudes) & (amplitudes > 0)
        ]

        onsets = window[
            "SCR_Onsets"
        ].to_numpy(dtype=float)

        peaks = window[
            "SCR_Peaks"
        ].to_numpy(dtype=float)

        # --------------------------------------------------
        # FEATURE EXTRACTION
        # --------------------------------------------------

        rows.append({

            # EDA statistical features
            "eda_mean": float(np.mean(clean)),
            "eda_std": float(np.std(clean)),
            "eda_median": float(np.median(clean)),

            "eda_iqr": float(
                np.percentile(clean, 75)
                - np.percentile(clean, 25)
            ),

            "eda_range": float(np.ptp(clean)),

            # Tonic features
            "eda_tonic_mean": float(np.mean(tonic)),
            "eda_tonic_std": float(np.std(tonic)),

            "eda_tonic_slope": float(
                np.polyfit(
                    np.arange(len(tonic)),
                    tonic,
                    1
                )[0]
            ),

            # Phasic features
            "eda_phasic_mean": float(np.mean(phasic)),
            "eda_phasic_std": float(np.std(phasic)),

            # Event-only SCR amplitude
            "eda_scr_event_amplitude_mean": (
                float(np.mean(valid_amplitudes))
                if len(valid_amplitudes) > 0
                else np.nan
            ),

            "eda_scr_valid_event_count": int(
                len(valid_amplitudes)
            ),

            # SCR counts retained for diagnostics
            "eda_scr_onsets_diagnostic": int(
                np.nansum(onsets)
            ),

            "eda_scr_peaks_diagnostic": int(
                np.nansum(peaks)
            ),

            # Raw signal quality indicators
            "eda_raw_std": float(np.std(raw)),

            "eda_unique_fraction": float(
                unique_fraction
            ),

            "eda_flat_step_fraction": flat_step_fraction,

            "eda_median_abs_difference": (
                median_abs_difference
            ),

            "eda_max_abs_jump": max_abs_jump,

            "eda_jump_ratio": float(jump_ratio),

        })

    return rows
