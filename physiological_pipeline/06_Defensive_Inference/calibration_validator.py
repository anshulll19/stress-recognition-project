"""Defensive participant-baseline calibration validation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


V5_FEATURES = [
    "eda_mean",
    "eda_std",
    "eda_median",
    "eda_iqr",
    "eda_range",
    "eda_tonic_mean",
    "eda_tonic_std",
    "eda_tonic_slope",
    "eda_phasic_mean",
    "eda_phasic_std",
    "eda_scr_valid_event_count",
    "eda_scr_onsets_diagnostic",
    "eda_scr_peaks_diagnostic",
    "eda_raw_std",
    "eda_unique_fraction",
    "eda_flat_step_fraction",
    "eda_median_abs_difference",
    "eda_max_abs_jump",
    "eda_jump_ratio",
]

EXPECTED_CALIBRATION_WINDOWS = 12


class CalibrationError(ValueError):
    """Raised when baseline calibration is not usable."""


@dataclass
class CalibrationResult:
    accepted: bool
    reasons: list[str]
    median: pd.Series | None = None
    scale: pd.Series | None = None
    metadata: dict | None = None


def validate_calibration(features: pd.DataFrame) -> CalibrationResult:
    """
    Validate calibration features and calculate V5-compatible
    participant-specific median/IQR normalization statistics.

    Missing individual values are tolerated when a finite median
    can be calculated, consistent with pandas median behaviour.
    """

    if not isinstance(features, pd.DataFrame):
        return CalibrationResult(
            accepted=False,
            reasons=["calibration_must_be_dataframe"],
        )

    if features.columns.duplicated().any():
        return CalibrationResult(
            accepted=False,
            reasons=["duplicate_feature_columns"],
        )

    missing = [name for name in V5_FEATURES if name not in features.columns]

    if missing:
        return CalibrationResult(
            accepted=False,
            reasons=["missing_features:" + ",".join(missing)],
        )

    if len(features) != EXPECTED_CALIBRATION_WINDOWS:
        return CalibrationResult(
            accepted=False,
            reasons=["incorrect_calibration_window_count"],
        )

    try:
        frame = features[V5_FEATURES].astype(float)
    except (TypeError, ValueError):
        return CalibrationResult(
            accepted=False,
            reasons=["non_numeric_calibration_features"],
        )

    median = frame.median()
    iqr = frame.quantile(0.75) - frame.quantile(0.25)
    std = frame.std(ddof=0)

    invalid_medians = [
        name for name in V5_FEATURES
        if not np.isfinite(median[name])
    ]

    if invalid_medians:
        return CalibrationResult(
            accepted=False,
            reasons=["features_without_finite_median:" + ",".join(invalid_medians)],
        )

    scale = iqr.where(
        np.isfinite(iqr) & (iqr > 1e-10),
        std,
    )

    scale = scale.where(
        np.isfinite(scale) & (scale > 1e-10),
        1.0,
    )

    if not np.isfinite(scale.to_numpy()).all():
        return CalibrationResult(
            accepted=False,
            reasons=["invalid_calibration_scale"],
        )

    return CalibrationResult(
        accepted=True,
        reasons=[],
        median=median,
        scale=scale,
        metadata={
            "method": "participant-specific median/IQR",
            "fallback": "standard deviation, then 1.0",
            "calibration_windows": len(frame),
            "feature_count": len(V5_FEATURES),
            "missing_values_present": bool(frame.isna().any().any()),
        },
    )
