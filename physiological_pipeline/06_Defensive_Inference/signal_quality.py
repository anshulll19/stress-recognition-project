"""Defensive input validation for V6 EDA inference."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SignalValidation:
    accepted: bool
    reasons: list[str]
    sampling_rate: float
    duration_seconds: float
    sample_count: int
    standard_deviation: float | None


def validate_eda_signal(
    signal,
    sampling_rate: float,
    minimum_duration_seconds: float = 85.0,
) -> SignalValidation:
    """
    Validate raw EDA input before feature extraction.

    This checks structural validity only. It does not certify
    physiological signal quality or sensor accuracy.
    """

    reasons = []

    try:
        fs = float(sampling_rate)
    except (TypeError, ValueError):
        fs = float("nan")

    if not np.isfinite(fs) or fs <= 0:
        return SignalValidation(
            accepted=False,
            reasons=["invalid_sampling_rate"],
            sampling_rate=fs,
            duration_seconds=0.0,
            sample_count=0,
            standard_deviation=None,
        )

    try:
        values = np.asarray(signal, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return SignalValidation(
            accepted=False,
            reasons=["signal_not_numeric"],
            sampling_rate=fs,
            duration_seconds=0.0,
            sample_count=0,
            standard_deviation=None,
        )

    count = int(values.size)
    duration = count / fs

    if count == 0:
        reasons.append("empty_signal")
    elif not np.isfinite(values).all():
        reasons.append("non_finite_values")

    if duration < minimum_duration_seconds:
        reasons.append("insufficient_duration")

    std = None

    if count > 0 and np.isfinite(values).all():
        std = float(np.std(values))

        if std == 0:
            reasons.append("flat_signal")

    return SignalValidation(
        accepted=len(reasons) == 0,
        reasons=reasons,
        sampling_rate=fs,
        duration_seconds=float(duration),
        sample_count=count,
        standard_deviation=std,
    )
