"""Calibration stability diagnostics for V6.

This module reports calibration scale behaviour.
It does not decide whether a participant should be rejected.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from calibration_validator import V5_FEATURES


@dataclass
class CalibrationStabilityResult:
    accepted_for_diagnostics: bool
    feature_diagnostics: dict
    warning_features: list
    iqr_count: int
    std_fallback_count: int
    unit_fallback_count: int
    interpretation: str


def inspect_calibration_stability(
    features: pd.DataFrame,
    numerical_epsilon: float = 1e-10,
) -> CalibrationStabilityResult:

    if not isinstance(features, pd.DataFrame):
        raise TypeError("Calibration features must be a DataFrame.")

    if features.columns.duplicated().any():
        raise ValueError("Duplicate calibration feature columns.")

    missing = [f for f in V5_FEATURES if f not in features.columns]
    if missing:
        raise ValueError(f"Missing calibration features: {missing}")

    if len(features) != 12:
        raise ValueError(
            f"Expected exactly 12 calibration windows; got {len(features)}."
        )

    frame = features[V5_FEATURES].astype(float)

    diagnostics = {}
    warnings = []

    iqr_count = 0
    std_count = 0
    unit_count = 0

    for feature in V5_FEATURES:
        values = frame[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]

        if len(finite) != len(values):
            diagnostics[feature] = {
                "method": "INVALID",
                "finite_count": int(len(finite)),
                "nonfinite_count": int(len(values) - len(finite)),
                "warning": "non_finite_calibration_values",
            }
            warnings.append(feature)
            continue

        series = pd.Series(finite)
        iqr = float(
            series.quantile(0.75) - series.quantile(0.25)
        )
        std = float(series.std(ddof=0))

        if np.isfinite(iqr) and iqr > numerical_epsilon:
            method = "IQR"
            scale = iqr
            iqr_count += 1
        elif np.isfinite(std) and std > numerical_epsilon:
            method = "STD"
            scale = std
            std_count += 1
            warnings.append(feature)
        else:
            method = "FALLBACK_1"
            scale = 1.0
            unit_count += 1
            warnings.append(feature)

        diagnostics[feature] = {
            "iqr": iqr,
            "std": std,
            "scale_used": scale,
            "method": method,
            "warning": None if method == "IQR" else (
                "standard_deviation_fallback"
                if method == "STD"
                else "unit_scale_fallback"
            ),
        }

    return CalibrationStabilityResult(
        accepted_for_diagnostics=True,
        feature_diagnostics=diagnostics,
        warning_features=warnings,
        iqr_count=iqr_count,
        std_fallback_count=std_count,
        unit_fallback_count=unit_count,
        interpretation=(
            "Diagnostic only. Scale fallbacks indicate low calibration "
            "variation; they do not independently establish invalid "
            "physiology or justify automatic abstention."
        ),
    )
