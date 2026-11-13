"""Diagnostic feature-distribution shift monitoring for V6."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibration_validator import V5_FEATURES


class DomainShiftError(ValueError):
    """Raised when domain-shift diagnostics cannot be performed."""


class DomainShiftMonitor:

    def __init__(self, reference_path):
        self.reference_path = Path(reference_path)

        if not self.reference_path.exists():
            raise FileNotFoundError(self.reference_path)

        with self.reference_path.open() as f:
            self.reference = json.load(f)

        if self.reference.get("features") != V5_FEATURES:
            raise DomainShiftError(
                "Reference feature schema does not match V5."
            )

        self.feature_reference = self.reference["feature_reference"]

    def inspect(self, features):
        """Compare normalized feature windows with V5 reference quantiles."""

        if not isinstance(features, pd.DataFrame):
            raise DomainShiftError(
                "Input features must be a pandas DataFrame."
            )

        if features.columns.duplicated().any():
            raise DomainShiftError("Duplicate feature columns detected.")

        missing = [
            feature for feature in V5_FEATURES
            if feature not in features.columns
        ]

        if missing:
            raise DomainShiftError(
                f"Missing V5 features: {missing}"
            )

        frame = features[V5_FEATURES].copy()

        try:
            frame = frame.astype(float)
        except (TypeError, ValueError) as exc:
            raise DomainShiftError(
                "Features contain non-numeric values."
            ) from exc

        diagnostics = {}

        for feature in V5_FEATURES:
            values = frame[feature].to_numpy(dtype=float)
            finite_mask = np.isfinite(values)
            finite_values = values[finite_mask]

            reference = self.feature_reference[feature]
            quantiles = reference["quantiles"]

            lower = float(quantiles["0.01"])
            upper = float(quantiles["0.99"])

            outside_count = int(
                np.sum(
                    (finite_values < lower)
                    | (finite_values > upper)
                )
            )

            finite_count = len(finite_values)

            diagnostics[feature] = {
                "total_windows": int(len(values)),
                "finite_windows": int(finite_count),
                "nonfinite_windows": int(len(values) - finite_count),
                "reference_q01": lower,
                "reference_q99": upper,
                "outside_reference_count": outside_count,
                "outside_reference_fraction": (
                    float(outside_count / finite_count)
                    if finite_count else None
                ),
            }

        return {
            "reference_name": self.reference.get("reference_name"),
            "reference_type": self.reference.get("reference_type"),
            "input_windows": int(len(frame)),
            "feature_diagnostics": diagnostics,
            "interpretation": (
                "Diagnostic only. Reference-range exceedance does not "
                "establish invalid signal or justify automatic abstention."
            ),
        }
