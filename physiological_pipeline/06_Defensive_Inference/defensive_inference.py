"""Defensive orchestration layer for the frozen UBFC Phys V5 EDA model."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_DIR = ROOT / "physiological_pipeline/05_UBFC_Deployment"
DEFENSIVE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(DEPLOYMENT_DIR))
sys.path.insert(0, str(DEFENSIVE_DIR))

from eda_svm_v5_inference import (  # noqa: E402
    EDASVMV5Inference,
    V5InferenceError,
    EDA_SAMPLING_RATE,
    WINDOW_SECONDS,
)
from signal_quality import validate_eda_signal  # noqa: E402
from calibration_validator import validate_calibration  # noqa: E402
from calibration_stability import inspect_calibration_stability  # noqa: E402
from domain_shift import DomainShiftMonitor  # noqa: E402


REFERENCE_PATH = DEFENSIVE_DIR / "v5_training_reference.json"


class DefensiveInference:
    """Apply mandatory validation and diagnostic checks around V5."""

    def __init__(self, model_path=None):
        self.engine = (
            EDASVMV5Inference(model_path)
            if model_path is not None
            else EDASVMV5Inference()
        )
        self.domain_monitor = DomainShiftMonitor(REFERENCE_PATH)
        self.calibrated = False
        self.calibration_diagnostics = None

    @staticmethod
    def _abstain(reason, diagnostics=None):
        return {
            "status": "abstained",
            "abstained": True,
            "reasons": [str(reason)],
            "diagnostics": diagnostics or {},
        }

    def calibrate(self, t1_eda):
        """Validate and calibrate using the first 85 seconds of T1."""

        self.calibrated = False
        self.calibration_diagnostics = None

        signal_check = validate_eda_signal(
            t1_eda,
            sampling_rate=EDA_SAMPLING_RATE,
            minimum_duration_seconds=85.0,
        )

        if not signal_check.accepted:
            return self._abstain(
                "invalid_calibration_signal",
                {"signal_validation_reasons": signal_check.reasons},
            )

        signal = np.asarray(t1_eda, dtype=float).reshape(-1)
        calibration_samples = int(85.0 * EDA_SAMPLING_RATE)
        calibration_signal = signal[:calibration_samples]

        try:
            calibration_features = self.engine._extract(
                calibration_signal
            )
        except Exception as exc:
            return self._abstain(
                "calibration_feature_extraction_failed",
                {"error": str(exc)},
            )

        validation = validate_calibration(calibration_features)

        if not validation.accepted:
            return self._abstain(
                "calibration_validation_failed",
                {"validation_reasons": validation.reasons},
            )

        stability = inspect_calibration_stability(
            calibration_features
        )

        try:
            adapter_metadata = self.engine.calibrate(signal)
        except V5InferenceError as exc:
            return self._abstain(
                "v5_calibration_failed",
                {"error": str(exc)},
            )

        # Confirm the defensive validator and frozen V5 adapter
        # calculate identical baseline statistics.
        medians_match = np.allclose(
            validation.median.to_numpy(dtype=float),
            self.engine.baseline_median.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
        scales_match = np.allclose(
            validation.scale.to_numpy(dtype=float),
            self.engine.baseline_scale.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )

        if not medians_match or not scales_match:
            self.engine.baseline_median = None
            self.engine.baseline_scale = None
            return self._abstain(
                "calibration_parity_mismatch",
                {
                    "medians_match": bool(medians_match),
                    "scales_match": bool(scales_match),
                },
            )

        self.calibrated = True

        warnings = []
        if stability.warning_features:
            warnings.append("calibration_scale_fallbacks_detected")

        self.calibration_diagnostics = {
            "adapter_metadata": adapter_metadata,
            "calibration_validation": validation.metadata,
            "calibration_stability": {
                "iqr_count": stability.iqr_count,
                "std_fallback_count": stability.std_fallback_count,
                "unit_fallback_count": stability.unit_fallback_count,
                "warning_features": stability.warning_features,
                "interpretation": stability.interpretation,
            },
            "calibration_medians_match": bool(medians_match),
            "calibration_scales_match": bool(scales_match),
        }

        return {
            "status": (
                "accepted_with_warnings" if warnings else "accepted"
            ),
            "abstained": False,
            "warnings": warnings,
            "diagnostics": self.calibration_diagnostics,
        }

    def predict_task(self, task_eda):
        """Run V5 task inference with structural and shift diagnostics."""

        if not self.calibrated:
            return self._abstain("participant_calibration_required")

        signal_check = validate_eda_signal(
            task_eda,
            sampling_rate=EDA_SAMPLING_RATE,
            minimum_duration_seconds=float(WINDOW_SECONDS),
        )

        if not signal_check.accepted:
            return self._abstain(
                "invalid_task_signal",
                {"signal_validation_reasons": signal_check.reasons},
            )

        signal = np.asarray(task_eda, dtype=float).reshape(-1)

        try:
            features = self.engine._extract(signal)

            expected_windows = (
                (len(signal) - WINDOW_SECONDS * EDA_SAMPLING_RATE)
                // (self.engine._extract.__globals__["STEP_SECONDS"]
                    * EDA_SAMPLING_RATE)
            ) + 1

            if len(features) != expected_windows:
                return self._abstain(
                    "task_window_alignment_mismatch",
                    {
                        "expected_windows": int(expected_windows),
                        "extracted_windows": int(len(features)),
                    },
                )

            normalized = (
                features - self.engine.baseline_median
            ) / self.engine.baseline_scale

            shift = self.domain_monitor.inspect(normalized)

            predictions = self.engine.predict_task(signal)

        except V5InferenceError as exc:
            return self._abstain("v5_inference_failed", {"error": str(exc)})
        except Exception as exc:
            return self._abstain(
                "defensive_inference_failed",
                {"error": str(exc)},
            )

        outside_features = [
            name
            for name, item in shift["feature_diagnostics"].items()
            if item["outside_reference_count"] > 0
        ]

        warnings = []
        if outside_features:
            warnings.append("training_reference_range_exceeded")

        # Do not expose the adapter's unvalidated is_reliable flag.
        cleaned_predictions = []
        for result in predictions:
            item = dict(result)
            item.pop("is_reliable", None)
            item.pop("is_calibrated", None)
            item["baseline_calibrated"] = True
            item["probability_calibrated"] = False
            cleaned_predictions.append(item)

        return {
            "status": (
                "accepted_with_warnings" if warnings else "accepted"
            ),
            "abstained": False,
            "warnings": warnings,
            "predictions": cleaned_predictions,
            "diagnostics": {
                "domain_shift": shift,
                "features_outside_reference": outside_features,
                "signal_duration_seconds": signal_check.duration_seconds,
                "window_count": len(cleaned_predictions),
            },
        }
