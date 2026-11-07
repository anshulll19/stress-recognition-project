"""
automatic_calibration.py
========================

Automatic resting baseline calibration module for UBFC-Phys multimodal signals.
Calibrates participant-specific tonic autonomic physiology using Task 1 (Rest).
Enforces the original 180-second baseline protocol (31 overlapping 30s windows with 5s stride)
and guards against sensor dropout corrupting baseline estimation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from .calibration import RestCalibrator
    from .physiology_inference import UBFC_FEATURES, UBFCInferenceEngine, PredictionResult
    from .physiological_streaming import slice_physiological_windows, PhysiologicalWindow
except ImportError:
    from calibration import RestCalibrator
    from physiology_inference import UBFC_FEATURES, UBFCInferenceEngine, PredictionResult
    from physiological_streaming import slice_physiological_windows, PhysiologicalWindow


REQUIRED_T1_BASELINE_SECONDS: float = 180.0
EXPECTED_T1_WINDOWS: int = 31  # (180 - 30) / 5 + 1 = 31
MIN_RELIABLE_T1_WINDOWS: int = 25  # At least 25 reliable windows (~80% sensor integrity)


class CalibrationFailureError(RuntimeError):
    """Raised when resting baseline calibration cannot be reliably computed."""
    pass


@dataclass
class CalibrationResult:
    """Standardized output of the automated resting baseline calibration."""
    is_calibrated: bool
    participant_id: Optional[str]
    n_total_windows: int
    n_reliable_windows: int
    n_unreliable_windows: int
    baseline_duration_s: float
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]
    calibrator: RestCalibrator
    baseline_predictions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_calibrated": self.is_calibrated,
            "participant_id": self.participant_id,
            "n_total_windows": self.n_total_windows,
            "n_reliable_windows": self.n_reliable_windows,
            "n_unreliable_windows": self.n_unreliable_windows,
            "baseline_duration_s": round(self.baseline_duration_s, 2),
            "feature_means": {k: round(v, 6) for k, v in self.feature_means.items()},
            "feature_stds": {k: round(v, 6) for k, v in self.feature_stds.items()},
            "metadata": self.metadata,
        }


def calibrate_t1_baseline(
    bvp_signal: np.ndarray,
    eda_signal: np.ndarray,
    participant_id: Optional[str] = None,
    bvp_fs: float = 64.0,
    eda_fs: float = 4.0,
    required_baseline_s: float = REQUIRED_T1_BASELINE_SECONDS,
    min_reliable_windows: int = MIN_RELIABLE_T1_WINDOWS,
) -> CalibrationResult:
    """
    Perform automatic participant-specific resting calibration on Task 1 (Rest).

    Enforces the research training protocol:
    1. Validates signal duration >= 180.0s (yielding 31 overlapping 30s windows with 5s stride).
    2. Identifies and isolates reliable physiological windows, discarding dropouts.
    3. Requires at least min_reliable_windows (default 25) reliable windows.
    4. Extracts the 12 statistical UBFC features for each reliable window.
    5. Computes baseline resting distribution (means and standard deviations) with epsilon guards.
    6. Fits and returns a RestCalibrator instance ready to be attached to inference engines for T2 and T3.
    """
    dur_b = len(bvp_signal) / bvp_fs
    dur_e = len(eda_signal) / eda_fs
    common_duration = min(dur_b, dur_e)

    if common_duration < required_baseline_s - 1e-3:
        raise CalibrationFailureError(
            f"T1 baseline duration ({common_duration:.1f}s) is shorter than the required "
            f"{required_baseline_s:.0f} seconds (expected {EXPECTED_T1_WINDOWS} windows @ 30s window/5s stride). "
            f"The research protocol requires 180 seconds of resting baseline to estimate tonic distributions."
        )

    windows = slice_physiological_windows(
        bvp_signal=bvp_signal,
        eda_signal=eda_signal,
        bvp_fs=bvp_fs,
        eda_fs=eda_fs,
        window_s=30.0,
        stride_s=5.0,
        extract_features=True,
    )

    if not windows:
        raise CalibrationFailureError("T1 signal duration is insufficient to generate any 30-second windows.")

    reliable_windows = [w for w in windows if w.is_reliable and w.features is not None]
    unreliable_count = len(windows) - len(reliable_windows)

    if len(reliable_windows) < min_reliable_windows:
        raise CalibrationFailureError(
            f"T1 baseline calibration failed: only {len(reliable_windows)} reliable windows "
            f"were extracted ({unreliable_count} dropouts/unreliable). Minimum required for reliable baseline estimation is {min_reliable_windows}."
        )

    # Compile feature records
    feature_rows = []
    for w in reliable_windows:
        feature_rows.append(w.features)

    df_feats = pd.DataFrame(feature_rows)[UBFC_FEATURES]

    # Fit the RestCalibrator
    calibrator = RestCalibrator(features=UBFC_FEATURES, eps=1e-6)
    calibrator.fit(df_feats)

    # Evaluate baseline windows with calibrated model
    engine = UBFCInferenceEngine(calibrator=calibrator)
    baseline_predictions: List[Dict[str, Any]] = []

    for w in windows:
        if w.is_reliable and w.features is not None:
            pred_res = engine.predict_features(w.features, use_calibration=True)
            baseline_predictions.append({
                "window_index": w.window_index,
                "start_s": w.start_s,
                "end_s": w.end_s,
                "is_reliable": True,
                "prediction": pred_res.prediction,
                "probability": round(pred_res.probability, 4),
                "confidence": round(pred_res.confidence, 4),
                "quality_flag": pred_res.quality_flag,
                "latency_ms": round(pred_res.latency_ms, 2),
            })
        else:
            baseline_predictions.append({
                "window_index": w.window_index,
                "start_s": w.start_s,
                "end_s": w.end_s,
                "is_reliable": False,
                "prediction": None,  # NEVER treat unreliable windows as non-stress (0)
                "probability": None,
                "confidence": None,
                "quality_flag": w.quality_flag,
                "quality_reason": w.quality_reason,
                "latency_ms": 0.0,
            })

    return CalibrationResult(
        is_calibrated=True,
        participant_id=participant_id,
        n_total_windows=len(windows),
        n_reliable_windows=len(reliable_windows),
        n_unreliable_windows=unreliable_count,
        baseline_duration_s=common_duration,
        feature_means=calibrator.means,
        feature_stds=calibrator.stds,
        calibrator=calibrator,
        baseline_predictions=baseline_predictions,
        metadata={
            "features": UBFC_FEATURES,
            "window_size_s": 30.0,
            "stride_s": 5.0,
            "expected_windows": EXPECTED_T1_WINDOWS,
            "n_baseline_samples": calibrator.n_baseline_samples,
        },
    )
