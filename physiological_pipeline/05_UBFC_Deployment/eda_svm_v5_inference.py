"""Independent inference adapter for the UBFC-Phys V5 EDA SVM."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models/ubfc_deployment/ubfc_phys_eda_svm_v5.joblib"

EXTRACTOR_DIR = PROJECT_ROOT / "physiological_pipeline/03_NeuroKit_V2"
sys.path.insert(0, str(EXTRACTOR_DIR))

from neurokit_eda_v2 import (  # noqa: E402
    extract_eda_features,
    EDA_SAMPLING_RATE,
    WINDOW_SECONDS,
    STEP_SECONDS,
)


FEATURES = [
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

CALIBRATION_END_SECONDS = 85.0
MIN_CALIBRATION_WINDOWS = 12


class V5InferenceError(RuntimeError):
    """Raised when V5 inference cannot be performed safely."""


class EDASVMV5Inference:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)

        artifact = joblib.load(self.model_path)

        if artifact.get("features") != FEATURES:
            raise V5InferenceError(
                "Saved model feature schema does not match V5 adapter."
            )

        self.model = artifact["model"]

        if hasattr(self.model, "predict_proba"):
            raise V5InferenceError(
                "Unexpected probability interface. Inspect model before deployment."
            )

        if not hasattr(self.model, "decision_function"):
            raise V5InferenceError(
                "Model does not expose decision_function."
            )

        self.baseline_median = None
        self.baseline_scale = None
        self.calibration_metadata = None

    @staticmethod
    def _validate_signal(signal, minimum_seconds):
        signal = np.asarray(signal, dtype=float).ravel()

        if signal.size < minimum_seconds * EDA_SAMPLING_RATE:
            raise V5InferenceError(
                f"EDA recording must contain at least {minimum_seconds} seconds."
            )

        if not np.isfinite(signal).all():
            raise V5InferenceError(
                "EDA signal contains NaN or infinite values."
            )

        if np.std(signal) == 0:
            raise V5InferenceError("EDA signal is flat.")

        return signal

    @staticmethod
    def _extract(signal):
        try:
            rows = extract_eda_features(
                signal,
                sampling_rate=EDA_SAMPLING_RATE,
            )
        except Exception as exc:
            raise V5InferenceError(
                f"EDA feature extraction failed: {exc}"
            ) from exc

        if not rows:
            raise V5InferenceError(
                "No valid EDA feature windows were extracted."
            )

        frame = pd.DataFrame(rows)

        missing = [f for f in FEATURES if f not in frame.columns]
        if missing:
            raise V5InferenceError(
                f"Extractor is missing V5 features: {missing}"
            )

        return frame[FEATURES].astype(float)

    def calibrate(self, t1_eda):
        """Calibrate using the first 12 T1 windows (ending <= 85 seconds)."""

        signal = self._validate_signal(
            t1_eda,
            minimum_seconds=85,
        )

        # Only the first 85 seconds belong to V5 calibration.
        calibration_samples = int(
            CALIBRATION_END_SECONDS * EDA_SAMPLING_RATE
        )
        calibration_signal = signal[:calibration_samples]

        calibration = self._extract(calibration_signal)

        if len(calibration) != MIN_CALIBRATION_WINDOWS:
            raise V5InferenceError(
                f"Expected exactly {MIN_CALIBRATION_WINDOWS} calibration "
                f"windows from the first 85 seconds; got {len(calibration)}. "
                "Calibration aborted."
            )

        median = calibration.median()
        iqr = (
            calibration.quantile(0.75)
            - calibration.quantile(0.25)
        )
        std = calibration.std(ddof=0)

        scale = iqr.where(
            np.isfinite(iqr) & (iqr > 1e-10),
            std,
        )
        scale = scale.where(
            np.isfinite(scale) & (scale > 1e-10),
            1.0,
        )

        if not np.isfinite(median.to_numpy(dtype=float)).all():
            raise V5InferenceError(
                "Calibration contains invalid feature medians."
            )

        if not np.isfinite(scale.to_numpy(dtype=float)).all():
            raise V5InferenceError(
                "Calibration contains invalid feature scales."
            )

        self.baseline_median = median
        self.baseline_scale = scale

        self.calibration_metadata = {
            "calibrated": True,
            "method": "participant-specific median/IQR",
            "calibration_windows": len(calibration),
            "calibration_window_end_max_seconds": 85.0,
            "sampling_rate_hz": EDA_SAMPLING_RATE,
        }

        return self.calibration_metadata.copy()

    def predict_task(self, task_eda):
        """Predict task windows using the fitted participant baseline."""

        if self.baseline_median is None:
            raise V5InferenceError(
                "T1 calibration must be completed before task inference."
            )

        signal = self._validate_signal(
            task_eda,
            minimum_seconds=WINDOW_SECONDS,
        )

        frame = self._extract(signal)

        expected_windows = (
            (len(signal) - WINDOW_SECONDS * EDA_SAMPLING_RATE)
            // (STEP_SECONDS * EDA_SAMPLING_RATE)
        ) + 1

        # The extractor skips windows if its processed signals contain
        # non-finite values. Do not guess their timestamps or labels.
        if len(frame) != expected_windows:
            raise V5InferenceError(
                f"Expected {expected_windows} task windows but extracted "
                f"{len(frame)}. Inference abstained because window alignment "
                "could not be guaranteed."
            )

        normalized = (
            frame - self.baseline_median
        ) / self.baseline_scale

        predictions = self.model.predict(normalized)
        scores = self.model.decision_function(normalized)

        results = []

        for index, (prediction, score) in enumerate(
            zip(predictions, scores)
        ):
            start_s = index * STEP_SECONDS

            results.append({
                "window_index": index,
                "start_s": float(start_s),
                "end_s": float(start_s + WINDOW_SECONDS),
                "prediction": int(prediction),
                "class_name": (
                    "distress" if prediction == 1 else "rest"
                ),
                "decision_score": float(score),
                "score_type": "raw_svm_decision_function",
                "probability": None,
                "is_calibrated": True,
                "is_reliable": True,
            })

        return results
