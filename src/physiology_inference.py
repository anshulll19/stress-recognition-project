"""
physiology_inference.py
========================

Production-grade, reusable physiological inference engine for stress recognition.

Supported Input Modalities and Adapters:
1. WESAD Wrist BVP (Empatica E4, 64 Hz, 60-second windows = 3840 samples)
   - Evaluated by CNN-TCN-LSTM deployment candidate checkpoint (classifier.keras).
   - Supported inputs:
       * Preprocessed window array: (3840, 1), (3840,), or (N, 3840, 1)
       * Continuous raw BVP time series (at 64 Hz or arbitrary fs resampled to 64 Hz)
2. UBFC-Phys Multimodal BVP + EDA
   - Evaluated by UBFC Random Forest deployment models (calibrated or uncalibrated).
   - Supported inputs:
       * 12-dimensional statistical feature vector or DataFrame
       * Raw signal pair: 30s BVP window (64 Hz, 1920 samples) + 30s EDA window (4 Hz, 120 samples)
3. Rest Calibration:
   - Integrates with `RestCalibrator` from `calibration.py` to remove inter-subject baseline shifts.

Domain Separation:
- Strictly enforces dataset-model compatibility. Cross-dataset input mixing is rejected with descriptive errors.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from scipy.stats import iqr

from calibration import RestCalibrator


@dataclass
class PredictionResult:
    """Standardized prediction output across all physiological adapters."""
    prediction: int                       # 0 = non-stress, 1 = stress
    probability: float                    # P(stress) in [0.0, 1.0]
    class_name: str                       # 'non-stress' or 'stress'
    confidence: float                     # max(P(stress), 1 - P(stress))
    modality: str                         # 'WESAD_BVP', 'UBFC_PHYS', etc.
    is_calibrated: bool                   # whether rest calibration was applied
    latency_ms: float                     # model execution time in milliseconds
    quality_flag: str = "OK"              # 'OK', 'SUSPECT', 'DROPOUT'
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "probability": round(self.probability, 4),
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "modality": self.modality,
            "is_calibrated": self.is_calibrated,
            "latency_ms": round(self.latency_ms, 2),
            "quality_flag": self.quality_flag,
            "metadata": self.metadata,
        }


# ============================================================================
# 1. WESAD Physiological Inference Engine
# ============================================================================

class WESADInferenceEngine:
    """
    Inference adapter for WESAD 64 Hz wrist BVP.
    Uses the trained CNN-TCN-LSTM deployment candidate checkpoint.
    """

    TARGET_FS = 64
    WINDOW_SECONDS = 60
    EXPECTED_SAMPLES = TARGET_FS * WINDOW_SECONDS  # 3840

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None):
        if checkpoint_path is None:
            checkpoint_path = Path("models/deployment_candidate/classifier.keras")
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"WESAD checkpoint not found at: {self.checkpoint_path}. "
                f"Ensure models/deployment_candidate/classifier.keras is present."
            )
        import keras
        self.model = keras.models.load_model(str(self.checkpoint_path))

    @staticmethod
    def bandpass_filter(signal: np.ndarray, fs: float = 64.0, low_hz: float = 0.7, high_hz: float = 3.7, order: int = 3) -> np.ndarray:
        """Standard 3rd-order Butterworth bandpass for continuous BVP."""
        nyq = fs / 2.0
        b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
        return filtfilt(b, a, signal)

    @staticmethod
    def zscore_normalize_window(window: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Z-score standardize a single window.
        Detects flat / zero-variance signals (sensor dropouts).
        """
        std = np.std(window)
        if std == 0 or np.isnan(std) or std < 1e-7:
            return np.zeros_like(window), "DROPOUT"
        norm_w = (window - np.mean(window)) / std
        return norm_w, "OK"

    def predict_window(self, window: np.ndarray) -> PredictionResult:
        """
        Run inference on a single 60s BVP window.

        Parameters
        ----------
        window : np.ndarray
            Array of shape (3840,), (3840, 1), or (1, 3840, 1).

        Returns
        -------
        PredictionResult
        """
        t_start = time.perf_counter()
        arr = np.asarray(window, dtype=np.float32)

        # Handle shapes
        if arr.ndim == 1:
            arr = arr.reshape(1, -1, 1)
        elif arr.ndim == 2:
            if arr.shape == (self.EXPECTED_SAMPLES, 1):
                arr = arr.reshape(1, self.EXPECTED_SAMPLES, 1)
            elif arr.shape[0] == 1 and arr.shape[1] == self.EXPECTED_SAMPLES:
                arr = arr.reshape(1, self.EXPECTED_SAMPLES, 1)
            else:
                raise ValueError(
                    f"Invalid 2D window shape: {arr.shape}. Expected ({self.EXPECTED_SAMPLES}, 1) or (1, {self.EXPECTED_SAMPLES})."
                )
        elif arr.ndim == 3:
            if arr.shape[1:] != (self.EXPECTED_SAMPLES, 1):
                raise ValueError(f"Invalid 3D window shape: {arr.shape}. Expected (1, {self.EXPECTED_SAMPLES}, 1).")
        else:
            raise ValueError(f"Unsupported array dimensions: {arr.ndim}")

        # Quality check: Finite values
        if not np.isfinite(arr).all():
            raise ValueError("Input window contains NaN or infinite values.")

        # Quality check: Variance & Z-score normalization
        norm_window, quality = self.zscore_normalize_window(arr[0, :, 0])
        if quality == "DROPOUT":
            latency = (time.perf_counter() - t_start) * 1000.0
            return PredictionResult(
                prediction=0,
                probability=0.0,
                class_name="non-stress",
                confidence=0.5,
                modality="WESAD_BVP",
                is_calibrated=False,
                latency_ms=latency,
                quality_flag="DROPOUT",
                metadata={"error": "Zero variance detected in BVP window (sensor disconnection)."},
            )

        model_input = norm_window.reshape(1, self.EXPECTED_SAMPLES, 1)
        preds = self.model.predict(model_input, verbose=0)
        p_stress = float(preds[0, 1])
        prediction = int(p_stress >= 0.5)
        class_name = "stress" if prediction == 1 else "non-stress"
        confidence = float(max(p_stress, 1.0 - p_stress))
        latency = (time.perf_counter() - t_start) * 1000.0

        return PredictionResult(
            prediction=prediction,
            probability=p_stress,
            class_name=class_name,
            confidence=confidence,
            modality="WESAD_BVP",
            is_calibrated=False,
            latency_ms=latency,
            quality_flag=quality,
            metadata={"raw_probs": [float(preds[0, 0]), float(preds[0, 1])]},
        )

    def predict_continuous(self, signal: np.ndarray, fs: float = 64.0, step_s: float = 5.0) -> List[PredictionResult]:
        """
        Process a continuous raw BVP stream, bandpass filter, slice sliding windows, and predict.
        """
        sig = np.asarray(signal, dtype=np.float32).ravel()
        if fs != self.TARGET_FS:
            n_target = int(round(len(sig) * self.TARGET_FS / fs))
            sig = resample(sig, n_target)

        filtered = self.bandpass_filter(sig, fs=self.TARGET_FS)
        step_samples = int(step_s * self.TARGET_FS)
        results = []
        n_samples = len(filtered)
        start = 0

        while start + self.EXPECTED_SAMPLES <= n_samples:
            w = filtered[start:start + self.EXPECTED_SAMPLES]
            res = self.predict_window(w)
            res.metadata["start_s"] = start / self.TARGET_FS
            res.metadata["end_s"] = (start + self.EXPECTED_SAMPLES) / self.TARGET_FS
            results.append(res)
            start += step_samples

        return results


# ============================================================================
# 2. UBFC-Phys Multimodal Inference Engine
# ============================================================================

UBFC_FEATURES = [
    "bvp_mean", "bvp_std", "bvp_median", "bvp_iqr", "bvp_slope_per_sample", "bvp_range",
    "eda_mean", "eda_std", "eda_median", "eda_iqr", "eda_slope_per_sample", "eda_range"
]

class UBFCInferenceEngine:
    """
    Inference adapter for UBFC-Phys multimodal physiological signals (BVP + EDA).
    Supports both calibrated and uncalibrated Random Forest deployment checkpoints.
    """

    BVP_FS = 64.0
    EDA_FS = 4.0
    WINDOW_S = 30.0

    def __init__(
        self,
        calibrated_checkpoint: Optional[Union[str, Path]] = None,
        uncalibrated_checkpoint: Optional[Union[str, Path]] = None,
        calibrator: Optional[RestCalibrator] = None,
    ):
        if calibrated_checkpoint is None:
            calibrated_checkpoint = Path("models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib")
        if uncalibrated_checkpoint is None:
            uncalibrated_checkpoint = Path("models/ubfc_deployment/ubfc_phys_uncalibrated_rf.joblib")

        self.calibrated_checkpoint = Path(calibrated_checkpoint)
        self.uncalibrated_checkpoint = Path(uncalibrated_checkpoint)
        self.calibrator = calibrator

        self.cal_model = None
        self.uncal_model = None
        self._load_models()

    def _load_models(self):
        if self.calibrated_checkpoint.exists():
            self.cal_model = joblib.load(self.calibrated_checkpoint)
        if self.uncalibrated_checkpoint.exists():
            self.uncal_model = joblib.load(self.uncalibrated_checkpoint)

        if self.cal_model is None and self.uncal_model is None:
            raise FileNotFoundError(
                "Neither calibrated nor uncalibrated UBFC deployment models were found. "
                "Ensure models/ubfc_deployment/ contains model artifacts."
            )

    def attach_calibrator(self, calibrator: RestCalibrator):
        """Attach a fitted RestCalibrator instance."""
        if not calibrator.is_fitted:
            raise ValueError("Provided RestCalibrator is not fitted.")
        self.calibrator = calibrator

    @staticmethod
    def extract_features(bvp_window: np.ndarray, eda_window: np.ndarray) -> Dict[str, float]:
        """
        Extract the 12 statistical/time-domain features from 30s BVP and EDA windows.
        """
        b = np.asarray(bvp_window, dtype=float).ravel()
        e = np.asarray(eda_window, dtype=float).ravel()

        if len(b) < 10 or len(e) < 5:
            raise ValueError("Window too short for reliable feature extraction.")

        b_slope = float(np.polyfit(np.arange(len(b)), b, 1)[0]) if len(b) > 1 else 0.0
        e_slope = float(np.polyfit(np.arange(len(e)), e, 1)[0]) if len(e) > 1 else 0.0

        return {
            "bvp_mean": float(np.mean(b)),
            "bvp_std": float(np.std(b)),
            "bvp_median": float(np.median(b)),
            "bvp_iqr": float(iqr(b)),
            "bvp_slope_per_sample": b_slope,
            "bvp_range": float(np.ptp(b)),
            "eda_mean": float(np.mean(e)),
            "eda_std": float(np.std(e)),
            "eda_median": float(np.median(e)),
            "eda_iqr": float(iqr(e)),
            "eda_slope_per_sample": e_slope,
            "eda_range": float(np.ptp(e)),
        }

    def predict_features(
        self,
        features: Union[Dict[str, float], pd.DataFrame, np.ndarray],
        use_calibration: Optional[bool] = None,
    ) -> PredictionResult:
        """
        Run inference on a 12-feature physiological representation.

        Parameters
        ----------
        features : Dict, DataFrame, or ndarray
            12 features in order: bvp_mean, bvp_std, bvp_median, bvp_iqr, bvp_slope_per_sample,
            bvp_range, eda_mean, eda_std, eda_median, eda_iqr, eda_slope_per_sample, eda_range.
        use_calibration : bool, optional
            Whether to apply rest calibration. If None, auto-selects based on whether
            a calibrator is attached and calibrated model is available.
        """
        t_start = time.perf_counter()

        # Format input into array of shape (1, 12)
        if isinstance(features, pd.Series):
            features = features.to_dict()

        if isinstance(features, dict):
            missing = [f for f in UBFC_FEATURES if f not in features]
            if missing:
                raise ValueError(f"Feature dictionary missing required keys: {missing}")
            vec = np.array([[float(features[f]) for f in UBFC_FEATURES]])
        elif isinstance(features, pd.DataFrame):
            missing = [f for f in UBFC_FEATURES if f not in features.columns]
            if missing:
                raise ValueError(f"Feature DataFrame missing required columns: {missing}")
            vec = features[UBFC_FEATURES].to_numpy(dtype=float)
            if len(vec) != 1:
                vec = vec[0:1]
        elif isinstance(features, np.ndarray):
            vec = np.asarray(features, dtype=float)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)
            if vec.shape[1] != len(UBFC_FEATURES):
                raise ValueError(
                    f"Array shape mismatch: expected {len(UBFC_FEATURES)} columns, got {vec.shape[1]}"
                )
        else:
            raise TypeError(f"Unsupported features type: {type(features)}")

        if not np.isfinite(vec).all():
            raise ValueError("Input features contain NaN or infinite values.")

        # Determine calibration mode
        if use_calibration is None:
            use_calib = (self.calibrator is not None and self.calibrator.is_fitted and self.cal_model is not None)
        else:
            use_calib = use_calibration

        if use_calib:
            if self.cal_model is None:
                raise RuntimeError("Calibrated deployment model is not loaded.")
            if self.calibrator is None or not self.calibrator.is_fitted:
                raise RuntimeError("use_calibration=True requested, but no fitted RestCalibrator is attached.")
            proc_vec = self.calibrator.transform(vec)
            model = self.cal_model
        else:
            if self.uncal_model is None:
                raise RuntimeError("Uncalibrated deployment model is not loaded.")
            proc_vec = vec
            model = self.uncal_model

        probs = model.predict_proba(proc_vec)
        p_stress = float(probs[0, 1])
        prediction = int(p_stress >= 0.5)
        class_name = "stress" if prediction == 1 else "non-stress"
        confidence = float(max(p_stress, 1.0 - p_stress))
        latency = (time.perf_counter() - t_start) * 1000.0

        return PredictionResult(
            prediction=prediction,
            probability=p_stress,
            class_name=class_name,
            confidence=confidence,
            modality="UBFC_PHYS",
            is_calibrated=use_calib,
            latency_ms=latency,
            quality_flag="OK",
            metadata={"features": {UBFC_FEATURES[i]: float(vec[0, i]) for i in range(len(UBFC_FEATURES))}},
        )

    def predict_raw_windows(
        self,
        bvp_window: np.ndarray,
        eda_window: np.ndarray,
        use_calibration: Optional[bool] = None,
    ) -> PredictionResult:
        """
        Extract features from raw 30s BVP and EDA snippets, then run model prediction.
        """
        feats = self.extract_features(bvp_window, eda_window)
        return self.predict_features(feats, use_calibration=use_calibration)


# ============================================================================
# 3. Unified Multimodal Physiological Dispatcher
# ============================================================================

def predict_stress(
    dataset: str,
    input_data: Union[np.ndarray, pd.DataFrame, Dict],
    calibrator: Optional[RestCalibrator] = None,
    **kwargs,
) -> PredictionResult:
    """
    Unified entry point for physiological stress prediction on supported new inputs.

    Parameters
    ----------
    dataset : str
        Either 'WESAD' or 'UBFC' / 'UBFC-Phys'.
    input_data : np.ndarray, pd.DataFrame, or Dict
        Input window or feature set matching the dataset specification.
    calibrator : RestCalibrator, optional
        Baseline calibrator for UBFC inputs.
    """
    ds = dataset.upper().strip()
    if ds == "WESAD":
        engine = WESADInferenceEngine()
        return engine.predict_window(input_data)
    elif ds in ("UBFC", "UBFC-PHYS", "UBFC_PHYS"):
        engine = UBFCInferenceEngine(calibrator=calibrator)
        if isinstance(input_data, tuple) and len(input_data) == 2:
            return engine.predict_raw_windows(input_data[0], input_data[1])
        return engine.predict_features(input_data)
    elif ds == "CLAS":
        raise ValueError(
            "CLAS stress prediction is BLOCKED: CLAS artifacts in this repository represent "
            "self-supervised MAE encoder representations without a trained classifier head. "
            "Cross-dataset zero-shot transfer from WESAD to CLAS failed (AUROC=0.367) due to "
            "sensor physics differences and protocol mismatch."
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported datasets: 'WESAD', 'UBFC-Phys'.")
