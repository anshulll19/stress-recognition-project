"""WESAD physiological inference adapter for CNN-TCN-LSTM deployment model."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from demo.adapters.base import BaseInferenceAdapter, PredictionResult, ValidationResult


class WESADInferenceAdapter(BaseInferenceAdapter):
    """Adapter for WESAD 64 Hz BVP signal classification via CNN-TCN-LSTM."""

    CLASS_NAMES = {
        0: "Non-stress (Baseline/Amusement)",
        1: "Stress (TSST Protocol)",
    }

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or Path(__file__).resolve().parents[2]
        self.data_path = self.root / "data/wesad/wesad_binary.npz"
        self.model_path = self.root / "models/deployment_candidate/classifier.keras"
        self.manifest_path = self.root / "models/deployment_candidate/manifest.json"

        self._model = None
        self._manifest = None
        self._dataset_cache = None

        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self._manifest = json.load(f)

    @property
    def dataset_name(self) -> str:
        return "WESAD (Wearable Stress and Affect Detection)"

    @property
    def expected_input_shape(self) -> Tuple[Optional[int], ...]:
        return (3840, 1)

    @property
    def sampling_rate_hz(self) -> Dict[str, float]:
        return {"BVP": 64.0}

    def is_model_available(self) -> bool:
        return self.model_path.exists()

    def get_manifest(self) -> Optional[Dict[str, Any]]:
        return self._manifest

    def load_model(self):
        """Lazy load Keras model."""
        if self._model is None and self.is_model_available():
            import tensorflow as tf
            self._model = tf.keras.models.load_model(str(self.model_path))
        return self._model

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and cache WESAD binary dataset windows, labels, and subject IDs."""
        if self._dataset_cache is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"WESAD dataset not found at {self.data_path}")
            with np.load(self.data_path, allow_pickle=True) as data:
                windows = data["windows"].astype(np.float32)
                labels = data["labels"].astype(int)
                subjects = data["subject_ids"].astype(str)
                self._dataset_cache = (windows, labels, subjects)
        return self._dataset_cache

    def get_subject_list(self) -> List[str]:
        """Return sorted unique list of subjects in the dataset."""
        _, _, subjects = self.load_dataset()
        return sorted(np.unique(subjects).tolist())

    get_subjects = get_subject_list

    def get_subject_windows(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (indices, windows, labels) for a specific subject."""
        windows, labels, subjects = self.load_dataset()
        indices = np.flatnonzero(subjects == subject_id)
        return indices, windows[indices], labels[indices]

    def validate_input(self, data: Any) -> ValidationResult:
        """Validate WESAD input array against contract."""
        errors = []
        warnings = []

        if not isinstance(data, np.ndarray):
            return ValidationResult(is_valid=False, errors=["Input must be a numpy ndarray."])

        # Allow (3840,), (3840, 1), or (1, 3840, 1)
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != 3840:
                errors.append(f"Expected 3840 samples (60 s @ 64 Hz), got shape {arr.shape}.")
            arr = arr.reshape(1, 3840, 1)
        elif arr.ndim == 2:
            if arr.shape == (3840, 1):
                arr = arr.reshape(1, 3840, 1)
            elif arr.shape == (1, 3840):
                arr = arr.reshape(1, 3840, 1)
            else:
                errors.append(f"Invalid 2D shape {arr.shape}. Expected (3840, 1) or (1, 3840).")
        elif arr.ndim == 3:
            if arr.shape[1:] != (3840, 1):
                errors.append(f"Invalid 3D shape {arr.shape}. Expected (N, 3840, 1).")
        else:
            errors.append(f"Unexpected dimensionality {arr.ndim}. Expected 1D, 2D, or 3D.")

        if not np.all(np.isfinite(arr)):
            errors.append("Input contains non-finite values (NaN or Inf).")

        std_val = float(np.std(arr))
        if std_val < 1e-6:
            warnings.append("Near-zero signal variance detected (flatline or detached sensor).")

        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            metadata={"processed_shape": arr.shape if len(errors) == 0 else None, "std": std_val}
        )

    def predict(self, window_data: Any, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Perform CNN-TCN-LSTM model inference on a single 3840-sample window."""
        val = self.validate_input(window_data)
        if not val.is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(val.errors)}")

        model = self.load_model()
        if model is None:
            raise RuntimeError("Model checkpoint could not be loaded.")

        arr = np.asarray(window_data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, 3840, 1)
        elif arr.ndim == 2:
            arr = arr.reshape(1, 3840, 1)

        probs = model.predict(arr, verbose=0)[0]
        # Softmax outputs sum to 1
        probs = np.clip(probs, 0.0, 1.0)
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs = probs / prob_sum

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        ground_truth = metadata.get("ground_truth") if metadata else None
        gt_label = self.CLASS_NAMES.get(ground_truth) if ground_truth is not None else None

        warnings = list(val.warnings)
        warnings.append(
            "Validation-participant score note: softmax probabilities are uncalibrated and reflect model logits."
        )

        return PredictionResult(
            predicted_class=predicted_class,
            class_label=self.CLASS_NAMES[predicted_class],
            probabilities=probs,
            confidence_score=confidence,
            is_calibrated=False,
            ground_truth=ground_truth,
            ground_truth_label=gt_label,
            window_metadata=metadata or {},
            signal_quality_ok=(len(val.warnings) == 0),
            warnings=warnings,
        )

