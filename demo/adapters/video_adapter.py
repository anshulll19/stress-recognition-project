"""Video-based facial distress recognition adapter using Random Forest baseline."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from demo.adapters.base import BaseInferenceAdapter, PredictionResult, ValidationResult


class VideoInferenceAdapter(BaseInferenceAdapter):
    """Adapter for offline and recorded video facial distress recognition.

    Uses the 23-feature Random Forest baseline (models/video_distress_baseline.joblib)
    trained on UBFC-Phys video recordings across 18 participants.
    """

    CLASS_NAMES = {
        0: "Non-stress (Rest Task T1 Proxy)",
        1: "Distress / Stress-task (Speech T2 / Math T3 Proxy)",
    }

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or Path(__file__).resolve().parents[2]
        self.model_path = self.root / "models/video_distress_baseline.joblib"
        self.feature_names_path = self.root / "models/video_feature_names.json"
        self.metadata_path = self.root / "models/video_model_metadata.json"
        self.windows_path = self.root / "results/video/video_windows.csv"
        self.landmarker_path = self.root / "models/face_landmarker.task"

        self._model = None
        self._feature_names = None
        self._metadata = None
        self._windows_df = None

        if self.feature_names_path.exists():
            with open(self.feature_names_path, "r") as f:
                self._feature_names = json.load(f)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self._metadata = json.load(f)

    @property
    def dataset_name(self) -> str:
        return "UBFC-Phys Video (Facial & Behavioral Analysis)"

    @property
    def expected_input_shape(self) -> Tuple[Optional[int], ...]:
        return (23,)

    @property
    def sampling_rate_hz(self) -> Dict[str, float]:
        return {"Video": 30.0}

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None and self.feature_names_path.exists():
            with open(self.feature_names_path, "r") as f:
                self._feature_names = json.load(f)
        return list(self._feature_names or [])

    def is_model_available(self) -> bool:
        return self.model_path.exists() and self.feature_names_path.exists()

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    def load_model(self):
        """Lazy load and patch scikit-learn 1.9 compatibility."""
        if self._model is None and self.is_model_available():
            pipeline = joblib.load(self.model_path)
            # Patch SimpleImputer missing attribute across scikit-learn versions
            if hasattr(pipeline, "named_steps") and "simpleimputer" in pipeline.named_steps:
                imputer = pipeline.named_steps["simpleimputer"]
                if not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "statistics_"):
                    imputer._fill_dtype = imputer.statistics_.dtype
            self._model = pipeline
        return self._model

    def load_windows(self) -> pd.DataFrame:
        """Load and cache precomputed video windows."""
        if self._windows_df is None:
            if not self.windows_path.exists():
                raise FileNotFoundError(f"Video windows file not found at {self.windows_path}")
            self._windows_df = pd.read_csv(self.windows_path)
        return self._windows_df

    def get_subjects(self) -> List[str]:
        """Return list of subject IDs in recorded video dataset (e.g. 's3'..'s20')."""
        df = self.load_windows()
        return sorted(df["subject_id"].unique().tolist())

    def get_subject_windows(self, subject_id: str) -> pd.DataFrame:
        """Return all 54 windows for a given subject."""
        df = self.load_windows()
        return df[df["subject_id"] == subject_id].reset_index(drop=True)

    def validate_input(self, data: Any) -> ValidationResult:
        """Validate 23-feature vector or single-row DataFrame."""
        errors = []
        warnings = []

        feats = self.feature_names

        if isinstance(data, pd.DataFrame):
            missing = [f for f in feats if f not in data.columns]
            if missing:
                errors.append(f"Missing required feature columns: {missing[:5]}...")
            if len(data) == 0:
                errors.append("DataFrame is empty.")
            arr = data[feats].astype(float).values if not missing else None
        elif isinstance(data, pd.Series):
            missing = [f for f in feats if f not in data.index]
            if missing:
                errors.append(f"Missing required features: {missing[:5]}...")
            arr = data[feats].astype(float).values.reshape(1, -1) if not missing else None
        elif isinstance(data, (list, tuple, np.ndarray)):
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                if len(arr) != 23:
                    errors.append(f"Expected 23 features, got length {len(arr)}.")
                arr = arr.reshape(1, -1)
            elif arr.ndim == 2:
                if arr.shape[1] != 23:
                    errors.append(f"Expected shape (N, 23), got {arr.shape}.")
            else:
                errors.append(f"Expected 1D or 2D array, got {arr.ndim}D.")
        else:
            errors.append(f"Unsupported data type: {type(data)}")
            arr = None

        if arr is not None:
            if not np.all(np.isfinite(arr)):
                warnings.append("Feature vector contains NaN or Inf values; imputer will handle if valid.")

        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            metadata={"expected_features": len(feats)}
        )

    def predict(self, window_data: Any, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Perform Random Forest inference on a validated 23-feature vector."""
        val = self.validate_input(window_data)
        if not val.is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(val.errors)}")

        model = self.load_model()
        feats = self.feature_names

        if isinstance(window_data, pd.DataFrame):
            X = window_data[feats].iloc[:1]
        elif isinstance(window_data, pd.Series):
            X = window_data[feats].to_frame().T
        else:
            arr = np.asarray(window_data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X = pd.DataFrame(arr[:1], columns=feats)

        probs = model.predict_proba(X)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        meta = metadata or {}
        ground_truth = meta.get("ground_truth")
        gt_label = self.CLASS_NAMES.get(ground_truth) if ground_truth is not None else None

        warnings = list(val.warnings)
        warnings.append(
            "Video baseline notice: High performance (Bal-Acc 0.967) strongly correlates with mouth/speech movement "
            "during speech task (T2) vs rest (T1). Do not treat as validated internal psychological stress measurement."
        )

        return PredictionResult(
            predicted_class=predicted_class,
            class_label=self.CLASS_NAMES[predicted_class],
            probabilities=probs,
            confidence_score=confidence,
            is_calibrated=False,
            ground_truth=ground_truth,
            ground_truth_label=gt_label,
            window_metadata=meta,
            signal_quality_ok=(meta.get("detection_fraction", 1.0) >= 0.8),
            warnings=warnings,
        )
