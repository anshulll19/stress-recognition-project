"""UBFC-Phys physiological data and temporal model inference/replay adapter."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from demo.adapters.base import BaseInferenceAdapter, PredictionResult, ValidationResult


class UBFCPhysInferenceAdapter(BaseInferenceAdapter):
    """Adapter for UBFC-Phys multimodal physiological signals (BVP @ 64 Hz, EDA @ 4 Hz).

    Replays verified out-of-fold predictions from TCN / LSTM temporal models with
    sample-synchronous raw physiological waveforms.
    """

    CLASS_NAMES = {
        0: "Non-stress (Rest Task 1 Proxy)",
        1: "Stress-related (Speech T2 / Math T3 Proxy)",
    }

    TASK_NAMES = {
        1: "T1 — Rest (Baseline)",
        2: "T2 — Speech Preparation / Public Speaking",
        3: "T3 — Mental Arithmetic Stress",
    }

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or Path(__file__).resolve().parents[2]
        self.raw_data_path = self.root / "results/physiology/raw_signal_windows.npz"
        self.metadata_path = self.root / "results/physiology/raw_signal_window_metadata.csv"
        self.oof_pred_path = self.root / "results/temporal_models/tcn_lstm_oof_predictions_full.csv"

        self._raw_cache = None
        self._meta_cache = None
        self._oof_cache = None

    @property
    def dataset_name(self) -> str:
        return "UBFC-Phys (Physiological Stress Recognition)"

    @property
    def expected_input_shape(self) -> Dict[str, Tuple[Optional[int], ...]]:
        return {
            "BVP": (640, 1),  # 10 s @ 64 Hz
            "EDA": (40, 1),   # 10 s @ 4 Hz
        }

    @property
    def sampling_rate_hz(self) -> Dict[str, float]:
        return {
            "BVP": 64.0,
            "EDA": 4.0,
        }

    def is_model_available(self) -> bool:
        return self.oof_pred_path.exists() and self.raw_data_path.exists()

    def load_dataset(self) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
        """Load and cache raw arrays, metadata, and OOF predictions."""
        if self._raw_cache is None:
            if not self.raw_data_path.exists():
                raise FileNotFoundError(f"UBFC raw signals not found at {self.raw_data_path}")
            with np.load(self.raw_data_path, allow_pickle=True) as data:
                self._raw_cache = {
                    "X_bvp": data["X_bvp"].astype(np.float32),
                    "X_eda": data["X_eda"].astype(np.float32),
                    "y": data["y"].astype(int),
                    "groups": data["groups"].astype(int),
                }

        if self._meta_cache is None:
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"UBFC metadata not found at {self.metadata_path}")
            self._meta_cache = pd.read_csv(self.metadata_path)

        if self._oof_cache is None:
            if not self.oof_pred_path.exists():
                raise FileNotFoundError(f"UBFC OOF predictions not found at {self.oof_pred_path}")
            self._oof_cache = pd.read_csv(self.oof_pred_path)

        return self._raw_cache, self._meta_cache, self._oof_cache

    def get_participants(self) -> List[int]:
        """Return sorted list of participant IDs (1..56)."""
        raw, _, _ = self.load_dataset()
        return sorted(np.unique(raw["groups"]).tolist())

    def get_models_available(self) -> List[str]:
        """Return available temporal model architectures (LSTM, TCN)."""
        _, _, oof = self.load_dataset()
        return sorted(oof["model"].unique().tolist())

    def get_participant_windows(
        self, participant_id: int, model_name: str = "LSTM"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Return synchronous BVP, EDA, labels, metadata, and OOF predictions for a participant."""
        raw, meta, oof = self.load_dataset()
        p_mask = (raw["groups"] == participant_id)
        indices = np.flatnonzero(p_mask)

        bvp_sub = raw["X_bvp"][indices]
        eda_sub = raw["X_eda"][indices]
        y_sub = raw["y"][indices]

        meta_sub = meta[meta["participant"] == participant_id].reset_index(drop=True)
        oof_sub = oof[
            (oof["participant"] == participant_id) & (oof["model"] == model_name)
        ].reset_index(drop=True)

        return indices, bvp_sub, eda_sub, y_sub, meta_sub, oof_sub

    def validate_input(self, data: Any) -> ValidationResult:
        """Validate BVP and EDA input arrays."""
        errors = []
        warnings = []

        if not isinstance(data, dict):
            return ValidationResult(is_valid=False, errors=["Data must be a dict with 'bvp' and 'eda' arrays."])

        bvp = data.get("bvp")
        eda = data.get("eda")

        if bvp is None or not isinstance(bvp, np.ndarray):
            errors.append("Missing or invalid 'bvp' numpy array.")
        else:
            if bvp.size != 640:
                errors.append(f"Expected BVP window size 640 (10 s @ 64 Hz), got {bvp.size}.")
            if not np.all(np.isfinite(bvp)):
                errors.append("BVP contains non-finite values (NaN/Inf).")
            elif float(np.std(bvp)) < 1e-6:
                warnings.append("Near-zero BVP variance detected.")

        if eda is None or not isinstance(eda, np.ndarray):
            errors.append("Missing or invalid 'eda' numpy array.")
        else:
            if eda.size != 40:
                errors.append(f"Expected EDA window size 40 (10 s @ 4 Hz), got {eda.size}.")
            if not np.all(np.isfinite(eda)):
                errors.append("EDA contains non-finite values (NaN/Inf).")
            elif float(np.std(eda)) < 1e-6:
                warnings.append("Near-zero EDA variance detected.")

        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            metadata={"bvp_size": bvp.size if bvp is not None else None, "eda_size": eda.size if eda is not None else None}
        )

    def predict(self, window_data: Any, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Retrieve verified out-of-fold temporal model prediction for a specific window index."""
        meta = metadata or {}
        window_idx = meta.get("window_index")
        model_name = meta.get("model_name", "LSTM")

        if window_idx is None:
            raise ValueError("window_index required in metadata for UBFC OOF replay.")

        _, _, oof = self.load_dataset()
        sub = oof[(oof["window_index"] == window_idx) & (oof["model"] == model_name)]

        if len(sub) == 0:
            raise KeyError(f"No prediction found for window_index={window_idx}, model={model_name}")

        row = sub.iloc[0]
        prob_class1 = float(row["probability"])
        prob_class0 = 1.0 - prob_class1
        probs = np.array([prob_class0, prob_class1], dtype=np.float32)

        pred = int(row["prediction"])
        gt = int(row["true"])

        warnings = []
        if isinstance(window_data, dict):
            val = self.validate_input(window_data)
            warnings.extend(val.warnings)

        warnings.append(
            f"UBFC-Phys Out-of-Fold Replay: Evaluated via Leave-One-Subject-Out cross-validation ({model_name})."
        )

        return PredictionResult(
            predicted_class=pred,
            class_label=self.CLASS_NAMES[pred],
            probabilities=probs,
            confidence_score=float(probs[pred]),
            is_calibrated=False,
            ground_truth=gt,
            ground_truth_label=self.CLASS_NAMES[gt],
            window_metadata={
                "participant": int(row["participant"]),
                "window_index": window_idx,
                "model": model_name,
                "task": meta.get("task"),
                "task_name": self.TASK_NAMES.get(meta.get("task")),
                "start_s": meta.get("start_s"),
                "end_s": meta.get("end_s"),
            },
            signal_quality_ok=(len(warnings) <= 1),
            warnings=warnings,
        )

