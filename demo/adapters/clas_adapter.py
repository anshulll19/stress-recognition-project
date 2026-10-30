"""CLAS dataset and pretrained encoder audit adapter.

Enforces project integrity rule: Prevents fabricated classifier predictions
when only self-supervised MAE encoder weights without a fine-tuned head exist.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from demo.adapters.base import BaseInferenceAdapter, PredictionResult, ValidationResult


class CLASInferenceAdapter(BaseInferenceAdapter):
    """Adapter auditing CLAS artifacts and documenting inference limitations."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or Path(__file__).resolve().parents[2]
        self.encoder_weights_path = self.root / "models/pretrained_clas_mae_encoder.weights.h5"
        self.encoder_meta_path = self.root / "models/pretrained_clas_mae_encoder.json"
        self.dataset_path = self.root / "data/clas/clas_binary.npz"
        self.experiment_log_path = self.root / "docs/experiment_log.csv"

        self._encoder_metadata = None
        if self.encoder_meta_path.exists():
            with open(self.encoder_meta_path, "r") as f:
                self._encoder_metadata = json.load(f)

    @property
    def dataset_name(self) -> str:
        return "CLAS (Cognitive Load, Affect and Stress)"

    @property
    def expected_input_shape(self) -> Tuple[Optional[int], ...]:
        return (3840, 1)  # 60 s @ 64 Hz PPG

    @property
    def sampling_rate_hz(self) -> Dict[str, float]:
        return {"PPG": 64.0}

    def is_model_available(self) -> bool:
        """CLAS has only self-supervised MAE encoder weights; no trained classifier head exists."""
        return False

    def get_encoder_metadata(self) -> Optional[Dict[str, Any]]:
        return self._encoder_metadata

    def get_audit_summary(self) -> Dict[str, Any]:
        """Return factual inventory of CLAS artifacts and why inference is disabled."""
        return {
            "has_dataset": self.dataset_path.exists(),
            "dataset_path": str(self.dataset_path),
            "has_encoder_weights": self.encoder_weights_path.exists(),
            "encoder_architecture": self._encoder_metadata.get("model_architecture") if self._encoder_metadata else None,
            "labels_used_in_training": self._encoder_metadata.get("labels_used", False) if self._encoder_metadata else False,
            "has_classifier_head": False,
            "cross_dataset_results": {
                "E2_wesad_to_clas": {"accuracy": 0.6265, "auc": 0.3670, "status": "Sub-random / Domain Shift"},
                "E3_clas_to_wesad": {"accuracy": 0.2389, "auc": 0.2226, "status": "Inverted Ranking / Severe Mismatch"}
            },
            "status": "INFERENCE_DISABLED",
            "reason": (
                "The repository contains self-supervised Masked Autoencoder (MAE) encoder weights "
                "(models/pretrained_clas_mae_encoder.weights.h5) trained without labels (labels_used: false). "
                "No trained binary classification head exists. Furthermore, cross-dataset transfer experiments "
                "(E2 and E3 in docs/experiment_log.csv) revealed severe signal morphology and protocol divergence "
                "(AUC 0.223 - 0.367). Fabricated predictions are strictly prohibited."
            )
        }

    def validate_input(self, data: Any) -> ValidationResult:
        return ValidationResult(
            is_valid=False,
            errors=["CLAS inference is disabled: no trained classifier head exists in the repository."],
            warnings=[
                "Self-supervised MAE encoder weights cannot be used directly as a calibrated binary stress classifier."
            ]
        )

    def predict(self, window_data: Any, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        raise NotImplementedError(
            "CLAS stress inference is deliberately disabled because only self-supervised MAE "
            "encoder weights are present in the repository, with no fine-tuned classification head."
        )

