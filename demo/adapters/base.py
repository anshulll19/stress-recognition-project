"""Base adapter interfaces and validation contracts for stress recognition models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ValidationResult:
    """Result of data contract and input validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Standardized model prediction result container."""
    predicted_class: int                   # 0 = non-stress, 1 = stress
    class_label: str                       # Human-readable semantic label
    probabilities: np.ndarray              # [prob_class_0, prob_class_1]
    confidence_score: float                # probability of predicted class
    is_calibrated: bool                    # whether probabilities are empirically calibrated
    ground_truth: Optional[int] = None     # True label if available
    ground_truth_label: Optional[str] = None
    window_metadata: Dict[str, Any] = field(default_factory=dict)
    signal_quality_ok: bool = True
    warnings: List[str] = field(default_factory=list)


class BaseInferenceAdapter(ABC):
    """Abstract base class for dataset-specific inference adapters."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name of the dataset (e.g. 'UBFC-Phys', 'WESAD')."""
        pass

    @property
    @abstractmethod
    def expected_input_shape(self) -> Tuple[Optional[int], ...]:
        """Expected single-window or batch input shape."""
        pass

    @property
    @abstractmethod
    def sampling_rate_hz(self) -> Dict[str, float]:
        """Expected sampling rate per channel."""
        pass

    @abstractmethod
    def is_model_available(self) -> bool:
        """Whether a valid model or saved predictions exist for inference."""
        pass

    @abstractmethod
    def validate_input(self, data: Any) -> ValidationResult:
        """Validate shapes, finite values, and channel counts before inference."""
        pass

    @abstractmethod
    def predict(self, window_data: Any, metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """Perform model inference on a validated window."""
        pass

