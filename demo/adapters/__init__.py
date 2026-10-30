"""Inference adapters package for multimodal stress recognition."""
from demo.adapters.base import BaseInferenceAdapter, PredictionResult, ValidationResult
from demo.adapters.wesad_adapter import WESADInferenceAdapter
from demo.adapters.ubfc_adapter import UBFCPhysInferenceAdapter
from demo.adapters.video_adapter import VideoInferenceAdapter
from demo.adapters.fusion_adapter import MultimodalFusionAdapter
from demo.adapters.trend_adapter import TemporalTrendAdapter, TrendConfig
from demo.adapters.clas_adapter import CLASInferenceAdapter

__all__ = [
    "BaseInferenceAdapter",
    "PredictionResult",
    "ValidationResult",
    "WESADInferenceAdapter",
    "UBFCPhysInferenceAdapter",
    "VideoInferenceAdapter",
    "MultimodalFusionAdapter",
    "TemporalTrendAdapter",
    "TrendConfig",
    "CLASInferenceAdapter",
]

