"""
multimodal_fusion.py
=====================
Physiology-first, video-assisted multimodal stress prediction fusion engine.

Architecture:
1. Physiological inference primary (BVP+EDA via UBFC or WESAD model)
2. Upper-face video inference secondary (19 non-confounded features)
3. Dynamic quality-aware blend:
   - If physio quality OK and face detected:    0.70 physio + 0.30 video
   - If physio degraded (DROPOUT) and face OK:  0.00 physio + 1.00 video (full fallback)
   - If physio OK and no face:                  1.00 physio + 0.00 video
   - If both degraded:                          Return quality_flag='BOTH_DEGRADED'

Features loaded from saved artifacts - does NOT require live hardware.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root (one level above src/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Lazy import of physiology engines (avoids circular imports)
# ---------------------------------------------------------------------------
def _import_physiology():
    """Return (WESADInferenceEngine, UBFCInferenceEngine, PredictionResult) lazily."""
    import importlib, sys

    # Make sure src/ is on the path when imported as a standalone script
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    mod = importlib.import_module("physiology_inference")
    return mod.WESADInferenceEngine, mod.UBFCInferenceEngine, mod.PredictionResult


# ============================================================================
# FusionResult dataclass
# ============================================================================

@dataclass
class FusionResult:
    """Standardized output of the multimodal fusion engine."""

    prediction: int              # 0 = non-stress, 1 = stress
    probability: float           # P(stress), fused
    class_name: str              # 'stress' or 'non-stress'
    confidence: float            # max(P, 1-P)
    p_physio: Optional[float]    # physio stream P(stress), or None
    p_video: Optional[float]     # video stream P(stress), or None
    physio_quality: str          # 'OK', 'DROPOUT', 'UNAVAILABLE'
    video_quality: str           # 'OK', 'NO_FACE', 'UNAVAILABLE'
    blend_weights: tuple         # (w_physio, w_video) actually used
    video_feature_set: str       # 'upper_face_19' or 'full_23'
    modality: str                # 'PHYSIO_ONLY', 'VIDEO_ONLY', 'FUSED'
    latency_ms: float            # wall-clock time for the fused prediction
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "prediction": self.prediction,
            "probability": round(self.probability, 4),
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "p_physio": round(self.p_physio, 4) if self.p_physio is not None else None,
            "p_video": round(self.p_video, 4) if self.p_video is not None else None,
            "physio_quality": self.physio_quality,
            "video_quality": self.video_quality,
            "blend_weights": list(self.blend_weights),
            "video_feature_set": self.video_feature_set,
            "modality": self.modality,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }


# ============================================================================
# sklearn imputer compatibility patch
# ============================================================================

def _patch_imputer(pipeline_or_model):
    """
    Apply `_fill_dtype` patch to any SimpleImputer inside a sklearn Pipeline
    to silence dtype-mismatch warnings in older scikit-learn versions.
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        steps = []
        if isinstance(pipeline_or_model, Pipeline):
            steps = [step for _, step in pipeline_or_model.steps]
        else:
            steps = [pipeline_or_model]

        for obj in steps:
            if isinstance(obj, SimpleImputer):
                if not hasattr(obj, "_fill_dtype"):
                    obj._fill_dtype = obj.statistics_.dtype
            # Also recurse into nested pipelines
            if isinstance(obj, Pipeline):
                _patch_imputer(obj)
    except Exception:
        pass  # Non-fatal: patch is best-effort


# ============================================================================
# 1. UpperFaceVideoEngine
# ============================================================================

class UpperFaceVideoEngine:
    """
    Inference engine for the upper-face video stress detection Random Forest.

    Loads `models/video_distress_upperface.joblib` (19 non-confounded upper-face
    features) or optionally the original 23-feature baseline model.

    Parameters
    ----------
    use_full_features : bool
        If True, load the 23-feature baseline model instead of the 19-feature
        upper-face model. Defaults to False (upper-face model is preferred).
    model_dir : Path | str, optional
        Override the default models/ directory.
    """

    UPPER_FACE_MODEL = "video_distress_upperface.joblib"
    UPPER_FACE_FEATURES_JSON = "video_upperface_features.json"
    FULL_MODEL = "video_distress_baseline.joblib"
    N_UPPER_FACE_FEATURES = 19
    N_FULL_FEATURES = 23

    def __init__(
        self,
        use_full_features: bool = False,
        model_dir: Optional[Union[str, Path]] = None,
    ):
        self.use_full_features = use_full_features
        self.model_dir = Path(model_dir) if model_dir else _PROJECT_ROOT / "models"

        self.model = None
        self.feature_names: List[str] = []
        self.n_features: int = 0
        self.video_feature_set: str = ""

        self._load()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self):
        """Load the appropriate model and feature list from disk."""
        if self.use_full_features:
            model_path = self.model_dir / self.FULL_MODEL
            self.n_features = self.N_FULL_FEATURES
            self.video_feature_set = "full_23"
            # Full model does not ship a dedicated feature-names JSON;
            # derive feature names from model internals if possible.
            self.feature_names = self._infer_feature_names(model_path, self.N_FULL_FEATURES)
        else:
            model_path = self.model_dir / self.UPPER_FACE_MODEL
            features_path = self.model_dir / self.UPPER_FACE_FEATURES_JSON
            self.n_features = self.N_UPPER_FACE_FEATURES
            self.video_feature_set = "upper_face_19"

            if not features_path.exists():
                raise FileNotFoundError(
                    f"Upper-face feature names JSON not found: {features_path}"
                )
            with features_path.open("r") as fh:
                self.feature_names = json.load(fh)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Video model not found: {model_path}. "
                "Ensure models/video_distress_upperface.joblib is present."
            )

        self.model = joblib.load(model_path)
        _patch_imputer(self.model)

    @staticmethod
    def _infer_feature_names(model_path: Path, n: int) -> List[str]:
        """Try to read feature names from model; fall back to generic names."""
        try:
            m = joblib.load(model_path)
            _patch_imputer(m)
            from sklearn.pipeline import Pipeline
            clf = m
            if isinstance(m, Pipeline):
                clf = m[-1]
            if hasattr(clf, "feature_names_in_"):
                return list(clf.feature_names_in_)
        except Exception:
            pass
        return [f"feature_{i}" for i in range(n)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_vector: Union[dict, np.ndarray, pd.Series],
    ) -> float:
        """
        Predict P(stress) from a feature vector.

        Parameters
        ----------
        feature_vector : dict, ndarray, or pd.Series
            - dict: keys must include all feature names (extras are ignored)
            - ndarray: shape (19,) or (1, 19)  [or (23,) for full model]
            - pd.Series: index must include all feature names

        Returns
        -------
        float
            P(stress) in [0.0, 1.0].
        """
        vec = self._coerce_to_array(feature_vector)
        proba = self.model.predict_proba(vec)
        return float(proba[0, 1])

    def _coerce_to_array(
        self,
        feature_vector: Union[dict, np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Convert various input types to a (1, n_features) float array."""
        if isinstance(feature_vector, pd.Series):
            feature_vector = feature_vector.to_dict()

        if isinstance(feature_vector, dict):
            missing = [f for f in self.feature_names if f not in feature_vector]
            if missing:
                raise ValueError(
                    f"Feature dict missing keys required by {self.video_feature_set} model: {missing}"
                )
            vec = np.array([[float(feature_vector[f]) for f in self.feature_names]])
        elif isinstance(feature_vector, np.ndarray):
            arr = np.asarray(feature_vector, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != self.n_features:
                raise ValueError(
                    f"Expected {self.n_features} features, got {arr.shape[1]}."
                )
            vec = arr
        else:
            raise TypeError(
                f"Unsupported feature_vector type: {type(feature_vector)}. "
                "Expected dict, np.ndarray, or pd.Series."
            )

        if not np.isfinite(vec).all():
            raise ValueError("Feature vector contains NaN or infinite values.")

        return vec


# ============================================================================
# 2. PhysiologyFirstFusionEngine
# ============================================================================

class PhysiologyFirstFusionEngine:
    """
    Physiology-first dynamic quality-aware fusion engine.

    Blending logic
    --------------
    +---------------------+-----------------+----------------------------+
    | physio_quality      | video_quality   | weights (physio, video)    |
    +=====================+=================+============================+
    | OK                  | OK              | physio_weight, video_weight|
    | OK                  | NO_FACE /       | 1.00, 0.00 (PHYSIO_ONLY)   |
    |                     | UNAVAILABLE     |                            |
    | DROPOUT / UNAVAIL.  | OK              | 0.00, 1.00 (VIDEO_ONLY)    |
    | DROPOUT / UNAVAIL.  | NO_FACE /       | quality_flag='BOTH_DEGRADED|
    |                     | UNAVAILABLE     |                            |
    +---------------------+-----------------+----------------------------+

    Parameters
    ----------
    physio_weight : float
        Default physio contribution when both modalities are OK (default 0.70).
    video_weight : float
        Default video contribution when both modalities are OK (default 0.30).
    physio_dropout_threshold : float
        Probability below which a physio DROPOUT result is clamped to 0.5
        before blending (default 0.5; effectively a no-signal prior).
    video_feature_set : str
        'upper_face_19' or 'full_23'. Recorded in FusionResult for traceability.
    """

    _PHYSIO_DEGRADED = {"DROPOUT", "UNAVAILABLE"}
    _VIDEO_DEGRADED = {"NO_FACE", "UNAVAILABLE"}

    def __init__(
        self,
        physio_weight: float = 0.70,
        video_weight: float = 0.30,
        physio_dropout_threshold: float = 0.5,
        video_feature_set: str = "upper_face_19",
    ):
        if abs(physio_weight + video_weight - 1.0) > 1e-6:
            raise ValueError(
                f"physio_weight + video_weight must sum to 1.0, "
                f"got {physio_weight} + {video_weight} = {physio_weight + video_weight}"
            )
        self.physio_weight = physio_weight
        self.video_weight = video_weight
        self.physio_dropout_threshold = physio_dropout_threshold
        self.video_feature_set = video_feature_set

    # ------------------------------------------------------------------
    # Primary public API: physio PredictionResult + optional video prob
    # ------------------------------------------------------------------

    def predict(
        self,
        physio_result,          # PredictionResult from physiology_inference
        video_prob: Optional[float] = None,
        video_quality: str = "OK",
    ) -> FusionResult:
        """
        Fuse a PredictionResult from a physiology engine with an optional
        video probability.

        Parameters
        ----------
        physio_result : PredictionResult
            Output of WESADInferenceEngine or UBFCInferenceEngine.
        video_prob : float, optional
            P(stress) from the video model in [0, 1], or None if unavailable.
        video_quality : str
            'OK', 'NO_FACE', or 'UNAVAILABLE'.

        Returns
        -------
        FusionResult
        """
        t0 = time.perf_counter()

        p_physio = physio_result.probability
        physio_quality = physio_result.quality_flag  # 'OK', 'DROPOUT', ...

        if video_prob is None:
            video_quality = "UNAVAILABLE"

        result = self.predict_from_oof(
            physio_prob=p_physio,
            video_prob=video_prob,
            physio_quality=physio_quality,
            video_quality=video_quality,
        )

        # Augment metadata with physio modality info
        result.metadata.update(
            {
                "physio_modality": physio_result.modality,
                "physio_is_calibrated": physio_result.is_calibrated,
                "physio_latency_ms": physio_result.latency_ms,
            }
        )
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------------
    # Secondary public API: work from pre-computed OOF probabilities
    # ------------------------------------------------------------------

    def predict_from_oof(
        self,
        physio_prob: Optional[float],
        video_prob: Optional[float],
        physio_quality: str = "OK",
        video_quality: str = "OK",
    ) -> FusionResult:
        """
        Fuse pre-computed physio and video P(stress) values using
        dynamic quality-aware weighting.

        Parameters
        ----------
        physio_prob : float or None
            P(stress) from the physiological model (OOF or live).
        video_prob : float or None
            P(stress) from the video model (OOF or live).
        physio_quality : str
            'OK', 'DROPOUT', or 'UNAVAILABLE'.
        video_quality : str
            'OK', 'NO_FACE', or 'UNAVAILABLE'.

        Returns
        -------
        FusionResult
        """
        t0 = time.perf_counter()

        # Normalise quality strings
        physio_quality = physio_quality.upper()
        video_quality = video_quality.upper()

        if physio_prob is None:
            physio_quality = "UNAVAILABLE"
        if video_prob is None:
            video_quality = "UNAVAILABLE"

        physio_ok = physio_quality not in self._PHYSIO_DEGRADED
        video_ok = video_quality not in self._VIDEO_DEGRADED

        # --- Quality gate ---
        if not physio_ok and not video_ok:
            # Both degraded — return a sentinel result
            latency = (time.perf_counter() - t0) * 1000.0
            return FusionResult(
                prediction=0,
                probability=0.5,
                class_name="non-stress",
                confidence=0.5,
                p_physio=physio_prob,
                p_video=video_prob,
                physio_quality=physio_quality,
                video_quality=video_quality,
                blend_weights=(0.0, 0.0),
                video_feature_set=self.video_feature_set,
                modality="BOTH_DEGRADED",
                latency_ms=latency,
                metadata={"quality_flag": "BOTH_DEGRADED"},
            )

        # --- Determine blend weights ---
        if physio_ok and video_ok:
            w_physio = self.physio_weight
            w_video = self.video_weight
            modality = "FUSED"
        elif physio_ok and not video_ok:
            w_physio = 1.0
            w_video = 0.0
            modality = "PHYSIO_ONLY"
        else:  # not physio_ok and video_ok
            w_physio = 0.0
            w_video = 1.0
            modality = "VIDEO_ONLY"

        # --- Compute fused probability ---
        p_physio_used = float(physio_prob) if physio_prob is not None else 0.5
        p_video_used = float(video_prob) if video_prob is not None else 0.5

        # Clamp physio dropout contribution to neutral prior
        if physio_quality == "DROPOUT" and w_physio > 0:
            p_physio_used = self.physio_dropout_threshold

        fused_prob = w_physio * p_physio_used + w_video * p_video_used
        fused_prob = float(np.clip(fused_prob, 0.0, 1.0))

        prediction = int(fused_prob >= 0.5)
        class_name = "stress" if prediction == 1 else "non-stress"
        confidence = float(max(fused_prob, 1.0 - fused_prob))
        latency = (time.perf_counter() - t0) * 1000.0

        return FusionResult(
            prediction=prediction,
            probability=fused_prob,
            class_name=class_name,
            confidence=confidence,
            p_physio=physio_prob,
            p_video=video_prob,
            physio_quality=physio_quality,
            video_quality=video_quality,
            blend_weights=(w_physio, w_video),
            video_feature_set=self.video_feature_set,
            modality=modality,
            latency_ms=latency,
            metadata={},
        )


# ============================================================================
# 3. MultimodalHUDEngine — top-level wrapper
# ============================================================================

class MultimodalHUDEngine:
    """
    Top-level multimodal HUD (Heads-Up Display) engine.

    Lazily loads sub-engines and provides convenience methods for OOF-based
    benchmarking and subject-level timelines.

    Parameters
    ----------
    physio_weight : float
        Default physio blend weight (default 0.70).
    video_weight : float
        Default video blend weight (default 0.30).
    use_full_video_features : bool
        If True, use the 23-feature baseline video model instead of the
        recommended 19-feature upper-face model (default False).
    oof_csv_path : Path | str, optional
        Override path to aligned_oof_predictions.csv.
    fusion_summary_path : Path | str, optional
        Override path to fusion_summary.csv.
    """

    _OOF_CSV_REL = "results/fusion_lstm/aligned_oof_predictions.csv"
    _FUSION_SUMMARY_REL = "results/fusion_lstm/fusion_summary.csv"

    def __init__(
        self,
        physio_weight: float = 0.70,
        video_weight: float = 0.30,
        use_full_video_features: bool = False,
        oof_csv_path: Optional[Union[str, Path]] = None,
        fusion_summary_path: Optional[Union[str, Path]] = None,
    ):
        self.physio_weight = physio_weight
        self.video_weight = video_weight
        self.use_full_video_features = use_full_video_features

        self.oof_csv_path = (
            Path(oof_csv_path) if oof_csv_path
            else _PROJECT_ROOT / self._OOF_CSV_REL
        )
        self.fusion_summary_path = (
            Path(fusion_summary_path) if fusion_summary_path
            else _PROJECT_ROOT / self._FUSION_SUMMARY_REL
        )

        # Lazily initialised sub-engines
        self._video_engine: Optional[UpperFaceVideoEngine] = None
        self._fusion_engine: Optional[PhysiologyFirstFusionEngine] = None
        self._oof_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    @property
    def video_engine(self) -> UpperFaceVideoEngine:
        """Lazy-load the video inference engine."""
        if self._video_engine is None:
            self._video_engine = UpperFaceVideoEngine(
                use_full_features=self.use_full_video_features
            )
        return self._video_engine

    @property
    def fusion_engine(self) -> PhysiologyFirstFusionEngine:
        """Lazy-load the fusion engine."""
        if self._fusion_engine is None:
            self._fusion_engine = PhysiologyFirstFusionEngine(
                physio_weight=self.physio_weight,
                video_weight=self.video_weight,
                video_feature_set=self.video_engine.video_feature_set,
            )
        return self._fusion_engine

    @property
    def oof_df(self) -> pd.DataFrame:
        """Lazy-load the aligned OOF predictions CSV."""
        if self._oof_df is None:
            if not self.oof_csv_path.exists():
                raise FileNotFoundError(
                    f"Aligned OOF predictions CSV not found: {self.oof_csv_path}"
                )
            self._oof_df = pd.read_csv(self.oof_csv_path)
        return self._oof_df

    # ------------------------------------------------------------------
    # OOF-based prediction (primary benchmarking path)
    # ------------------------------------------------------------------

    def predict_oof_window(self, row: pd.Series) -> FusionResult:
        """
        Produce a FusionResult from a single row of aligned_oof_predictions.csv.

        Expected columns in *row*:
          - ``p_physio``  : physio OOF P(stress)
          - ``p_video``   : video OOF P(stress)
          - ``detection_fraction`` : fraction of frames with a detected face (0–1)

        Quality flags are inferred:
          - physio: 'DROPOUT' if p_physio is NaN, else 'OK'
          - video:  'NO_FACE' if detection_fraction < 0.5, else 'OK'

        Parameters
        ----------
        row : pd.Series
            A single row from aligned_oof_predictions.csv (or equivalent).

        Returns
        -------
        FusionResult
        """
        t0 = time.perf_counter()

        # Extract physio probability and quality
        p_physio_raw = row.get("p_physio", None)
        if p_physio_raw is None or (isinstance(p_physio_raw, float) and np.isnan(p_physio_raw)):
            p_physio = None
            physio_quality = "UNAVAILABLE"
        else:
            p_physio = float(p_physio_raw)
            physio_quality = "OK"

        # Extract video probability and quality
        p_video_raw = row.get("p_video", None)
        det_frac = float(row.get("detection_fraction", 1.0))

        if p_video_raw is None or (isinstance(p_video_raw, float) and np.isnan(p_video_raw)):
            p_video = None
            video_quality = "UNAVAILABLE"
        elif det_frac < 0.5:
            p_video = float(p_video_raw)
            video_quality = "NO_FACE"
        else:
            p_video = float(p_video_raw)
            video_quality = "OK"

        result = self.fusion_engine.predict_from_oof(
            physio_prob=p_physio,
            video_prob=p_video,
            physio_quality=physio_quality,
            video_quality=video_quality,
        )

        # Enrich metadata from the OOF row
        result.metadata.update(
            {
                "subject_id": str(row.get("subject_id", "")),
                "task": str(row.get("task", "")),
                "window_start_sec": float(row.get("window_start_sec", 0.0)),
                "window_end_sec": float(row.get("window_end_sec", 0.0)),
                "y_true": int(row.get("y_true", -1)),
                "detection_fraction": det_frac,
            }
        )
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------------
    # Subject timeline
    # ------------------------------------------------------------------

    def get_oof_timeline(self, subject_id: str) -> List[FusionResult]:
        """
        Return a chronologically ordered list of FusionResult objects for all
        OOF windows belonging to *subject_id*.

        Parameters
        ----------
        subject_id : str
            Subject identifier as it appears in the 'subject_id' column of
            aligned_oof_predictions.csv (e.g., 's3').

        Returns
        -------
        List[FusionResult]
        """
        df = self.oof_df
        mask = df["subject_id"].astype(str) == str(subject_id)
        subject_df = df[mask].sort_values("window_start_sec", na_position="last")

        results: List[FusionResult] = []
        for _, row in subject_df.iterrows():
            results.append(self.predict_oof_window(row))
        return results

    # ------------------------------------------------------------------
    # Benchmark summary
    # ------------------------------------------------------------------

    def get_benchmark_summary(self) -> pd.DataFrame:
        """
        Load and return the fusion benchmark summary CSV.

        Returns
        -------
        pd.DataFrame
            Contents of results/fusion_lstm/fusion_summary.csv.

        Raises
        ------
        FileNotFoundError
            If fusion_summary.csv does not exist.
        """
        if not self.fusion_summary_path.exists():
            raise FileNotFoundError(
                f"Fusion summary CSV not found: {self.fusion_summary_path}"
            )
        return pd.read_csv(self.fusion_summary_path)


# ============================================================================
# Module-level convenience exports
# ============================================================================

__all__ = [
    "FusionResult",
    "UpperFaceVideoEngine",
    "PhysiologyFirstFusionEngine",
    "MultimodalHUDEngine",
]
