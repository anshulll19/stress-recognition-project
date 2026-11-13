"""
physiology.py
=============

Pydantic schemas for the physiological stress inference API.
Provides type validation, serialization models, and explicit schema contracts for
frontend integration with Next.js 16 / React 19 / TypeScript clients.

Enforces:
- 180-second resting baseline protocol (11,520 BVP samples @ 64 Hz, 720 EDA samples @ 4 Hz -> 31 overlapping windows).
- 40-second active task minimum (2,560 BVP samples @ 64 Hz, 160 EDA samples @ 4 Hz -> at least 3 sliding windows).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Core Feature and Signal Schemas
# ============================================================================

class PhysiologicalFeatures(BaseModel):
    """
    Standardized 12-dimensional statistical feature representation extracted
    from 30-second synchronized BVP (64 Hz) and EDA (4 Hz) windows.
    Exact order must match UBFC deployment candidates.
    """
    bvp_mean: float = Field(..., description="Mean blood volume pulse amplitude")
    bvp_std: float = Field(..., description="Standard deviation of BVP amplitude")
    bvp_median: float = Field(..., description="Median of BVP amplitude")
    bvp_iqr: float = Field(..., description="Interquartile range of BVP amplitude")
    bvp_slope_per_sample: float = Field(..., description="Linear trend slope across BVP window")
    bvp_range: float = Field(..., description="Peak-to-peak BVP amplitude range")
    eda_mean: float = Field(..., description="Mean skin conductance level (uS)")
    eda_std: float = Field(..., description="Standard deviation of skin conductance")
    eda_median: float = Field(..., description="Median skin conductance level (uS)")
    eda_iqr: float = Field(..., description="Interquartile range of EDA")
    eda_slope_per_sample: float = Field(..., description="Tonic drift slope across EDA window")
    eda_range: float = Field(..., description="Peak-to-peak skin conductance deflection (uS)")


class WindowResult(BaseModel):
    """V5 EDA-only inference result. Quality scope is recording-level, not per-window."""
    window_index: int
    start_s: float
    end_s: float
    is_reliable: bool = Field(..., description="Window accepted by inference; not independent sensor-quality validation")
    prediction: Optional[int] = Field(None, description="0=rest, 1=distress; None if inference abstained")
    probability: Optional[float] = None
    confidence: Optional[float] = None
    class_name: str
    quality_flag: str
    quality_reason: Optional[str] = None
    latency_ms: float = 0.0
    is_calibrated: bool = True
    features: Optional[PhysiologicalFeatures] = None
    decision_score: Optional[float] = Field(None, description="Raw SVM decision_function score; not a probability")
    score_type: Optional[str] = "raw_svm_decision_function"
    baseline_calibrated: bool = True
    probability_calibrated: bool = False
    quality_scope: str = "recording_level_only"


# ============================================================================
# Session Lifecycle Schemas
# ============================================================================

class SessionCreateRequest(BaseModel):
    """Request payload to initialize an in-memory physiological analysis session."""
    participant_id: str = Field(..., min_length=1, max_length=64, description="Unique participant or subject identifier (e.g. 'P01', 'sub-14')")
    notes: Optional[str] = Field(None, max_length=500, description="Optional metadata or experiment notes")
    model_version: str = Field("v5", description="Inference model: v5 (EDA-only) or v3 (BVP+EDA)")

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"v3", "v5"}:
            raise ValueError("model_version must be 'v3' or 'v5'")
        return value


class SessionResponse(BaseModel):
    """Session representation returned on creation or retrieval."""
    session_id: str = Field(..., description="Unique UUID identifying the session")
    participant_id: str = Field(..., description="Associated participant identifier")
    model_version: str = Field("v5", description="Selected inference model version")
    status: str = Field(..., description="Session state: 'created', 'baseline_uploaded', 'calibrated', 'completed'")
    is_calibrated: bool = Field(..., description="Whether T1 resting baseline calibration has been computed")
    created_at: datetime = Field(..., description="ISO 8601 session creation timestamp")
    updated_at: datetime = Field(..., description="ISO 8601 last activity timestamp")
    completed_tasks: List[str] = Field(default_factory=list, description="List of task identifiers completed in this session")


class SessionStatusResponse(BaseModel):
    """Current detailed calibration and task status for a session."""
    session_id: str = Field(..., description="Session identifier")
    participant_id: str = Field(..., description="Participant identifier")
    status: str = Field(..., description="Current lifecycle state")
    is_calibrated: bool = Field(..., description="Whether T1 baseline calibration has been completed")
    baseline_summary: Optional[Dict[str, Any]] = Field(None, description="Baseline calibration parameters if calibrated")
    completed_tasks: List[str] = Field(default_factory=list, description="Tasks executed so far")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SessionDeleteResponse(BaseModel):
    """Confirmation payload after clearing a session from memory."""
    session_id: str = Field(..., description="Cleared session identifier")
    cleared: bool = Field(..., description="True if the session was successfully removed")
    message: str = Field(..., description="Status message")


# ============================================================================
# Baseline Upload & Calibration Schemas
# ============================================================================

class BaselineUploadJSONRequest(BaseModel):
    """JSON request for T1 baseline physiological signals."""

    participant_id: str = Field(
        ..., description="Must match the session participant identifier"
    )
    bvp_samples: List[float] = Field(
        ..., min_length=1, description="Raw BVP samples"
    )
    eda_samples: List[float] = Field(
        ..., min_length=1, description="Raw EDA samples"
    )
    bvp_fs: float = Field(64.0, gt=0, allow_inf_nan=False)
    eda_fs: float = Field(4.0, gt=0, allow_inf_nan=False)
    task_id: str = Field("T1")

    @field_validator("task_id")
    @classmethod
    def validate_t1(cls, v: str) -> str:
        if v.upper() != "T1":
            raise ValueError(f"Baseline upload must specify task_id='T1', got '{v}'")
        return v.upper()

class BaselineUploadResponse(BaseModel):
    """Response returned upon successful ingestion of T1 baseline signals."""
    session_id: str = Field(..., description="Session identifier")
    participant_id: str = Field(..., description="Participant identifier")
    task_id: str = Field("T1", description="Task identifier")
    duration_s: float = Field(..., description="Synchronous duration of ingested signals in seconds")
    bvp_samples: int = Field(..., description="Total ingested BVP samples")
    eda_samples: int = Field(..., description="Total ingested EDA samples")
    bvp_fs: float = Field(64.0, description="BVP sampling frequency in Hz")
    eda_fs: float = Field(4.0, description="EDA sampling frequency in Hz")
    status: str = Field(..., description="Current session state (e.g. 'baseline_uploaded')")
    message: str = Field(..., description="Next step guidance (call /baseline/calibrate)")


class CalibrationResponse(BaseModel):
    """V5 EDA-only participant baseline calibration response."""

    session_id: str
    participant_id: str
    is_calibrated: bool = False

    status: str = "not_calibrated"
    abstained: bool = False
    warnings: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    calibration_windows: int = 0
    calibration_duration_s: float = 0.0
    feature_count: int = 19

    calibration_method: str = "participant-specific median/IQR"
    probability_calibrated: bool = False

    # Retained as optional legacy fields for frontend compatibility.
    n_total_windows: Optional[int] = None
    n_reliable_windows: Optional[int] = None
    n_unreliable_windows: Optional[int] = None
    baseline_duration_s: Optional[float] = None
    feature_means: Optional[Dict[str, float]] = None
    feature_stds: Optional[Dict[str, float]] = None
    baseline_stress_percentage: Optional[float] = None
    mean_baseline_probability: Optional[float] = None

    message: str


# ============================================================================
# Task Inference Schemas (T2 Speech, T3 Arithmetic)
# ============================================================================

class TaskInferenceJSONRequest(BaseModel):
    """JSON request for T2/T3 physiological inference."""

    participant_id: str = Field(
        ..., description="Must match the session participant identifier"
    )
    bvp_samples: List[float] = Field(..., min_length=1)
    eda_samples: List[float] = Field(..., min_length=1)
    bvp_fs: float = Field(64.0, gt=0, allow_inf_nan=False)
    eda_fs: float = Field(4.0, gt=0, allow_inf_nan=False)

class TaskSummaryMetrics(BaseModel):
    """Task summary. Stress percentage is the fraction of accepted windows labelled distress."""
    task_id: str
    task_name: str
    participant_id: str
    total_windows: int = 0
    reliable_windows: int = 0
    unreliable_windows: int = 0
    stress_windows: int = 0
    non_stress_windows: int = 0
    stress_percentage: Optional[float] = None
    mean_stress_probability: Optional[float] = None
    peak_stress_probability: Optional[float] = None
    min_stress_probability: Optional[float] = None
    mean_confidence: Optional[float] = None
    mean_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    dominant_state: str = "inconclusive"
    is_calibrated: bool = True
    mean_decision_score: Optional[float] = None
    min_decision_score: Optional[float] = None
    max_decision_score: Optional[float] = None
    score_type: str = "raw_svm_decision_function"
    probability_calibrated: bool = False


class TaskInferenceResponse(BaseModel):
    """V5 task inference response, including abstention and diagnostics."""
    session_id: str
    participant_id: str
    task_id: str
    summary: TaskSummaryMetrics
    windows: List[WindowResult]
    status: str = "accepted"
    abstained: bool = False
    warnings: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Session Multi-Task Summary Schemas
# ============================================================================

class ReactivityProfile(BaseModel):
    """Physiological reactivity comparing stress tasks relative to resting baseline."""
    baseline_t1_prob: Optional[float] = Field(None, description="Average P(stress) during T1 rest")
    speech_t2_prob: Optional[float] = Field(None, description="Average P(stress) during T2 speech")
    arithmetic_t3_prob: Optional[float] = Field(None, description="Average P(stress) during T3 arithmetic")
    delta_t2_vs_baseline: Optional[float] = Field(None, description="Change in P(stress) from T1 to T2")
    delta_t3_vs_baseline: Optional[float] = Field(None, description="Change in P(stress) from T1 to T3")
    reactivity_class: str = Field(..., description="'NORMAL', 'HIGH_REACTIVITY', 'LOW_REACTIVITY', or 'MODERATE_REACTIVITY'")


class SessionSummaryResponse(BaseModel):
    """Comprehensive multi-task summary across all completed tasks in a session."""
    session_id: str = Field(..., description="Session identifier")
    participant_id: str = Field(..., description="Participant identifier")
    is_calibrated: bool = Field(..., description="Whether baseline calibration is active")
    tasks_completed: List[str] = Field(..., description="Completed task identifiers (e.g. ['T1', 'T2', 'T3'])")
    tasks: Dict[str, TaskSummaryMetrics] = Field(..., description="Task-level summaries keyed by task ID")
    reactivity_profile: ReactivityProfile = Field(..., description="Autonomic reactivity analysis")
    interpretation: str = Field(..., description="Standardized research interpretation and disclaimers")


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Standardized client-facing error envelope."""
    error_code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable explanation")
    details: Optional[Dict[str, Any]] = Field(None, description="Contextual validation or failure details")
