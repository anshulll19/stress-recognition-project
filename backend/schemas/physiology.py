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
    """
    Inference result and signal quality evaluation for a single 30s sliding window.
    Crucial: Unreliable or dropout windows have prediction=None and probability=None,
    guaranteeing they are never treated as non-stress (0).
    """
    window_index: int = Field(..., description="Zero-indexed window sequence number")
    start_s: float = Field(..., description="Window start offset in seconds from recording onset")
    end_s: float = Field(..., description="Window end offset in seconds from recording onset")
    is_reliable: bool = Field(..., description="True if sensor contact was valid and variance was non-zero")
    prediction: Optional[int] = Field(None, description="0=non-stress, 1=stress, None=unreliable/dropout")
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="P(stress) in [0.0, 1.0], None if unreliable")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence max(P, 1-P), None if unreliable")
    class_name: str = Field(..., description="'stress', 'non-stress', or 'unreliable'")
    quality_flag: str = Field(..., description="Signal quality status: 'OK', 'DROPOUT', or 'UNRELIABLE'")
    quality_reason: Optional[str] = Field(None, description="Detailed explanation of sensor or extraction failure")
    latency_ms: float = Field(..., ge=0.0, description="Inference latency for this window in milliseconds")
    is_calibrated: bool = Field(..., description="Whether participant rest calibration was applied")
    features: Optional[PhysiologicalFeatures] = Field(None, description="Extracted 12 features for this window")


# ============================================================================
# Session Lifecycle Schemas
# ============================================================================

class SessionCreateRequest(BaseModel):
    """Request payload to initialize an in-memory physiological analysis session."""
    participant_id: str = Field(..., min_length=1, max_length=64, description="Unique participant or subject identifier (e.g. 'P01', 'sub-14')")
    notes: Optional[str] = Field(None, max_length=500, description="Optional metadata or experiment notes")


class SessionResponse(BaseModel):
    """Session representation returned on creation or retrieval."""
    session_id: str = Field(..., description="Unique UUID identifying the session")
    participant_id: str = Field(..., description="Associated participant identifier")
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
    """
    JSON request body for uploading T1 baseline signals when not using multipart file upload.
    Original UBFC Research Protocol: 180 seconds of resting baseline.
    BVP nominal rate: 64 Hz (11,520 samples). EDA nominal rate: 4 Hz (720 samples).
    """
    participant_id: str = Field(..., description="Must match the session participant identifier")
    bvp_samples: List[float] = Field(..., min_length=11520, description="Raw BVP samples (180s @ 64 Hz = 11,520 samples)")
    eda_samples: List[float] = Field(..., min_length=720, description="Raw EDA samples (180s @ 4 Hz = 720 samples)")
    task_id: str = Field("T1", description="Baseline task identifier, must be 'T1'")

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
    """Response returned upon successful execution of resting baseline calibration."""
    session_id: str = Field(..., description="Session identifier")
    participant_id: str = Field(..., description="Participant identifier")
    is_calibrated: bool = Field(True, description="Always true upon successful calibration")
    n_total_windows: int = Field(..., description="Total 30s windows sliced from T1 (expected 31 windows for 180s)")
    n_reliable_windows: int = Field(..., description="Count of reliable baseline windows used for calibration (>= 25)")
    n_unreliable_windows: int = Field(..., description="Count of rejected dropout/unreliable baseline windows")
    baseline_duration_s: float = Field(..., description="Baseline recording duration in seconds")
    feature_means: Dict[str, float] = Field(..., description="Resting baseline feature means (mu_rest)")
    feature_stds: Dict[str, float] = Field(..., description="Resting baseline feature standard deviations (sigma_rest)")
    baseline_stress_percentage: float = Field(..., description="Stress percentage observed during resting baseline (expected near 0%)")
    mean_baseline_probability: Optional[float] = Field(None, description="Average P(stress) during baseline")
    message: str = Field(..., description="Guidance for proceeding with T2/T3 stress inference")


# ============================================================================
# Task Inference Schemas (T2 Speech, T3 Arithmetic)
# ============================================================================

class TaskInferenceJSONRequest(BaseModel):
    """
    JSON request body for executing stress inference on active tasks (T2 or T3).
    Minimum duration: 40 seconds (2,560 BVP samples @ 64 Hz, 160 EDA samples @ 4 Hz).
    """
    participant_id: str = Field(..., description="Must match the session participant identifier")
    bvp_samples: List[float] = Field(..., min_length=2560, description="Raw BVP samples (minimum 40s @ 64 Hz = 2,560 samples)")
    eda_samples: List[float] = Field(..., min_length=160, description="Raw EDA samples (minimum 40s @ 4 Hz = 160 samples)")


class TaskSummaryMetrics(BaseModel):
    """Summary metrics aggregating predictions across all sliding windows in a task."""
    task_id: str = Field(..., description="'T1', 'T2', or 'T3'")
    task_name: str = Field(..., description="Descriptive task label")
    participant_id: str = Field(..., description="Participant identifier")
    total_windows: int = Field(..., description="Total 30s windows evaluated")
    reliable_windows: int = Field(..., description="Windows with valid sensor contact")
    unreliable_windows: int = Field(..., description="Windows with dropout or sensor artifacts")
    stress_windows: int = Field(..., description="Reliable windows classified as stress (prediction=1)")
    non_stress_windows: int = Field(..., description="Reliable windows classified as non-stress (prediction=0)")
    stress_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of reliable windows classified as stress")
    mean_stress_probability: Optional[float] = Field(None, description="Average P(stress) over reliable windows")
    peak_stress_probability: Optional[float] = Field(None, description="Maximum P(stress) over reliable windows")
    min_stress_probability: Optional[float] = Field(None, description="Minimum P(stress) over reliable windows")
    mean_confidence: Optional[float] = Field(None, description="Average model confidence over reliable windows")
    mean_latency_ms: float = Field(..., description="Average inference latency per window in milliseconds")
    total_latency_ms: float = Field(..., description="Total inference latency across all windows in milliseconds")
    dominant_state: str = Field(..., description="'stress', 'non-stress', or 'inconclusive'")
    is_calibrated: bool = Field(True, description="Whether T1 participant calibration was applied")


class TaskInferenceResponse(BaseModel):
    """Complete inference response for an executed task, including summary and window timeline."""
    session_id: str = Field(..., description="Session identifier")
    participant_id: str = Field(..., description="Participant identifier")
    task_id: str = Field(..., description="'T2' or 'T3'")
    summary: TaskSummaryMetrics = Field(..., description="Aggregated task-level metrics")
    windows: List[WindowResult] = Field(..., description="Per-window timeline with timestamps and quality flags")


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
