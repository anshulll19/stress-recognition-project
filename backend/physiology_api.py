"""
physiology_api.py
=================

FastAPI physiological stress inference backend for integration with Next.js 16 / React 19 frontend.
Provides session lifecycle management, T1 resting baseline calibration, and calibrated T2/T3 stress inference.
Preserves the 30-second window, 5-second stride, and exact 12-feature schema.
Enforces the 180-second resting baseline protocol (31 overlapping windows) and 40-second active task minimum.
Ensures sensor dropout/unreliable windows are never treated as non-stress predictions.

NOTE ON PERSISTENCE:
In-memory sessions are development-only and do not survive server restarts.
For multi-instance or durable production deployments, integrate Redis or a relational database.
"""

from datetime import datetime, timezone
import io
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Union
import uuid

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

from calibration import RestCalibrator
from physiology_inference import UBFC_FEATURES, UBFCInferenceEngine
from physiological_file_ingestion import (
    IngestionError,
    IngestedPhysiologyData,
    ingest_bvp_eda,
    MIN_BASELINE_DURATION_S,
    MIN_TASK_DURATION_S,
)
from physiological_streaming import (
    PhysiologicalWindow,
    slice_physiological_windows,
)
from automatic_calibration import (
    CalibrationFailureError,
    CalibrationResult,
    calibrate_t1_baseline,
)
from session_aggregation import (
    TaskSummary,
    WindowInferenceRecord,
    aggregate_session_summary,
    aggregate_task_results,
)
from backend.schemas.physiology import (
    BaselineUploadJSONRequest,
    BaselineUploadResponse,
    CalibrationResponse,
    ErrorResponse,
    PhysiologicalFeatures,
    SessionCreateRequest,
    SessionDeleteResponse,
    SessionResponse,
    SessionStatusResponse,
    SessionSummaryResponse,
    TaskInferenceJSONRequest,
    TaskInferenceResponse,
    TaskSummaryMetrics,
    WindowResult,
)


# ============================================================================
# FastAPI App Initialization & CORS Setup
# ============================================================================

app = FastAPI(
    title="VitalPulse Physiological Stress API",
    version="1.0.0",
    description=(
        "Production-grade FastAPI physiological stress recognition service. "
        "Evaluates 30-second multimodal BVP (64 Hz) and EDA (4 Hz) signals using "
        "individualized resting baseline calibration (T1 Rest 180s -> T2 Speech / T3 Arithmetic >=40s). "
        "Strictly enforces quality diagnostics: unreliable windows are never treated as non-stress."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration via Environment Variables
raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001")
allowed_origins = [orig.strip() for orig in raw_cors.split(",") if orig.strip()]
if "*" in allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if "*" not in allowed_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Path Masking & Security Helper
# ============================================================================

def sanitize_error_message(msg: str) -> str:
    """Sanitize internal filesystem paths from user-facing error messages."""
    return re.sub(r"(/[a-zA-Z0-9_\-\.]+)+", "[redacted_path]", str(msg))


# ============================================================================
# In-Memory Session Storage (Development Mode)
# ============================================================================

class SessionState:
    """Encapsulates isolated state, baseline calibration, and inference results for a session."""

    def __init__(self, session_id: str, participant_id: str, notes: Optional[str] = None):
        self.session_id: str = session_id
        self.participant_id: str = participant_id
        self.notes: Optional[str] = notes
        self.created_at: datetime = datetime.now(timezone.utc)
        self.updated_at: datetime = datetime.now(timezone.utc)
        self.status: str = "created"  # created, baseline_uploaded, calibrated, completed
        self.is_calibrated: bool = False

        # Ingested T1 baseline
        self.t1_data: Optional[IngestedPhysiologyData] = None

        # Calibration parameters
        self.calibration_result: Optional[CalibrationResult] = None
        self.calibrator: Optional[RestCalibrator] = None

        # Evaluated Task Summaries (T1, T2, T3)
        self.task_summaries: Dict[str, TaskSummary] = {}

    def touch(self):
        self.updated_at = datetime.now(timezone.utc)


_SESSION_STORE: Dict[str, SessionState] = {}
_STORE_LOCK = threading.Lock()


def get_session_or_404(session_id: str) -> SessionState:
    """Retrieve an active session or raise 404 HTTPException."""
    with _STORE_LOCK:
        sess = _SESSION_STORE.get(session_id)
    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. It may have expired or the backend was restarted.",
        )
    return sess


def validate_matching_participant(session: SessionState, req_participant_id: Optional[str]):
    """Ensure provided participant identifier matches the session participant."""
    if req_participant_id is not None:
        p_clean = req_participant_id.strip()
        if p_clean and p_clean != session.participant_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Participant ID mismatch: session belongs to '{session.participant_id}', "
                    f"but request specified '{p_clean}'."
                ),
            )


# ============================================================================
# Root & Health Endpoints
# ============================================================================

@app.get("/api/physiology/health", tags=["System"])
def health_check():
    """Health check validating models and deployment artifacts."""
    cal_ckpt = PROJECT_ROOT / "models/ubfc_deployment/ubfc_phys_calibrated_rf.joblib"
    uncal_ckpt = PROJECT_ROOT / "models/ubfc_deployment/ubfc_phys_uncalibrated_rf.joblib"

    return {
        "status": "healthy",
        "service": "vitalpulse-physiology-backend",
        "deployment_models_present": {
            "calibrated_rf": cal_ckpt.is_file(),
            "uncalibrated_rf": uncal_ckpt.is_file(),
        },
        "active_sessions": len(_SESSION_STORE),
        "persistence_mode": "in_memory_development_only",
        "protocol_durations": {
            "t1_baseline_min_seconds": MIN_BASELINE_DURATION_S,
            "task_min_seconds": MIN_TASK_DURATION_S,
        },
    }


# ============================================================================
# 1. Create Session
# ============================================================================

@app.post(
    "/api/physiology/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Sessions"],
    summary="Create a new physiological analysis session",
)
def create_session(request: SessionCreateRequest):
    """
    Initialize an isolated session for a participant.
    Subsequent baseline calibration and task inference must use the returned session ID.
    """
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    with _STORE_LOCK:
        sess = SessionState(
            session_id=session_id,
            participant_id=request.participant_id.strip(),
            notes=request.notes,
        )
        _SESSION_STORE[session_id] = sess

    return SessionResponse(
        session_id=sess.session_id,
        participant_id=sess.participant_id,
        status=sess.status,
        is_calibrated=sess.is_calibrated,
        created_at=sess.created_at,
        updated_at=sess.updated_at,
        completed_tasks=[],
    )


# ============================================================================
# 2. Upload T1 Baseline (BVP + EDA)
# ============================================================================

@app.post(
    "/api/physiology/sessions/{session_id}/baseline",
    response_model=BaselineUploadResponse,
    tags=["Baseline Calibration"],
    summary="Upload Task 1 (Rest) BVP and EDA signals",
)
async def upload_baseline(session_id: str, request: Request):
    """
    Upload continuous raw BVP (@ 64 Hz) and EDA (@ 4 Hz) for Task 1 (Rest).
    Enforces the research training protocol: minimum 180 seconds (11,520 BVP samples, 720 EDA samples).

    Accepts:
    - Multipart Form: `bvp_file`, `eda_file`, and optional `participant_id`.
    - Application JSON: `participant_id`, `bvp_samples`, `eda_samples`, `task_id="T1"`.

    Validates duration compatibility, 180s baseline requirement, and finite numbers.
    Explicitly cleans up uploaded temporary file handles.
    """
    sess = get_session_or_404(session_id)
    content_type = request.headers.get("content-type", "").lower()

    bvp_raw: Union[bytes, List[float]]
    eda_raw: Union[bytes, List[float]]
    participant_id_in_req: Optional[str] = None

    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid multipart form: {str(e)}")

        bvp_file = form.get("bvp_file")
        eda_file = form.get("eda_file")
        participant_id_in_req = form.get("participant_id")

        if bvp_file is None or eda_file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multipart upload requires both 'bvp_file' and 'eda_file' form fields.",
            )

        try:
            bvp_raw = await bvp_file.read()
            eda_raw = await eda_file.read()
        finally:
            # Explicit cleanup of temporary spooled file handles
            if hasattr(bvp_file, "close"):
                await bvp_file.close()
            if hasattr(eda_file, "close"):
                await eda_file.close()

    elif "application/json" in content_type:
        try:
            body = await request.json()
            parsed = BaselineUploadJSONRequest(**body)
            bvp_raw = parsed.bvp_samples
            eda_raw = parsed.eda_samples
            participant_id_in_req = parsed.participant_id
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid JSON payload: {str(e)}")
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be 'multipart/form-data' or 'application/json'.",
        )

    validate_matching_participant(sess, participant_id_in_req)

    try:
        ingested = ingest_bvp_eda(
            bvp_data=bvp_raw,
            eda_data=eda_raw,
            participant_id=sess.participant_id,
            task_id="T1",
            min_duration_s=MIN_BASELINE_DURATION_S,
        )
    except IngestionError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_error_message(str(e)))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signal ingestion failed: {sanitize_error_message(str(e))}",
        )

    with _STORE_LOCK:
        sess.t1_data = ingested
        sess.status = "baseline_uploaded"
        sess.touch()

    return BaselineUploadResponse(
        session_id=sess.session_id,
        participant_id=sess.participant_id,
        task_id="T1",
        duration_s=ingested.duration_s,
        bvp_samples=ingested.n_bvp_samples,
        eda_samples=ingested.n_eda_samples,
        bvp_fs=ingested.bvp_fs,
        eda_fs=ingested.eda_fs,
        status=sess.status,
        message="T1 baseline signals ingested successfully (180s protocol verified). Next step: invoke POST /baseline/calibrate.",
    )


# ============================================================================
# 3. Calibrate T1 Baseline
# ============================================================================

@app.post(
    "/api/physiology/sessions/{session_id}/baseline/calibrate",
    response_model=CalibrationResponse,
    tags=["Baseline Calibration"],
    summary="Compute participant rest calibration parameters from T1",
)
def calibrate_baseline(session_id: str):
    """
    Execute resting baseline calibration using the uploaded T1 signals.

    Slices T1 into 30s windows (stride 5s -> 31 overlapping windows across 180s),
    isolates reliable windows (requiring >= 25 reliable windows), extracts 12 features,
    and calculates resting baseline means and standard deviations (with zero-variance protection).
    Attaches the fitted calibrator to the session for reuse by T2 and T3.
    """
    sess = get_session_or_404(session_id)

    if sess.t1_data is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No T1 baseline data uploaded for this session. Call POST /baseline first.",
        )

    try:
        cal_res = calibrate_t1_baseline(
            bvp_signal=sess.t1_data.bvp,
            eda_signal=sess.t1_data.eda,
            participant_id=sess.participant_id,
            bvp_fs=sess.t1_data.bvp_fs,
            eda_fs=sess.t1_data.eda_fs,
            required_baseline_s=MIN_BASELINE_DURATION_S,
        )
    except CalibrationFailureError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Calibration failed: {sanitize_error_message(str(e))}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected calibration failure: {sanitize_error_message(str(e))}",
        )

    # Convert T1 baseline predictions into WindowInferenceRecord for task summary
    records: List[WindowInferenceRecord] = []
    for p in cal_res.baseline_predictions:
        is_rel = p["is_reliable"]
        records.append(
            WindowInferenceRecord(
                window_index=p["window_index"],
                start_s=p["start_s"],
                end_s=p["end_s"],
                is_reliable=is_rel,
                prediction=p["prediction"],
                probability=p["probability"],
                confidence=p["confidence"],
                class_name="stress" if p["prediction"] == 1 else ("non-stress" if p["prediction"] == 0 else "unreliable"),
                quality_flag=p["quality_flag"],
                quality_reason=p.get("quality_reason"),
                latency_ms=p["latency_ms"],
                is_calibrated=True,
                features=None,
            )
        )

    t1_summary = aggregate_task_results(
        task_id="T1",
        window_records=records,
        participant_id=sess.participant_id,
        is_calibrated=True,
    )

    with _STORE_LOCK:
        sess.calibration_result = cal_res
        sess.calibrator = cal_res.calibrator
        sess.is_calibrated = True
        sess.status = "calibrated"
        sess.task_summaries["T1"] = t1_summary
        sess.touch()

    return CalibrationResponse(
        session_id=sess.session_id,
        participant_id=sess.participant_id,
        is_calibrated=True,
        n_total_windows=cal_res.n_total_windows,
        n_reliable_windows=cal_res.n_reliable_windows,
        n_unreliable_windows=cal_res.n_unreliable_windows,
        baseline_duration_s=cal_res.baseline_duration_s,
        feature_means=cal_res.feature_means,
        feature_stds=cal_res.feature_stds,
        baseline_stress_percentage=t1_summary.stress_percentage,
        mean_baseline_probability=t1_summary.mean_stress_probability,
        message="T1 baseline successfully calibrated (180s protocol verified). Ready for active task inference (T2 Speech, T3 Arithmetic).",
    )


# ============================================================================
# 4. Session Status
# ============================================================================

@app.get(
    "/api/physiology/sessions/{session_id}/status",
    response_model=SessionStatusResponse,
    tags=["Sessions"],
    summary="Get calibration status and completed tasks",
)
def get_session_status(session_id: str):
    """Retrieve session calibration status and completed task inventory."""
    sess = get_session_or_404(session_id)

    baseline_info = None
    if sess.is_calibrated and sess.calibration_result is not None:
        baseline_info = sess.calibration_result.to_dict()

    return SessionStatusResponse(
        session_id=sess.session_id,
        participant_id=sess.participant_id,
        status=sess.status,
        is_calibrated=sess.is_calibrated,
        baseline_summary=baseline_info,
        completed_tasks=list(sess.task_summaries.keys()),
        created_at=sess.created_at,
        updated_at=sess.updated_at,
    )


# ============================================================================
# 5. Generic Task Inference Handler (T2 Speech, T3 Arithmetic)
# ============================================================================

async def _process_task_inference(
    session_id: str,
    task_id: str,
    request: Request,
) -> TaskInferenceResponse:
    """
    Internal execution engine for stress tasks T2 and T3.
    Enforces participant rest calibration, minimum 40s recording (>= 3 windows @ 30s/5s stride),
    exact 12-feature schema, and safety against dropout misclassification.
    Explicitly cleans up uploaded temporary file handles.
    """
    sess = get_session_or_404(session_id)

    if not sess.is_calibrated or sess.calibrator is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session is not calibrated. You must upload T1 baseline and run /baseline/calibrate before running stress tasks.",
        )

    content_type = request.headers.get("content-type", "").lower()
    bvp_raw: Union[bytes, List[float]]
    eda_raw: Union[bytes, List[float]]
    participant_id_in_req: Optional[str] = None

    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid multipart form: {str(e)}")

        bvp_file = form.get("bvp_file")
        eda_file = form.get("eda_file")
        participant_id_in_req = form.get("participant_id")

        if bvp_file is None or eda_file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multipart upload requires both 'bvp_file' and 'eda_file' form fields.",
            )

        try:
            bvp_raw = await bvp_file.read()
            eda_raw = await eda_file.read()
        finally:
            # Explicit cleanup of temporary spooled file handles
            if hasattr(bvp_file, "close"):
                await bvp_file.close()
            if hasattr(eda_file, "close"):
                await eda_file.close()

    elif "application/json" in content_type:
        try:
            body = await request.json()
            parsed = TaskInferenceJSONRequest(**body)
            bvp_raw = parsed.bvp_samples
            eda_raw = parsed.eda_samples
            participant_id_in_req = parsed.participant_id
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid JSON payload: {str(e)}")
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be 'multipart/form-data' or 'application/json'.",
        )

    validate_matching_participant(sess, participant_id_in_req)

    try:
        ingested = ingest_bvp_eda(
            bvp_data=bvp_raw,
            eda_data=eda_raw,
            participant_id=sess.participant_id,
            task_id=task_id,
            min_duration_s=MIN_TASK_DURATION_S,
        )
    except IngestionError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=sanitize_error_message(str(e)))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signal ingestion failed: {sanitize_error_message(str(e))}",
        )

    # Slice windows: 30s window, 5s stride
    windows = slice_physiological_windows(
        bvp_signal=ingested.bvp,
        eda_signal=ingested.eda,
        bvp_fs=ingested.bvp_fs,
        eda_fs=ingested.eda_fs,
        window_s=30.0,
        stride_s=5.0,
        extract_features=True,
    )

    if not windows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{task_id} signals too short to generate a single 30-second window.",
        )

    # Reuse the exact same T1 calibrator
    engine = UBFCInferenceEngine(calibrator=sess.calibrator)

    window_records: List[WindowInferenceRecord] = []
    window_responses: List[WindowResult] = []

    for w in windows:
        if w.is_reliable and w.features is not None:
            # Predict using calibrated deployment model
            pred = engine.predict_features(w.features, use_calibration=True)
            feat_model = PhysiologicalFeatures(**w.features)

            rec = WindowInferenceRecord(
                window_index=w.window_index,
                start_s=w.start_s,
                end_s=w.end_s,
                is_reliable=True,
                prediction=pred.prediction,
                probability=pred.probability,
                confidence=pred.confidence,
                class_name=pred.class_name,
                quality_flag=pred.quality_flag,
                quality_reason=None,
                latency_ms=pred.latency_ms,
                is_calibrated=True,
                features=w.features,
            )
            w_res = WindowResult(
                window_index=w.window_index,
                start_s=w.start_s,
                end_s=w.end_s,
                is_reliable=True,
                prediction=pred.prediction,
                probability=pred.probability,
                confidence=pred.confidence,
                class_name=pred.class_name,
                quality_flag=pred.quality_flag,
                quality_reason=None,
                latency_ms=pred.latency_ms,
                is_calibrated=True,
                features=feat_model,
            )
        else:
            # Unreliable / Dropout window: NEVER TREAT AS NON-STRESS (prediction=0)
            rec = WindowInferenceRecord(
                window_index=w.window_index,
                start_s=w.start_s,
                end_s=w.end_s,
                is_reliable=False,
                prediction=None,
                probability=None,
                confidence=None,
                class_name="unreliable",
                quality_flag=w.quality_flag,
                quality_reason=w.quality_reason,
                latency_ms=0.0,
                is_calibrated=True,
                features=None,
            )
            w_res = WindowResult(
                window_index=w.window_index,
                start_s=w.start_s,
                end_s=w.end_s,
                is_reliable=False,
                prediction=None,
                probability=None,
                confidence=None,
                class_name="unreliable",
                quality_flag=w.quality_flag,
                quality_reason=w.quality_reason,
                latency_ms=0.0,
                is_calibrated=True,
                features=None,
            )

        window_records.append(rec)
        window_responses.append(w_res)

    summary = aggregate_task_results(
        task_id=task_id,
        window_records=window_records,
        participant_id=sess.participant_id,
        is_calibrated=True,
    )

    summary_model = TaskSummaryMetrics(**summary.to_dict(include_windows=False))

    with _STORE_LOCK:
        sess.task_summaries[task_id] = summary
        sess.status = "completed"
        sess.touch()

    return TaskInferenceResponse(
        session_id=sess.session_id,
        participant_id=sess.participant_id,
        task_id=task_id,
        summary=summary_model,
        windows=window_responses,
    )


# ============================================================================
# 5. Run Speech Inference (Task T2)
# ============================================================================

@app.post(
    "/api/physiology/sessions/{session_id}/tasks/T2",
    response_model=TaskInferenceResponse,
    tags=["Task Inference"],
    summary="Execute calibrated speech stress inference (Task T2)",
)
async def run_speech_inference(session_id: str, request: Request):
    """
    Run physiological stress inference on Task 2 (Speech Stress Task).
    Requires minimum 40s recording (produces >= 3 windows @ 30s/5s stride).
    Reuses the fitted T1 calibrator to standardize autonomic features.
    """
    return await _process_task_inference(session_id=session_id, task_id="T2", request=request)


# ============================================================================
# 6. Run Arithmetic Inference (Task T3)
# ============================================================================

@app.post(
    "/api/physiology/sessions/{session_id}/tasks/T3",
    response_model=TaskInferenceResponse,
    tags=["Task Inference"],
    summary="Execute calibrated mental arithmetic inference (Task T3)",
)
async def run_arithmetic_inference(session_id: str, request: Request):
    """
    Run physiological stress inference on Task 3 (Mental Arithmetic Stress Task).
    Requires minimum 40s recording (produces >= 3 windows @ 30s/5s stride).
    Reuses the fitted T1 calibrator to standardize autonomic features.
    """
    return await _process_task_inference(session_id=session_id, task_id="T3", request=request)


# ============================================================================
# 7. Get Multi-Task Session Summary
# ============================================================================

@app.get(
    "/api/physiology/sessions/{session_id}/summary",
    response_model=SessionSummaryResponse,
    tags=["Sessions"],
    summary="Get multi-task comparison and reactivity summary",
)
def get_session_summary(session_id: str):
    """
    Aggregate cross-task results comparing resting baseline (T1) vs speech (T2) and arithmetic (T3).
    Computes reactivity deltas and classifies overall reactivity.
    """
    sess = get_session_or_404(session_id)

    raw_summary = aggregate_session_summary(
        session_id=sess.session_id,
        participant_id=sess.participant_id,
        task_summaries=sess.task_summaries,
        is_calibrated=sess.is_calibrated,
    )

    tasks_out: Dict[str, TaskSummaryMetrics] = {}
    for tid, t_dict in raw_summary["tasks"].items():
        tasks_out[tid] = TaskSummaryMetrics(**t_dict)

    return SessionSummaryResponse(
        session_id=sess.session_id,
        participant_id=sess.participant_id,
        is_calibrated=sess.is_calibrated,
        tasks_completed=raw_summary["tasks_completed"],
        tasks=tasks_out,
        reactivity_profile=raw_summary["reactivity_profile"],
        interpretation=raw_summary["interpretation"],
    )


# ============================================================================
# 8. Clear Session
# ============================================================================

@app.delete(
    "/api/physiology/sessions/{session_id}",
    response_model=SessionDeleteResponse,
    tags=["Sessions"],
    summary="Clear session state from memory",
)
def clear_session(session_id: str):
    """
    Remove an active session and release allocated buffers and calibrator state from memory.
    """
    with _STORE_LOCK:
        sess = _SESSION_STORE.pop(session_id, None)

    if sess is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    return SessionDeleteResponse(
        session_id=session_id,
        cleared=True,
        message="Session successfully cleared from memory.",
    )


# ============================================================================
# Global Exception Handler
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Intercept uncaught exceptions and format as standardized JSON, concealing internal paths."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": "HTTP_ERROR",
                "message": sanitize_error_message(exc.detail),
                "details": None,
            },
        )

    sanitized = sanitize_error_message(str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal physiological engine error occurred.",
            "details": {"error_type": type(exc).__name__, "info": sanitized},
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("backend.physiology_api:app", host=host, port=port, reload=True)
