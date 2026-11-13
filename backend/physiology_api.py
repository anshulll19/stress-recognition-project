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



# V5/V6 EDA-only inference

V6_DIR = PROJECT_ROOT / "physiological_pipeline" / "06_Defensive_Inference"

if str(V6_DIR) not in sys.path:

    sys.path.insert(0, str(V6_DIR))



from defensive_inference import DefensiveInference



# V3 BVP + EDA inference adapter (loaded dynamically to preserve V5 imports)

import importlib.util



V3_ADAPTER_PATH = (

PROJECT_ROOT

    / "physiological_pipeline"

    / "05_UBFC_Deployment"

    / "bvp_eda_rf_v3_inference.py"

)

_v3_spec = importlib.util.spec_from_file_location(

    "bvp_eda_rf_v3_inference", V3_ADAPTER_PATH

)

if _v3_spec is None or _v3_spec.loader is None:

    raise ImportError(f"Cannot load V3 adapter: {V3_ADAPTER_PATH}")

_v3_module = importlib.util.module_from_spec(_v3_spec)

_v3_spec.loader.exec_module(_v3_module)

BVPEDARFV3Inference = _v3_module.BVPEDARFV3Inference

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

    return re.sub(r"(/[a-zA-Z0-9_\-\\.]+)+", "[redacted_path]", str(msg))





# ============================================================================

# In-Memory Session Storage (Development Mode)

# ============================================================================



class SessionState:

    """Encapsulates isolated state, baseline calibration, and inference results for a session."""



    def __init__(

        self,

        session_id: str,

        participant_id: str,

        notes: Optional[str] = None,

        model_version: str = "v5",

    ):

        self.session_id: str = session_id

        self.participant_id: str = participant_id

        self.notes: Optional[str] = notes

        self.model_version: str = model_version

        self.created_at: datetime = datetime.now(timezone.utc)

        self.updated_at: datetime = datetime.now(timezone.utc)

        self.status: str = "created"  # created, baseline_uploaded, calibrated, completed

        self.is_calibrated: bool = False



        # Ingested T1 baseline

        self.t1_data: Optional[IngestedPhysiologyData] = None



        # Calibration parameters

        self.calibration_result: Optional[CalibrationResult] = None

        self.defensive_inference = None

        self.v3_inference = None

        self.calibration_diagnostics: Dict[str, Any] = {}

        self.calibration_warnings: List[str] = []



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

    """Health check validating V3 and V5 deployment artifacts."""



    v5_model = PROJECT_ROOT / "models/ubfc_deployment/ubfc_phys_eda_svm_v5.joblib"

    v5_manifest = PROJECT_ROOT / "models/ubfc_deployment/eda_svm_v5_manifest.json"



    v3_model = PROJECT_ROOT / "models/ubfc_deployment/ubfc_phys_bvp_eda_rf_v3.joblib"

    v3_manifest = PROJECT_ROOT / "models/ubfc_deployment/bvp_eda_rf_v3_manifest.json"



    deployment_models = {

        "eda_svm_v5": v5_model.is_file(),

        "eda_svm_v5_manifest": v5_manifest.is_file(),

        "bvp_eda_rf_v3": v3_model.is_file(),

        "bvp_eda_rf_v3_manifest": v3_manifest.is_file(),

    }



    artifacts_present = all(deployment_models.values())



    return {

        "status": "healthy" if artifacts_present else "degraded",

        "service": "vitalpulse-physiology-backend",

        "deployment_models_present": deployment_models,

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

            model_version=request.model_version,

        )

        _SESSION_STORE[session_id] = sess



    return SessionResponse(

        session_id=sess.session_id,

        participant_id=sess.participant_id,

        model_version=sess.model_version,

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
    """Upload T1 BVP/EDA data using the deployment's supported 64 Hz / 4 Hz profile."""
    sess = get_session_or_404(session_id)
    content_type = request.headers.get("content-type", "").lower()

    bvp_raw: Union[bytes, List[float]]
    eda_raw: Union[bytes, List[float]]
    participant_id_in_req: Optional[str] = None
    bvp_fs = 64.0
    eda_fs = 4.0

    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid multipart form: {sanitize_error_message(str(e))}")

        bvp_file = form.get("bvp_file")
        eda_file = form.get("eda_file")
        participant_id_in_req = form.get("participant_id")
        try:
            bvp_fs = float(form.get("bvp_fs", 64.0))
            eda_fs = float(form.get("eda_fs", 4.0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="bvp_fs and eda_fs must be valid numeric sampling rates.")

        if bvp_file is None or eda_file is None:
            raise HTTPException(status_code=400, detail="Multipart upload requires both bvp_file and eda_file.")
        try:
            bvp_raw = await bvp_file.read()
            eda_raw = await eda_file.read()
        finally:
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
            bvp_fs = parsed.bvp_fs
            eda_fs = parsed.eda_fs
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON payload: {sanitize_error_message(str(e))}")
    else:
        raise HTTPException(status_code=415, detail="Content-Type must be multipart/form-data or application/json.")

    validate_matching_participant(sess, participant_id_in_req)
    if not np.isfinite(bvp_fs) or not np.isfinite(eda_fs) or not np.isclose(bvp_fs, 64.0) or not np.isclose(eda_fs, 4.0):
        raise HTTPException(
            status_code=422,
            detail=("Unsupported sampling rates. Current deployment requires BVP at 64 Hz and EDA at 4 Hz. "
                    "Resampling for other devices is not yet implemented."),
        )

    try:
        ingested = ingest_bvp_eda(
            bvp_data=bvp_raw,
            eda_data=eda_raw,
            participant_id=sess.participant_id,
            task_id="T1",
            min_duration_s=MIN_BASELINE_DURATION_S,
        )
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_message(str(e)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal ingestion failed: {sanitize_error_message(str(e))}")

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
        message="T1 baseline signals ingested successfully. Next step: invoke POST /baseline/calibrate.",
    )


# 3. Calibrate T1 Baseline — V5/V6 EDA-only



@app.post(

    "/api/physiology/sessions/{session_id}/baseline/calibrate",

    response_model=CalibrationResponse,

    tags=["Baseline Calibration"],

    summary="Calibrate participant baseline using V5 EDA features",

)

def calibrate_baseline(session_id: str):

    sess = get_session_or_404(session_id)



    if sess.t1_data is None:

        raise HTTPException(

            status_code=400,

            detail="No T1 baseline uploaded. Upload T1 before calibration.",

        )



    try:

        if sess.model_version == "v3":

            inference = BVPEDARFV3Inference()

            result = inference.calibrate(

                np.asarray(sess.t1_data.bvp, dtype=float),

                np.asarray(sess.t1_data.eda, dtype=float),

                bvp_fs=sess.t1_data.bvp_fs,

                eda_fs=sess.t1_data.eda_fs,

            )



            if result.get("abstained", False):

                return CalibrationResponse(

                    session_id=sess.session_id,

                    participant_id=sess.participant_id,

                    is_calibrated=False,

                    status=result.get("status", "abstained"),

                    abstained=True,

                    warnings=result.get("warnings", []),

                    diagnostics=result,

                    calibration_windows=result.get("calibration_windows", 0),

                    calibration_duration_s=result.get("calibration_duration_s", 85.0),

                    feature_count=result.get("feature_count", 22),

                    calibration_method="participant T1 baseline mean/std",

                    probability_calibrated=False,

                    message="V3 baseline calibration abstained.",

                )



            with _STORE_LOCK:

                sess.v3_inference = inference

                sess.defensive_inference = None

                sess.calibration_diagnostics = result

                sess.calibration_warnings = result.get("warnings", [])

                sess.calibration_result = None

                sess.is_calibrated = True

                sess.status = "calibrated"

                sess.task_summaries.pop("T1", None)

                sess.touch()



            return CalibrationResponse(

                session_id=sess.session_id,

                participant_id=sess.participant_id,

                is_calibrated=True,

                status=result.get("status", "accepted"),

                abstained=False,

                warnings=result.get("warnings", []),

                diagnostics=result,

                calibration_windows=result.get("calibration_windows", 12),

                calibration_duration_s=result.get("calibration_duration_s", 85.0),

                feature_count=result.get("feature_count", 22),

                calibration_method="participant T1 baseline mean/std",

                probability_calibrated=False,

                message="V3 BVP+EDA baseline calibration successful. Ready for T2/T3 inference.",

            )



        inference = DefensiveInference()

        result = inference.calibrate(np.asarray(sess.t1_data.eda, dtype=float))



        if result.get("abstained", False):

            return CalibrationResponse(

                session_id=sess.session_id,

                participant_id=sess.participant_id,

                is_calibrated=False,

                status=result.get("status", "abstained"),

                abstained=True,

                warnings=result.get("warnings", []),

                diagnostics=result.get("diagnostics", {}),

                calibration_duration_s=85.0,

                message="Baseline calibration abstained. Check signal quality and diagnostics.",

            )



        diagnostics = result.get("diagnostics", {})

        adapter = diagnostics.get("adapter_metadata", {})

        validation = diagnostics.get("calibration_validation", {})



        with _STORE_LOCK:

            sess.defensive_inference = inference

            sess.calibration_diagnostics = diagnostics

            sess.calibration_warnings = result.get("warnings", [])

            sess.calibration_result = None

            sess.is_calibrated = True

            sess.status = "calibrated"

            sess.task_summaries.pop("T1", None)

            sess.touch()



        return CalibrationResponse(

            session_id=sess.session_id,

            participant_id=sess.participant_id,

            is_calibrated=True,

            status=result.get("status", "accepted"),

            abstained=False,

            warnings=result.get("warnings", []),

            diagnostics=diagnostics,

            calibration_windows=adapter.get("calibration_windows", 12),

            calibration_duration_s=adapter.get("calibration_window_end_max_seconds", 85.0),

            feature_count=validation.get("feature_count", 19),

            calibration_method=adapter.get("method", "participant-specific median/IQR"),

            probability_calibrated=False,

            message="V5 EDA baseline calibration successful. Ready for T2/T3 inference.",

        )



    except Exception as e:

        sess.is_calibrated = False

        sess.defensive_inference = None

        raise HTTPException(

            status_code=422,

            detail=f"{sess.model_version.upper()} calibration failed: {sanitize_error_message(str(e))}",

        )





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



    calibration_method = (

        "V3 BVP+EDA participant-specific T1 baseline mean/std calibration"

        if sess.model_version == "v3"

        else "V5 EDA-only participant-specific median/IQR calibration"

    )



    baseline_info = None

    if sess.is_calibrated:

        baseline_info = {

            "method": calibration_method,

            "diagnostics": sess.calibration_diagnostics,

            "warnings": sess.calibration_warnings,

            "probability_calibrated": False,

        }



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



    baseline_info = None

    if sess.is_calibrated:

        baseline_info ={

            "method": calibration_method,

            "diagnostics": sess.calibration_diagnostics,

            "warnings": sess.calibration_warnings,

            "probability_calibrated": False,

        }



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

    """Run V5/V6 EDA-only inference for T2 or T3."""

    import time



    sess = get_session_or_404(session_id)



    if sess.model_version == "v3":

        inference_engine = sess.v3_inference

    else:

        inference_engine = sess.defensive_inference



    if not sess.is_calibrated or inference_engine is None:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail="Session is not calibrated. Upload T1 baseline and run /baseline/calibrate first.",

        )



    content_type = request.headers.get("content-type", "").lower()

    bvp_raw: Union[bytes, List[float]]

    eda_raw: Union[bytes, List[float]]

    participant_id_in_req: Optional[str] = None

    bvp_fs = 64.0

    eda_fs = 4.0



    if content_type.startswith("multipart/form-data"):

        try:

            form = await request.form()

        except Exception as e:

            raise HTTPException(status_code=400, detail=f"Invalid multipart form: {sanitize_error_message(str(e))}")



        bvp_file = form.get("bvp_file")

        eda_file = form.get("eda_file")

        participant_id_in_req = form.get("participant_id")

        try:

            bvp_fs = float(form.get("bvp_fs", 64.0))

            eda_fs = float(form.get("eda_fs", 4.0))

        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="bvp_fs and eda_fs must be valid numeric sampling rates.",
            )

        if bvp_file is None or eda_file is None:

            raise HTTPException(status_code=400, detail="Multipart upload requires both bvp_file and eda_file.")



        try:

            bvp_raw = await bvp_file.read()

            eda_raw = await eda_file.read()

        finally:

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

            bvp_fs = parsed.bvp_fs

            eda_fs = parsed.eda_fs

        except Exception as e:

            raise HTTPException(status_code=422, detail=f"Invalid JSON payload: {sanitize_error_message(str(e))}")

    else:

        raise HTTPException(

            status_code=415,

            detail="Content-Type must be multipart/form-data or application/json.",

        )



    validate_matching_participant(sess, participant_id_in_req)

    if not np.isfinite(bvp_fs) or not np.isfinite(eda_fs) or not np.isclose(bvp_fs, 64.0) or not np.isclose(eda_fs, 4.0):
        raise HTTPException(
            status_code=422,
            detail=(
                "Unsupported sampling rates. Current deployment requires BVP at 64 Hz and EDA at 4 Hz. "
                "Resampling for other devices is not yet implemented."
            ),
        )

    try:
        ingested = ingest_bvp_eda(


            bvp_data=bvp_raw,

            eda_data=eda_raw,

            participant_id=sess.participant_id,

            task_id=task_id,

            min_duration_s=MIN_TASK_DURATION_S,

        )

    except IngestionError as e:

        raise HTTPException(status_code=400, detail=sanitize_error_message(str(e)))

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"Signal ingestion failed: {sanitize_error_message(str(e))}",

        )



    started = time.perf_counter()



    if sess.model_version == "v3":

        result = sess.v3_inference.predict_task(

            np.asarray(ingested.bvp, dtype=float),

            np.asarray(ingested.eda, dtype=float),

            bvp_fs=ingested.bvp_fs,

            eda_fs=ingested.eda_fs,

        )

        score_field = "distress_score"

        default_score_type = "uncalibrated_random_forest_probability"

        warnings = list(result.get("warnings", []))

        warnings.append(

            "Random Forest distress scores are uncalibrated and must not be interpreted as confidence."

        )

    else:

        result = sess.defensive_inference.predict_task(

            np.asarray(ingested.eda, dtype=float)

        )

        score_field = "decision_score"

        default_score_type = "raw_svm_decision_function"

        warnings = list(result.get("warnings", []))

        warnings.append("BVP is accepted for upload compatibility but is not used by V5 inference.")



    elapsed_ms = (time.perf_counter() - started) * 1000.0

    warnings.append(

        "Window-level sensor quality is not independently validated; quality checks are recording-level."

    )



    predictions = result.get("predictions", [])

    diagnostics = result.get("diagnostics", {})

    abstained = bool(result.get("abstained", False))

    accepted = [] if abstained else predictions



    window_responses = []

    for item in accepted:

        prediction = item.get("prediction")

        window_responses.append(

            WindowResult(

                window_index=int(item["window_index"]),

                start_s=float(item["start_s"]),

                end_s=float(item["end_s"]),

                is_reliable=True,

                prediction=int(prediction) if prediction is not None else None,

                probability=None,

                confidence=None,

                class_name=str(item.get("class_name", "inconclusive")),

                quality_flag="INFERENCE_ACCEPTED",

                quality_reason=None,

                latency_ms=elapsed_ms / max(len(accepted), 1),

                is_calibrated=True,

                features=None,

                decision_score=(

                    float(item[score_field])

                    if item.get(score_field) is not None

                    else None

                ),

                score_type=item.get("score_type", default_score_type),

                baseline_calibrated=True,

                probability_calibrated=False,

                quality_scope="recording_level_only",

            )

        )



    distress_count = sum(w.prediction == 1 for w in window_responses)

    rest_count = sum(w.prediction == 0 for w in window_responses)

    count = len(window_responses)

    scores = [w.decision_score for w in window_responses if w.decision_score is not None]

    task_name = "Speech task" if task_id == "T2" else "Arithmetic task"



    if abstained:

        status_value = str(result.get("status", "abstained"))

        dominant_state = "inconclusive"

        stress_percentage = None

    else:

        status_value = str(result.get("status", "accepted"))

        dominant_state = (

            "distress" if distress_count > rest_count

            else "rest" if rest_count > distress_count

            else "inconclusive"

        )

        stress_percentage = (100.0 * distress_count / count) if count else None



    summary_model = TaskSummaryMetrics(

        task_id=task_id,

        task_name=task_name,

        participant_id=sess.participant_id,

        total_windows=count,

        reliable_windows=count,

        unreliable_windows=0,

        stress_windows=distress_count,

        non_stress_windows=rest_count,

        stress_percentage=stress_percentage,

        mean_stress_probability=None,

        peak_stress_probability=None,

        min_stress_probability=None,

        mean_confidence=None,

        mean_latency_ms=elapsed_ms / max(count, 1),

        total_latency_ms=elapsed_ms,

        dominant_state=dominant_state,

        is_calibrated=True,

        mean_decision_score=float(np.mean(scores)) if scores else None,

        min_decision_score=float(np.min(scores)) if scores else None,

        max_decision_score=float(np.max(scores)) if scores else None,

        score_type=(

            "uncalibrated_random_forest_probability"

            if sess.model_version == "v3"

            else "raw_svm_decision_function"

        ),

        probability_calibrated=False,

    )



    with _STORE_LOCK:

        if not abstained:

            sess.task_summaries[task_id] = summary_model

            sess.status = "completed"

        sess.touch()



    return TaskInferenceResponse(

        session_id=sess.session_id,

        participant_id=sess.participant_id,

        task_id=task_id,

        summary=summary_model,

        windows=window_responses,

        status=status_value,

        abstained=abstained,

        warnings=warnings,

        diagnostics=diagnostics,

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

    """Return model-aware task summaries without invalid probability-based reactivity."""

    sess = get_session_or_404(session_id)



    tasks_out = {

        task_id: (

            summary if isinstance(summary, TaskSummaryMetrics)

            else TaskSummaryMetrics(**summary)

        )

        for task_id, summary in sess.task_summaries.items()

    }



    if sess.model_version == "v3":

        interpretation = (

            "V3 uses uncalibrated Random Forest scores, not calibrated probabilities. "

            "Probability-based physiological reactivity is therefore not computed. "

            "Task percentages represent the fraction of accepted windows classified as distress."

        )

    else:

        interpretation = (

            "V5 uses raw SVM decision scores, not calibrated probabilities. "

            "Probability-based physiological reactivity is therefore not computed. "

            "Task percentages represent the fraction of accepted windows classified as distress."

        )



    return SessionSummaryResponse(

        session_id=sess.session_id,

        participant_id=sess.participant_id,

        is_calibrated=sess.is_calibrated,

        tasks_completed=list(tasks_out.keys()),

        tasks=tasks_out,

        reactivity_profile={

            "baseline_t1_prob": None,

            "speech_t2_prob": None,

            "arithmetic_t3_prob": None,

            "delta_t2_vs_baseline": None,

            "delta_t3_vs_baseline": None,

            "reactivity_class": "NOT_COMPUTED",

        },

        interpretation=interpretation,

    )



    return SessionSummaryResponse(

        session_id=sess.session_id,

        participant_id=sess.participant_id,

        is_calibrated=sess.is_calibrated,

        tasks_completed=list(tasks_out.keys()),

        tasks=tasks_out,

        reactivity_profile={

            "baseline_t1_prob": None,

            "speech_t2_prob": None,

            "arithmetic_t3_prob": None,

            "delta_t2_vs_baseline": None,

            "delta_t3_vs_baseline": None,

            "reactivity_class": "NOT_COMPUTED",

        },

        interpretation=(

            "V5 uses raw SVM decision scores, not calibrated probabilities. "

            "Probability-based physiological reactivity is therefore not computed. "

            "Task percentages represent the fraction of accepted windows classified as distress."

        ),

    )





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
