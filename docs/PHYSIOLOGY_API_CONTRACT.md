# Physiological Stress Inference API Contract

**Service Name:** VitalPulse Physiological Stress Backend  
**Framework:** FastAPI / Pydantic (Python 3.12)  
**Standard Port:** `8000` (default) or `$PORT`  
**Base Path:** `/api/physiology`  
**Current Version:** `1.0.0`  

---

## 1. System Overview & Architecture

The VitalPulse Physiological API enables individual-specific, real-time physiological stress recognition. It ingests continuous Blood Volume Pulse (BVP @ 64 Hz) and Electrodermal Activity (EDA @ 4 Hz) signals, slices synchronized 30-second sliding windows with 5-second strides, standardizes signals against an individual's resting baseline (Task 1), and predicts autonomic stress states across active cognitive and social stress tasks (Task 2 Speech, Task 3 Arithmetic).

```
Session Lifecycle & State Machine:
┌──────────────────┐       POST /sessions
│   Uninitialized  │ ─────────────────────────► [CREATED]
└──────────────────┘                               │
                                                   │ POST /baseline (T1 BVP + EDA)
                                                   ▼
                                           [BASELINE_UPLOADED]
                                                   │
                                                   │ POST /baseline/calibrate
                                                   ▼
                                             [CALIBRATED]
                                                   │
                         ┌─────────────────────────┴────────────────────────┐
                         │ POST /tasks/T2 (Speech)                          │ POST /tasks/T3 (Arithmetic)
                         ▼                                                  ▼
                   [COMPLETED] ◄────────────────────────────────────── [COMPLETED]
                         │
                         ├─────────────────► GET /summary
                         │
                         └─────────────────► DELETE /sessions/{id} ──► [CLEARED]
```

### Safety and Clinical Integrity Rules
1. **Zero-Variance & Dropout Protection:** If sensor detachment or flatline occurs (`std < 1e-6` in BVP or EDA), the window is flagged as `is_reliable=false`, `quality_flag='DROPOUT'`, `prediction=null`, `probability=null`.
2. **Never Treat Dropout as Non-Stress:** Unreliable windows are strictly excluded from stress percentage calculations. They are NEVER classified as `0` (non-stress).
3. **Calibrator Isolation & Reuse:** Resting parameters ($\mu_{rest}, \sigma_{rest}$) computed during T1 calibration are pinned to the session and automatically reused for T2 and T3 inference.

---

## 2. Physiological Feature Specification

Each 30-second window is mapped to the 12 statistical/time-domain features required by the UBFC Random Forest deployment models (`models/ubfc_deployment/`):

| Index | Feature Name | Modality | Description |
|:---:|:---|:---:|:---|
| 0 | `bvp_mean` | BVP (64 Hz) | Mean blood volume pulse amplitude |
| 1 | `bvp_std` | BVP (64 Hz) | Standard deviation of BVP amplitude |
| 2 | `bvp_median` | BVP (64 Hz) | Median BVP value |
| 3 | `bvp_iqr` | BVP (64 Hz) | Interquartile range of BVP amplitude |
| 4 | `bvp_slope_per_sample` | BVP (64 Hz) | Linear trend slope across the 30s window |
| 5 | `bvp_range` | BVP (64 Hz) | Peak-to-peak amplitude range ($\max - \min$) |
| 6 | `eda_mean` | EDA (4 Hz) | Mean skin conductance level ($\mu S$) |
| 7 | `eda_std` | EDA (4 Hz) | Phasic/tonic skin conductance variability |
| 8 | `eda_median` | EDA (4 Hz) | Median skin conductance level ($\mu S$) |
| 9 | `eda_iqr` | EDA (4 Hz) | Interquartile range of skin conductance |
| 10 | `eda_slope_per_sample` | EDA (4 Hz) | Tonic drift slope across the 30s window |
| 11 | `eda_range` | EDA (4 Hz) | Peak-to-peak skin conductance deflection ($\mu S$) |

---

## 3. Endpoints Specification

### 3.1 Create Session
- **Method:** `POST`
- **Path:** `/api/physiology/sessions`
- **Purpose:** Initialize an isolated in-memory session for a participant.
- **Request Headers:** `Content-Type: application/json`

**Request Body:**
```json
{
  "participant_id": "P01",
  "notes": "Experimental pilot session"
}
```

**Success Response (`201 Created`):**
```json
{
  "session_id": "sess_2f7ad7985e0f",
  "participant_id": "P01",
  "status": "created",
  "is_calibrated": false,
  "created_at": "2026-11-07T11:31:37.892015Z",
  "updated_at": "2026-11-07T11:31:37.892015Z",
  "completed_tasks": []
}
```

---

### 3.2 Upload T1 Baseline
- **Method:** `POST`
- **Path:** `/api/physiology/sessions/{session_id}/baseline`
- **Purpose:** Ingest continuous BVP and EDA recordings for Task 1 (Rest). Minimum duration is 180.0 seconds (31 overlapping windows @ 30s window / 5s stride) to match the research training protocol.
- **Content Types Supported:**
  1. `multipart/form-data`: form fields `bvp_file`, `eda_file`, and optional `participant_id`.
  2. `application/json`: payload with `bvp_samples`, `eda_samples`, `participant_id`, `task_id="T1"`.

**JSON Request Body:**
```json
{
  "participant_id": "P01",
  "task_id": "T1",
  "bvp_samples": [0.12, 0.15, 0.18, "... (11,520 samples for 180s @ 64 Hz) ..."],
  "eda_samples": [0.31, 0.31, 0.32, "... (720 samples for 180s @ 4 Hz) ..."]
}
```

**Success Response (`200 OK`):**
```json
{
  "session_id": "sess_2f7ad7985e0f",
  "participant_id": "P01",
  "task_id": "T1",
  "duration_s": 180.0,
  "bvp_samples": 11520,
  "eda_samples": 720,
  "bvp_fs": 64.0,
  "eda_fs": 4.0,
  "status": "baseline_uploaded",
  "message": "T1 baseline signals ingested successfully (180s protocol verified). Next step: invoke POST /baseline/calibrate."
}
```

---

### 3.3 Calibrate T1 Baseline
- **Method:** `POST`
- **Path:** `/api/physiology/sessions/{session_id}/baseline/calibrate`
- **Purpose:** Slices 180s T1 into 31 overlapping 30s windows (stride 5s), assesses window quality, extracts 12 features, and fits `RestCalibrator`.
- **Request Body:** None required.

**Success Response (`200 OK`):**
```json
{
  "session_id": "sess_2f7ad7985e0f",
  "participant_id": "P01",
  "is_calibrated": true,
  "n_total_windows": 31,
  "n_reliable_windows": 31,
  "n_unreliable_windows": 0,
  "baseline_duration_s": 180.0,
  "feature_means": {
    "bvp_mean": 0.001245,
    "bvp_std": 0.712398,
    "bvp_median": 0.000124,
    "bvp_iqr": 0.998124,
    "bvp_slope_per_sample": 0.000001,
    "bvp_range": 2.214012,
    "eda_mean": 0.304512,
    "eda_std": 0.003112,
    "eda_median": 0.304410,
    "eda_iqr": 0.004120,
    "eda_slope_per_sample": 0.000021,
    "eda_range": 0.012450
  },
  "feature_stds": {
    "bvp_mean": 0.000512,
    "bvp_std": 0.012401,
    "bvp_median": 0.000410,
    "bvp_iqr": 0.015201,
    "bvp_slope_per_sample": 0.000001,
    "bvp_range": 0.045120,
    "eda_mean": 0.002891,
    "eda_std": 0.000214,
    "eda_median": 0.002750,
    "eda_iqr": 0.000312,
    "eda_slope_per_sample": 0.000002,
    "eda_range": 0.001102
  },
  "baseline_stress_percentage": 0.0,
  "mean_baseline_probability": 0.0412,
  "message": "T1 baseline successfully calibrated (180s protocol verified). Ready for active task inference (T2 Speech, T3 Arithmetic)."
}
```

---

### 3.4 Get Session Status
- **Method:** `GET`
- **Path:** `/api/physiology/sessions/{session_id}/status`
- **Purpose:** Inspect session progress, calibration status, and completed tasks.

**Success Response (`200 OK`):**
```json
{
  "session_id": "sess_2f7ad7985e0f",
  "participant_id": "P01",
  "status": "calibrated",
  "is_calibrated": true,
  "baseline_summary": {
    "is_calibrated": true,
    "participant_id": "P01",
    "n_total_windows": 4,
    "n_reliable_windows": 4,
    "n_unreliable_windows": 0,
    "baseline_duration_s": 45.0,
    "feature_means": { "...": 0.0 },
    "feature_stds": { "...": 0.0 },
    "metadata": {
      "features": ["bvp_mean", "bvp_std", "bvp_median", "bvp_iqr", "bvp_slope_per_sample", "bvp_range", "eda_mean", "eda_std", "eda_median", "eda_iqr", "eda_slope_per_sample", "eda_range"],
      "window_size_s": 30.0,
      "stride_s": 5.0,
      "n_baseline_samples": 4
    }
  },
  "completed_tasks": ["T1"],
  "created_at": "2026-11-07T11:31:37.892015Z",
  "updated_at": "2026-11-07T11:31:38.241012Z"
}
```

---

### 3.5 Run Speech Stress Inference (Task T2)
- **Method:** `POST`
- **Path:** `/api/physiology/sessions/{session_id}/tasks/T2`
- **Purpose:** Slices T2 into 30s windows (stride 5s), applies T1 calibration parameters, and runs inference. Minimum duration: 40 seconds (at least 3 windows: 0-30s, 5-35s, 10-40s).
- **Content Types:** `multipart/form-data` or `application/json`.

**JSON Request Body:**
```json
{
  "participant_id": "P01",
  "bvp_samples": [0.45, 0.48, "... (minimum 2560 samples for 40s @ 64 Hz) ..."],
  "eda_samples": [0.55, 0.56, "... (minimum 160 samples for 40s @ 4 Hz) ..."]
}
```

**Success Response (`200 OK`):**
```json
{
  "session_id": "sess_2f7ad7985e0f",
  "participant_id": "P01",
  "task_id": "T2",
  "summary": {
    "task_id": "T2",
    "task_name": "Speech Stress Task",
    "participant_id": "P01",
    "total_windows": 3,
    "reliable_windows": 3,
    "unreliable_windows": 0,
    "stress_windows": 3,
    "non_stress_windows": 0,
    "stress_percentage": 100.0,
    "mean_stress_probability": 0.8842,
    "peak_stress_probability": 0.9410,
    "min_stress_probability": 0.8250,
    "mean_confidence": 0.8842,
    "mean_latency_ms": 28.5,
    "total_latency_ms": 85.5,
    "dominant_state": "stress",
    "is_calibrated": true,
    "metadata": {}
  },
  "windows": [
    {
      "window_index": 0,
      "start_s": 0.0,
      "end_s": 30.0,
      "is_reliable": true,
      "prediction": 1,
      "probability": 0.8250,
      "confidence": 0.8250,
      "class_name": "stress",
      "quality_flag": "OK",
      "quality_reason": null,
      "latency_ms": 29.1,
      "is_calibrated": true,
      "features": {
        "bvp_mean": 0.021,
        "bvp_std": 1.15,
        "bvp_median": 0.012,
        "bvp_iqr": 1.48,
        "bvp_slope_per_sample": 0.0001,
        "bvp_range": 3.42,
        "eda_mean": 0.52,
        "eda_std": 0.025,
        "eda_median": 0.51,
        "eda_iqr": 0.031,
        "eda_slope_per_sample": 0.0002,
        "eda_range": 0.084
      }
    }
  ]
}
```

---

### 3.6 Run Mental Arithmetic Inference (Task T3)
- **Method:** `POST`
- **Path:** `/api/physiology/sessions/{session_id}/tasks/T3`
- **Purpose:** Executes mental arithmetic stress inference, reusing the same T1 baseline calibrator.
- **Request / Response:** Structurally identical to T2, with `task_id="T3"` and `task_name="Mental Arithmetic Stress Task"`.

---

### 3.7 Get Multi-Task Session Summary
- **Method:** `GET`
- **Path:** `/api/physiology/sessions/{session_id}/summary`
- **Purpose:** Cross-task comparison of baseline (T1) vs speech (T2) and arithmetic (T3), with reactivity deltas.

**Success Response (`200 OK`):**
```json
{
  "session_id": "sess_2f7ad7985e0f",
  "participant_id": "P01",
  "is_calibrated": true,
  "tasks_completed": ["T1", "T2", "T3"],
  "tasks": {
    "T1": {
      "task_id": "T1",
      "task_name": "Rest / Baseline",
      "participant_id": "P01",
      "total_windows": 4,
      "reliable_windows": 4,
      "unreliable_windows": 0,
      "stress_windows": 0,
      "non_stress_windows": 4,
      "stress_percentage": 0.0,
      "mean_stress_probability": 0.0412,
      "peak_stress_probability": 0.0620,
      "min_stress_probability": 0.0210,
      "mean_confidence": 0.9588,
      "mean_latency_ms": 24.1,
      "total_latency_ms": 96.4,
      "dominant_state": "non-stress",
      "is_calibrated": true
    },
    "T2": {
      "task_id": "T2",
      "task_name": "Speech Stress Task",
      "participant_id": "P01",
      "total_windows": 3,
      "reliable_windows": 3,
      "unreliable_windows": 0,
      "stress_windows": 3,
      "non_stress_windows": 0,
      "stress_percentage": 100.0,
      "mean_stress_probability": 0.8842,
      "peak_stress_probability": 0.9410,
      "min_stress_probability": 0.8250,
      "mean_confidence": 0.8842,
      "mean_latency_ms": 28.5,
      "total_latency_ms": 85.5,
      "dominant_state": "stress",
      "is_calibrated": true
    },
    "T3": {
      "task_id": "T3",
      "task_name": "Mental Arithmetic Stress Task",
      "participant_id": "P01",
      "total_windows": 3,
      "reliable_windows": 3,
      "unreliable_windows": 0,
      "stress_windows": 3,
      "non_stress_windows": 0,
      "stress_percentage": 100.0,
      "mean_stress_probability": 0.9120,
      "peak_stress_probability": 0.9650,
      "min_stress_probability": 0.8600,
      "mean_confidence": 0.9120,
      "mean_latency_ms": 27.8,
      "total_latency_ms": 83.4,
      "dominant_state": "stress",
      "is_calibrated": true
    }
  },
  "reactivity_profile": {
    "baseline_t1_prob": 0.0412,
    "speech_t2_prob": 0.8842,
    "arithmetic_t3_prob": 0.9120,
    "delta_t2_vs_baseline": 0.843,
    "delta_t3_vs_baseline": 0.8708,
    "reactivity_class": "HIGH_REACTIVITY"
  },
  "interpretation": "UBFC-Phys multimodal physiological stress evaluation calibrated relative to participant resting baseline (Task 1). Measures autonomic shifts during speech and arithmetic stressors. Research output; not a clinical diagnostic instrument."
}
```

---

### 3.8 Delete Session
- **Method:** `DELETE`
- **Path:** `/api/physiology/sessions/{session_id}`
- **Purpose:** Deallocate in-memory buffers and calibrator for the specified session.

**Success Response (`200 OK`):**
```json
{
  "session_id": "sess_2f7ad7985e0f",
  "cleared": true,
  "message": "Session successfully cleared from memory."
}
```

---

## 4. Standard Error Responses

All non-2xx responses return a structured error envelope:

```json
{
  "error_code": "HTTP_ERROR",
  "message": "Participant ID mismatch: session belongs to 'P01', but request specified 'P02'.",
  "details": null
}
```

| HTTP Status | Reason | Cause |
|---|---|---|
| `400 Bad Request` | Validation Failure | Mismatched participant ID, flatline recording, duration spread > 15s, or calling `/tasks/T2` prior to baseline calibration. |
| `404 Not Found` | Session Absent | Session ID does not exist or server restarted. |
| `415 Unsupported Media` | Content-Type Mismatch | Request is not `application/json` or `multipart/form-data`. |
| `422 Unprocessable` | Calibration / Signal Failure | Signal duration is < 30 seconds or all windows suffer sensor dropout. |
| `500 Server Error` | Engine Failure | Model checkpoint error or uncaught exception (file paths redacted). |

---

## 5. End-to-End Workflow via cURL

### Step 1: Create Session
```bash
SESSION_RESP=$(curl -s -X POST http://localhost:8000/api/physiology/sessions \
  -H "Content-Type: application/json" \
  -d '{"participant_id": "P01", "notes": "E2E cURL Test"}')
SESSION_ID=$(echo $SESSION_RESP | grep -o '"session_id":"[^"]*' | cut -d'"' -f4)
echo "Session ID: $SESSION_ID"
```

### Step 2: Upload T1 Baseline (Multipart File)
```bash
curl -X POST http://localhost:8000/api/physiology/sessions/${SESSION_ID}/baseline \
  -F "bvp_file=@data/pilot/bvp_s1_T1.csv" \
  -F "eda_file=@data/pilot/eda_s1_T1.csv" \
  -F "participant_id=P01"
```

### Step 3: Calibrate T1 Baseline
```bash
curl -X POST http://localhost:8000/api/physiology/sessions/${SESSION_ID}/baseline/calibrate
```

### Step 4: Check Session Status
```bash
curl http://localhost:8000/api/physiology/sessions/${SESSION_ID}/status
```

### Step 5: Execute Speech Inference (Task T2)
```bash
curl -X POST http://localhost:8000/api/physiology/sessions/${SESSION_ID}/tasks/T2 \
  -F "bvp_file=@data/pilot/bvp_s1_T2.csv" \
  -F "eda_file=@data/pilot/eda_s1_T2.csv" \
  -F "participant_id=P01"
```

### Step 6: Execute Arithmetic Inference (Task T3)
```bash
curl -X POST http://localhost:8000/api/physiology/sessions/${SESSION_ID}/tasks/T3 \
  -F "bvp_file=@data/pilot/bvp_s1_T3.csv" \
  -F "eda_file=@data/pilot/eda_s1_T3.csv" \
  -F "participant_id=P01"
```

### Step 7: Retrieve Multi-Task Session Summary
```bash
curl http://localhost:8000/api/physiology/sessions/${SESSION_ID}/summary
```

### Step 8: Clear Session
```bash
curl -X DELETE http://localhost:8000/api/physiology/sessions/${SESSION_ID}
```

