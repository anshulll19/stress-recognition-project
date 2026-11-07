# Next.js 16 / React 19 Frontend Integration Guide

**Integration Target:** Next.js 16 (App Router) + React 19 + TypeScript  
**Backend API:** VitalPulse FastAPI Physiological Backend (`http://localhost:8000/api/physiology`)  
**Pipeline:** UBFC-Phys Multimodal Rest Calibration & Inference (T1 Baseline → T2 Speech → T3 Arithmetic)  

---

## 1. Quick Start & Server Startup

### A. Start the Backend API
From the project root:

```bash
# Using Python virtual environment
./venv/bin/uvicorn backend.physiology_api:app --host 0.0.0.0 --port 8000 --reload
```

### B. Configure Frontend Environment (`.env.local`)
In your Next.js project root:

```env
NEXT_PUBLIC_PHYSIOLOGY_API_URL=http://localhost:8000/api/physiology
```

### C. CORS Configuration
If running Next.js on a non-standard port (e.g. `http://localhost:3000` or `http://localhost:3001`), pass `CORS_ORIGINS` when launching uvicorn:

```bash
CORS_ORIGINS="http://localhost:3000,http://localhost:3001" ./venv/bin/uvicorn backend.physiology_api:app --port 8000 --reload
```

### D. Enforced Research Protocol Durations
- **Task 1 (Rest Baseline):** Minimum **180 seconds** (11,520 BVP samples @ 64 Hz, 720 EDA samples @ 4 Hz). Produces 31 overlapping 30s windows (stride 5s) to establish a stable tonic autonomic baseline matching the trained model's benchmark.
- **Task 2 / Task 3 (Active Stressors):** Minimum **40 seconds** (2,560 BVP samples @ 64 Hz, 160 EDA samples @ 4 Hz). Produces at least 3 sliding windows (0–30s, 5–35s, 10–40s) for trend evaluation.

> [!IMPORTANT]
> **In-Memory Session Notice:** Sessions are stored in server memory during development. If the backend is restarted or reloaded, previous session IDs will return `404 Not Found`. Design your frontend state to handle 404 by offering a "Start New Session" action.

---

## 2. Complete TypeScript Type Definitions (`types/physiology.ts`)

Save the following file in your frontend repository at `types/physiology.ts`:

```typescript
/**
 * Standard 12-dimensional statistical feature representation
 * extracted from 30-second synchronized BVP (64 Hz) and EDA (4 Hz) windows.
 */
export interface PhysiologicalFeatures {
  bvp_mean: number;
  bvp_std: number;
  bvp_median: number;
  bvp_iqr: number;
  bvp_slope_per_sample: number;
  bvp_range: number;
  eda_mean: number;
  eda_std: number;
  eda_median: number;
  eda_iqr: number;
  eda_slope_per_sample: number;
  eda_range: number;
}

/**
 * Quality evaluation and inference result for a single 30s sliding window.
 */
export interface WindowResult {
  window_index: number;
  start_s: number;
  end_s: number;
  is_reliable: boolean;
  prediction: number | null;        // 1 = stress, 0 = non-stress, null = unreliable
  probability: number | null;       // P(stress) in [0.0, 1.0], null if unreliable
  confidence: number | null;        // max(P, 1-P), null if unreliable
  class_name: 'stress' | 'non-stress' | 'unreliable';
  quality_flag: 'OK' | 'DROPOUT' | 'UNRELIABLE';
  quality_reason: string | null;
  latency_ms: number;
  is_calibrated: boolean;
  features: PhysiologicalFeatures | null;
}

/**
 * Aggregated summary metrics across all evaluated windows in a task.
 */
export interface TaskSummaryMetrics {
  task_id: 'T1' | 'T2' | 'T3' | string;
  task_name: string;
  participant_id: string;
  total_windows: number;
  reliable_windows: number;
  unreliable_windows: number;
  stress_windows: number;
  non_stress_windows: number;
  stress_percentage: number;        // Based solely on reliable windows
  mean_stress_probability: number | null;
  peak_stress_probability: number | null;
  min_stress_probability: number | null;
  mean_confidence: number | null;
  mean_latency_ms: number;
  total_latency_ms: number;
  dominant_state: 'stress' | 'non-stress' | 'inconclusive';
  is_calibrated: boolean;
  metadata?: Record<string, unknown>;
}

/**
 * Active session representation.
 */
export interface SessionResponse {
  session_id: string;
  participant_id: string;
  status: 'created' | 'baseline_uploaded' | 'calibrated' | 'completed';
  is_calibrated: boolean;
  created_at: string;
  updated_at: string;
  completed_tasks: string[];
}

export interface SessionStatusResponse extends SessionResponse {
  baseline_summary: Record<string, unknown> | null;
}

export interface BaselineUploadResponse {
  session_id: string;
  participant_id: string;
  task_id: 'T1';
  duration_s: number;
  bvp_samples: number;
  eda_samples: number;
  bvp_fs: number;
  eda_fs: number;
  status: string;
  message: string;
}

export interface CalibrationResponse {
  session_id: string;
  participant_id: string;
  is_calibrated: boolean;
  n_total_windows: number;
  n_reliable_windows: number;
  n_unreliable_windows: number;
  baseline_duration_s: number;
  feature_means: Record<string, number>;
  feature_stds: Record<string, number>;
  baseline_stress_percentage: number;
  mean_baseline_probability: number | null;
  message: string;
}

export interface TaskInferenceResponse {
  session_id: string;
  participant_id: string;
  task_id: 'T2' | 'T3';
  summary: TaskSummaryMetrics;
  windows: WindowResult[];
}

export interface ReactivityProfile {
  baseline_t1_prob: number | null;
  speech_t2_prob: number | null;
  arithmetic_t3_prob: number | null;
  delta_t2_vs_baseline: number | null;
  delta_t3_vs_baseline: number | null;
  reactivity_class: 'NORMAL' | 'HIGH_REACTIVITY' | 'LOW_REACTIVITY' | 'MODERATE_REACTIVITY';
}

export interface SessionSummaryResponse {
  session_id: string;
  participant_id: string;
  is_calibrated: boolean;
  tasks_completed: string[];
  tasks: Record<string, TaskSummaryMetrics>;
  reactivity_profile: ReactivityProfile;
  interpretation: string;
}

export interface ApiErrorPayload {
  error_code: string;
  message: string;
  details?: Record<string, unknown> | null;
}
```

---

## 3. Type-Safe Client Implementation (`lib/api/physiologyClient.ts`)

```typescript
import {
  SessionResponse,
  SessionStatusResponse,
  BaselineUploadResponse,
  CalibrationResponse,
  TaskInferenceResponse,
  SessionSummaryResponse,
  ApiErrorPayload,
} from '@/types/physiology';

const BASE_URL = process.env.NEXT_PUBLIC_PHYSIOLOGY_API_URL || 'http://localhost:8000/api/physiology';

class ApiClientError extends Error {
  status: number;
  payload: ApiErrorPayload;

  constructor(status: number, payload: ApiErrorPayload) {
    super(payload.message || `API Error (${status})`);
    this.status = status;
    this.payload = payload;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const res = await fetch(url, options);

  if (!res.ok) {
    let payload: ApiErrorPayload;
    try {
      payload = await res.json();
    } catch {
      payload = { error_code: 'UNKNOWN_ERROR', message: res.statusText };
    }
    throw new ApiClientError(res.status, payload);
  }

  return res.json() as Promise<T>;
}

export const physiologyApi = {
  // 1. Create Session
  createSession: (participantId: string, notes?: string) =>
    request<SessionResponse>('/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ participant_id: participantId, notes }),
    }),

  // 2. Upload T1 Baseline (Multipart)
  uploadBaselineMultipart: (sessionId: string, bvpFile: File, edaFile: File, participantId: string) => {
    const formData = new FormData();
    formData.append('bvp_file', bvpFile);
    formData.append('eda_file', edaFile);
    formData.append('participant_id', participantId);

    return request<BaselineUploadResponse>(`/sessions/${sessionId}/baseline`, {
      method: 'POST',
      body: formData,
    });
  },

  // 3. Calibrate T1 Baseline
  calibrateBaseline: (sessionId: string) =>
    request<CalibrationResponse>(`/sessions/${sessionId}/baseline/calibrate`, {
      method: 'POST',
    }),

  // 4. Session Status
  getStatus: (sessionId: string) =>
    request<SessionStatusResponse>(`/sessions/${sessionId}/status`),

  // 5. Run Speech Task (T2)
  runSpeechTask: (sessionId: string, bvpFile: File, edaFile: File, participantId: string) => {
    const formData = new FormData();
    formData.append('bvp_file', bvpFile);
    formData.append('eda_file', edaFile);
    formData.append('participant_id', participantId);

    return request<TaskInferenceResponse>(`/sessions/${sessionId}/tasks/T2`, {
      method: 'POST',
      body: formData,
    });
  },

  // 6. Run Arithmetic Task (T3)
  runArithmeticTask: (sessionId: string, bvpFile: File, edaFile: File, participantId: string) => {
    const formData = new FormData();
    formData.append('bvp_file', bvpFile);
    formData.append('eda_file', edaFile);
    formData.append('participant_id', participantId);

    return request<TaskInferenceResponse>(`/sessions/${sessionId}/tasks/T3`, {
      method: 'POST',
      body: formData,
    });
  },

  // 7. Multi-Task Session Summary
  getSessionSummary: (sessionId: string) =>
    request<SessionSummaryResponse>(`/sessions/${sessionId}/summary`),

  // 8. Delete / Clear Session
  clearSession: (sessionId: string) =>
    request<{ session_id: string; cleared: boolean; message: string }>(`/sessions/${sessionId}`, {
      method: 'DELETE',
    }),
};
```

---

## 4. React 19 State Management Hook (`hooks/usePhysiologySession.ts`)

```typescript
'use client';

import { useState, useCallback } from 'react';
import { physiologyApi } from '@/lib/api/physiologyClient';
import {
  SessionResponse,
  CalibrationResponse,
  TaskInferenceResponse,
  SessionSummaryResponse,
} from '@/types/physiology';

export function usePhysiologySession() {
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResponse | null>(null);
  const [t2Result, setT2Result] = useState<TaskInferenceResponse | null>(null);
  const [t3Result, setT3Result] = useState<TaskInferenceResponse | null>(null);
  const [sessionSummary, setSessionSummary] = useState<SessionSummaryResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Step 1: Start Session
  const startSession = useCallback(async (participantId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const sess = await physiologyApi.createSession(participantId);
      setSession(sess);
      setCalibration(null);
      setT2Result(null);
      setT3Result(null);
      setSessionSummary(null);
      return sess;
    } catch (err: any) {
      setError(err.message || 'Failed to initialize session');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Step 2 & 3: Upload & Calibrate T1 Baseline
  const calibrateT1 = useCallback(async (bvpFile: File, edaFile: File) => {
    if (!session) throw new Error('No active session.');
    setIsLoading(true);
    setError(null);
    try {
      await physiologyApi.uploadBaselineMultipart(session.session_id, bvpFile, edaFile, session.participant_id);
      const calRes = await physiologyApi.calibrateBaseline(session.session_id);
      setCalibration(calRes);
      return calRes;
    } catch (err: any) {
      setError(err.message || 'Calibration failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [session]);

  // Step 4: Run Speech (T2)
  const executeT2 = useCallback(async (bvpFile: File, edaFile: File) => {
    if (!session) throw new Error('No active session.');
    if (!calibration?.is_calibrated) throw new Error('T1 calibration required before executing T2.');
    setIsLoading(true);
    setError(null);
    try {
      const res = await physiologyApi.runSpeechTask(session.session_id, bvpFile, edaFile, session.participant_id);
      setT2Result(res);
      return res;
    } catch (err: any) {
      setError(err.message || 'Speech task inference failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [session, calibration]);

  // Step 5: Run Arithmetic (T3)
  const executeT3 = useCallback(async (bvpFile: File, edaFile: File) => {
    if (!session) throw new Error('No active session.');
    if (!calibration?.is_calibrated) throw new Error('T1 calibration required before executing T3.');
    setIsLoading(true);
    setError(null);
    try {
      const res = await physiologyApi.runArithmeticTask(session.session_id, bvpFile, edaFile, session.participant_id);
      setT3Result(res);
      return res;
    } catch (err: any) {
      setError(err.message || 'Arithmetic task inference failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [session, calibration]);

  // Step 6: Fetch Multi-Task Summary
  const fetchSummary = useCallback(async () => {
    if (!session) return;
    try {
      const sum = await physiologyApi.getSessionSummary(session.session_id);
      setSessionSummary(sum);
      return sum;
    } catch (err: any) {
      setError(err.message || 'Failed to fetch session summary');
    }
  }, [session]);

  return {
    session,
    calibration,
    t2Result,
    t3Result,
    sessionSummary,
    isLoading,
    error,
    startSession,
    calibrateT1,
    executeT2,
    executeT3,
    fetchSummary,
  };
}
```

---

## 5. UI Rendering & Critical Safety Rules

### A. How to Render the 30s Sliding Window Timeline
Windows slide in 5-second steps (`start_s` to `end_s`):
- `Window 0: 0.0s – 30.0s`
- `Window 1: 5.0s – 35.0s`
- `Window 2: 10.0s – 40.0s`

When rendering timeline graphs (e.g. using Recharts or Chart.js):
```tsx
// Example timeline item mapping:
const chartData = taskResponse.windows.map((w) => ({
  time: `${w.start_s.toFixed(0)}s–${w.end_s.toFixed(0)}s`,
  probability: w.is_reliable ? (w.probability! * 100) : null,
  isReliable: w.is_reliable,
  status: w.quality_flag,
}));
```

### B. Safety Rule: Handling Sensor Dropout
When `w.is_reliable === false`:
- `w.prediction` is `null`
- `w.probability` is `null`
- `w.quality_flag` is `'DROPOUT'` or `'UNRELIABLE'`

> [!CAUTION]
> **NEVER DISPLAY SENSOR DROPOUT AS NON-STRESS (CALM):**  
> If `w.prediction === null`, do NOT render a green "Non-stress" badge or set probability to 0%. Instead, render an amber/red warning pill:  
> `<Badge variant="warning">Sensor Contact Lost / Dropout</Badge>`

