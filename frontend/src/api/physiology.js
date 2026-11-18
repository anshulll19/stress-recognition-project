// Client for the physiology FastAPI backend (backend/physiology_api.py).
// Paths and field names follow docs/PHYSIOLOGY_API_CONTRACT.md and backend/schemas/physiology.py.
const BASE = (import.meta.env.VITE_PHYSIOLOGY_API_URL ?? "").replace(/\/+$/, "");

export const apiConfigured = BASE !== "";

// The model every session uses. v5 is the EDA-only SVM (BVP is accepted but not used).
export const MODEL_VERSION = "v5";

export class ApiError extends Error {
  constructor(message, { status = null, kind = "http" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    // "not_configured" | "network" | "session_expired" | "http"
    this.kind = kind;
  }
}

// FastAPI sends a string for most errors and a list of {loc, msg} for validation errors.
function describeDetail(detail, status) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail.map((d) => d?.msg).filter(Boolean);
    if (msgs.length) return msgs.join(" ");
  }
  if (detail) return JSON.stringify(detail);
  return `Request failed (${status})`;
}

async function request(path, options = {}) {
  if (!apiConfigured) {
    throw new ApiError("The physiology API address is not set (VITE_PHYSIOLOGY_API_URL).", {
      kind: "not_configured",
    });
  }

  let res;
  try {
    res = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new ApiError("Could not reach the physiology backend. Is it running?", {
      kind: "network",
    });
  }

  const body = await res.json().catch(() => null);

  if (!res.ok) {
    // Sessions are kept in server memory, so a restart makes the id return 404.
    if (res.status === 404 && path.startsWith("/sessions/")) {
      throw new ApiError("This session no longer exists. Start a new session.", {
        status: 404,
        kind: "session_expired",
      });
    }
    const detail = body?.detail ?? body?.message ?? body?.error;
    throw new ApiError(describeDetail(detail, res.status), { status: res.status });
  }
  return body;
}

const postJson = (data) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

const sid = (id) => encodeURIComponent(id);

const SIGNAL_TASKS = ["T2", "T3"];

function recordingForm(bvpFile, edaFile, participantId) {
  const form = new FormData();
  form.append("bvp_file", bvpFile);
  form.append("eda_file", edaFile);
  if (participantId) form.append("participant_id", participantId);
  return form;
}

export const getHealth = () => request("/health");

export const createSession = (participantId, notes, modelVersion = MODEL_VERSION) =>
  request(
    "/sessions",
    postJson({
      participant_id: participantId,
      notes: notes || null,
      model_version: modelVersion,
    }),
  );

export const getSessionStatus = (sessionId) => request(`/sessions/${sid(sessionId)}/status`);

export const deleteSession = (sessionId) =>
  request(`/sessions/${sid(sessionId)}`, { method: "DELETE" });

// Rest baseline (T1). Needs at least 180 s of continuous BVP (64 Hz) and EDA (4 Hz).
// No Content-Type header is set on purpose: the browser adds the multipart boundary itself.
export const uploadBaseline = (sessionId, bvpFile, edaFile, participantId) =>
  request(`/sessions/${sid(sessionId)}/baseline`, {
    method: "POST",
    body: recordingForm(bvpFile, edaFile, participantId),
  });

// Fits the rest calibrator on the uploaded T1 baseline. No request body.
export const calibrateBaseline = (sessionId) =>
  request(`/sessions/${sid(sessionId)}/baseline/calibrate`, { method: "POST" });

// Stress task inference (T2 speech, T3 arithmetic). Needs at least 40 s.
// The server returns 400 if the baseline has not been calibrated first.
export function runTask(sessionId, taskId, bvpFile, edaFile, participantId) {
  if (!SIGNAL_TASKS.includes(taskId)) {
    throw new ApiError(`Unknown task "${taskId}". Use T2 or T3.`);
  }
  return request(`/sessions/${sid(sessionId)}/tasks/${taskId}`, {
    method: "POST",
    body: recordingForm(bvpFile, edaFile, participantId),
  });
}

// Cross-task summary. With v5 this holds per-task results and an explanation,
// not probability-based reactivity.
export const getSummary = (sessionId) => request(`/sessions/${sid(sessionId)}/summary`);
