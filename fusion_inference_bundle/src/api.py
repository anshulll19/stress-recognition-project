from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "deployment_candidate" / "classifier.keras"

app = FastAPI(
    title="Multimodal Stress Research Platform",
    version="0.1.0",
    description="Research API; not a medical diagnostic system.",
)


class WindowRequest(BaseModel):
    samples: list[float] = Field(min_length=3840, max_length=3840)


@app.get("/")
def root():
    return {
        "name": "Multimodal Stress Research Platform",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "research_api_ready",
        "prediction_enabled": False,
        "checkpoint_exists": CHECKPOINT.is_file(),
        "label_mapping": "unverified",
        "message": (
            "Predictions are disabled until label provenance, preprocessing "
            "compatibility, and checkpoint validation are established."
        ),
    }


@app.post("/validate-window")
def validate_window(request: WindowRequest):
    x = np.asarray(request.samples, dtype=np.float64)

    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="Samples must be finite.")

    signal_std = float(x.std())
    if signal_std < 1e-8:
        raise HTTPException(
            status_code=422,
            detail="Near-zero variance: check for a flatline or sensor dropout.",
        )

    return {
        "valid": True,
        "sample_count": int(x.size),
        "sampling_rate_hz": 64,
        "window_seconds": 60,
        "signal_std": signal_std,
        "prediction_performed": False,
        "message": "Shape and numerical checks passed; no stress prediction was made.",
    }

import pandas as pd

FEATURES_PATH = (
    ROOT / "ubfc_pipeline" / "out" / "features_full" / "features.csv"
)


@app.get("/pilot/features/{window_index}")
def pilot_features(window_index: int):
    if window_index < 0:
        raise HTTPException(status_code=404, detail="Window not found.")

    if not FEATURES_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Pilot feature table is unavailable.",
        )

    df = pd.read_csv(FEATURES_PATH)

    if window_index >= len(df):
        raise HTTPException(status_code=404, detail="Window not found.")

    row = df.iloc[window_index]

    def select(prefix):
        return {
            str(k): float(v)
            for k, v in row.items()
            if str(k).startswith(prefix) and pd.notna(v)
        }

    return {
        "source": "UBFC-Phys single-participant T1 pilot",
        "window": {
            "index": int(window_index),
            "subject_id": int(row["sid"]),
            "task": str(row["task"]),
            "start_seconds": float(row["t0"]),
            "end_seconds": float(row["t1"]),
        },
        "physiological_features": select("p_"),
        "video_features": select("v_"),
        "quality_features": select("q_"),
        "prediction": None,
        "interpretation": (
            "Experimental features from one participant and one task. "
            "Not a validated stress or distress assessment."
        ),
    }
