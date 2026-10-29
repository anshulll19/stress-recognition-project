"""Adapters: turn this repo's real model outputs into ``BranchPrediction``.

Everything here reads PRERECORDED data (UBFC-Phys), so every prediction is
labelled ``DataSource.REPLAY``. Labels are task-derived proxies, not validated
stress labels (see models/video_model_metadata.json).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Sequence

from src.prediction_schema import Branch, BranchPrediction, DataSource, utc_now

# Class 0 / 1 as defined in models/video_model_metadata.json
CLASS_NAMES = ("rest_task_proxy", "stress_task_proxy")

# Same flatline rule as the existing API (src/api.py /validate-window)
FLATLINE_STD = 1e-8
LOW_DETECTION_FRACTION = 0.8


def _probs(p_stress: float) -> dict:
    p = min(max(float(p_stress), 0.0), 1.0)
    return {CLASS_NAMES[0]: 1.0 - p, CLASS_NAMES[1]: p}


# --------------------------------------------------------------- physiology
def signal_quality_flatline(samples: Sequence[float]) -> tuple[float, List[str]]:
    """Minimal physiology quality check: finite values and non-zero variance.

    This only detects dropout/flatline. It is NOT a full signal-quality index;
    replace it when Member 1 supplies one.
    """
    vals = [float(x) for x in samples]
    if not vals or not all(math.isfinite(v) for v in vals):
        return 0.0, ["signal_lost"]
    mean = sum(vals) / len(vals)
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
    if std < FLATLINE_STD:
        return 0.0, ["signal_lost"]
    return 1.0, ["quality_flatline_check_only"]


def physiology_from_probability(p_stress: float, quality: float,
                                quality_flags: Optional[List[str]] = None,
                                model_version: str = "unknown",
                                window_ref: Optional[str] = None,
                                source: DataSource = DataSource.REPLAY,
                                timestamp: Optional[str] = None) -> BranchPrediction:
    return BranchPrediction(
        branch=Branch.PHYSIOLOGY, available=True, source=source,
        probabilities=_probs(p_stress), quality=quality,
        quality_flags=list(quality_flags or []),
        timestamp=timestamp or utc_now().isoformat(),
        model_version=model_version, window_ref=window_ref)


def load_oof_predictions(csv_path: Path, model: str = "LSTM"):
    """Load out-of-fold physiology predictions (results/temporal_models/...)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df[df["model"] == model].reset_index(drop=True)


def physiology_from_oof(df, participant: int, window_index: int, quality: float,
                        quality_flags: Optional[List[str]] = None) -> BranchPrediction:
    """One row of the OOF file -> BranchPrediction. The file has no quality
    column, so the caller must supply ``quality``."""
    row = df[(df["participant"] == participant) & (df["window_index"] == window_index)]
    if row.empty:
        return unavailable(Branch.PHYSIOLOGY, f"p{participant}/w{window_index}",
                           "window_not_found")
    r = row.iloc[0]
    return physiology_from_probability(
        r["probability"], quality, quality_flags,
        model_version=f"{r['model']}_oof",
        window_ref=f"p{participant}/w{window_index}")


# -------------------------------------------------------------------- video
def load_video_predictor(model_path: Path, feature_names_path: Path):
    """Load the saved video baseline. Returns (model, feature_names) or
    (None, reason) if it cannot be loaded (e.g. scikit-learn version mismatch)."""
    try:
        import joblib
        names = json.loads(Path(feature_names_path).read_text(encoding="utf-8"))
        return joblib.load(model_path), names
    except Exception as exc:  # noqa: BLE001 - any load failure must not crash the demo
        return None, f"{type(exc).__name__}: {exc}"


def video_quality_from_window(row: Any) -> tuple[float, List[str]]:
    """Quality from the face-detection columns in results/video/video_windows.csv."""
    flags: List[str] = []
    frac = float(row["detection_fraction"])
    if not bool(row["window_complete"]):
        flags.append("window_incomplete")
    if not bool(row["window_reliable"]):
        flags.append("window_unreliable")
    if frac <= 0.0:
        flags.append("face_not_detected")
    elif frac < LOW_DETECTION_FRACTION:
        flags.append("low_face_detection")
    quality = 0.0 if flags and ("window_unreliable" in flags or "window_incomplete" in flags) else frac
    return quality, flags


def video_from_window_row(row: Any, model: Any, feature_names: List[str]) -> BranchPrediction:
    ref = f"{row['subject_id']}/{row['session_id']}/{row['window_start_sec']:g}-{row['window_end_sec']:g}s"
    quality, flags = video_quality_from_window(row)
    try:
        import pandas as pd
        X = pd.DataFrame([[row[n] for n in feature_names]], columns=feature_names)
        p_stress = float(model.predict_proba(X)[0][1])
    except Exception as exc:  # noqa: BLE001
        return unavailable(Branch.VIDEO, ref, f"model_error:{type(exc).__name__}")
    return BranchPrediction(
        branch=Branch.VIDEO, available=True, source=DataSource.REPLAY,
        probabilities=_probs(p_stress), quality=quality, quality_flags=flags,
        timestamp=utc_now().isoformat(), model_version="video_rf_baseline",
        window_ref=ref)


# ------------------------------------------------------------------ shared
def unavailable(branch: Branch, window_ref: Optional[str], reason: str) -> BranchPrediction:
    return BranchPrediction(branch=branch, available=False, source=DataSource.REPLAY,
                            quality=0.0, quality_flags=[reason],
                            timestamp=utc_now().isoformat(), window_ref=window_ref)
