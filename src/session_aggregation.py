"""
session_aggregation.py
======================

Task and multi-task session aggregation engine for physiological inference.
Calculates high-integrity summary metrics (stress percentage, latency, mean/peak probabilities).
Strictly enforces safety: unreliable/dropout windows are NEVER counted as non-stress.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class WindowInferenceRecord:
    """Detailed record for a single physiological inference window."""
    window_index: int
    start_s: float
    end_s: float
    is_reliable: bool
    prediction: Optional[int]            # 1 = stress, 0 = non-stress, None = unreliable/dropout
    probability: Optional[float]        # P(stress) in [0.0, 1.0], None if unreliable
    confidence: Optional[float]         # max(P, 1-P), None if unreliable
    class_name: str                     # 'stress', 'non-stress', or 'unreliable'
    quality_flag: str                   # 'OK', 'DROPOUT', 'UNRELIABLE'
    quality_reason: Optional[str]
    latency_ms: float
    is_calibrated: bool
    features: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "window_index": self.window_index,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "is_reliable": self.is_reliable,
            "prediction": self.prediction,
            "probability": round(self.probability, 4) if self.probability is not None else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "class_name": self.class_name,
            "quality_flag": self.quality_flag,
            "quality_reason": self.quality_reason,
            "latency_ms": round(self.latency_ms, 2),
            "is_calibrated": self.is_calibrated,
            "features": {k: round(v, 6) for k, v in self.features.items()} if self.features else None,
        }


@dataclass
class TaskSummary:
    """Aggregated physiological summary for a specific task (T1, T2, or T3)."""
    task_id: str
    task_name: str
    participant_id: Optional[str]
    total_windows: int
    reliable_windows: int
    unreliable_windows: int
    stress_windows: int
    non_stress_windows: int
    stress_percentage: float            # Based solely on reliable windows
    mean_stress_probability: Optional[float]
    peak_stress_probability: Optional[float]
    min_stress_probability: Optional[float]
    mean_confidence: Optional[float]
    mean_latency_ms: float
    total_latency_ms: float
    dominant_state: str                 # 'stress', 'non-stress', or 'inconclusive'
    is_calibrated: bool
    windows: List[WindowInferenceRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_windows: bool = True) -> dict:
        data = {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "participant_id": self.participant_id,
            "total_windows": self.total_windows,
            "reliable_windows": self.reliable_windows,
            "unreliable_windows": self.unreliable_windows,
            "stress_windows": self.stress_windows,
            "non_stress_windows": self.non_stress_windows,
            "stress_percentage": round(self.stress_percentage, 2),
            "mean_stress_probability": round(self.mean_stress_probability, 4) if self.mean_stress_probability is not None else None,
            "peak_stress_probability": round(self.peak_stress_probability, 4) if self.peak_stress_probability is not None else None,
            "min_stress_probability": round(self.min_stress_probability, 4) if self.min_stress_probability is not None else None,
            "mean_confidence": round(self.mean_confidence, 4) if self.mean_confidence is not None else None,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "dominant_state": self.dominant_state,
            "is_calibrated": self.is_calibrated,
            "metadata": self.metadata,
        }
        if include_windows:
            data["windows"] = [w.to_dict() for w in self.windows]
        return data


TASK_NAMES: Dict[str, str] = {
    "T1": "Rest / Baseline",
    "T2": "Speech Stress Task",
    "T3": "Mental Arithmetic Stress Task",
}


def aggregate_task_results(
    task_id: str,
    window_records: List[WindowInferenceRecord],
    participant_id: Optional[str] = None,
    is_calibrated: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> TaskSummary:
    """
    Compute task-level summary metrics from individual window records.

    Crucial Guarantee:
    Unreliable windows (e.g. sensor dropout) have prediction=None and probability=None.
    They are NEVER counted in non_stress_windows or used to dilute stress_percentage.
    """
    total_windows = len(window_records)
    reliable_records = [w for w in window_records if w.is_reliable and w.prediction is not None]
    reliable_count = len(reliable_records)
    unreliable_count = total_windows - reliable_count

    stress_count = sum(1 for w in reliable_records if w.prediction == 1)
    non_stress_count = sum(1 for w in reliable_records if w.prediction == 0)

    if reliable_count > 0:
        stress_pct = (stress_count / reliable_count) * 100.0
        probs = [w.probability for w in reliable_records if w.probability is not None]
        confs = [w.confidence for w in reliable_records if w.confidence is not None]
        mean_prob = float(np.mean(probs)) if probs else None
        peak_prob = float(np.max(probs)) if probs else None
        min_prob = float(np.min(probs)) if probs else None
        mean_conf = float(np.mean(confs)) if confs else None
        dominant = "stress" if stress_pct >= 50.0 else "non-stress"
    else:
        stress_pct = 0.0
        mean_prob = None
        peak_prob = None
        min_prob = None
        mean_conf = None
        dominant = "inconclusive"

    latencies = [w.latency_ms for w in window_records]
    total_lat = float(np.sum(latencies)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0

    return TaskSummary(
        task_id=task_id.upper(),
        task_name=TASK_NAMES.get(task_id.upper(), f"Task {task_id}"),
        participant_id=participant_id,
        total_windows=total_windows,
        reliable_windows=reliable_count,
        unreliable_windows=unreliable_count,
        stress_windows=stress_count,
        non_stress_windows=non_stress_count,
        stress_percentage=stress_pct,
        mean_stress_probability=mean_prob,
        peak_stress_probability=peak_prob,
        min_stress_probability=min_prob,
        mean_confidence=mean_conf,
        mean_latency_ms=mean_lat,
        total_latency_ms=total_lat,
        dominant_state=dominant,
        is_calibrated=is_calibrated,
        windows=window_records,
        metadata=metadata or {},
    )


def aggregate_session_summary(
    session_id: str,
    participant_id: Optional[str],
    task_summaries: Dict[str, TaskSummary],
    is_calibrated: bool,
) -> Dict[str, Any]:
    """
    Produce unified multi-task session summary comparing baseline (T1) vs stressors (T2, T3).
    """
    tasks_dict = {}
    for tid, summary in task_summaries.items():
        tasks_dict[tid] = summary.to_dict(include_windows=False)

    t1_prob = task_summaries.get("T1").mean_stress_probability if "T1" in task_summaries else None
    t2_prob = task_summaries.get("T2").mean_stress_probability if "T2" in task_summaries else None
    t3_prob = task_summaries.get("T3").mean_stress_probability if "T3" in task_summaries else None

    # Compute delta relative to baseline
    delta_t2 = round(t2_prob - t1_prob, 4) if (t2_prob is not None and t1_prob is not None) else None
    delta_t3 = round(t3_prob - t1_prob, 4) if (t3_prob is not None and t1_prob is not None) else None

    # Overall reactivity profile
    reactivity = "NORMAL"
    if delta_t2 is not None and delta_t3 is not None:
        if delta_t2 > 0.30 or delta_t3 > 0.30:
            reactivity = "HIGH_REACTIVITY"
        elif delta_t2 < 0.10 and delta_t3 < 0.10:
            reactivity = "LOW_REACTIVITY"
        else:
            reactivity = "MODERATE_REACTIVITY"

    return {
        "session_id": session_id,
        "participant_id": participant_id,
        "is_calibrated": is_calibrated,
        "tasks_completed": list(task_summaries.keys()),
        "tasks": tasks_dict,
        "reactivity_profile": {
            "baseline_t1_prob": t1_prob,
            "speech_t2_prob": t2_prob,
            "arithmetic_t3_prob": t3_prob,
            "delta_t2_vs_baseline": delta_t2,
            "delta_t3_vs_baseline": delta_t3,
            "reactivity_class": reactivity,
        },
        "interpretation": (
            "UBFC-Phys multimodal physiological stress evaluation calibrated relative to participant "
            "resting baseline (Task 1). Measures autonomic shifts during speech and arithmetic stressors. "
            "Research output; not a clinical diagnostic instrument."
        ),
    }

