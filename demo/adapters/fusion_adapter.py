"""Multimodal decision-level late fusion adapter for UBFC-Phys.

Integrates verified out-of-fold predictions from physiological (LSTM) and video (RandomForest)
modalities on 972 synchronized windows across 18 participants.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from demo.adapters.base import PredictionResult, ValidationResult


class MultimodalFusionAdapter:
    """Adapter for multimodal decision-level late fusion on aligned windows."""

    CLASS_NAMES = {
        0: "Non-stress (Rest Task T1 Proxy)",
        1: "Stress-related (Speech T2 / Math T3 Proxy)",
    }

    FUSION_METHODS = [
        "blend_0.3_uf",    # Physio-First with clean Upper-Face video (no mouth confound)
        "blend_0.3",       # Original Late Fusion with full face
        "blend_0.2",
        "blend_0.5",
        "quality_aware",
        "stacked_loso",
        "video_upperface",
        "video_only",
        "physio_only",
    ]

    METHOD_DESCRIPTIONS = {
        "blend_0.3_uf": "Physio-First (70%) + Upper-Face Video (30% Clean) · AUC: 0.933",
        "blend_0.3": "Fixed Late Fusion (30% Physio + 70% Full-Face Video) · AUC: 0.996 ⚠",
        "blend_0.2": "Fixed Late Fusion (20% Physio + 80% Full-Face Video) · AUC: 0.995",
        "blend_0.5": "Equal Weighting (50% Physio + 50% Video) · AUC: 0.995",
        "quality_aware": "Dynamic Quality-Aware Weighting (gated by detection fraction) · AUC: 0.995",
        "stacked_loso": "Stacked Logistic Regression Meta-Learner (Inner-LOSO) · AUC: 0.996",
        "video_upperface": "Unimodal Upper-Face Video (19 Features, Clean) · AUC: 0.968",
        "video_only": "Unimodal Full-Face Video (23 Features, Confounded) · AUC: 0.994",
        "physio_only": "Unimodal Physiology Baseline (LSTM) · AUC: 0.814",
    }

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or Path(__file__).resolve().parents[2]
        self.aligned_path = self.root / "results/fusion_lstm/aligned_oof_predictions.csv"
        self.summary_path = self.root / "results/fusion_lstm/fusion_summary.csv"
        self.per_subject_path = self.root / "results/fusion_lstm/per_participant_fusion_metrics.csv"
        self.confound_path = self.root / "results/fusion_lstm/fusion_confound_analysis.csv"

        self._aligned_df = None
        self._summary_df = None
        self._per_subject_df = None
        self._confound_df = None

    def load_aligned_data(self) -> pd.DataFrame:
        if self._aligned_df is None:
            if not self.aligned_path.exists():
                raise FileNotFoundError(f"Aligned fusion file not found at {self.aligned_path}")
            self._aligned_df = pd.read_csv(self.aligned_path)
        return self._aligned_df

    def load_summary(self) -> pd.DataFrame:
        if self._summary_df is None and self.summary_path.exists():
            self._summary_df = pd.read_csv(self.summary_path)
        return self._summary_df

    def load_per_subject(self) -> pd.DataFrame:
        if self._per_subject_df is None and self.per_subject_path.exists():
            self._per_subject_df = pd.read_csv(self.per_subject_path)
        return self._per_subject_df

    def load_confounds(self) -> pd.DataFrame:
        if self._confound_df is None and self.confound_path.exists():
            self._confound_df = pd.read_csv(self.confound_path)
        return self._confound_df

    def get_subjects(self) -> List[str]:
        df = self.load_aligned_data()
        return sorted(df["subject_id"].unique().tolist())

    def get_subject_windows(self, subject_id: str) -> pd.DataFrame:
        df = self.load_aligned_data()
        return df[df["subject_id"] == subject_id].reset_index(drop=True)

    def validate_alignment(
        self, physio_row: pd.Series, video_row: pd.Series
    ) -> ValidationResult:
        """Enforce strict temporal, subject, and task alignment before fusing."""
        errors = []
        warnings = []

        sub_p = str(physio_row.get("subject_id", "")).strip().lower()
        sub_v = str(video_row.get("subject_id", "")).strip().lower()
        if sub_p != sub_v:
            errors.append(f"Subject mismatch: Physio={sub_p} vs Video={sub_v}")

        task_p = physio_row.get("task")
        task_v = video_row.get("task")
        if task_p is not None and task_v is not None and int(task_p) != int(task_v):
            errors.append(f"Task mismatch: Physio task={task_p} vs Video task={task_v}")

        start_p = physio_row.get("window_start_sec", physio_row.get("start_s"))
        start_v = video_row.get("window_start_sec", video_row.get("start_s"))
        if start_p is not None and start_v is not None and abs(float(start_p) - float(start_v)) > 0.01:
            errors.append(f"Timestamp mismatch: Physio start={start_p}s vs Video start={start_v}s")

        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            metadata={"subject": sub_p, "task": task_p, "start_sec": start_p}
        )

    def get_fusion_prediction(
        self, row: pd.Series, method: str = "blend_0.3"
    ) -> PredictionResult:
        """Produce multimodal prediction for an aligned window record."""
        p_col = f"p_{method}"
        if p_col not in row:
            raise KeyError(f"Fusion column '{p_col}' not found. Available: {list(row.index)}")

        prob_class1 = float(row[p_col])
        prob_class1 = float(np.clip(prob_class1, 0.0, 1.0))
        prob_class0 = 1.0 - prob_class1
        probs = np.array([prob_class0, prob_class1], dtype=np.float32)

        pred = 1 if prob_class1 >= 0.5 else 0
        gt = int(row["y_true"]) if "y_true" in row else None

        detection_frac = float(row.get("detection_fraction", 1.0))

        return PredictionResult(
            predicted_class=pred,
            class_label=self.CLASS_NAMES[pred],
            probabilities=probs,
            confidence_score=float(probs[pred]),
            is_calibrated=False,
            ground_truth=gt,
            ground_truth_label=self.CLASS_NAMES[gt] if gt is not None else None,
            window_metadata={
                "subject_id": row["subject_id"],
                "task": int(row["task"]),
                "window_start_sec": float(row["window_start_sec"]),
                "window_end_sec": float(row["window_end_sec"]),
                "p_physio": float(row["p_physio"]),
                "p_video": float(row["p_video"]),
                "p_fused": prob_class1,
                "fusion_method": method,
                "detection_fraction": detection_frac,
            },
            signal_quality_ok=(detection_frac >= 0.8),
            warnings=[
                f"Fusion method: {self.METHOD_DESCRIPTIONS.get(method, method)}",
                "Video modality subject to speech confound during T2 (Public Speaking).",
            ],
        )

