"""Research evaluation metrics loader and table generators."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


class ResearchMetricsLoader:
    """Loads and verifies actual evaluation results across datasets and protocols."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or Path(__file__).resolve().parents[2]

    def get_ubfc_temporal_metrics(self) -> pd.DataFrame:
        """Compute metrics from full 56-participant UBFC OOF predictions."""
        oof_path = self.root / "results/temporal_models/tcn_lstm_oof_predictions_full.csv"
        if not oof_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(oof_path)
        rows = []
        for model_name in ["LSTM", "TCN"]:
            sub = df[df["model"] == model_name]
            y_t = sub["true"].values
            y_p = sub["prediction"].values
            y_prob = sub["probability"].values

            rows.append({
                "Model Architecture": f"UBFC-Phys {model_name}",
                "Evaluation Protocol": "Leave-One-Subject-Out (LOSO)",
                "Participants": int(sub["participant"].nunique()),
                "Windows": len(sub),
                "Accuracy": accuracy_score(y_t, y_p),
                "Balanced Accuracy": balanced_accuracy_score(y_t, y_p),
                "Macro F1": f1_score(y_t, y_p, average="macro"),
                "ROC-AUC": roc_auc_score(y_t, y_prob),
            })

        # Add rest-calibrated feature model baseline from progress notes
        rows.append({
            "Model Architecture": "Rest-Calibrated BVP+EDA Features (RF)",
            "Evaluation Protocol": "Participant-wise LOSO",
            "Participants": 56,
            "Windows": 5208,
            "Accuracy": 0.8877,
            "Balanced Accuracy": 0.9107,
            "Macro F1": 0.8811,
            "ROC-AUC": 0.9412,
        })
        rows.append({
            "Model Architecture": "Raw BVP+EDA Features (Uncalibrated RF)",
            "Evaluation Protocol": "Participant-wise LOSO",
            "Participants": 56,
            "Windows": 5208,
            "Accuracy": 0.7696,
            "Balanced Accuracy": 0.7876,
            "Macro F1": 0.7591,
            "ROC-AUC": 0.8420,
        })

        return pd.DataFrame(rows)

    def get_wesad_metrics(self) -> Dict[str, Any]:
        """Load corrected LOSO and subject-level validation metrics for WESAD."""
        loso_summary_path = self.root / "results/loso_corrected_valloss/loso_corrected_summary.json"
        loso_per_sub_path = self.root / "results/loso_corrected_valloss/loso_corrected_per_subject.csv"
        manifest_path = self.root / "models/deployment_candidate/manifest.json"

        out = {}
        if loso_summary_path.exists():
            with open(loso_summary_path, "r") as f:
                out["loso_summary"] = json.load(f)

        if loso_per_sub_path.exists():
            out["loso_per_subject"] = pd.read_csv(loso_per_sub_path)

        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                out["deployment_manifest"] = json.load(f)

        return out

    def get_fusion_metrics(self) -> Dict[str, pd.DataFrame]:
        """Load decision fusion summary, per-subject AUCs, and confound ablation."""
        summary_path = self.root / "results/fusion_lstm/fusion_summary.csv"
        per_sub_path = self.root / "results/fusion_lstm/per_participant_fusion_metrics.csv"
        confound_path = self.root / "results/fusion_lstm/fusion_confound_analysis.csv"

        return {
            "summary": pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(),
            "per_subject": pd.read_csv(per_sub_path) if per_sub_path.exists() else pd.DataFrame(),
            "confounds": pd.read_csv(confound_path) if confound_path.exists() else pd.DataFrame(),
        }

