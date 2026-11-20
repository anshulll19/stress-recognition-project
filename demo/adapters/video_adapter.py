import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from src.video_feature_extractor import VideoFeatureExtractor


class VideoStressAdapter:
    def __init__(
        self,
        model_path="models/video_distress_upperface.joblib",
        landmarker_path="models/face_landmarker.task",
    ):
        model_path = Path(model_path)

        if model_path.is_dir():
            model_path = model_path / "models" / "video_distress_upperface.joblib"

        if not model_path.is_file():
            model_path = Path("models/video_distress_upperface.joblib")

        if not model_path.is_file():
            raise FileNotFoundError(f"Model bundle not found: {model_path}")

        if not Path(landmarker_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe landmarker not found: {landmarker_path}"
            )

        bundle = joblib.load(model_path)

        required_keys = {"model", "scaler", "features"}
        if not required_keys.issubset(bundle):
            raise ValueError(f"Model bundle must contain {required_keys}")

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_cols = list(bundle["features"])

        if len(self.feature_cols) != 19:
            raise ValueError(
                f"Expected 19 upper-face features, got {len(self.feature_cols)}"
            )

        if self.scaler.n_features_in_ != len(self.feature_cols):
            raise ValueError("Scaler feature count does not match model schema.")

        if self.model.n_features_in_ != len(self.feature_cols):
            raise ValueError("Model feature count does not match model schema.")

        if 1 not in self.model.classes_:
            raise ValueError("Model does not contain distress class 1.")

        self.distress_class_index = list(self.model.classes_).index(1)

        self.extractor = VideoFeatureExtractor(
            model_path=landmarker_path
        )

        print(f"Loaded video model: {model_path}")
        print(f"Loaded {len(self.feature_cols)} model features.")

    def process_video(self, video_path, participant_id="runtime_user"):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        windows_df = self.extractor.extract_video_features(
            video_path,
            participant_id=participant_id,
        )

        if windows_df.empty:
            raise ValueError("No video windows were extracted.")

        # Confirm the exact model feature schema.
        missing = [
            col for col in self.feature_cols
            if col not in windows_df.columns
        ]

        if missing:
            raise ValueError(f"Missing model features: {missing}")

        # Use the extractor's actual reliability field.
        reliable_mask = windows_df["reliable"].fillna(False).astype(bool)

        reliable_df = windows_df.loc[reliable_mask].copy()

        # Initialize every window, including unreliable ones.
        window_outputs = []

        for _, row in windows_df.iterrows():
            window_outputs.append({
                "participant_id": str(row["participant_id"]),
                "window_index": int(row["window_index"]),
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "detection_fraction": float(row["detection_fraction"]),
                "reliable": bool(row["reliable"]),
                "prediction": None,
                "stress_probability_score": None,
                "smoothed_probability_score": None,
            })

        if reliable_df.empty:
            return {
                "participant_id": participant_id,
                "status": "warning",
                "message": "No windows passed the video reliability checks.",
                "total_windows": len(windows_df),
                "reliable_windows": 0,
                "unreliable_windows": len(windows_df),
                "window_predictions": window_outputs,
            }

        # Never silently fill missing features with zero.
        X_live = reliable_df[self.feature_cols].to_numpy(dtype=float)

        if not np.isfinite(X_live).all():
            raise ValueError(
                "Reliable video windows contain NaN or infinite model features."
            )

        X_scaled = self.scaler.transform(X_live)

        predictions = self.model.predict(X_scaled)
        scores = self.model.predict_proba(X_scaled)[
            :, self.distress_class_index
        ]

        # These are RF class scores, not calibrated probabilities.
        reliable_df["prediction"] = predictions.astype(int)
        reliable_df["stress_probability_score"] = scores

        reliable_df["smoothed_probability_score"] = (
            reliable_df["stress_probability_score"]
            .rolling(window=3, min_periods=1)
            .mean()
        )

        # Map predictions back to their original window positions.
        for position, (_, row) in enumerate(reliable_df.iterrows()):
            original_index = windows_df.index.get_loc(row.name)

            window_outputs[original_index].update({
                "prediction": int(row["prediction"]),
                "stress_probability_score": float(
                    row["stress_probability_score"]
                ),
                "smoothed_probability_score": float(
                    row["smoothed_probability_score"]
                ),
            })

        mean_score = float(
            reliable_df["smoothed_probability_score"].mean()
        )

        return {
            "participant_id": participant_id,
            "status": "success",
            "total_windows": len(windows_df),
            "reliable_windows": len(reliable_df),
            "unreliable_windows": len(windows_df) - len(reliable_df),
            "window_predictions": window_outputs,
            "session_mean_stress_probability_score": mean_score,
            "score_interpretation": (
                "Uncalibrated Random Forest class-1 score; "
                "not a calibrated probability."
            ),
            "overall_session_assessment": (
                "High Stress Indicator"
                if mean_score > 0.5
                else "Calm / Baseline"
            ),
        }


VideoInferenceAdapter = VideoStressAdapter


if __name__ == "__main__":
    adapter = VideoStressAdapter()
    print("Video inference adapter initialized successfully.")