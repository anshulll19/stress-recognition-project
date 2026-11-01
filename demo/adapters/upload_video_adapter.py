"""Raw-video inference using the trained 19-feature upper-face model."""

import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


class UploadVideoAdapter:
    FEATURES = [
        "f_motion_mean",
        "f_motion_std",
        "f_ear_mean_mean",
        "f_ear_mean_std",
        "f_brow_eye_mean",
        "f_brow_eye_std",
        "f_turn_proxy_mean",
        "f_turn_proxy_std",
        "f_nod_proxy_mean",
        "f_nod_proxy_std",
        "f_tilt_proxy_mean",
        "f_tilt_proxy_std",
        "f_turn_depth_proxy_mean",
        "f_turn_depth_proxy_std",
        "f_turn_proxy_absdiff",
        "f_nod_proxy_absdiff",
        "f_tilt_proxy_absdiff",
        "f_motion_max",
        "f_motion_p90",
    ]

    SIGNALS = [
        "motion", "ear_mean", "brow_eye", "turn_proxy",
        "nod_proxy", "tilt_proxy", "turn_depth_proxy",
    ]

    NEEDED = [
        1, 10, 152, 234, 454, 33, 133, 159, 145,
        263, 362, 386, 374, 13, 14, 61, 291, 105, 334
    ]

    def __init__(self, project_root):
        self.root = Path(project_root)
        self.model_path = self.root / "models/video_distress_upperface.joblib"
        self.features_path = self.root / "models/video_upperface_features.json"
        self.landmarker_path = self.root / "models/face_landmarker.task"

        for path in [
            self.model_path, self.features_path, self.landmarker_path
        ]:
            if not path.exists():
                raise FileNotFoundError(path)

        saved_features = json.loads(self.features_path.read_text())
        if saved_features != self.FEATURES:
            raise ValueError("Upper-face feature order does not match model.")

        self.model = joblib.load(self.model_path)

    @staticmethod
    def _distance(points, a, b):
        return float(np.linalg.norm(points[a][:2] - points[b][:2]))

    def _extract_frame(self, frame, detector, previous_points):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )

        if not result.face_landmarks:
            return None, None

        landmarks = result.face_landmarks[0]
        p = {
            i: np.array(
                [landmarks[i].x, landmarks[i].y, landmarks[i].z],
                dtype=float
            )
            for i in self.NEEDED
        }

        d = lambda a, b: self._distance(p, a, b)

        face_w = d(234, 454)
        face_h = d(10, 152)
        dx = p[454][0] - p[234][0]
        dy = p[454][1] - p[234][1]
        eye_left = d(33, 133)
        eye_right = d(263, 362)
        face_height_signed = p[152][1] - p[10][1]

        if min(
            face_w, face_h, abs(dx), eye_left, eye_right,
            abs(face_height_signed)
        ) < 1e-6:
            return None, None

        motion = np.nan
        if previous_points is not None:
            motion = float(np.mean([
                np.linalg.norm(p[i] - previous_points[i])
                for i in self.NEEDED
            ]))

        values = {
            "motion": motion,
            "ear_mean": (
                d(159, 145) / eye_left
                + d(386, 374) / eye_right
            ) / 2,
            "brow_eye": (
                d(105, 159) + d(334, 386)
            ) / 2 / face_h,
            "turn_proxy": (p[1][0] - p[234][0]) / dx - 0.5,
            "nod_proxy": (
                p[1][1] - p[10][1]
            ) / face_height_signed,
            "tilt_proxy": float(
                np.degrees(np.arctan2(dy, abs(dx)))
            ),
            "turn_depth_proxy": (
                p[454][2] - p[234][2]
            ) / face_w,
        }

        return values, p

    def _aggregate(self, rows):
        w = pd.DataFrame(rows)
        out = {}

        for signal in self.SIGNALS:
            values = w[signal].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            out[f"f_{signal}_mean"] = float(values.mean())
            out[f"f_{signal}_std"] = float(values.std())

        for signal in ["turn_proxy", "nod_proxy", "tilt_proxy"]:
            values = w[signal].to_numpy(dtype=float)
            differences = np.abs(np.diff(values))
            differences = differences[np.isfinite(differences)]
            out[f"f_{signal}_absdiff"] = float(differences.mean())

        motion = w["motion"].to_numpy(dtype=float)
        motion = motion[np.isfinite(motion)]
        out["f_motion_max"] = float(motion.max())
        out["f_motion_p90"] = float(np.percentile(motion, 90))

        return pd.DataFrame(
            [[out[name] for name in self.FEATURES]],
            columns=self.FEATURES,
        )

    def predict_video(self, video_bytes):
        if not video_bytes:
            raise ValueError("Uploaded video is empty.")

        import tempfile

        temp_path = None
        detector = None
        cap = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            ) as temp:
                temp.write(video_bytes)
                temp_path = Path(temp.name)

            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                raise ValueError("Could not decode uploaded video.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Invalid video frame rate.")

            duration = total_frames / fps
            if duration < 10:
                raise ValueError(
                    f"Video is {duration:.2f}s. At least 10 seconds required."
                )

            detector = mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(self.landmarker_path)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                )
            )

            target_indices = np.rint(
                np.arange(0, 10, 0.2) * fps
            ).astype(int)

            rows = []
            previous_points = None
            sampled = 0

            for frame_index in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()

                if not ok:
                    previous_points = None
                    continue

                sampled += 1
                values, points = self._extract_frame(
                    frame, detector, previous_points
                )

                if values is None:
                    previous_points = None
                else:
                    rows.append(values)
                    previous_points = points

            valid = len(rows)
            detection_fraction = valid / max(sampled, 1)

            if sampled != 50:
                raise ValueError(
                    f"Could only decode {sampled}/50 sampled frames."
                )

            if valid < 20:
                raise ValueError(
                    f"Only {valid}/50 frames had a valid face. "
                    "At least 20 valid frames are required."
                )

            X = self._aggregate(rows)

            if not np.isfinite(X.to_numpy()).all():
                raise ValueError("Extracted features contain invalid values.")

            probabilities = self.model.predict_proba(X)[0]
            prediction = int(self.model.predict(X)[0])

            return {
                "prediction": prediction,
                "class_label": (
                    "Stress-task proxy" if prediction == 1
                    else "Rest-task proxy"
                ),
                "class_0_probability": float(probabilities[0]),
                "class_1_probability": float(probabilities[1]),
                "sampled_frames": sampled,
                "valid_face_frames": valid,
                "detection_fraction": detection_fraction,
                "duration_sec": duration,
                "features": X.iloc[0].to_dict(),
            }

        finally:
            if cap is not None:
                cap.release()
            if detector is not None:
                detector.close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
