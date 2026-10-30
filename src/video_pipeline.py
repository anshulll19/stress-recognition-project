"""
Video preprocessing and facial feature extraction pipeline for stress recognition.
Preserves exact geometry, 5 FPS sampling rate, face-bounding-box normalization,
and 10-second window aggregation from the verified research pipeline.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

# Standard MediaPipe FaceMesh landmark indices used in research pipeline
LM_NEEDED = [
    1, 10, 152, 234, 454,            # nose tip, forehead, chin, left/right cheek
    33, 133, 159, 145,               # eye 1: outer, inner, upper lid, lower lid
    263, 362, 386, 374,              # eye 2
    13, 14, 61, 291,                 # mouth: upper lip, lower lip, corners
    105, 334                         # eyebrows
]

SIGNALS = [
    "motion", "ear_mean", "mouth_open", "mouth_width", "brow_eye",
    "turn_proxy", "nod_proxy", "tilt_proxy", "turn_depth_proxy"
]

FEATURE_NAMES_23 = [
    "f_motion_mean", "f_motion_std", "f_ear_mean_mean", "f_ear_mean_std",
    "f_mouth_open_mean", "f_mouth_open_std", "f_mouth_width_mean", "f_mouth_width_std",
    "f_brow_eye_mean", "f_brow_eye_std", "f_turn_proxy_mean", "f_turn_proxy_std",
    "f_nod_proxy_mean", "f_nod_proxy_std", "f_tilt_proxy_mean", "f_tilt_proxy_std",
    "f_turn_depth_proxy_mean", "f_turn_depth_proxy_std", "f_turn_proxy_absdiff",
    "f_nod_proxy_absdiff", "f_tilt_proxy_absdiff", "f_motion_max", "f_motion_p90"
]

FEATURE_NAMES_19 = [
    "f_motion_mean", "f_motion_std", "f_ear_mean_mean", "f_ear_mean_std",
    "f_brow_eye_mean", "f_brow_eye_std", "f_turn_proxy_mean", "f_turn_proxy_std",
    "f_nod_proxy_mean", "f_nod_proxy_std", "f_tilt_proxy_mean", "f_tilt_proxy_std",
    "f_turn_depth_proxy_mean", "f_turn_depth_proxy_std", "f_turn_proxy_absdiff",
    "f_nod_proxy_absdiff", "f_tilt_proxy_absdiff", "f_motion_max", "f_motion_p90"
]

WINDOW_SEC = 10.0
STRIDE_SEC = 10.0
SAMPLE_FPS = 5.0
MIN_VALID_FRAMES = 3
MIN_DETECTION_FRACTION = 0.8
MIN_WINDOW_FILL = 0.9


def compute_frame_signals(coords: Optional[np.ndarray], motion: float) -> Dict[str, float]:
    """Compute the 9 geometric signals from normalized landmark coordinates.

    coords shape: (478, 3) normalized by face bounding box.
    """
    if coords is None:
        return {s: np.nan for s in SIGNALS}

    def dist(a: int, b: int) -> float:
        return float(np.hypot(coords[a, 0] - coords[b, 0], coords[a, 1] - coords[b, 1]))

    face_w = dist(234, 454)
    face_h = dist(10, 152)
    dx = coords[454, 0] - coords[234, 0]
    dy = coords[454, 1] - coords[234, 1]

    if face_w < 1e-6 or face_h < 1e-6 or abs(dx) < 1e-6:
        return {s: np.nan for s in SIGNALS}

    left_ear = dist(159, 145) / max(dist(33, 133), 1e-6)
    right_ear = dist(386, 374) / max(dist(263, 362), 1e-6)
    ear_mean = (left_ear + right_ear) / 2.0

    mouth_open = dist(13, 14) / max(dist(61, 291), 1e-6)
    mouth_width = dist(61, 291) / face_w
    brow_eye = (dist(105, 159) + dist(334, 386)) / 2.0 / face_h

    turn_proxy = (coords[1, 0] - coords[234, 0]) / dx - 0.5
    nod_proxy = (coords[1, 1] - coords[10, 1]) / max(abs(coords[152, 1] - coords[10, 1]), 1e-6)
    tilt_proxy = float(np.degrees(np.arctan2(dy, abs(dx))))
    turn_depth_proxy = (coords[454, 2] - coords[234, 2]) / face_w

    sig = {
        "motion": motion,
        "ear_mean": ear_mean,
        "mouth_open": mouth_open,
        "mouth_width": mouth_width,
        "brow_eye": brow_eye,
        "turn_proxy": turn_proxy,
        "nod_proxy": nod_proxy,
        "tilt_proxy": tilt_proxy,
        "turn_depth_proxy": turn_depth_proxy,
    }
    return sig


def aggregate_window_features(df_window: pd.DataFrame) -> Dict[str, float]:
    """Aggregate per-frame signal rows inside one window into f_* features.

    Follows the exact rules from pipeline/common.py:
    - Never fill with zeros; missing values remain np.nan.
    - Requires at least MIN_VALID_FRAMES (3) to compute any statistic.
    """
    out: Dict[str, float] = {}

    for s in SIGNALS:
        v = pd.to_numeric(df_window[s], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        ok = len(v) >= MIN_VALID_FRAMES
        out[f"f_{s}_mean"] = float(v.mean()) if ok else np.nan
        out[f"f_{s}_std"] = float(v.std()) if ok else np.nan

    for s in ["turn_proxy", "nod_proxy", "tilt_proxy"]:
        v = pd.to_numeric(df_window[s], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) >= MIN_VALID_FRAMES:
            dd = np.abs(np.diff(v))
            dd = dd[np.isfinite(dd)]
            out[f"f_{s}_absdiff"] = float(dd.mean()) if len(dd) >= 1 else np.nan
        else:
            out[f"f_{s}_absdiff"] = np.nan

    m = pd.to_numeric(df_window["motion"], errors="coerce").to_numpy(float)
    m = m[np.isfinite(m)]
    out["f_motion_max"] = float(m.max()) if len(m) >= MIN_VALID_FRAMES else np.nan
    out["f_motion_p90"] = float(np.percentile(m, 90)) if len(m) >= MIN_VALID_FRAMES else np.nan

    return out


class VideoPipeline:
    """Video preprocessing and feature extraction using MediaPipe FaceLandmarker."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        project_root = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path) if model_path else (project_root / "models/face_landmarker.task")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {self.model_path}. "
                "Ensure models/face_landmarker.task is present."
            )

    def extract_features_from_video(
        self,
        video_path: Union[str, Path],
        max_duration_sec: Optional[float] = None,
        sample_fps: float = SAMPLE_FPS,
        window_sec: float = WINDOW_SEC,
        stride_sec: float = STRIDE_SEC,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Extract frame signals and window features from a video file.

        Parameters
        ----------
        video_path : Union[str, Path]
            Path to video file (.mp4, .avi, .mov).
        max_duration_sec : Optional[float]
            If provided, process only up to this many seconds (for faster testing).
        sample_fps : float
            Sampling rate in frames per second (default 5.0 FPS).
        window_sec : float
            Window length in seconds (default 10.0 s).
        stride_sec : float
            Window stride in seconds (default 10.0 s, non-overlapping).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
            (windows_df_23, windows_df_19, summary_metrics)
        """
        import mediapipe as mp

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if orig_fps <= 0 or total_frame_count <= 0:
            cap.release()
            raise ValueError(f"Invalid video metadata (fps={orig_fps}, frames={total_frame_count})")

        video_duration = total_frame_count / orig_fps
        target_duration = min(video_duration, max_duration_sec) if max_duration_sec else video_duration

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        interval = 1.0 / sample_fps
        timestamp = 0.0
        frame_rows: List[Dict[str, Any]] = []
        previous_coords: Optional[np.ndarray] = None
        previous_timestamp: Optional[float] = None
        sampled_frames = 0
        detected_frames = 0

        try:
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
                while timestamp < target_duration - 1e-4:
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                    ok, frame = cap.read()
                    if not ok:
                        break

                    sampled_frames += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, int(round(timestamp * 1000.0)))

                    row: Dict[str, Any] = {
                        "timestamp_sec": timestamp,
                        "face_detected": 0,
                        "motion": np.nan,
                    }

                    if result.face_landmarks:
                        detected_frames += 1
                        landmarks = result.face_landmarks[0]
                        coords = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)

                        # Bounding-box normalization
                        x_min, y_min = coords[:, :2].min(axis=0)
                        x_max, y_max = coords[:, :2].max(axis=0)
                        scale = max(x_max - x_min, y_max - y_min, 1e-8)
                        coords[:, 0] = (coords[:, 0] - x_min) / scale
                        coords[:, 1] = (coords[:, 1] - y_min) / scale
                        coords[:, 2] = coords[:, 2] / scale

                        row["face_detected"] = 1

                        # Motion calculation (only for adjacent sampled timestamps)
                        if (
                            previous_coords is not None
                            and previous_timestamp is not None
                            and abs(timestamp - previous_timestamp - interval) <= max(1e-6, interval * 0.05)
                        ):
                            row["motion"] = float(np.linalg.norm(coords - previous_coords, axis=1).mean())

                        previous_coords = coords
                        previous_timestamp = timestamp

                        # Signals
                        sig = compute_frame_signals(coords, row["motion"])
                        row.update(sig)
                    else:
                        previous_coords = None
                        previous_timestamp = None
                        row.update({s: np.nan for s in SIGNALS})

                    frame_rows.append(row)
                    timestamp += interval
        finally:
            cap.release()

        frames_df = pd.DataFrame(frame_rows)

        # ---------------------------------------------------------
        # Window Generation (Strict adherence to pipeline/common.py)
        # ---------------------------------------------------------
        expected_rows_per_window = int(round(window_sec * sample_fps))
        window_rows: List[Dict[str, Any]] = []

        t0 = 0.0
        win_idx = 0
        while t0 + window_sec <= target_duration + 1e-4:
            t1 = t0 + window_sec
            w = frames_df[(frames_df["timestamp_sec"] >= t0 - 1e-5) & (frames_df["timestamp_sec"] < t1 - 1e-5)]
            n_rows = len(w)
            n_detected = int((w["face_detected"] == 1).sum())
            det_frac = (n_detected / n_rows) if n_rows > 0 else 0.0
            is_complete = (n_rows >= int(round(expected_rows_per_window * MIN_WINDOW_FILL)))

            # Training discarded incomplete/partial windows; do not predict on partial windows
            if is_complete:
                feats = aggregate_window_features(w)
                row_w = {
                    "window_idx": win_idx,
                    "window_start_sec": t0,
                    "window_end_sec": t1,
                    "n_rows": n_rows,
                    "n_detected": n_detected,
                    "detection_fraction": det_frac,
                    "window_complete": is_complete,
                    "window_reliable": (det_frac >= MIN_DETECTION_FRACTION),
                }
                row_w.update(feats)
                window_rows.append(row_w)
                win_idx += 1

            t0 += stride_sec

        if not window_rows:
            # Video too short for a full window or no frames decoded
            windows_23 = pd.DataFrame(columns=["window_idx", "window_start_sec", "window_end_sec"] + FEATURE_NAMES_23)
            windows_19 = pd.DataFrame(columns=["window_idx", "window_start_sec", "window_end_sec"] + FEATURE_NAMES_19)
        else:
            df_full = pd.DataFrame(window_rows)
            meta_cols = ["window_idx", "window_start_sec", "window_end_sec", "n_rows", "n_detected",
                         "detection_fraction", "window_complete", "window_reliable"]
            windows_23 = df_full[meta_cols + FEATURE_NAMES_23].copy()
            windows_19 = df_full[meta_cols + FEATURE_NAMES_19].copy()

        summary = {
            "video_path": str(video_path),
            "video_duration_sec": video_duration,
            "processed_duration_sec": target_duration,
            "sampled_frames": sampled_frames,
            "detected_frames": detected_frames,
            "detection_rate": (detected_frames / sampled_frames) if sampled_frames > 0 else 0.0,
            "total_windows": len(window_rows),
            "reliable_windows": sum(r["window_reliable"] for r in window_rows) if window_rows else 0,
        }

        return windows_23, windows_19, summary

