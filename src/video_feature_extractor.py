"""Standalone MediaPipe facial-video feature extractor.

Produces the project's 23-feature window schema and its 19-feature upper-face
subset. Defaults: 5 FPS, 10-second non-overlapping windows, 80% detection
fraction, at least 20 valid frames. Requires OpenCV, NumPy, Pandas and the
MediaPipe Tasks API; model asset defaults to models/face_landmarker.task.

NOTE: Validate numerical parity against the original notebook/reference
extractor before using these features with an already-trained classifier.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import pandas as pd

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    mp_python = vision = None
    _MP_IMPORT_ERROR = exc
else:
    _MP_IMPORT_ERROR = None


FULL_FEATURES = [
    "f_motion_mean", "f_motion_std",
    "f_ear_mean_mean", "f_ear_mean_std",
    "f_mouth_open_mean", "f_mouth_open_std",
    "f_mouth_width_mean", "f_mouth_width_std",
    "f_brow_eye_mean", "f_brow_eye_std",
    "f_turn_proxy_mean", "f_turn_proxy_std",
    "f_nod_proxy_mean", "f_nod_proxy_std",
    "f_tilt_proxy_mean", "f_tilt_proxy_std",
    "f_turn_depth_proxy_mean", "f_turn_depth_proxy_std",
    "f_turn_proxy_absdiff", "f_nod_proxy_absdiff",
    "f_tilt_proxy_absdiff", "f_motion_max", "f_motion_p90",
]
MOUTH_FEATURES = {
    "f_mouth_open_mean", "f_mouth_open_std",
    "f_mouth_width_mean", "f_mouth_width_std",
}
UPPERFACE_FEATURES = [f for f in FULL_FEATURES if f not in MOUTH_FEATURES]

SIGNALS = [
    "motion", "ear_mean", "mouth_open", "mouth_width", "brow_eye",
    "turn_proxy", "nod_proxy", "tilt_proxy", "turn_depth_proxy",
]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-8 else float("nan")


def _points(landmarks: Sequence) -> np.ndarray:
    return np.asarray([(p.x, p.y, p.z) for p in landmarks], dtype=np.float64)


def _frame_features(p: np.ndarray, previous: Optional[np.ndarray]) -> Dict[str, float]:
    if p.ndim != 2 or p.shape[0] <= 454 or p.shape[1] < 3 or not np.isfinite(p).all():
        raise ValueError("Invalid or incomplete face landmarks.")
    fw = _dist(p[234], p[454])
    fh = abs(float(p[152, 1] - p[10, 1]))
    if fw <= 1e-8 or fh <= 1e-8:
        raise ValueError("Degenerate face dimensions.")

    ear_l = _ratio(_dist(p[159], p[145]), _dist(p[33], p[133]))
    ear_r = _ratio(_dist(p[386], p[374]), _dist(p[263], p[362]))
    ear = float(np.nanmean([ear_l, ear_r]))
    mouth_open = _ratio(_dist(p[13], p[14]), _dist(p[61], p[291]))
    mouth_width = _ratio(_dist(p[61], p[291]), fw)
    brow_eye = _ratio((_dist(p[105], p[159]) + _dist(p[334], p[386])) / 2, fh)
    turn = _ratio(float(p[1, 0] - p[234, 0]), float(p[454, 0] - p[234, 0])) - 0.5
    nod = _ratio(float(p[1, 1] - p[10, 1]), abs(float(p[152, 1] - p[10, 1])))
    tilt = math.degrees(math.atan2(float(p[454, 1] - p[234, 1]),
                                   abs(float(p[454, 0] - p[234, 0]))))
    depth = _ratio(float(p[454, 2] - p[234, 2]), fw)

    motion = float("nan")
    if previous is not None and previous.shape == p.shape:
        motion = float(np.mean(np.linalg.norm(p[:, :2] - previous[:, :2], axis=1)) / fw)

    return dict(motion=motion, ear_mean=ear, mouth_open=mouth_open,
                mouth_width=mouth_width, brow_eye=brow_eye, turn_proxy=turn,
                nod_proxy=nod, tilt_proxy=tilt, turn_depth_proxy=depth)


def _aggregate(rows: List[Dict[str, float]], detection_fraction: float,
               min_detection_fraction: float, min_valid_frames: int) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for signal in SIGNALS:
        v = np.asarray([r.get(signal, np.nan) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        prefix = "ear_mean" if signal == "ear_mean" else signal
        out[f"f_{prefix}_mean"] = float(np.mean(v)) if len(v) >= 3 else np.nan
        out[f"f_{prefix}_std"] = float(np.std(v, ddof=0)) if len(v) >= 3 else np.nan
    for signal in ("turn_proxy", "nod_proxy", "tilt_proxy"):
        v = np.asarray([r.get(signal, np.nan) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out[f"f_{signal}_absdiff"] = float(np.mean(np.abs(np.diff(v)))) if len(v) >= 2 else np.nan
    motion = np.asarray([r.get("motion", np.nan) for r in rows], dtype=float)
    motion = motion[np.isfinite(motion)]
    out["f_motion_max"] = float(np.max(motion)) if len(motion) else np.nan
    out["f_motion_p90"] = float(np.percentile(motion, 90)) if len(motion) else np.nan
    valid_count = sum(bool(any(np.isfinite(r.get(s, np.nan)) for s in SIGNALS)) for r in rows)
    out = {name: out.get(name, np.nan) for name in FULL_FEATURES} | {
        "detection_fraction": float(detection_fraction),
        "valid_frame_count": int(valid_count),
        "reliable": bool(detection_fraction >= min_detection_fraction and valid_count >= min_valid_frames),
    }
    return out


class VideoFeatureExtractor:
    def __init__(
        self,
        model_path: Union[str, Path] = "models/face_landmarker.task",
        target_fps: float = 5.0,
        window_duration: float = 10.0,
        window_stride: Optional[float] = None,
        min_detection_fraction: float = 0.80,
        min_valid_frames: int = 20,
        max_frame_dimension: Optional[int] = 960,
        upperface_schema_path: Optional[Union[str, Path]] = "models/video_upperface_features.json",
    ):
        if target_fps <= 0 or window_duration <= 0:
            raise ValueError("target_fps and window_duration must be positive.")
        self.model_path = Path(model_path)
        self.target_fps = float(target_fps)
        self.window_duration = float(window_duration)
        self.window_stride = float(window_stride or window_duration)
        self.min_detection_fraction = float(min_detection_fraction)
        self.min_valid_frames = int(min_valid_frames)
        self.max_frame_dimension = max_frame_dimension
        if not 0 < self.min_detection_fraction <= 1 or self.min_valid_frames < 1 or self.window_stride <= 0:
            raise ValueError("Invalid reliability threshold, minimum frames, or window stride.")
        self.upperface_features = list(UPPERFACE_FEATURES)
        if upperface_schema_path and Path(upperface_schema_path).is_file():
            schema = json.loads(Path(upperface_schema_path).read_text())
            if schema != UPPERFACE_FEATURES:
                raise ValueError("Saved upper-face schema differs from this extractor's 19-feature order.")
            self.upperface_features = schema

    def _landmarker(self):
        if _MP_IMPORT_ERROR is not None:
            raise ImportError("Install a compatible mediapipe package to use VideoFeatureExtractor.") from _MP_IMPORT_ERROR
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MediaPipe model not found: {self.model_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def extract_video_features(self, video_path: Union[str, Path],
                               participant_id: Optional[str] = None) -> pd.DataFrame:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError(f"Invalid source FPS: {fps}")
        duration = nframes / fps if nframes > 0 else 0.0
        if duration < self.window_duration:
            cap.release()
            raise ValueError(f"Video duration {duration:.2f}s is shorter than one {self.window_duration:.2f}s window.")

        frame_step = max(1, int(round(fps / self.target_fps)))
        sampled = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_step == 0:
                ts = idx / fps
                if self.max_frame_dimension:
                    h, w = frame.shape[:2]
                    scale = min(1.0, self.max_frame_dimension / max(h, w))
                    if scale < 1:
                        frame = cv2.resize(frame, (round(w * scale), round(h * scale)),
                                           interpolation=cv2.INTER_AREA)
                sampled.append((ts, frame))
            idx += 1
        cap.release()

        landmarker = self._landmarker()
        records = []
        previous = None
        last_ms = -1
        try:
            for ts, bgr in sampled:
                ms = max(int(round(ts * 1000)), last_ms + 1)
                last_ms = ms
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp_python.Image(image_format=mp_python.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, ms)
                if not result.face_landmarks:
                    records.append((ts, None))
                    continue
                try:
                    pts = _points(result.face_landmarks[0])
                    feat = _frame_features(pts, previous)
                    previous = pts
                    records.append((ts, feat))
                except (ValueError, IndexError, FloatingPointError):
                    records.append((ts, None))
        finally:
            landmarker.close()

        output = []
        start = 0.0
        win_idx = 0
        while start + self.window_duration <= duration + 1e-6:
            end = start + self.window_duration
            in_window = [r for r in records if start <= r[0] < end]
            detected = [r for _, r in in_window if r is not None]
            expected = max(1, int(round(self.window_duration * self.target_fps)))
            detection_fraction = min(1.0, len(detected) / expected)
            agg = _aggregate(detected, detection_fraction, self.min_detection_fraction,
                             self.min_valid_frames)
            output.append({
                "participant_id": participant_id or video_path.stem,
                "video_path": str(video_path),
                "window_index": win_idx,
                "start_s": float(start),
                "end_s": float(end),
                "n_sampled_frames": len(in_window),
                **agg,
            })
            win_idx += 1
            start += self.window_stride

        if not output:
            raise ValueError("No complete windows were generated.")
        columns = ["participant_id", "video_path", "window_index", "start_s", "end_s",
                   "n_sampled_frames", *FULL_FEATURES, "detection_fraction",
                   "valid_frame_count", "reliable"]
        return pd.DataFrame(output, columns=columns)

    def extract_upperface_features(self, video_path: Union[str, Path],
                                   participant_id: Optional[str] = None) -> pd.DataFrame:
        df = self.extract_video_features(video_path, participant_id)
        metadata = ["participant_id", "video_path", "window_index", "start_s", "end_s",
                    "n_sampled_frames", "detection_fraction", "valid_frame_count", "reliable"]
        return df[metadata + self.upperface_features].copy()
