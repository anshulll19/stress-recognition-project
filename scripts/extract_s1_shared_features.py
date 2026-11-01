from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = Path.home() / "Downloads/ubfc_test_upload."
LANDMARKER_PATH = ROOT / "models/face_landmarker.task"
OUTPUT_PATH = ROOT / "results/local_s1/shared_s1_features_23.csv"

VIDEO_FILES = {
    "T1": VIDEO_DIR / "vid_s1_T1.mp4",
    "T3": VIDEO_DIR / "vid_s1_T3.mp4",
}

NEEDED = [
    1, 10, 152, 234, 454, 33, 133, 159, 145,
    263, 362, 386, 374, 13, 14, 61, 291, 105, 334
]

SIGNALS = [
    "motion",
    "ear_mean",
    "mouth_open",
    "mouth_width",
    "brow_eye",
    "turn_proxy",
    "nod_proxy",
    "tilt_proxy",
    "turn_depth_proxy",
]

FEATURES_23 = [
    "f_motion_mean",
    "f_motion_std",
    "f_ear_mean_mean",
    "f_ear_mean_std",
    "f_mouth_open_mean",
    "f_mouth_open_std",
    "f_mouth_width_mean",
    "f_mouth_width_std",
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

SAMPLE_FPS = 5.0
WINDOW_SECONDS = 10.0
MIN_VALID_FRAMES = 20


def extract_frame_features(frame, detector, previous_points):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    result = detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )

    if not result.face_landmarks:
        return None, None

    lm = result.face_landmarks[0]
    p = {
        i: np.array([lm[i].x, lm[i].y, lm[i].z], dtype=float)
        for i in NEEDED
    }

    def d(a, b):
        return float(np.linalg.norm(p[a][:2] - p[b][:2]))

    face_w = d(234, 454)
    face_h = d(10, 152)
    dx = p[454][0] - p[234][0]
    dy = p[454][1] - p[234][1]
    eye_left = d(33, 133)
    eye_right = d(263, 362)
    mouth_width_px = d(61, 291)
    face_height_signed = p[152][1] - p[10][1]

    if min(
        face_w, face_h, abs(dx), eye_left, eye_right,
        mouth_width_px, abs(face_height_signed)
    ) < 1e-6:
        return None, None

    motion = np.nan
    if previous_points is not None:
        motion = float(np.mean([
            np.linalg.norm(p[i] - previous_points[i])
            for i in NEEDED
        ]))

    values = {
        "motion": motion,
        "ear_mean": (
            d(159, 145) / eye_left
            + d(386, 374) / eye_right
        ) / 2,
        "mouth_open": d(13, 14) / mouth_width_px,
        "mouth_width": mouth_width_px / face_w,
        "brow_eye": (
            d(105, 159) + d(334, 386)
        ) / 2 / face_h,
        "turn_proxy": (p[1][0] - p[234][0]) / dx - 0.5,
        "nod_proxy": (p[1][1] - p[10][1]) / face_height_signed,
        "tilt_proxy": float(np.degrees(np.arctan2(dy, abs(dx)))),
        "turn_depth_proxy": (p[454][2] - p[234][2]) / face_w,
    }

    return values, p


def make_window_features(rows):
    w = pd.DataFrame(rows)
    out = {}

    for signal in SIGNALS:
        values = w[signal].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        out[f"f_{signal}_mean"] = (
            float(values.mean()) if len(values) >= 3 else np.nan
        )
        out[f"f_{signal}_std"] = (
            float(values.std()) if len(values) >= 3 else np.nan
        )

    for signal in ["turn_proxy", "nod_proxy", "tilt_proxy"]:
        values = w[signal].to_numpy(dtype=float)
        differences = np.abs(np.diff(values))
        differences = differences[np.isfinite(differences)]

        out[f"f_{signal}_absdiff"] = (
            float(differences.mean()) if len(differences) >= 3 else np.nan
        )

    motion = w["motion"].to_numpy(dtype=float)
    motion = motion[np.isfinite(motion)]

    out["f_motion_max"] = (
        float(motion.max()) if len(motion) >= 3 else np.nan
    )
    out["f_motion_p90"] = (
        float(np.percentile(motion, 90))
        if len(motion) >= 3 else np.nan
    )

    return out


def process_video(task, video_path, detector):
    print(f"\nProcessing {task}: {video_path.name}")

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"FPS: {fps:.3f} | Frames: {total_frames} | Duration: {duration:.3f}s")

    if fps <= 0 or duration < WINDOW_SECONDS:
        cap.release()
        raise ValueError("Video is shorter than 10 seconds.")

    target_times = np.arange(0, WINDOW_SECONDS, 1.0 / SAMPLE_FPS)
    target_indices = np.rint(target_times * fps).astype(int)

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
        values, points = extract_frame_features(
            frame, detector, previous_points
        )

        if values is None:
            previous_points = None
        else:
            rows.append(values)
            previous_points = points

    cap.release()

    valid = len(rows)
    print(f"Sampled: {sampled} | Valid face frames: {valid}/{sampled}")

    if sampled != 50:
        raise RuntimeError(f"Expected 50 sampled frames, got {sampled}.")
    if valid < MIN_VALID_FRAMES:
        raise RuntimeError(f"Only {valid} valid frames; minimum is 20.")

    features = make_window_features(rows)

    if list(features.keys()) != FEATURES_23:
        raise RuntimeError("Internal 23-feature order mismatch.")

    if not np.isfinite(list(features.values())).all():
        bad = [k for k, v in features.items() if not np.isfinite(v)]
        raise RuntimeError(f"Non-finite features: {bad}")

    record = {
        "participant": "s1",
        "task": task,
        "video": str(video_path),
        "duration_sec": duration,
        "sampled_frames": sampled,
        "valid_face_frames": valid,
        "face_detection_fraction": valid / sampled,
    }
    record.update(features)
    return record


def main():
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(LANDMARKER_PATH)

    detector = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(LANDMARKER_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )

    results = []
    try:
        for task, path in VIDEO_FILES.items():
            results.append(process_video(task, path, detector))
    finally:
        detector.close()

    output = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nEXTRACTION COMPLETE")
    print("Feature count:", len(FEATURES_23))
    print(output[["task", "valid_face_frames", "face_detection_fraction"]])
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
