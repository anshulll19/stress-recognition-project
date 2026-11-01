import json
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = Path.home() / "Downloads/ubfc_test_upload."
MODEL_PATH = ROOT / "models/video_distress_upperface.joblib"
FEATURE_PATH = ROOT / "models/video_upperface_features.json"
LANDMARKER_PATH = ROOT / "models/face_landmarker.task"

OUTPUT_PATH = ROOT / "results/local_s1/upperface_s1_predictions.csv"

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
    "brow_eye",
    "turn_proxy",
    "nod_proxy",
    "tilt_proxy",
    "turn_depth_proxy",
]

EXPECTED_FEATURES = [
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

SAMPLE_FPS = 5.0
WINDOW_SECONDS = 10.0
MIN_VALID_FRAMES = 20


def extract_frame_features(frame, detector, previous_points):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    result = detector.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
    )

    if not result.face_landmarks:
        return None, None

    lm = result.face_landmarks[0]

    p = {
        i: np.array(
            [lm[i].x, lm[i].y, lm[i].z],
            dtype=float
        )
        for i in NEEDED
    }

    def d(a, b):
        return float(np.linalg.norm(p[a][:2] - p[b][:2]))

    face_w = d(234, 454)
    face_h = d(10, 152)
    dx = p[454][0] - p[234][0]
    dy = p[454][1] - p[234][1]

    if min(
        face_w,
        face_h,
        abs(dx),
        d(33, 133),
        d(263, 362),
        d(61, 291),
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
            d(159, 145) / d(33, 133)
            + d(386, 374) / d(263, 362)
        ) / 2,
        "brow_eye": (
            d(105, 159) + d(334, 386)
        ) / 2 / face_h,
        "turn_proxy": (p[1][0] - p[234][0]) / dx - 0.5,
        "nod_proxy": (
            (p[1][1] - p[10][1])
            / (p[152][1] - p[10][1])
        ),
        "tilt_proxy": float(
            np.degrees(np.arctan2(dy, abs(dx)))
        ),
        "turn_depth_proxy": (
            p[454][2] - p[234][2]
        ) / face_w,
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
            float(differences.mean())
            if len(differences) >= 3
            else np.nan
        )

    motion = w["motion"].to_numpy(dtype=float)
    motion = motion[np.isfinite(motion)]

    out["f_motion_max"] = (
        float(motion.max()) if len(motion) >= 3 else np.nan
    )
    out["f_motion_p90"] = (
        float(np.percentile(motion, 90))
        if len(motion) >= 3
        else np.nan
    )

    return pd.DataFrame(
        [[out[name] for name in EXPECTED_FEATURES]],
        columns=EXPECTED_FEATURES,
    )


def process_video(task, video_path, detector, model, features):
    print(f"\n{'=' * 60}")
    print(f"Processing {task}: {video_path.name}")

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video FPS: {fps:.3f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.3f} seconds")

    if fps <= 0 or duration < WINDOW_SECONDS:
        cap.release()
        raise ValueError("Video is shorter than the required 10-second window.")

    # Select the frame nearest each target timestamp:
    # 0.0, 0.2, 0.4, ... 9.8 seconds.
    target_times = np.arange(
        0,
        WINDOW_SECONDS,
        1.0 / SAMPLE_FPS
    )

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
            frame,
            detector,
            previous_points
        )

        if values is None:
            previous_points = None
        else:
            rows.append(values)
            previous_points = points

    cap.release()

    print(f"Sampled frames: {sampled}")
    print(f"Valid face frames: {len(rows)}")
    print(
        "Face detection fraction:",
        f"{len(rows) / max(sampled, 1):.1%}"
    )

    if sampled != 50:
        raise RuntimeError(
            f"Expected 50 sampled frames, got {sampled}."
        )

    if len(rows) < MIN_VALID_FRAMES:
        raise RuntimeError(
            f"Only {len(rows)} valid face frames. "
            f"Minimum required: {MIN_VALID_FRAMES}."
        )

    X = make_window_features(rows)

    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise RuntimeError(
            f"Non-finite features found; prediction stopped: {missing}"
        )

    if list(X.columns) != features:
        raise RuntimeError("Feature order mismatch.")

    probabilities = model.predict_proba(X)[0]
    prediction = int(model.predict(X)[0])

    print("\nPrediction:")
    print(f"Class 0 — rest-task proxy: {probabilities[0]:.2%}")
    print(f"Class 1 — stress-task proxy: {probabilities[1]:.2%}")
    print(f"Predicted class: {prediction}")

    record = {
        "participant": "s1",
        "task": task,
        "video": str(video_path),
        "duration_sec": duration,
        "sampled_frames": sampled,
        "valid_face_frames": len(rows),
        "face_detection_fraction": len(rows) / sampled,
        "prediction": prediction,
        "class_0_probability": float(probabilities[0]),
        "class_1_probability": float(probabilities[1]),
    }

    record.update(X.iloc[0].to_dict())

    return record


def main():
    for path in [MODEL_PATH, FEATURE_PATH, LANDMARKER_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    features = json.loads(FEATURE_PATH.read_text())

    if features != EXPECTED_FEATURES:
        raise RuntimeError(
            "The feature JSON differs from the expected 19-feature order."
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba.")

    print("Upper-face model loaded:", type(model).__name__)
    print("Feature count:", len(features))

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
        for task, video_path in VIDEO_FILES.items():
            results.append(
                process_video(
                    task,
                    video_path,
                    detector,
                    model,
                    features,
                )
            )
    finally:
        detector.close()

    output = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print(output[
        [
            "task",
            "valid_face_frames",
            "prediction",
            "class_0_probability",
            "class_1_probability",
        ]
    ].to_string(index=False))

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
