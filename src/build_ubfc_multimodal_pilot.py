from pathlib import Path
import csv
import numpy as np

root = Path("data/ubfc_phys_subset/s1")
frames_dir = root / "frames"
phys_csv = root / "pilot" / "t1_physiological_windows.csv"
face_csv = frames_dir / "face_detection_results.csv"
out_csv = root / "pilot" / "t1_multimodal_pilot.csv"

# Frame samples were extracted at 1 frame per 10 seconds.
FRAME_INTERVAL_S = 10.0
IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 640.0

with face_csv.open(newline="") as f:
    detections = list(csv.DictReader(f))

frame_rows = []
for row in detections:
    frame_index = int(Path(row["frame"]).stem.split("_")[-1])
    timestamp = (frame_index - 1) * FRAME_INTERVAL_S

    if row["status"] == "detected":
        x, y = float(row["x"]), float(row["y"])
        w, h = float(row["width"]), float(row["height"])
        frame_rows.append({
            "timestamp_s": timestamp,
            "face_detected": 1,
            "face_center_x_norm": (x + w / 2) / IMAGE_WIDTH,
            "face_center_y_norm": (y + h / 2) / IMAGE_HEIGHT,
            "face_width_norm": w / IMAGE_WIDTH,
            "face_height_norm": h / IMAGE_HEIGHT,
            "face_area_fraction": (w * h) / (IMAGE_WIDTH * IMAGE_HEIGHT),
        })
    else:
        frame_rows.append({
            "timestamp_s": timestamp,
            "face_detected": 0,
            "face_center_x_norm": None,
            "face_center_y_norm": None,
            "face_width_norm": None,
            "face_height_norm": None,
            "face_area_fraction": None,
        })

feature_names = [
    "face_detected",
    "face_center_x_norm",
    "face_center_y_norm",
    "face_width_norm",
    "face_height_norm",
    "face_area_fraction",
]

with phys_csv.open(newline="") as f:
    windows = list(csv.DictReader(f))

merged = []
for window in windows:
    start = float(window["start_s"])
    end = float(window["end_s"])

    # Half-open interval avoids double-counting frames at boundaries.
    selected = [
        r for r in frame_rows
        if start <= r["timestamp_s"] < end
    ]

    output = dict(window)
    output["video_frames_sampled"] = len(selected)

    for name in feature_names:
        values = [
            r[name] for r in selected
            if r[name] is not None
        ]
        output[f"video_{name}_mean"] = (
            float(np.mean(values)) if values else ""
        )

    output["video_features_valid"] = int(
        len(selected) > 0 and any(
            r["face_detected"] == 1 for r in selected
        )
    )
    merged.append(output)

if not merged:
    raise SystemExit("No physiological windows found.")

with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
    writer.writeheader()
    writer.writerows(merged)

print(f"Physiological windows: {len(windows)}")
print(f"Video frames available: {len(frame_rows)}")
print(f"Merged windows: {len(merged)}")
print(f"Windows with video samples: {sum(int(r['video_frames_sampled']) > 0 for r in merged)}")
print(f"Output: {out_csv}")
print("These are descriptive features, not validated stress predictions.")
