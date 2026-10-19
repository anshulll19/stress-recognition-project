from pathlib import Path
import csv
import subprocess
import json
import numpy as np

root = Path("data/ubfc_phys_subset/s1")
out_dir = root / "pilot"
out_dir.mkdir(parents=True, exist_ok=True)

# Load physiological recordings
bvp = np.loadtxt(root / "bvp_s1_T1.csv", delimiter=",")
eda = np.loadtxt(root / "eda_s1_T1.csv", delimiter=",")

BVP_FS = 64.0
EDA_FS = 4.0
WINDOW_S = 30.0
STEP_S = 5.0

# Obtain video duration from its container metadata
result = subprocess.run(
    [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(root / "vid_s1_T1.avi"),
    ],
    capture_output=True, text=True, check=True,
)
video_duration = float(json.loads(result.stdout)["format"]["duration"])

bvp_duration = len(bvp) / BVP_FS
eda_duration = len(eda) / EDA_FS
common_duration = min(video_duration, bvp_duration, eda_duration)

rows = []
start = 0.0

while start + WINDOW_S <= common_duration:
    end = start + WINDOW_S

    bvp_start = int(round(start * BVP_FS))
    bvp_end = int(round(end * BVP_FS))
    eda_start = int(round(start * EDA_FS))
    eda_end = int(round(end * EDA_FS))

    bvp_window = bvp[bvp_start:bvp_end]
    eda_window = eda[eda_start:eda_end]

    rows.append({
        "start_s": start,
        "end_s": end,
        "bvp_samples": len(bvp_window),
        "eda_samples": len(eda_window),
        "bvp_mean": float(np.mean(bvp_window)),
        "bvp_std": float(np.std(bvp_window)),
        "bvp_min": float(np.min(bvp_window)),
        "bvp_max": float(np.max(bvp_window)),
        "eda_mean": float(np.mean(eda_window)),
        "eda_std": float(np.std(eda_window)),
        "eda_min": float(np.min(eda_window)),
        "eda_max": float(np.max(eda_window)),
        "label": "unlabelled",
    })

    start += STEP_S

output_csv = out_dir / "t1_physiological_windows.csv"

with output_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Video duration: {video_duration:.3f} s")
print(f"Common usable duration: {common_duration:.3f} s")
print(f"Window length: {WINDOW_S:.0f} s")
print(f"Window step: {STEP_S:.0f} s")
print(f"Windows created: {len(rows)}")
print(f"Output: {output_csv}")
print("Labels remain unlabelled; no continuous anxiety labels were inferred.")
