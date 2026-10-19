from pathlib import Path
import csv
import json
import subprocess
import numpy as np

root = Path("data/ubfc_phys_subset/s1")

def video_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])

video = root / "vid_s1_T1.avi"
bvp = np.loadtxt(root / "bvp_s1_T1.csv", delimiter=",")
eda = np.loadtxt(root / "eda_s1_T1.csv", delimiter=",")

# Dataset documentation gives BVP=64 Hz and EDA=4 Hz.
bvp_fs = 64.0
eda_fs = 4.0

video_s = video_duration(video)
bvp_s = len(bvp) / bvp_fs
eda_s = len(eda) / eda_fs

print("T1 RECORDING DURATION CHECK")
print(f"Video: {video_s:.3f} s")
print(f"BVP:   {bvp_s:.3f} s ({len(bvp)} samples at {bvp_fs:g} Hz)")
print(f"EDA:   {eda_s:.3f} s ({len(eda)} samples at {eda_fs:g} Hz)")
print()
print(f"Video - BVP: {video_s - bvp_s:+.3f} s")
print(f"Video - EDA: {video_s - eda_s:+.3f} s")
print()
print("IMPORTANT: Similar durations do not prove exact timestamp alignment.")
