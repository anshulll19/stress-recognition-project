from pathlib import Path
import csv
import cv2
import numpy as np

frames_dir = Path("data/ubfc_phys_subset/s1/frames")
output_csv = frames_dir / "visual_quality_features.csv"

rows = []

for path in sorted(frames_dir.glob("frame_*.jpg")):
    image = cv2.imread(str(path))

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rows.append({
        "frame": path.name,
        "timestamp_s": (int(path.stem.split("_")[-1]) - 1) * 10.0,
        "mean_intensity": float(gray.mean()),
        "intensity_std": float(gray.std()),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "width": image.shape[1],
        "height": image.shape[0],
    })

if not rows:
    raise SystemExit("No readable frames found.")

with output_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Frames processed: {len(rows)}")
print(f"Output: {output_csv}")

for key in ("mean_intensity", "intensity_std", "laplacian_variance"):
    values = np.array([r[key] for r in rows])
    print(f"{key}: min={values.min():.3f}, max={values.max():.3f}, std={values.std():.3f}")
