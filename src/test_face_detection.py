from pathlib import Path
import cv2
import csv

frames_dir = Path("data/ubfc_phys_subset/s1/frames")
output_csv = frames_dir / "face_detection_results.csv"

image_paths = sorted(frames_dir.glob("frame_*.jpg"))

if not image_paths:
    raise SystemExit(f"No frames found in {frames_dir}")

# OpenCV's bundled Haar cascade face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

if detector.empty():
    raise SystemExit("Could not load the Haar cascade detector.")

results = []

for path in image_paths:
    image = cv2.imread(str(path))

    if image is None:
        results.append([path.name, "unreadable", 0, "", "", "", ""])
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    # Store the largest detected face, if any
    if len(faces):
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        results.append([path.name, "detected", len(faces), x, y, w, h])
    else:
        results.append([path.name, "not_detected", 0, "", "", "", ""])

with output_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame", "status", "faces_detected",
        "x", "y", "width", "height"
    ])
    writer.writerows(results)

detected = sum(row[1] == "detected" for row in results)

print(f"Frames checked: {len(results)}")
print(f"Frames with a detected face: {detected}")
print(f"Frames without a detected face: {len(results) - detected}")
print(f"Results saved to: {output_csv}")
