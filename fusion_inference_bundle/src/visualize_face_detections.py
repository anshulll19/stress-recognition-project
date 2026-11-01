from pathlib import Path
import csv
import cv2
import math

frames_dir = Path("data/ubfc_phys_subset/s1/frames")
csv_path = frames_dir / "face_detection_results.csv"
output_path = frames_dir / "face_detection_contact_sheet.jpg"

with csv_path.open(newline="") as f:
    results = list(csv.DictReader(f))

tiles = []
for row in results:
    image_path = frames_dir / row["frame"]
    image = cv2.imread(str(image_path))
    if image is None:
        continue

    if row["status"] == "detected":
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    image = cv2.resize(image, (240, 240))
    cv2.putText(
        image, row["frame"], (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
    )
    tiles.append(image)

if not tiles:
    raise SystemExit("No readable frames found.")

columns = 3
rows = math.ceil(len(tiles) / columns)
sheet = 255 * __import__("numpy").ones(
    (rows * 260, columns * 240, 3), dtype="uint8"
)

for i, tile in enumerate(tiles):
    y = (i // columns) * 260
    x = (i % columns) * 240
    sheet[y:y + 240, x:x + 240] = tile

cv2.imwrite(str(output_path), sheet)
print(f"Contact sheet: {output_path}")
print(f"Frames shown: {len(tiles)}")
