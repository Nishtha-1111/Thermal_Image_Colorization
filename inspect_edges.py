
import cv2
import numpy as np
import glob
import os
import yaml

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ir_dir = cfg["dataset"]["train_ir"]

files = sorted(
    glob.glob(os.path.join(ir_dir, "*.jpg")) +
    glob.glob(os.path.join(ir_dir, "*.png"))
)

if not files:
    raise FileNotFoundError(ir_dir)

# Select 10 images spread across the dataset
indices = np.linspace(
    0, len(files) - 1, min(10, len(files)), dtype=int
)

rows = []

for idx in indices:
    img = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (256, 256))

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    dilated = cv2.dilate(canny, kernel, iterations=1)

    density_before = np.mean(canny > 0) * 100
    density_after = np.mean(dilated > 0) * 100

    print(
        os.path.basename(files[idx]),
        f"Before: {density_before:.2f}%",
        f"After: {density_after:.2f}%"
    )

    panels = []

    for image, label in [
        (img, "IR INPUT"),
        (canny, "CANNY"),
        (dilated, "AFTER DILATION")
    ]:
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            panel, label, (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1
        )
        panels.append(panel)

    rows.append(cv2.hconcat(panels))

if rows:
    output = cv2.vconcat(rows)
    cv2.imwrite("edge_inspection.jpg", output)
    print("\nSaved: edge_inspection.jpg")
