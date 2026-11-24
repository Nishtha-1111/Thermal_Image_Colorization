
import cv2
import numpy as np
import glob
import os
import yaml

# Read dataset path from existing configuration
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ir_dir = cfg["dataset"]["train_ir"]

files = sorted(
    glob.glob(os.path.join(ir_dir, "*.jpg")) +
    glob.glob(os.path.join(ir_dir, "*.png"))
)

if not files:
    raise FileNotFoundError(f"No images found in: {ir_dir}")

densities = []

for path in files[:100]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (256, 256))

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )

    edges = cv2.dilate(edges, kernel, iterations=1)

    density = np.mean(edges > 0) * 100
    densities.append(density)

print("\n--- EDGE DENSITY REPORT ---")
print("Directory:", ir_dir)
print("Images checked:", len(densities))
print("Average:", round(np.mean(densities), 2), "%")
print("Minimum:", round(np.min(densities), 2), "%")
print("Maximum:", round(np.max(densities), 2), "%")
