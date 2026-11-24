
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

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

print("IMAGE | BASELINE | NO DILATION | HIGH THRESHOLDS")

for path in files[:100]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    img = cv2.resize(img, (256, 256))
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    baseline = cv2.Canny(blurred, 50, 150)
    baseline = cv2.dilate(baseline, kernel, iterations=1)

    no_dilation = cv2.Canny(blurred, 50, 150)

    high_threshold = cv2.Canny(blurred, 80, 180)

    d1 = np.mean(baseline > 0) * 100
    d2 = np.mean(no_dilation > 0) * 100
    d3 = np.mean(high_threshold > 0) * 100

    print(
        os.path.basename(path),
        f"{d1:.2f}%",
        f"{d2:.2f}%",
        f"{d3:.2f}%"
    )
