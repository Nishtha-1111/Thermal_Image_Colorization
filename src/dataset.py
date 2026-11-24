"""
dataset.py – LLVIP paired IR↔Visible dataset loader with edge maps, VOC annotation support,
and standardized preprocessing for training, validation, and inference.
"""

import os
import csv
import glob
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# VOC annotation helper
# ---------------------------------------------------------------------------

def parse_voc_annotation(xml_path):
    """Return list of (xmin, ymin, xmax, ymax) boxes from VOC XML."""
    if not os.path.exists(xml_path):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.iter("object"):
        bbox = obj.find("bndbox")
        boxes.append((
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ))
    return boxes


# ---------------------------------------------------------------------------
# Standardized Preprocessing Pipeline (Canonical for Training & Inference)
# ---------------------------------------------------------------------------

def compute_edge_map(gray_np: np.ndarray) -> np.ndarray:
    """
    Canny Edge Pipeline:
      1. Input: Resized thermal grayscale image (uint8).
      2. Gaussian Blur (3 x 3, sigma=0).
      3. Canny edge detection (low=50, high=150).
      4. Morphological dilation with elliptical structuring element (3 x 3).
    Returns float32 edge map in [0, 1].
    """
    blurred = cv2.GaussianBlur(gray_np, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150).astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def preprocess_thermal(ir_input, image_size=(256, 256)):
    """
    Thermal Input Pipeline:
      1. Load / accept thermal image.
      2. Convert to grayscale uint8.
      3. Resize to 256 x 256 (cv2.INTER_AREA).
      4. Normalize to [-1, 1] tensor (1, H, W).
    Returns:
      gray_256 (np.ndarray uint8 [256, 256]),
      ir_tensor (torch.Tensor float32 [1, 256, 256] in [-1, 1])
    """
    if isinstance(ir_input, str):
        bgr = cv2.imread(ir_input)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read thermal image: {ir_input}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif isinstance(ir_input, np.ndarray):
        if ir_input.ndim == 3:
            gray = cv2.cvtColor(ir_input, cv2.COLOR_BGR2GRAY)
        else:
            gray = ir_input.copy()
    else:
        raise TypeError(f"Unsupported thermal input type: {type(ir_input)}")

    h, w = image_size
    if gray.shape[:2] != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)

    ir_t = torch.from_numpy(gray.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
    return gray, ir_t


def preprocess_visible(vis_input, image_size=(256, 256)):
    """
    Visible RGB Target Pipeline:
      1. Load / accept visible image.
      2. Convert to RGB uint8.
      3. Resize to 256 x 256 (cv2.INTER_AREA).
      4. Normalize to [-1, 1] tensor (3, H, W).
    Returns:
      rgb_256 (np.ndarray uint8 [256, 256, 3]),
      vis_tensor (torch.Tensor float32 [3, 256, 256] in [-1, 1])
    """
    if isinstance(vis_input, str):
        bgr = cv2.imread(vis_input)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read visible image: {vis_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(vis_input, np.ndarray):
        if vis_input.ndim == 3 and vis_input.shape[2] == 3:
            rgb = cv2.cvtColor(vis_input, cv2.COLOR_BGR2RGB)
        else:
            rgb = vis_input.copy()
    else:
        raise TypeError(f"Unsupported visible input type: {type(vis_input)}")

    h, w = image_size
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    vis_t = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
    return rgb, vis_t


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class LLVIPDataset(Dataset):
    """
    Paired LLVIP Dataset Loader supporting both CSV-partitioned loading
    (train/validation/test) and legacy directory paths.

    Returns dicts with keys:
      ir          – (1, H, W) float32 tensor in [-1, 1]   (grayscale IR)
      vis         – (3, H, W) float32 tensor in [-1, 1]   (target visible RGB)
      edge        – (1, H, W) float32 tensor in [0, 1]    (Canny edge map of IR)
      filename    – stem of the image file
    """

    def __init__(self, ir_dir=None, vis_dir=None, ann_dir=None,
                 csv_file=None, split=None,
                 image_size=(256, 256), augment=False, max_samples=None):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.ann_dir = ann_dir
        self.image_size = image_size   # (H, W)
        self.augment = augment
        self.pairs = []

        # Mode A: Load from paired CSV split
        if csv_file is not None and os.path.exists(csv_file):
            valid_splits = {split.lower()} if split else {"train", "validation", "test"}
            with open(csv_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_split = row["split"].lower()
                    if split is None or row_split in valid_splits:
                        self.pairs.append((row["thermal_path"], row["visible_path"], row["filename"]))

        # Mode B: Fallback / Legacy directory scanning
        elif ir_dir is not None and vis_dir is not None:
            ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.jpg")) +
                              glob.glob(os.path.join(ir_dir, "*.png")))
            vis_stems = {os.path.splitext(os.path.basename(f))[0]
                         for f in glob.glob(os.path.join(vis_dir, "*.jpg")) +
                                  glob.glob(os.path.join(vis_dir, "*.png"))}

            for ir_path in ir_files:
                stem = os.path.splitext(os.path.basename(ir_path))[0]
                if stem in vis_stems:
                    for ext in (".jpg", ".png"):
                        vis_path = os.path.join(vis_dir, stem + ext)
                        if os.path.exists(vis_path):
                            self.pairs.append((ir_path, vis_path, stem))
                            break
        else:
            raise ValueError("Must provide either valid 'csv_file' or both 'ir_dir' and 'vis_dir'.")

        if max_samples:
            self.pairs = self.pairs[:max_samples]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ir_path, vis_path, stem = self.pairs[idx]

        # 1. Load & preprocess thermal IR
        ir_bgr = cv2.imread(ir_path)
        if ir_bgr is None:
            raise FileNotFoundError(f"Cannot read thermal image: {ir_path}")
        ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Load visible RGB
        vis_bgr = cv2.imread(vis_path)
        if vis_bgr is None:
            raise FileNotFoundError(f"Cannot read visible image: {vis_path}")
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        # 3. Resize to model input size (256 x 256)
        h, w = self.image_size
        ir_gray = cv2.resize(ir_gray, (w, h), interpolation=cv2.INTER_AREA)
        vis_rgb = cv2.resize(vis_rgb, (w, h), interpolation=cv2.INTER_AREA)

        # 4. Compute Canny edge map directly on resized thermal image
        edge = compute_edge_map(ir_gray)   # (H, W) float32 [0, 1]

        # 5. Data Augmentation: Random horizontal flip (applied synchronously)
        if self.augment and random.random() > 0.5:
            ir_gray = np.fliplr(ir_gray).copy()
            vis_rgb = np.fliplr(vis_rgb).copy()
            edge = np.fliplr(edge).copy()

        # 6. Normalize tensors: IR to [-1, 1], Vis to [-1, 1], Edge to [0, 1]
        ir_t = torch.from_numpy(ir_gray.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        vis_t = torch.from_numpy(vis_rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        edge_t = torch.from_numpy(edge).unsqueeze(0)

        return {
            "ir": ir_t,
            "vis": vis_t,
            "edge": edge_t,
            "filename": stem,
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_p = "datasets/llvip_pairs.csv"
    for sp in ["train", "validation", "test"]:
        ds = LLVIPDataset(csv_file=csv_p, split=sp)
        print(f"Split [{sp}]: {len(ds)} samples")
    sample = ds[0]
    print("Sample keys & shapes:")
    print({k: (v.shape if hasattr(v, 'shape') else v) for k, v in sample.items()})
