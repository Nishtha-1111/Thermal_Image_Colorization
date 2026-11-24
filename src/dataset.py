"""
dataset.py – LLVIP paired IR↔Visible dataset loader with edge maps and VOC annotation support.
"""

import os
import glob
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


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
# Edge-map helper (Canny on infrared)
# ---------------------------------------------------------------------------

def compute_edge_map(gray_np: np.ndarray) -> np.ndarray:
    """
    Compute a soft edge map from a uint8 single-channel image.
    Returns float32 in [0,1].
    """
    # Canny on slightly blurred image
    blurred = cv2.GaussianBlur(gray_np, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150).astype(np.float32) / 255.0
    # Dilate slightly for training signal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class LLVIPDataset(Dataset):
    """
    Returns dicts with keys:
      ir          – (1, H, W) float32 tensor in [-1,1]   (grayscale IR)
      vis         – (3, H, W) float32 tensor in [-1,1]   (target visible RGB)
      edge        – (1, H, W) float32 tensor in [0,1]    (Canny edge map of IR)
      filename    – stem of the image file
    """

    def __init__(self, ir_dir, vis_dir, ann_dir=None,
                 image_size=(256, 256), augment=False, max_samples=None):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.ann_dir = ann_dir
        self.image_size = image_size   # (H, W)
        self.augment = augment

        # Collect paired filenames
        ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.jpg")) +
                          glob.glob(os.path.join(ir_dir, "*.png")))
        vis_stems = {os.path.splitext(os.path.basename(f))[0]
                     for f in glob.glob(os.path.join(vis_dir, "*.jpg")) +
                              glob.glob(os.path.join(vis_dir, "*.png"))}

        self.pairs = []
        for ir_path in ir_files:
            stem = os.path.splitext(os.path.basename(ir_path))[0]
            if stem in vis_stems:
                # find visible file (jpg or png)
                for ext in (".jpg", ".png"):
                    vis_path = os.path.join(vis_dir, stem + ext)
                    if os.path.exists(vis_path):
                        self.pairs.append((ir_path, vis_path, stem))
                        break

        if max_samples:
            self.pairs = self.pairs[:max_samples]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ir_path, vis_path, stem = self.pairs[idx]

        # Load IR as grayscale uint8
        ir_bgr = cv2.imread(ir_path)
        if ir_bgr is None:
            raise FileNotFoundError(f"Cannot read {ir_path}")
        ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)   # (H, W)

        # Load visible as RGB
        vis_bgr = cv2.imread(vis_path)
        if vis_bgr is None:
            raise FileNotFoundError(f"Cannot read {vis_path}")
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)   # (H, W, 3)

        # Resize to model input size
        h, w = self.image_size
        ir_gray = cv2.resize(ir_gray, (w, h), interpolation=cv2.INTER_AREA)
        vis_rgb = cv2.resize(vis_rgb, (w, h), interpolation=cv2.INTER_AREA)

        # Compute edge map (before augmentation so edges match the IR)
        edge = compute_edge_map(ir_gray)   # (H, W) float32 [0,1]

        # Augmentation: random horizontal flip (must be applied consistently)
        if self.augment and random.random() > 0.5:
            ir_gray = np.fliplr(ir_gray).copy()
            vis_rgb = np.fliplr(vis_rgb).copy()
            edge = np.fliplr(edge).copy()

        # Convert to tensors in [-1,1] / [0,1]
        ir_t = torch.from_numpy(ir_gray.astype(np.float32) / 255.0)   # [0,1]
        ir_t = ir_t.unsqueeze(0)                                       # (1,H,W)
        ir_t = ir_t * 2.0 - 1.0                                       # [-1,1]

        vis_t = torch.from_numpy(vis_rgb.astype(np.float32) / 255.0)  # [0,1]
        vis_t = vis_t.permute(2, 0, 1)                                 # (3,H,W)
        vis_t = vis_t * 2.0 - 1.0                                     # [-1,1]

        edge_t = torch.from_numpy(edge).unsqueeze(0)                   # (1,H,W)

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
    import yaml
    cfg = yaml.safe_load(open("configs/config.yaml"))
    ds = LLVIPDataset(
        ir_dir=cfg["dataset"]["train_ir"],
        vis_dir=cfg["dataset"]["train_vis"],
        ann_dir=cfg["dataset"]["annotations"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        augment=True,
    )
    print(f"Train samples: {len(ds)}")
    sample = ds[0]
    print({k: (v.shape if hasattr(v, 'shape') else v) for k, v in sample.items()})
