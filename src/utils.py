"""
utils.py – Shared utilities for LLVIP pipeline.
"""

import os
import random
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Image I/O
# ============================================================

def read_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def tensor_to_numpy(t, denorm=True):
    """(C,H,W) tensor [-1,1] → uint8 numpy (H,W,C) RGB."""
    img = t.detach().cpu().float().numpy()
    if denorm:
        img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    return img


# ============================================================
# Visualization
# ============================================================

def make_grid(images: list, titles: list = None, cols=4, figsize_per=3):
    """
    Create a matplotlib figure grid.
    images: list of numpy uint8 arrays (H,W,3) or (H,W)
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(figsize_per * cols, figsize_per * rows))
    axes = np.array(axes).flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


# ============================================================
# Checkpoint helpers
# ============================================================

def latest_checkpoint(ckpt_dir):
    """Return path to the most recent checkpoint, or None."""
    import glob
    files = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.pth")))
    return files[-1] if files else None


# ============================================================
# CUDA / device info
# ============================================================

def device_info():
    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        return (f"CUDA device: {d.name}  "
                f"({d.total_memory // 1024**2} MB VRAM)")
    return "CPU only"


if __name__ == "__main__":
    print(device_info())
