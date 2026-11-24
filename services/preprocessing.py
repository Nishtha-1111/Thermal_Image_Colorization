"""
services/preprocessing.py
All image preprocessing matching the training pipeline exactly.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataset import compute_edge_map


CANNY_LOW  = 50
CANNY_HIGH = 150
DILATE_ITERS = 1


def read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def uploaded_to_bgr(uploaded_file) -> np.ndarray:
    import numpy as np
    data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, h: int = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=h,
                                     templateWindowSize=7,
                                     searchWindowSize=21)


def apply_clahe(gray: np.ndarray, clip: float = 2.0,
                grid: tuple = (8, 8)) -> np.ndarray:
    clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe_obj.apply(gray)


def extract_edges(gray: np.ndarray,
                  low: int = CANNY_LOW,
                  high: int = CANNY_HIGH,
                  dilate_iters: int = DILATE_ITERS) -> np.ndarray:
    """
    Reproduce compute_edge_map from src/dataset.py:
      Gaussian blur → Canny → dilate with ellipse kernel.
    Returns float32 [0, 1].
    """
    return compute_edge_map(gray)   # delegates to canonical implementation


def extract_edges_visual(gray: np.ndarray,
                         low: int = CANNY_LOW,
                         high: int = CANNY_HIGH) -> dict:
    """
    Returns intermediate visual stages for the UI.
    """
    blurred  = cv2.GaussianBlur(gray, (3, 3), 0)
    raw_canny = cv2.Canny(blurred, low, high)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated  = cv2.dilate(raw_canny, kernel, iterations=DILATE_ITERS)
    soft     = dilated.astype(np.float32) / 255.0

    # Overlay: highlight edges in cyan on gray
    overlay  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[dilated > 0] = [255, 180, 0]   # orange-cyan overlay

    return {
        "blurred":    blurred,
        "raw_canny":  raw_canny,
        "dilated":    dilated,
        "soft":       soft,
        "overlay":    overlay,
        "density_pct": round(100.0 * (dilated > 0).mean(), 2),
    }


def preprocess_for_model(bgr: np.ndarray,
                          size: int = 256,
                          denoise_h: int = 10,
                          clahe_clip: float = 2.0) -> dict:
    """
    Full preprocessing pipeline → returns tensors + all intermediate images.
    """
    import torch

    gray_full  = to_grayscale(bgr)
    gray_rs    = cv2.resize(gray_full, (size, size), interpolation=cv2.INTER_AREA)
    denoised   = denoise(gray_rs, h=denoise_h)
    clahe_out  = apply_clahe(denoised, clip=clahe_clip)
    edge_info  = extract_edges_visual(clahe_out)
    soft_edge  = extract_edges(clahe_out)   # canonical float [0,1]

    # Normalize to [-1, 1] matching training
    ir_t   = torch.from_numpy(gray_rs.astype(np.float32) / 127.5 - 1.0) \
                   .unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    edge_t = torch.from_numpy(soft_edge) \
                   .unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

    return {
        "gray_full":  gray_full,
        "gray_rs":    gray_rs,
        "denoised":   denoised,
        "clahe":      clahe_out,
        "edge_info":  edge_info,
        "soft_edge":  soft_edge,
        "ir_tensor":  ir_t,
        "edge_tensor": edge_t,
        "size":       size,
    }


# ── Color utilities ─────────────────────────────────────────────
def apply_colormap(gray: np.ndarray,
                   cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    return cv2.applyColorMap(gray, cmap)


def bgr_to_pil(bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def gray_to_pil(gray: np.ndarray):
    from PIL import Image
    if gray.dtype != np.uint8:
        gray = (gray * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(gray)


def pil_to_bgr(pil_img) -> np.ndarray:
    import numpy as np
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
