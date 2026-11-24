"""
enhancement.py – Traditional IR enhancement pipeline.

Steps:
  1. CLAHE for local contrast enhancement
  2. Fast Non-Local Means denoising
  3. Pseudo-colorization (colormap-based OR histogram-transfer)
  4. Edge overlay for visualization

Usage as a library or run standalone:
  python src/enhancement.py --input path/to/ir.jpg --output results/traditional/
"""

import os
import argparse
import glob

import cv2
import numpy as np


# ============================================================
# Core enhancement functions
# ============================================================

def apply_clahe(gray: np.ndarray, clip_limit=2.0, tile_grid=(8, 8)) -> np.ndarray:
    """Apply CLAHE to a uint8 grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def denoise_nlm(gray: np.ndarray, h=10) -> np.ndarray:
    """Fast Non-Local Means denoising for a grayscale image."""
    return cv2.fastNlMeansDenoising(gray, None, h=h,
                                     templateWindowSize=7, searchWindowSize=21)


def pseudo_colorize(gray: np.ndarray, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """Apply OpenCV colormap to a uint8 grayscale image → BGR."""
    return cv2.applyColorMap(gray, colormap)


def histogram_color_transfer(source_gray: np.ndarray,
                              reference_bgr: np.ndarray) -> np.ndarray:
    """
    Transfer the color distribution from a reference visible image to the
    source grayscale IR image using histogram matching in LAB space.

    source_gray:   (H, W)    uint8 grayscale
    reference_bgr: (H, W, 3) uint8 BGR visible image
    Returns:       (H, W, 3) uint8 BGR colorized image
    """
    # Convert reference to LAB
    ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    # Build the output LAB image: L from source, A/B from reference mean+std
    source_norm = source_gray.astype(np.float32)
    source_norm = (source_norm - source_norm.mean()) / (source_norm.std() + 1e-5)

    out_lab = np.zeros((*source_gray.shape, 3), dtype=np.float32)
    # L channel: scale to LAB range [0,100]
    out_lab[:, :, 0] = source_norm * ref_lab[:, :, 0].std() + ref_lab[:, :, 0].mean()
    # A, B channels: replicate reference mean (simple global transfer)
    out_lab[:, :, 1] = ref_lab[:, :, 1].mean()
    out_lab[:, :, 2] = ref_lab[:, :, 2].mean()

    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_Lab2BGR)


def overlay_edges(bgr: np.ndarray, gray: np.ndarray,
                  low_thresh=50, high_thresh=150,
                  edge_color=(0, 255, 255), alpha=0.5) -> np.ndarray:
    """
    Overlay Canny edges on a BGR colorized image.
    edge_color: BGR tuple, alpha: blend weight for edges.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    result = bgr.copy()
    mask = edges > 0
    result[mask] = (1 - alpha) * result[mask] + alpha * np.array(edge_color)
    return result.astype(np.uint8)


# ============================================================
# Full pipeline
# ============================================================

def enhance_ir(ir_path: str,
               clahe_clip=2.0, clahe_grid=(8, 8),
               denoise_h=10,
               colormap=cv2.COLORMAP_INFERNO,
               add_edge_overlay=True,
               reference_bgr=None) -> dict:
    """
    Full enhancement pipeline for a single IR image.
    Returns a dict of intermediate and final results (all BGR or gray uint8).
    """
    bgr = cv2.imread(ir_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read {ir_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Step 1: denoise
    denoised = denoise_nlm(gray, h=denoise_h)

    # Step 2: CLAHE
    clahe_out = apply_clahe(denoised, clip_limit=clahe_clip,
                             tile_grid=clahe_grid)

    # Step 3: pseudo-colorization
    colorized = pseudo_colorize(clahe_out, colormap=colormap)

    # Optional: histogram color transfer from a reference visible image
    if reference_bgr is not None:
        colorized = histogram_color_transfer(clahe_out, reference_bgr)

    result = {
        "gray_original":  gray,
        "denoised":       denoised,
        "clahe":          clahe_out,
        "colorized":      colorized,
    }

    # Step 4: edge overlay
    if add_edge_overlay:
        result["edge_overlay"] = overlay_edges(colorized, clahe_out)

    return result


# ============================================================
# Batch processing
# ============================================================

def batch_enhance(ir_dir: str, out_dir: str,
                  clahe_clip=2.0, denoise_h=10, max_samples=None):
    """Process all IR images in ir_dir and save to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ir_dir, "*.jpg")) +
                   glob.glob(os.path.join(ir_dir, "*.png")))
    if max_samples:
        files = files[:max_samples]

    print(f"[Enhancement] Processing {len(files)} images → {out_dir}")
    for i, f in enumerate(files):
        try:
            res = enhance_ir(f, clahe_clip=clahe_clip, denoise_h=denoise_h)
            stem = os.path.splitext(os.path.basename(f))[0]
            cv2.imwrite(os.path.join(out_dir, f"{stem}_enhanced.jpg"), res["colorized"])
            cv2.imwrite(os.path.join(out_dir, f"{stem}_edge.jpg"), res.get("edge_overlay", res["colorized"]))
        except Exception as e:
            print(f"  [WARN] {f}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} done")

    print("[Enhancement] Done.")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR Enhancement Pipeline")
    parser.add_argument("--input",   required=True,
                        help="Path to single IR image OR directory of IR images")
    parser.add_argument("--output",  default="results/traditional",
                        help="Output directory")
    parser.add_argument("--clahe_clip", type=float, default=2.0)
    parser.add_argument("--denoise_h",  type=float, default=10.0)
    parser.add_argument("--max",  type=int, default=None,
                        help="Max images to process (for quick tests)")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_enhance(args.input, args.output,
                      clahe_clip=args.clahe_clip,
                      denoise_h=args.denoise_h,
                      max_samples=args.max)
    else:
        res = enhance_ir(args.input,
                         clahe_clip=args.clahe_clip,
                         denoise_h=args.denoise_h)
        os.makedirs(args.output, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        for k, v in res.items():
            cv2.imwrite(os.path.join(args.output, f"{stem}_{k}.jpg"), v)
        print(f"Saved to {args.output}")
