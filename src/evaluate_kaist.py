"""
src/evaluate_kaist.py – Zero-Shot Cross-Dataset Evaluation on KAIST Multispectral Dataset.

Rules for scientific rigor:
  - DO NOT train on KAIST.
  - DO NOT fine-tune on KAIST.
  - DO NOT select checkpoints based on KAIST results.
  - Evaluates models trained strictly on LLVIP.

Usage:
  python src/evaluate_kaist.py --kaist_dir path/to/KAIST
"""

import os
import sys
import time
import json
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).parent))
from dataset import compute_edge_map
from evaluate_models import load_model, compute_edge_f1, tensor_to_bgr


def main():
    parser = argparse.ArgumentParser(description="Zero-shot KAIST Evaluation")
    parser.add_argument("--kaist_dir", type=str, default="KAIST", help="Root folder of KAIST dataset")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max test images to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("ZERO-SHOT CROSS-DATASET EVALUATION ON KAIST")
    print(f"Device: {device} | KAIST path: {args.kaist_dir}")
    print("=" * 70)

    kaist_path = Path(args.kaist_dir)
    if not kaist_path.exists():
        print(f"\n[NOTE] KAIST dataset directory '{args.kaist_dir}' is not found yet.")
        print("When KAIST is downloaded, place it in 'KAIST/' or specify with --kaist_dir <path>.")
        print("Expected structure inside KAIST/:")
        print("  KAIST/")
        print("    ├── infrared/ (or thermal/)")
        print("    └── visible/")
        return

    # Scan for paired files
    ir_candidates = sorted(glob.glob(str(kaist_path / "**" / "*lwir*.*"), recursive=True) +
                           glob.glob(str(kaist_path / "**" / "*thermal*.*"), recursive=True) +
                           glob.glob(str(kaist_path / "**" / "*ir*.*"), recursive=True))

    print(f"Found {len(ir_candidates)} potential infrared files in KAIST.")

    experiments = [
        {"id": "E1", "name": "Baseline (No Sobel)", "ckpt": "experiments/E1_baseline/checkpoints/best_model.pth"},
        {"id": "E2", "name": "Original + Sobel", "ckpt": "experiments/E2_sobel/checkpoints/best_model.pth"},
        {"id": "E3", "name": "Gated U-Net (No Sobel)", "ckpt": "experiments/E3_gated/checkpoints/best_model.pth"},
        {"id": "E4", "name": "Gated U-Net + Sobel", "ckpt": "experiments/E4_proposed/checkpoints/best_model.pth"},
    ]

    # Fallback for E2 if baseline best_model.pth exists
    if not os.path.exists(experiments[1]["ckpt"]) and os.path.exists("checkpoints/best_model.pth"):
        experiments[1]["ckpt"] = "checkpoints/best_model.pth"

    results = {}
    for exp in experiments:
        exp_id = exp["id"]
        ckpt = exp["ckpt"]
        if not os.path.exists(ckpt):
            print(f"[{exp_id}] Checkpoint missing: {ckpt}")
            continue

        model = load_model(exp_id, ckpt, device)
        print(f"[{exp_id}] Ready for zero-shot evaluation on KAIST.")

    out_json = Path("results/kaist_zero_shot_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
