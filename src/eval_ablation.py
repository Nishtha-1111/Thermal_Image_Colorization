"""
src/eval_ablation.py
Ablation evaluation script:
  1. Runs batch inference on LLVIP test set for a given experiment checkpoint.
  2. Computes PSNR, SSIM, and Canny Edge F1 matching Phase 2 evaluation exactly.
  3. Saves outputs to experiments/<exp>/results/test_gen and summary_metrics.json.

Usage:
  python -m src.eval_ablation --config experiments/A_baseline/config.yaml --checkpoint experiments/A_baseline/checkpoints/best_model.pth
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path

import yaml
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LLVIPDataset, compute_edge_map
from models import UNetGenerator


def tensor_to_bgr(t):
    img = t.cpu().numpy()
    img = ((img + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def evaluate_experiment(cfg_path: str, checkpoint_path: str, max_samples: int = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment", {}).get("name", "exp")
    use_edge_input = cfg.get("experiment", {}).get("use_edge_input", False)
    in_channels_G = cfg.get("model", {}).get("in_channels_G", 2 if use_edge_input else 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating Ablation Experiment: {exp_name} ===")
    print(f"Device: {device} | in_channels_G: {in_channels_G} | use_edge: {use_edge_input}")

    # Load generator
    netG = UNetGenerator(
        in_channels=in_channels_G,
        out_channels=cfg.get("model", {}).get("out_channels_G", 3),
        ngf=cfg.get("model", {}).get("ngf", 64),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["netG"] if "netG" in state else state
    netG.load_state_dict(state_dict)
    netG.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Results dir
    result_dir = cfg.get("output", {}).get("result_dir", f"experiments/{exp_name}/results")
    out_dir = os.path.join(result_dir, "test_gen")
    os.makedirs(out_dir, exist_ok=True)

    img_size = cfg["dataset"]["image_size"]
    img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)

    ds = LLVIPDataset(
        ir_dir=cfg["dataset"]["test_ir"],
        vis_dir=cfg["dataset"]["test_vis"],
        image_size=img_size,
        augment=False,
        max_samples=max_samples,
    )

    print(f"Running inference on {len(ds)} test samples -> {out_dir}")
    gt_map = {}
    test_vis_dir = Path(cfg["dataset"]["test_vis"])
    for f in test_vis_dir.glob("*.jpg"):
        gt_map[f.stem] = f

    psnr_vals, ssim_vals, f1_vals = [], [], []

    def edges(x):
        b = cv2.GaussianBlur(x, (3, 3), 0)
        e = cv2.Canny(b, 50, 150)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.dilate(e, k, iterations=1) > 0

    with torch.no_grad():
        for i, sample in enumerate(ds):
            ir = sample["ir"].unsqueeze(0).to(device)
            edge = sample["edge"].unsqueeze(0).to(device)

            if use_edge_input:
                gen_input = torch.cat([ir, edge], dim=1)
            else:
                gen_input = ir

            fake = netG(gen_input)[0]
            pred_bgr = tensor_to_bgr(fake)
            fname = sample["filename"]
            out_file = os.path.join(out_dir, f"{fname}.jpg")
            cv2.imwrite(out_file, pred_bgr)

            # Compute metrics against GT
            if fname in gt_map:
                gt_bgr = cv2.imread(str(gt_map[fname]))
                if gt_bgr is not None:
                    pg = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY)
                    gg = cv2.cvtColor(cv2.resize(gt_bgr, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

                    mse = np.mean((pg.astype(np.float64) - gg.astype(np.float64)) ** 2)
                    p_val = 100.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
                    s_val = float(ssim_fn(pg, gg, data_range=255))

                    pe, ge_ = edges(pg), edges(gg)
                    tp = int((pe & ge_).sum()); fp = int((pe & ~ge_).sum()); fn = int((~pe & ge_).sum())
                    pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8)
                    f_val = float(2 * pr * rc / (pr + rc + 1e-8))

                    psnr_vals.append(p_val)
                    ssim_vals.append(s_val)
                    f1_vals.append(f_val)

            if (i + 1) % 500 == 0 or (i + 1) == len(ds):
                print(f"  [{i+1}/{len(ds)}] PSNR: {np.mean(psnr_vals):.2f} | SSIM: {np.mean(ssim_vals):.4f} | F1: {np.mean(f1_vals):.4f}")

    n = len(psnr_vals)
    summary = {
        "experiment": exp_name,
        "n": n,
        "psnr": round(float(np.mean(psnr_vals)) if n else 0.0, 4),
        "ssim": round(float(np.mean(ssim_vals)) if n else 0.0, 4),
        "edge_f1": round(float(np.mean(f1_vals)) if n else 0.0, 4),
    }

    summary_file = os.path.join(result_dir, "summary_metrics.json")
    with open(summary_file, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 50)
    print(f"RESULTS FOR {exp_name}:")
    print(f"  PSNR:    {summary['psnr']} dB")
    print(f"  SSIM:    {summary['ssim']}")
    print(f"  Edge F1: {summary['edge_f1']}")
    print(f"Saved summary to {summary_file}")
    print("=" * 50)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ablation Model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    evaluate_experiment(args.config, args.checkpoint, args.max_samples)
