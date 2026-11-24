"""
src/evaluate_models.py – Benchmark evaluation on LLVIP test set (N=3,463) for E1, E2, E3, and E4.

Usage:
  python src/evaluate_models.py
  python src/evaluate_models.py --save_visuals --num_visuals 20
  python src/evaluate_models.py --exp E2 --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LLVIPDataset, compute_edge_map
from models import UNetGenerator
from gated_models import GatedUNetGenerator


def compute_edge_f1(pred_gray: np.ndarray, gt_gray: np.ndarray) -> float:
    def get_binary_edges(img):
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return dilated > 0

    p_e = get_binary_edges(pred_gray)
    g_e = get_binary_edges(gt_gray)

    tp = int((p_e & g_e).sum())
    fp = int((p_e & ~g_e).sum())
    fn = int((~p_e & g_e).sum())

    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return float(2.0 * prec * rec / (prec + rec + 1e-8))


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().float().numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_model(exp_id: str, ckpt_path: str, device: torch.device):
    if exp_id in ["E3", "E4"]:
        model = GatedUNetGenerator(in_channels=2, out_channels=3, ngf=64).to(device)
    else:
        model = UNetGenerator(in_channels=2, out_channels=3, ngf=64).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {exp_id}: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["netG"] if "netG" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_single_model(model, test_ds, device, warmup=50):
    psnr_list, ssim_list, f1_list, latencies = [], [], [], []

    # Warmup for timing
    dummy = torch.randn(1, 2, 256, 256, device=device)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        for i, sample in enumerate(test_ds):
            ir_t = sample["ir"].unsqueeze(0).to(device)
            edge_t = sample["edge"].unsqueeze(0).to(device)
            vis_t = sample["vis"]  # (3, 256, 256)
            inp = torch.cat([ir_t, edge_t], dim=1)

            t0 = time.perf_counter()
            pred = model(inp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(t_ms)

            # Evaluate metrics on float64 grayscale
            pred_bgr = tensor_to_bgr(pred[0])
            gt_bgr = tensor_to_bgr(vis_t)

            pred_gray = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY)
            gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)

            # PSNR
            mse = np.mean((pred_gray.astype(np.float64) - gt_gray.astype(np.float64)) ** 2)
            psnr_v = 100.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
            # SSIM
            ssim_v = float(ssim_fn(pred_gray, gt_gray, data_range=255))
            # Edge F1
            f1_v = compute_edge_f1(pred_gray, gt_gray)

            psnr_list.append(psnr_v)
            ssim_list.append(ssim_v)
            f1_list.append(f1_v)

            if (i + 1) % 500 == 0:
                print(f"    Evaluated {i+1}/{len(test_ds)} images...")

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "edge_f1": float(np.mean(f1_list)),
        "latency_ms": float(np.mean(latencies[warmup:])),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Colorization Models on LLVIP")
    parser.add_argument("--save_visuals", action="store_true", default=True)
    parser.add_argument("--num_visuals", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running LLVIP evaluation on device: {device}")

    test_ds = LLVIPDataset(csv_file="datasets/llvip_pairs.csv", split="test", image_size=(256, 256))
    print(f"Test samples: {len(test_ds)}")

    experiments = [
        {"id": "E1", "name": "Baseline (No Sobel)", "ckpt": "experiments/E1_baseline/checkpoints/best_model.pth"},
        {"id": "E2", "name": "Original + Sobel", "ckpt": "experiments/E2_sobel/checkpoints/best_model.pth"},
        {"id": "E3", "name": "Gated U-Net (No Sobel)", "ckpt": "experiments/E3_gated/checkpoints/best_model.pth"},
        {"id": "E4", "name": "Gated U-Net + Sobel", "ckpt": "experiments/E4_proposed/checkpoints/best_model.pth"},
    ]

    # Check fallback for E2 if baseline best_model.pth is available
    if not os.path.exists(experiments[1]["ckpt"]) and os.path.exists("checkpoints/best_model.pth"):
        experiments[1]["ckpt"] = "checkpoints/best_model.pth"

    results = {}
    loaded_models = {}

    for exp in experiments:
        exp_id = exp["id"]
        ckpt = exp["ckpt"]
        print(f"\n[{exp_id}] {exp['name']}")
        if not os.path.exists(ckpt):
            print(f"  --> Checkpoint not found: {ckpt} (Skipping until trained)")
            continue

        model = load_model(exp_id, ckpt, device)
        loaded_models[exp_id] = model
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        metrics = evaluate_single_model(model, test_ds, device)
        metrics["params_m"] = round(params_m, 2)
        results[exp_id] = metrics

        print(f"  PSNR: {metrics['psnr']:.4f} dB | SSIM: {metrics['ssim']:.4f} | "
              f"Edge F1: {metrics['edge_f1']:.4f} | Latency: {metrics['latency_ms']:.2f} ms | Params: {metrics['params_m']} M")

    # Save visual comparison panels
    if args.save_visuals and len(loaded_models) > 0:
        vis_dir = Path("results/visual_comparisons")
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating {args.num_visuals} visual comparison panels -> {vis_dir}")

        for idx in range(min(args.num_visuals, len(test_ds))):
            sample = test_ds[idx]
            stem = sample["filename"]
            ir_t = sample["ir"].unsqueeze(0).to(device)
            edge_t = sample["edge"].unsqueeze(0).to(device)
            vis_t = sample["vis"]
            inp = torch.cat([ir_t, edge_t], dim=1)

            ir_bgr = cv2.cvtColor(((ir_t[0, 0].cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_bgr = tensor_to_bgr(vis_t)

            panels = [ir_bgr, gt_bgr]
            for exp_id in ["E1", "E2", "E3", "E4"]:
                if exp_id in loaded_models:
                    with torch.no_grad():
                        pred = loaded_models[exp_id](inp)
                    panels.append(tensor_to_bgr(pred[0]))
                else:
                    panels.append(np.zeros((256, 256, 3), dtype=np.uint8))

            grid = np.hstack(panels)
            cv2.imwrite(str(vis_dir / f"{stem}_comparison.jpg"), grid)

    # Save Summary Table
    out_json = Path("results/llvip_evaluation_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Table 1: LLVIP Test Results Summary")
    print("=" * 70)
    print(f"{'Model':<8} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'Edge F1 ↑':<10} | {'Latency (ms) ↓':<15} | {'Params (M)':<10}")
    print("-" * 75)
    for exp_id, m in results.items():
        print(f"{exp_id:<8} | {m['psnr']:<12.4f} | {m['ssim']:<10.4f} | {m['edge_f1']:<10.4f} | {m['latency_ms']:<15.2f} | {m['params_m']:<10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
