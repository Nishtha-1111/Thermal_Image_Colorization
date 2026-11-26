"""
src/evaluate_kaist.py – Zero-Shot Cross-Dataset Evaluation on KAIST Multispectral Dataset.

Scientific constraints:
  - DO NOT train on KAIST.
  - DO NOT fine-tune on KAIST.
  - DO NOT select checkpoints based on KAIST results.
  - Evaluates models trained strictly on LLVIP.

Usage:
  python src/evaluate_kaist.py --kaist_dir KAIST
  python src/evaluate_kaist.py --kaist_dir KAIST --max_samples 500 --save_visuals
"""

import os
import sys
import time
import json
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).parent))
from dataset import compute_edge_map, preprocess_thermal, preprocess_visible
from evaluate_models import load_model, compute_edge_f1, tensor_to_bgr


def find_kaist_pairs(kaist_root: Path):
    """
    Robustly locate paired (LWIR/thermal, Visible) images from any KAIST structure:
    - Standard hierarchical: setXX/V00X/lwir/I00000.jpg <-> setXX/V00X/visible/I00000.jpg
    - Subdirectory layout: KAIST/lwir/ <-> KAIST/visible/
    - Flat naming: stem_lwir.jpg <-> stem_visible.jpg
    """
    pairs = []
    seen = set()

    # Strategy 1: Path keyword replacement (hierarchical layout)
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        for ir_path in kaist_root.rglob(ext):
            if ir_path.name.startswith("._") or "/._" in str(ir_path).replace("\\", "/"):
                continue
            p_str = str(ir_path)
            for tag in ["lwir", "thermal", "infrared", "ir"]:
                if f"/{tag}/" in p_str.replace("\\", "/") or f"\\{tag}\\" in p_str:
                    for vis_tag in ["visible", "vis", "rgb"]:
                        cand_str = p_str.replace(f"/{tag}/", f"/{vis_tag}/").replace(f"\\{tag}\\", f"\\{vis_tag}\\")
                        if cand_str != p_str and os.path.exists(cand_str) and not os.path.basename(cand_str).startswith("._") and p_str not in seen:
                            stem = ir_path.stem
                            pairs.append((p_str, cand_str, stem))
                            seen.add(p_str)
                            break
                    break

    # Strategy 2: Common stem matching between visible and lwir directories
    if not pairs:
        ir_files = {}
        vis_files = {}
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for f in kaist_root.rglob(ext):
                if f.name.startswith("._"):
                    continue
                p_lower = str(f).lower().replace("\\", "/")
                if any(k in p_lower for k in ["lwir", "thermal", "infrared"]):
                    ir_files[f.stem] = str(f)
                elif any(k in p_lower for k in ["visible", "vis", "rgb"]):
                    vis_files[f.stem] = str(f)

        common_stems = sorted(set(ir_files.keys()).intersection(vis_files.keys()))
        for s in common_stems:
            pairs.append((ir_files[s], vis_files[s], s))

    return sorted(pairs, key=lambda x: x[0])


class KAISTDataset(Dataset):
    def __init__(self, pairs, image_size=(256, 256)):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ir_path, vis_path, stem = self.pairs[idx]

        # 1. Thermal IR
        ir_bgr = cv2.imread(ir_path)
        if ir_bgr is None:
            raise FileNotFoundError(f"Failed to read IR: {ir_path}")
        ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)
        ir_resized = cv2.resize(ir_gray, self.image_size, interpolation=cv2.INTER_LINEAR)
        ir_tensor = torch.from_numpy(ir_resized.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)

        # 2. Canny Edge Map directly on resized thermal
        edge_map = compute_edge_map(ir_resized)
        edge_tensor = torch.from_numpy(edge_map.astype(np.float32)).unsqueeze(0)

        # 3. Ground Truth Visible RGB
        vis_bgr = cv2.imread(vis_path)
        if vis_bgr is None:
            raise FileNotFoundError(f"Failed to read Visible: {vis_path}")
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        vis_resized = cv2.resize(vis_rgb, self.image_size, interpolation=cv2.INTER_LINEAR)
        vis_tensor = torch.from_numpy(vis_resized.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)

        return {
            "ir": ir_tensor,
            "edge": edge_tensor,
            "vis": vis_tensor,
            "filename": stem,
            "ir_path": ir_path,
            "vis_path": vis_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot KAIST Cross-Dataset Evaluation")
    parser.add_argument("--kaist_dir", type=str, default="KAIST", help="Root folder of KAIST dataset")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max test images to evaluate")
    parser.add_argument("--save_visuals", action="store_true", default=True, help="Save visual comparison panels")
    parser.add_argument("--num_visuals", type=int, default=20, help="Number of visual panels to save")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75, flush=True)
    print("PHASE 4: ZERO-SHOT CROSS-DATASET EVALUATION ON KAIST", flush=True)
    print(f"Device: {device} | KAIST path: {args.kaist_dir}", flush=True)
    print("=" * 75, flush=True)

    kaist_path = Path(args.kaist_dir)
    if not kaist_path.exists():
        print(f"\n[NOTE] KAIST dataset directory '{args.kaist_dir}' does not exist yet.", flush=True)
        print("Please download KAIST and place it inside the workspace folder 'KAIST/'.", flush=True)
        print("Expected formats supported automatically:", flush=True)
        print("  1. Standard KAIST archive: KAIST/setXX/V00X/lwir/*.jpg and KAIST/setXX/V00X/visible/*.jpg")
        print("  2. Flat directory: KAIST/lwir/*.jpg and KAIST/visible/*.jpg")
        return

    pairs = find_kaist_pairs(kaist_path)
    if not pairs:
        print(f"[ERROR] No paired IR/Visible images found in '{args.kaist_dir}'.", flush=True)
        print("Please check that both thermal (lwir/thermal/ir) and visible (visible/vis) files are present.", flush=True)
        return

    if args.max_samples and len(pairs) > args.max_samples:
        print(f"Subsampling {args.max_samples} pairs from total {len(pairs)} KAIST pairs...", flush=True)
        # Uniform sampling across the dataset
        step = max(1, len(pairs) // args.max_samples)
        pairs = pairs[::step][:args.max_samples]

    print(f"Loaded {len(pairs)} verified KAIST image pairs for zero-shot testing.\n", flush=True)

    test_ds = KAISTDataset(pairs, image_size=(256, 256))

    experiments = [
        {"id": "E1", "name": "Baseline (No Sobel)", "ckpt": "experiments/E1_baseline/checkpoints/best_model.pth"},
        {"id": "E2", "name": "Original + Sobel", "ckpt": "experiments/E2_sobel/checkpoints/best_model.pth"},
        {"id": "E3", "name": "Gated U-Net (No Sobel)", "ckpt": "experiments/E3_gated/checkpoints/best_model.pth"},
        {"id": "E4", "name": "Gated U-Net + Sobel", "ckpt": "experiments/E4_proposed/checkpoints/best_model.pth"},
    ]

    # Fallback for E2 if baseline best_model.pth is available
    if not os.path.exists(experiments[1]["ckpt"]) and os.path.exists("checkpoints/best_model.pth"):
        experiments[1]["ckpt"] = "checkpoints/best_model.pth"

    results = {}
    loaded_models = {}

    for exp in experiments:
        exp_id = exp["id"]
        ckpt = exp["ckpt"]
        print(f"\n[{exp_id}] {exp['name']}", flush=True)
        if not os.path.exists(ckpt):
            print(f"  --> Checkpoint missing: {ckpt} (Skipping until trained)", flush=True)
            continue

        model = load_model(exp_id, ckpt, device)
        loaded_models[exp_id] = model
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        # Warmup and latency benchmark
        latencies = []
        dummy = torch.randn(1, 2, 256, 256, device=device)
        for _ in range(30):
            with torch.no_grad():
                _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for _ in range(50):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

        # Batch evaluation
        loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
        psnr_list, ssim_list, f1_list = [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                ir_t = batch["ir"].to(device)
                edge_t = batch["edge"].to(device)
                vis_t = batch["vis"]
                inp = torch.cat([ir_t, edge_t], dim=1)

                preds = model(inp).cpu()

                for b in range(preds.shape[0]):
                    pred_bgr = tensor_to_bgr(preds[b])
                    gt_bgr = tensor_to_bgr(vis_t[b])

                    pred_gray = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY)
                    gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)

                    mse = np.mean((pred_gray.astype(np.float64) - gt_gray.astype(np.float64)) ** 2)
                    psnr_v = 100.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
                    ssim_v = float(ssim_fn(pred_gray, gt_gray, data_range=255))
                    f1_v = compute_edge_f1(pred_gray, gt_gray)

                    psnr_list.append(psnr_v)
                    ssim_list.append(ssim_v)
                    f1_list.append(f1_v)

                evaluated_count = len(psnr_list)
                if (batch_idx + 1) % 10 == 0 or evaluated_count >= len(test_ds):
                    print(f"    Evaluated {evaluated_count}/{len(test_ds)} KAIST pairs | "
                          f"PSNR: {np.mean(psnr_list):.4f} dB | SSIM: {np.mean(ssim_list):.4f} | Edge F1: {np.mean(f1_list):.4f}", flush=True)

        metrics = {
            "psnr": float(np.mean(psnr_list)),
            "ssim": float(np.mean(ssim_list)),
            "edge_f1": float(np.mean(f1_list)),
            "latency_ms": float(np.mean(latencies)),
            "params_m": round(params_m, 2),
            "num_samples": len(test_ds),
        }
        results[exp_id] = metrics
        print(f"  --> Final KAIST Zero-Shot: PSNR: {metrics['psnr']:.4f} dB | SSIM: {metrics['ssim']:.4f} | "
              f"Edge F1: {metrics['edge_f1']:.4f} | Latency: {metrics['latency_ms']:.2f} ms", flush=True)

    # Save visual comparison panels
    if args.save_visuals and len(loaded_models) > 0:
        vis_dir = Path("results/kaist_visual_comparisons")
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating {args.num_visuals} KAIST comparison panels -> {vis_dir}", flush=True)

        for idx in range(min(args.num_visuals, len(test_ds))):
            sample = test_ds[idx]
            stem = sample["filename"]
            ir_t = sample["ir"].unsqueeze(0).to(device)
            edge_t = sample["edge"].unsqueeze(0).to(device)
            vis_t = sample["vis"]
            inp = torch.cat([ir_t, edge_t], dim=1)

            ir_bgr = cv2.cvtColor(((ir_t[0, 0].cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            gt_bgr = tensor_to_bgr(vis_t)

            col_labels = ["KAIST IR", "Ground Truth", "E1: Baseline", "E2: Original+Sobel", "E3: Gated U-Net", "E4: Proposed"]
            panels = [ir_bgr, gt_bgr]
            for exp_id in ["E1", "E2", "E3", "E4"]:
                if exp_id in loaded_models:
                    with torch.no_grad():
                        pred = loaded_models[exp_id](inp)
                    panels.append(tensor_to_bgr(pred[0]))
                else:
                    placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Pending Train", (45, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1, cv2.LINE_AA)
                    panels.append(placeholder)

            header_height = 36
            total_width = 256 * len(panels)
            header_bar = np.zeros((header_height, total_width, 3), dtype=np.uint8) + 30
            for col_idx, label in enumerate(col_labels):
                cv2.putText(header_bar, label, (col_idx * 256 + 15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            images_row = np.hstack(panels)
            grid = np.vstack([header_bar, images_row])
            cv2.imwrite(str(vis_dir / f"kaist_{stem}_comparison.jpg"), grid)

    # Save KAIST summary JSON
    out_json = Path("results/kaist_zero_shot_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Table 2
    print("\n" + "=" * 75, flush=True)
    print("Table 2: KAIST Zero-Shot Cross-Dataset Results", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Model':<8} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'Edge F1 ↑':<10} | {'Latency (ms) ↓':<15} | {'Params (M)':<10}", flush=True)
    print("-" * 75, flush=True)
    for exp_id, m in results.items():
        print(f"{exp_id:<8} | {m['psnr']:<12.4f} | {m['ssim']:<10.4f} | {m['edge_f1']:<10.4f} | {m['latency_ms']:<15.2f} | {m['params_m']:<10.2f}", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
