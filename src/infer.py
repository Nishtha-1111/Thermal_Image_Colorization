"""
infer.py – Standalone batch inference script for Pix2Pix.

Usage:
  python src/infer.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
  python src/infer.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --input path/to/single.jpg
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LLVIPDataset, compute_edge_map
from models import UNetGenerator


def tensor_to_bgr(t):
    img = t.cpu().numpy()
    img = ((img + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)              # RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def infer_single(netG, device, ir_path, out_path, img_size=(256, 256)):
    bgr = cv2.imread(ir_path)
    if bgr is None:
        raise FileNotFoundError(ir_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_size
    gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    edge = compute_edge_map(gray)

    ir_t   = torch.from_numpy(gray.astype(np.float32) / 127.5 - 1.0)
    edge_t = torch.from_numpy(edge)
    ir_t   = ir_t.unsqueeze(0).unsqueeze(0).to(device)
    edge_t = edge_t.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = netG(torch.cat([ir_t, edge_t], dim=1))[0]

    cv2.imwrite(out_path, tensor_to_bgr(fake))
    print(f"Saved: {out_path}")


def infer_dataset(netG, device, cfg, max_samples=None):
    pairs_csv = cfg["dataset"].get("pairs_csv")
    if pairs_csv and os.path.exists(pairs_csv):
        ds = LLVIPDataset(
            csv_file=pairs_csv,
            split="test",
            image_size=tuple(cfg["dataset"]["image_size"]),
            augment=False,
            max_samples=max_samples,
        )
    else:
        ds = LLVIPDataset(
            ir_dir=cfg["dataset"]["test_ir"],
            vis_dir=cfg["dataset"]["test_vis"],
            image_size=tuple(cfg["dataset"]["image_size"]),
            augment=False,
            max_samples=max_samples,
        )
    out_dir = os.path.join(cfg["paths"]["results_pix2pix"], "test_gen")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Inferring {len(ds)} test images → {out_dir}")
    with torch.no_grad():
        for i, sample in enumerate(ds):
            ir   = sample["ir"].unsqueeze(0).to(device)
            edge = sample["edge"].unsqueeze(0).to(device)
            fake = netG(torch.cat([ir, edge], dim=1))[0]
            out_path = os.path.join(out_dir, f"{sample['filename']}.jpg")
            cv2.imwrite(out_path, tensor_to_bgr(fake))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(ds)}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input",      default=None,
                        help="Single IR image path; omit for full test-set inference")
    parser.add_argument("--output",     default=None,
                        help="Output path for single-image mode")
    parser.add_argument("--max",        type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    netG = UNetGenerator(in_channels=2, out_channels=3,
                         ngf=cfg["model"]["ngf"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(state["netG"])
    netG.eval()

    if args.input:
        out = args.output or os.path.splitext(args.input)[0] + "_generated.jpg"
        infer_single(netG, device, args.input, out,
                     img_size=tuple(cfg["dataset"]["image_size"]))
    else:
        infer_dataset(netG, device, cfg, max_samples=args.max)
