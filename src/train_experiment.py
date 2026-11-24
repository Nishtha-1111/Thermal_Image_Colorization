"""
src/train_experiment.py – Unified training harness for Experiments E1, E2, E3, and E4.

Usage:
  python src/train_experiment.py --config configs/E1_baseline.yaml
  python src/train_experiment.py --config configs/E2_sobel.yaml
  python src/train_experiment.py --config configs/E3_gated.yaml
  python src/train_experiment.py --config configs/E4_proposed.yaml
  python src/train_experiment.py --exp E1   # shortcut for configs/E1_baseline.yaml
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import cv2
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LLVIPDataset
from models import UNetGenerator, PatchGANDiscriminator, init_weights
from gated_models import GatedUNetGenerator
from losses import GANLoss, CompositeGeneratorLoss


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("train_exp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().float().numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_samples(ir: torch.Tensor, vis: torch.Tensor, fake: torch.Tensor, edge: torch.Tensor,
                 out_dir: Path, epoch: int, step: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr_vis = tensor_to_bgr(vis[0])
    bgr_fake = tensor_to_bgr(fake[0])

    ir_gray = ((ir[0, 0].cpu().numpy() + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    bgr_ir = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)

    edge_gray = (edge[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    bgr_edge = cv2.cvtColor(edge_gray, cv2.COLOR_GRAY2BGR)

    # Grid: [IR | Canny | Fake Visible | Real GT Visible]
    grid = np.hstack([bgr_ir, bgr_edge, bgr_fake, bgr_vis])
    out_file = out_dir / f"epoch_{epoch:03d}_step_{step:06d}.jpg"
    cv2.imwrite(str(out_file), grid)
    return out_file


def train(cfg_path: str, resume_path: str = None):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_info = cfg["experiment"]
    exp_id = exp_info["id"]
    exp_name = exp_info["name"]
    generator_type = exp_info.get("generator_type", "original_unet")
    use_sobel_loss = exp_info.get("use_sobel_loss", True)

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    log_dir = Path(cfg["paths"]["logs"])
    sample_dir = Path(cfg["paths"]["samples"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(log_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info(f"STARTING EXPERIMENT: {exp_id} ({exp_name})")
    logger.info(f"Generator Architecture : {generator_type}")
    logger.info(f"Generator Input        : IR + Canny (2 channels)")
    logger.info(f"Sobel Edge Loss        : {'ENABLED (weight 10.0)' if use_sobel_loss else 'DISABLED'}")
    logger.info(f"Discriminator Input    : IR + Canny + RGB (5 channels, UNCHANGED)")
    logger.info(f"Device                 : {device}")
    logger.info("=" * 70)

    # 1. Dataset & Loaders
    pairs_csv = cfg["dataset"]["pairs_csv"]
    train_ds = LLVIPDataset(
        csv_file=pairs_csv,
        split="train",
        image_size=tuple(cfg["dataset"]["image_size"]),
        augment=True,
    )
    val_ds = LLVIPDataset(
        csv_file=pairs_csv,
        split="validation",
        image_size=tuple(cfg["dataset"]["image_size"]),
        augment=False,
    )

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataset"].get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    logger.info(f"Dataset: {pairs_csv}")
    logger.info(f"Training partition   : {len(train_ds):,} pairs ({len(train_loader)} batches/epoch)")
    logger.info(f"Validation partition : {len(val_ds):,} pairs (reserved for checkpoint selection)")
    logger.info(f"Testing partition    : 3,463 pairs (STRICTLY WITHHELD)")

    # 2. Build Models
    in_channels_G = exp_info.get("in_channels_G", 2)
    out_channels_G = exp_info.get("out_channels_G", 3)
    ngf = exp_info.get("ngf", 64)

    if generator_type == "gated_unet":
        netG = GatedUNetGenerator(in_channels=in_channels_G, out_channels=out_channels_G, ngf=ngf).to(device)
    else:
        netG = UNetGenerator(in_channels=in_channels_G, out_channels=out_channels_G, ngf=ngf).to(device)

    # Discriminator: Unchanged 5 channels [IR(1) + Canny(1) + RGB(3)]
    netD = PatchGANDiscriminator(in_channels=5, ndf=64).to(device)

    init_weights(netG, "normal")
    init_weights(netD, "normal")

    # 3. Loss Functions & Optimizers
    loss_D_fn = GANLoss(mode=cfg["loss"].get("gan_mode", "lsgan"), device=device)
    comp_loss_G = CompositeGeneratorLoss(
        use_sobel_loss=use_sobel_loss,
        lambda_gan=cfg["loss"].get("lambda_gan", 1.0),
        lambda_l1=cfg["loss"].get("lambda_l1", 100.0),
        lambda_sobel=cfg["loss"].get("lambda_sobel", 10.0),
        lambda_vgg=cfg["loss"].get("lambda_vgg", 5.0),
        gan_mode=cfg["loss"].get("gan_mode", "lsgan"),
        device=device,
    )

    lr_G = cfg["training"]["lr_G"]
    lr_D = cfg["training"]["lr_D"]
    beta1 = cfg["training"]["beta1"]
    beta2 = cfg["training"]["beta2"]

    optG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, beta2))
    optD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, beta2))

    total_epochs = cfg["training"]["epochs"]
    decay_epoch = cfg["training"].get("lr_decay_start", 50)

    def lr_lambda(epoch):
        return 1.0 - max(0, epoch + 1 - decay_epoch) / float(total_epochs - decay_epoch + 1)

    schedG = optim.lr_scheduler.LambdaLR(optG, lr_lambda=lr_lambda)
    schedD = optim.lr_scheduler.LambdaLR(optD, lr_lambda=lr_lambda)

    start_epoch = 1
    best_val_loss = float("inf")

    # Resume from checkpoint if requested
    if resume_path == "auto" or (resume_path is not None and not os.path.exists(resume_path)):
        # Auto-detect latest checkpoint or best_model
        if (ckpt_dir / "checkpoint_latest.pth").exists():
            resume_path = str(ckpt_dir / "checkpoint_latest.pth")
        elif (ckpt_dir / "best_model.pth").exists():
            resume_path = str(ckpt_dir / "best_model.pth")
        else:
            ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pth"))
            resume_path = str(ckpts[-1]) if ckpts else None

    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])
        optG.load_state_dict(ckpt["optG"])
        optD.load_state_dict(ckpt["optD"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        for _ in range(start_epoch - 1):
            schedG.step()
            schedD.step()
        logger.info(f"Resumed at epoch {start_epoch} (best_val_loss: {best_val_loss:.4f})")

    # 4. Training Loop
    sample_every = cfg["training"].get("sample_every", 500)
    save_every = cfg["training"].get("save_every", 10)
    global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        netG.train()
        netD.train()

        running_loss_D = 0.0
        running_loss_G = 0.0
        running_loss_l1 = 0.0
        running_loss_sobel = 0.0
        running_loss_vgg = 0.0

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            ir = batch["ir"].to(device)
            vis = batch["vis"].to(device)
            edge = batch["edge"].to(device)
            gen_in = torch.cat([ir, edge], dim=1)  # (B, 2, 256, 256)

            # --- Update Discriminator ---
            optD.zero_grad()
            with torch.no_grad():
                fake_vis = netG(gen_in)

            # 5-channel real and fake inputs
            real_D_in = torch.cat([ir, edge, vis], dim=1)
            fake_D_in = torch.cat([ir, edge, fake_vis], dim=1)

            loss_D_real = loss_D_fn(netD(real_D_in), True)
            loss_D_fake = loss_D_fn(netD(fake_D_in), False)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optD.step()

            # --- Update Generator ---
            optG.zero_grad()
            fake_vis = netG(gen_in)
            pred_fake_G = netD(torch.cat([ir, edge, fake_vis], dim=1))

            loss_G, loss_dict = comp_loss_G(pred_fake_G, fake_vis, vis)
            loss_G.backward()
            optG.step()

            running_loss_D += loss_D.item()
            running_loss_G += loss_G.item()
            running_loss_l1 += loss_dict["loss_l1"]
            running_loss_sobel += loss_dict["loss_sobel"]
            running_loss_vgg += loss_dict["loss_vgg"]

            if global_step % sample_every == 0:
                with torch.no_grad():
                    sample_file = save_samples(ir, vis, fake_vis, edge, sample_dir, epoch, global_step)
                logger.info(f"  [Sample Saved] {sample_file.name}")

        avg_loss_D = running_loss_D / len(train_loader)
        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_l1 = running_loss_l1 / len(train_loader)
        avg_loss_sobel = running_loss_sobel / len(train_loader)
        avg_loss_vgg = running_loss_vgg / len(train_loader)

        # --- Validation Evaluation ---
        netG.eval()
        val_l1_sum = 0.0
        val_sobel_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for val_batch in val_loader:
                ir_v = val_batch["ir"].to(device)
                vis_v = val_batch["vis"].to(device)
                edge_v = val_batch["edge"].to(device)
                fake_v = netG(torch.cat([ir_v, edge_v], dim=1))

                val_l1_sum += torch.abs(fake_v - vis_v).mean().item()
                if use_sobel_loss:
                    val_sobel_sum += comp_loss_G.criterion_sobel(fake_v, vis_v).item()
                val_batches += 1

        val_l1 = val_l1_sum / val_batches
        val_sobel = (val_sobel_sum / val_batches) if use_sobel_loss else 0.0
        val_total = 100.0 * val_l1 + (10.0 * val_sobel if use_sobel_loss else 0.0)

        elapsed = time.time() - t0
        curr_lr = schedG.get_last_lr()[0]

        logger.info(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] ({elapsed:.1f}s, lr={curr_lr:.6f}) | "
            f"Train D: {avg_loss_D:.4f} | Train G: {avg_loss_G:.4f} (L1: {avg_loss_l1:.4f}, Sobel: {avg_loss_sobel:.4f}, VGG: {avg_loss_vgg:.4f}) | "
            f"Val L1: {val_l1:.4f} (Val Total: {val_total:.2f})"
        )

        # Save Best Checkpoint (based on validation loss)
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = ckpt_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "best_val_loss": best_val_loss,
                "config": cfg,
            }, str(best_path))
            logger.info(f"  ★ New best model saved -> {best_path.name} (val_total: {best_val_loss:.4f})")

        # Save Latest Checkpoint (ensures zero progress lost if interrupted)
        latest_path = ckpt_dir / "checkpoint_latest.pth"
        torch.save({
            "epoch": epoch,
            "netG": netG.state_dict(),
            "netD": netD.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg,
        }, str(latest_path))

        # Save Periodic Checkpoint
        if epoch % save_every == 0 or epoch == total_epochs:
            ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pth"
            torch.save({
                "epoch": epoch,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optG": optG.state_dict(),
                "optD": optD.state_dict(),
                "best_val_loss": best_val_loss,
                "config": cfg,
            }, str(ckpt_path))
            logger.info(f"  Checkpoint saved -> {ckpt_path.name}")

        schedG.step()
        schedD.step()

    logger.info("=" * 70)
    logger.info(f"EXPERIMENT {exp_id} COMPLETED SUCCESSFULLY!")
    logger.info(f"Best validation loss : {best_val_loss:.4f}")
    logger.info(f"Best checkpoint path : {ckpt_dir / 'best_model.pth'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLVIP Thermal-to-Visible Experiments")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config")
    parser.add_argument("--exp", type=str, default=None, choices=["E1", "E2", "E3", "E4"],
                        help="Experiment shortcut (E1, E2, E3, or E4)")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Path to checkpoint to resume training (or omit path to auto-detect latest/best)")
    args = parser.parse_args()

    config_map = {
        "E1": "configs/E1_baseline.yaml",
        "E2": "configs/E2_sobel.yaml",
        "E3": "configs/E3_gated.yaml",
        "E4": "configs/E4_proposed.yaml",
    }

    if args.exp:
        target_cfg = config_map[args.exp]
    elif args.config:
        target_cfg = args.config
    else:
        target_cfg = "configs/E1_baseline.yaml"

    train(target_cfg, resume_path=args.resume)
