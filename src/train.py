"""
train.py – Pix2Pix training for LLVIP IR → Visible colorization.

Edge-preserving additions:
  - Input to G: IR image (1ch) concatenated with Canny edge map (1ch) → 2ch
  - Input to D: [IR, edge, RGB] → 5ch
  - Losses: GAN + L1 + Edge-Preserving + VGG Perceptual

Usage:
  python src/train.py --config configs/config.yaml
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

import yaml
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except Exception:
    HAS_TB = False
    class SummaryWriter:        # fallback no-op writer
        def __init__(self, **kwargs): pass
        def add_scalar(self, *a, **k): pass
        def close(self): pass

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset import LLVIPDataset
from models import (
    UNetGenerator, PatchGANDiscriminator,
    GANLoss, EdgePreservingLoss, VGGPerceptualLoss,
    init_weights,
)


# ============================================================
# Utility helpers
# ============================================================

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("train")


def tensor2img(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor in [-1,1] to uint8 BGR numpy image."""
    img = t.detach().cpu().float().numpy()
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)          # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[0] == 1:
        img = img[0]                           # grayscale
    return img


def save_samples(ir_batch, vis_batch, fake_batch, edge_batch,
                 out_dir, epoch, step, n=4):
    """Save a grid of [IR | Edge | Fake | Real] for visual inspection."""
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, ir_batch.size(0))
    rows = []
    for i in range(n):
        ir_img = tensor2img(ir_batch[i].expand(3, -1, -1))
        edge_img = (edge_batch[i, 0].cpu().numpy() * 255).astype(np.uint8)
        edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        fake_img = tensor2img(fake_batch[i])
        real_img = tensor2img(vis_batch[i])
        row = np.concatenate([ir_img, edge_img, fake_img, real_img], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    path = os.path.join(out_dir, f"epoch{epoch:04d}_step{step:06d}.jpg")
    cv2.imwrite(path, grid)
    return path


def save_checkpoint(netG, netD, optG, optD, epoch, cfg, is_best=False):
    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch":        epoch,
        "netG":         netG.state_dict(),
        "netD":         netD.state_dict(),
        "optG":         optG.state_dict(),
        "optD":         optD.state_dict(),
    }
    path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:04d}.pth")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))
    return path


def load_checkpoint(path, netG, netD, optG, optD, device):
    state = torch.load(path, map_location=device)
    netG.load_state_dict(state["netG"])
    netD.load_state_dict(state["netD"])
    optG.load_state_dict(state["optG"])
    optD.load_state_dict(state["optD"])
    return state["epoch"]


# ============================================================
# Training step helpers
# ============================================================

def set_requires_grad(nets, requires_grad):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = requires_grad


def forward_pass(batch, netG, device):
    """Return fake image given a batch dict."""
    ir    = batch["ir"].to(device)        # (B, 1, H, W)
    edge  = batch["edge"].to(device)      # (B, 1, H, W)
    gen_input = torch.cat([ir, edge], dim=1)   # (B, 2, H, W)
    fake  = netG(gen_input)               # (B, 3, H, W)
    return ir, edge, fake


# ============================================================
# Main training loop
# ============================================================

def train(cfg_path: str, resume: str = None):
    # --- Config ---
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(cfg["paths"]["logs"])
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg_path}")

    # --- Datasets & Loaders ---
    pairs_csv = cfg["dataset"].get("pairs_csv")
    if pairs_csv and os.path.exists(pairs_csv):
        train_ds = LLVIPDataset(
            csv_file=pairs_csv,
            split="train",
            ann_dir=cfg["dataset"].get("annotations"),
            image_size=tuple(cfg["dataset"]["image_size"]),
            augment=True,
        )
        val_ds = LLVIPDataset(
            csv_file=pairs_csv,
            split="validation",
            image_size=tuple(cfg["dataset"]["image_size"]),
            augment=False,
        )
        logger.info(f"Loaded dataset from CSV split: {pairs_csv}")
        logger.info(f"Train partition: {len(train_ds)} samples (90% of official train)")
        logger.info(f"Validation partition: {len(val_ds)} samples (10% of official train)")
    else:
        train_ds = LLVIPDataset(
            ir_dir=cfg["dataset"]["train_ir"],
            vis_dir=cfg["dataset"]["train_vis"],
            ann_dir=cfg["dataset"].get("annotations"),
            image_size=tuple(cfg["dataset"]["image_size"]),
            augment=True,
        )
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
    ) if val_ds is not None else None

    logger.info(f"Train samples: {len(train_ds)}  |  Batches/epoch: {len(train_loader)}")

    # --- Models ---
    netG = UNetGenerator(in_channels=2, out_channels=3,
                         ngf=cfg["model"]["ngf"]).to(device)
    netD = PatchGANDiscriminator(in_channels=5,          # IR(1)+edge(1)+RGB(3)
                                  ndf=cfg["model"]["ndf"]).to(device)
    init_weights(netG, "normal")
    init_weights(netD, "normal")

    # --- Losses ---
    criterion_GAN  = GANLoss(mode=cfg["model"]["gan_mode"], device=device)
    criterion_L1   = torch.nn.L1Loss()
    criterion_edge = EdgePreservingLoss().to(device)
    criterion_vgg  = VGGPerceptualLoss(device=device)

    lam_L1   = cfg["training"]["lambda_L1"]
    lam_edge = cfg["training"]["lambda_edge"]
    lam_vgg  = cfg["training"]["lambda_perceptual"]

    # --- Optimizers ---
    optG = optim.Adam(netG.parameters(),
                      lr=cfg["training"]["lr_G"],
                      betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]))
    optD = optim.Adam(netD.parameters(),
                      lr=cfg["training"]["lr_D"],
                      betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]))

    # LR schedulers (linear decay after decay_epoch)
    n_epochs    = cfg["training"]["num_epochs"]
    decay_start = cfg["training"]["decay_epoch"]

    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / float(n_epochs - decay_start))

    schedG = optim.lr_scheduler.LambdaLR(optG, lr_lambda)
    schedD = optim.lr_scheduler.LambdaLR(optD, lr_lambda)

    # --- Optionally resume ---
    start_epoch = 0
    if resume and os.path.exists(resume):
        start_epoch = load_checkpoint(resume, netG, netD, optG, optD, device)
        logger.info(f"Resumed from {resume} at epoch {start_epoch}")

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=cfg["paths"]["logs"])

    sample_dir = cfg["paths"]["results_pix2pix"]
    global_step = 0

    # ---- Training loop ----
    for epoch in range(start_epoch, n_epochs):
        netG.train(); netD.train()
        epoch_loss_G = epoch_loss_D = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            ir   = batch["ir"].to(device)        # (B,1,H,W)
            vis  = batch["vis"].to(device)        # (B,3,H,W)
            edge = batch["edge"].to(device)       # (B,1,H,W)

            gen_input = torch.cat([ir, edge], dim=1)  # (B,2,H,W)
            fake_vis  = netG(gen_input)                # (B,3,H,W)

            # ---- Update D ----
            set_requires_grad([netD], True)
            optD.zero_grad()

            real_pair = torch.cat([ir, edge, vis],      dim=1)  # (B,5,H,W)
            fake_pair = torch.cat([ir, edge, fake_vis.detach()], dim=1)

            loss_D_real = criterion_GAN(netD(real_pair), True)
            loss_D_fake = criterion_GAN(netD(fake_pair), False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optD.step()

            # ---- Update G ----
            set_requires_grad([netD], False)
            optG.zero_grad()

            fake_pair_g = torch.cat([ir, edge, fake_vis], dim=1)
            loss_GAN  = criterion_GAN(netD(fake_pair_g), True)
            loss_L1   = criterion_L1(fake_vis, vis) * lam_L1
            loss_edge = criterion_edge(fake_vis, vis) * lam_edge
            loss_vgg  = criterion_vgg(fake_vis, vis) * lam_vgg
            loss_G    = loss_GAN + loss_L1 + loss_edge + loss_vgg
            loss_G.backward()
            optG.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            global_step  += 1

            # Logging
            if (batch_idx + 1) % cfg["training"]["log_every"] == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{n_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"D: {loss_D.item():.4f}  "
                    f"G: {loss_G.item():.4f}  "
                    f"L1: {loss_L1.item():.4f}  "
                    f"Edge: {loss_edge.item():.4f}  "
                    f"VGG: {loss_vgg.item():.4f}"
                )
                writer.add_scalar("Loss/D",    loss_D.item(),    global_step)
                writer.add_scalar("Loss/G",    loss_G.item(),    global_step)
                writer.add_scalar("Loss/L1",   loss_L1.item(),   global_step)
                writer.add_scalar("Loss/Edge", loss_edge.item(), global_step)
                writer.add_scalar("Loss/VGG",  loss_vgg.item(),  global_step)

            # Sample images
            if (batch_idx + 1) % cfg["training"]["sample_every"] == 0:
                netG.eval()
                with torch.no_grad():
                    sample_path = save_samples(
                        ir, vis, fake_vis, edge,
                        sample_dir, epoch + 1, global_step,
                    )
                logger.info(f"  Sample saved: {sample_path}")
                netG.train()

        # End of epoch
        avg_G = epoch_loss_G / len(train_loader)
        avg_D = epoch_loss_D / len(train_loader)
        elapsed = time.time() - t0
        logger.info(
            f"==> Epoch {epoch+1}/{n_epochs} done in {elapsed:.1f}s  "
            f"avg G: {avg_G:.4f}  avg D: {avg_D:.4f}"
        )
        writer.add_scalar("Epoch/G", avg_G, epoch + 1)
        writer.add_scalar("Epoch/D", avg_D, epoch + 1)
        writer.add_scalar("LR/G", schedG.get_last_lr()[0], epoch + 1)

        # Save checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            ckpt = save_checkpoint(netG, netD, optG, optD, epoch + 1, cfg)
            logger.info(f"  Checkpoint: {ckpt}")

        schedG.step()
        schedD.step()

    # Save final model
    save_checkpoint(netG, netD, optG, optD, n_epochs, cfg, is_best=True)
    writer.close()
    logger.info("Training complete.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.resume)
