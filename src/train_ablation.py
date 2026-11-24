"""
src/train_ablation.py
Modular training runner for Ablation Study (Experiments A, B, C, D).

Preserves all original training details:
  - Optimizer: Adam (lr_G=0.0002, lr_D=0.0002, beta1=0.5, beta2=0.999)
  - Scheduler: Linear decay starting from lr_decay_start (epoch 50)
  - Weight init: normal (0, 0.02)
  - Loss formulation: GANLoss (LSGAN) + L1 (x100) + VGG (x5) + Sobel (x lambda_edge)
  - Augmentation: LLVIPDataset with augment=True for training

Channel Routing:
  - When use_edge_input is False (Exp A & C):
      Generator receives: ir (1 ch)
      Discriminator condition: ir (1 ch) -> total D input: [ir, vis] (4 ch)
  - When use_edge_input is True (Exp B & D):
      Generator receives: [ir, edge] (2 ch)
      Discriminator condition: [ir, edge] (2 ch) -> total D input: [ir, edge, vis] (5 ch)

Usage:
  python -m src.train_ablation --config experiments/A_baseline/config.yaml
"""

import os
import sys
import time
import json
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
    class SummaryWriter:
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


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def tensor2img(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().float().numpy()
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[0] == 1:
        img = img[0]
    return img


def save_samples(ir_batch, vis_batch, fake_batch, edge_batch,
                 out_dir, epoch, step, use_edge=True, n=4):
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, ir_batch.size(0))
    rows = []
    for i in range(n):
        ir_img = tensor2img(ir_batch[i].expand(3, -1, -1))
        fake_img = tensor2img(fake_batch[i])
        real_img = tensor2img(vis_batch[i])
        if use_edge and edge_batch is not None:
            edge_img = (edge_batch[i, 0].cpu().numpy() * 255).astype(np.uint8)
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
            row = np.concatenate([ir_img, edge_img, fake_img, real_img], axis=1)
        else:
            row = np.concatenate([ir_img, fake_img, real_img], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    path = os.path.join(out_dir, f"epoch{epoch:04d}_step{step:06d}.jpg")
    cv2.imwrite(path, grid)
    return path


def save_checkpoint(netG, netD, optG, optD, epoch, ckpt_dir, is_best=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
    }
    path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:04d}.pth")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(ckpt_dir, "best_model.pth"))
    return path


def set_requires_grad(nets, requires_grad):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = requires_grad


def train_ablation(cfg_path: str, resume: str = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment", {}).get("name", "ablation_exp")
    use_edge_input = cfg.get("experiment", {}).get("use_edge_input", False)
    use_edge_loss  = cfg.get("loss", {}).get("use_edge_loss", True)
    seed = cfg.get("experiment", {}).get("seed", 42)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Paths
    ckpt_dir = cfg.get("output", {}).get("checkpoint_dir", f"experiments/{exp_name}/checkpoints")
    result_dir = cfg.get("output", {}).get("result_dir", f"experiments/{exp_name}/results")
    log_dir = cfg.get("output", {}).get("log_dir", f"experiments/{exp_name}/logs")
    sample_dir = os.path.join(result_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    logger = setup_logger(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Starting Ablation Run: {exp_name} ===")
    logger.info(f"Device: {device}")
    logger.info(f"Use Edge Input: {use_edge_input} | Use Edge Loss: {use_edge_loss}")

    # Dataset
    img_size = cfg["dataset"]["image_size"]
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    else:
        img_size = tuple(img_size)

    train_ds = LLVIPDataset(
        ir_dir=cfg["dataset"]["train_ir"],
        vis_dir=cfg["dataset"]["train_vis"],
        ann_dir=cfg["dataset"].get("annotations"),
        image_size=img_size,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"].get("num_workers", 4),
        pin_memory=True,
    )

    # Channel configuration
    in_channels_G = cfg.get("model", {}).get("in_channels_G", 2 if use_edge_input else 1)
    in_channels_D = cfg.get("model", {}).get("in_channels_D", 5 if use_edge_input else 4)

    logger.info(f"Generator in_channels: {in_channels_G} | Discriminator in_channels: {in_channels_D}")

    netG = UNetGenerator(
        in_channels=in_channels_G,
        out_channels=cfg.get("model", {}).get("out_channels_G", 3),
        ngf=cfg.get("model", {}).get("ngf", 64),
    ).to(device)

    netD = PatchGANDiscriminator(
        in_channels=in_channels_D,
        ndf=cfg.get("model", {}).get("ndf", 64),
    ).to(device)

    init_weights(netG, "normal")
    init_weights(netD, "normal")

    # Losses
    gan_mode = cfg.get("loss", {}).get("gan_mode", cfg.get("model", {}).get("gan_mode", "lsgan"))
    criterion_GAN  = GANLoss(mode=gan_mode, device=device)
    criterion_L1   = torch.nn.L1Loss()
    criterion_edge = EdgePreservingLoss().to(device)
    criterion_vgg  = VGGPerceptualLoss(device=device)

    lam_L1   = cfg["training"]["lambda_L1"]
    lam_edge = cfg["training"]["lambda_edge"] if use_edge_loss else 0.0
    lam_vgg  = cfg["training"]["lambda_perceptual"]

    logger.info(f"Loss weights -> L1: {lam_L1}, Edge: {lam_edge}, VGG: {lam_vgg}")

    # Optimizers
    optG = optim.Adam(
        netG.parameters(),
        lr=cfg["training"]["lr_G"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
    )
    optD = optim.Adam(
        netD.parameters(),
        lr=cfg["training"]["lr_D"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
    )

    # Schedulers
    n_epochs = cfg["training"].get("epochs", cfg["training"].get("num_epochs", 100))
    decay_start = cfg["training"].get("lr_decay_start", cfg["training"].get("decay_epoch", 50))

    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / float(n_epochs - decay_start))

    schedG = optim.lr_scheduler.LambdaLR(optG, lr_lambda)
    schedD = optim.lr_scheduler.LambdaLR(optD, lr_lambda)

    # Resume checkpoint if specified
    start_epoch = 0
    if resume and os.path.exists(resume):
        state = torch.load(resume, map_location=device)
        netG.load_state_dict(state["netG"])
        netD.load_state_dict(state["netD"])
        optG.load_state_dict(state["optG"])
        optD.load_state_dict(state["optD"])
        start_epoch = state.get("epoch", 0)
        logger.info(f"Resumed from {resume} at epoch {start_epoch}")

    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0
    history = []

    log_every = cfg["training"].get("log_every", 100)
    sample_every = cfg["training"].get("sample_every", 500)
    save_every = cfg["training"].get("save_every", 10)

    # Training loop
    for epoch in range(start_epoch, n_epochs):
        netG.train()
        netD.train()
        epoch_loss_G = epoch_loss_D = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            ir   = batch["ir"].to(device)
            vis  = batch["vis"].to(device)
            edge = batch["edge"].to(device)

            # Assemble conditioned inputs
            if use_edge_input:
                gen_input = torch.cat([ir, edge], dim=1)
                disc_cond = torch.cat([ir, edge], dim=1)
            else:
                gen_input = ir
                disc_cond = ir

            fake_vis = netG(gen_input)

            # --- Update D ---
            set_requires_grad([netD], True)
            optD.zero_grad()

            real_pair = torch.cat([disc_cond, vis], dim=1)
            fake_pair = torch.cat([disc_cond, fake_vis.detach()], dim=1)

            loss_D_real = criterion_GAN(netD(real_pair), True)
            loss_D_fake = criterion_GAN(netD(fake_pair), False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optD.step()

            # --- Update G ---
            set_requires_grad([netD], False)
            optG.zero_grad()

            fake_pair_g = torch.cat([disc_cond, fake_vis], dim=1)
            loss_GAN = criterion_GAN(netD(fake_pair_g), True)
            loss_L1  = criterion_L1(fake_vis, vis) * lam_L1

            if lam_edge > 0.0:
                loss_edge = criterion_edge(fake_vis, vis) * lam_edge
            else:
                loss_edge = torch.tensor(0.0, device=device)

            loss_vgg = criterion_vgg(fake_vis, vis) * lam_vgg
            loss_G   = loss_GAN + loss_L1 + loss_edge + loss_vgg
            loss_G.backward()
            optG.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            global_step += 1

            if (batch_idx + 1) % log_every == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{n_epochs}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"D: {loss_D.item():.4f} "
                    f"G: {loss_G.item():.4f} "
                    f"L1: {loss_L1.item():.4f} "
                    f"Edge: {loss_edge.item():.4f} "
                    f"VGG: {loss_vgg.item():.4f}"
                )
                writer.add_scalar("Loss/D", loss_D.item(), global_step)
                writer.add_scalar("Loss/G", loss_G.item(), global_step)
                writer.add_scalar("Loss/L1", loss_L1.item(), global_step)
                writer.add_scalar("Loss/Edge", loss_edge.item(), global_step)
                writer.add_scalar("Loss/VGG", loss_vgg.item(), global_step)

            if (batch_idx + 1) % sample_every == 0:
                netG.eval()
                with torch.no_grad():
                    save_samples(ir, vis, fake_vis, edge if use_edge_input else None,
                                 sample_dir, epoch + 1, global_step, use_edge=use_edge_input)
                netG.train()

        avg_G = epoch_loss_G / len(train_loader)
        avg_D = epoch_loss_D / len(train_loader)
        elapsed = time.time() - t0
        logger.info(
            f"==> Epoch {epoch+1}/{n_epochs} done in {elapsed:.1f}s | "
            f"avg G: {avg_G:.4f} | avg D: {avg_D:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "avg_G": avg_G,
            "avg_D": avg_D,
            "time_sec": elapsed,
        })
        with open(os.path.join(result_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

        if (epoch + 1) % save_every == 0:
            ckpt = save_checkpoint(netG, netD, optG, optD, epoch + 1, ckpt_dir)
            logger.info(f"  Checkpoint saved: {ckpt}")

        schedG.step()
        schedD.step()

    # Save final best model
    save_checkpoint(netG, netD, optG, optD, n_epochs, ckpt_dir, is_best=True)
    writer.close()
    logger.info(f"=== Ablation Run Complete: {exp_name} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Training Runner")
    parser.add_argument("--config", required=True, help="Path to experiment config.yaml")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from")
    args = parser.parse_args()
    train_ablation(args.config, args.resume)
