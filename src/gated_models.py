"""
src/gated_models.py – Gated U-Net Generator with Additive Attention Gates on Skip Connections.

Architecture:
  - 8-stage Encoder (identical to baseline U-Net 256)
  - 8-stage Decoder with Additive Spatial Attention Gates on all skip paths
  - Base filters: ngf = 64
  - Batch normalization and dropout (p=0.5) identical to baseline
  - Final Tanh output in [-1, 1]
  - Supports Configuration A (Thermal only, in_channels=1)
    and Configuration B (Thermal + Canny, in_channels=2)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from attention_gate import AttentionGate
from models import (
    UNetDown,
    PatchGANDiscriminator,
    GANLoss,
    EdgePreservingLoss,
    VGGPerceptualLoss,
    init_weights,
)


# ============================================================
# Gated Decoder Block (GatedUNetUp)
# ============================================================

class GatedUNetUp(nn.Module):
    """
    Decoder block with integrated additive attention gating on the skip connection:
      1. Decoder feature upsampling: TransposeConv -> BatchNorm -> ReLU (-> Dropout)
         producing gating signal g of shape (B, out_c, H, W).
      2. Attention gating: AttentionGate(skip, g)
         producing attention map alpha in [0, 1] and gated skip: F_gated = alpha * skip.
      3. Gated feature concatenation: torch.cat([g, F_gated], dim=1).
    """

    def __init__(self, in_c: int, out_c: int, skip_c: int, dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.up = nn.Sequential(*layers)
        self.gate = AttentionGate(in_channels_x=skip_c, in_channels_g=out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, return_attention: bool = False):
        g = self.up(x)
        if return_attention:
            gated_skip, alpha = self.gate(skip, g, return_attention=True)
            return torch.cat([g, gated_skip], dim=1), alpha
        else:
            gated_skip = self.gate(skip, g, return_attention=False)
            return torch.cat([g, gated_skip], dim=1)


# ============================================================
# Gated U-Net Generator
# ============================================================

class GatedUNetGenerator(nn.Module):
    """
    Gated U-Net Generator for Thermal-to-Visible Colorization.

    Supports two input configurations:
      - Configuration A: Thermal only (in_channels=1)
      - Configuration B: Thermal + Canny (in_channels=2)

    Skip connections are passed through additive spatial attention gates
    conditioned on the upsampled decoder feature map.
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 3, ngf: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf

        # --- 8-stage Encoder (256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1) ---
        self.d1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )                                                       # 128x128, channels: ngf (64)
        self.d2 = UNetDown(ngf,       ngf * 2)                  # 64x64,   channels: ngf*2 (128)
        self.d3 = UNetDown(ngf * 2,   ngf * 4)                  # 32x32,   channels: ngf*4 (256)
        self.d4 = UNetDown(ngf * 4,   ngf * 8)                  # 16x16,   channels: ngf*8 (512)
        self.d5 = UNetDown(ngf * 8,   ngf * 8)                  # 8x8,     channels: ngf*8 (512)
        self.d6 = UNetDown(ngf * 8,   ngf * 8)                  # 4x4,     channels: ngf*8 (512)
        self.d7 = UNetDown(ngf * 8,   ngf * 8)                  # 2x2,     channels: ngf*8 (512)
        self.d8 = UNetDown(ngf * 8,   ngf * 8, normalize=False) # 1x1,     channels: ngf*8 (512, bottleneck)

        # --- 8-stage Decoder with Attention Gates on all 7 skip connections ---
        # u1: in 512, out 512, skip d7 (512) -> concat -> 1024 (2x2)
        self.u1 = GatedUNetUp(ngf * 8,  ngf * 8, skip_c=ngf * 8, dropout=True)
        # u2: in 1024, out 512, skip d6 (512) -> concat -> 1024 (4x4)
        self.u2 = GatedUNetUp(ngf * 16, ngf * 8, skip_c=ngf * 8, dropout=True)
        # u3: in 1024, out 512, skip d5 (512) -> concat -> 1024 (8x8)
        self.u3 = GatedUNetUp(ngf * 16, ngf * 8, skip_c=ngf * 8, dropout=True)
        # u4: in 1024, out 512, skip d4 (512) -> concat -> 1024 (16x16)
        self.u4 = GatedUNetUp(ngf * 16, ngf * 8, skip_c=ngf * 8, dropout=False)
        # u5: in 1024, out 256, skip d3 (256) -> concat -> 512 (32x32)
        self.u5 = GatedUNetUp(ngf * 16, ngf * 4, skip_c=ngf * 4, dropout=False)
        # u6: in 512, out 128, skip d2 (128) -> concat -> 256 (64x64)
        self.u6 = GatedUNetUp(ngf * 8,  ngf * 2, skip_c=ngf * 2, dropout=False)
        # u7: in 256, out 64, skip d1 (64) -> concat -> 128 (128x128)
        self.u7 = GatedUNetUp(ngf * 4,  ngf,     skip_c=ngf,     dropout=False)

        # Final reconstruction stage (128x128 -> 256x256, out_channels=3)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, edge: torch.Tensor = None, return_attentions: bool = False):
        """
        Forward pass supporting:
          - x: (B, in_channels, H, W)
          - Or x: (B, 1, H, W) and edge: (B, 1, H, W) when in_channels=2
        """
        if edge is not None:
            if x.shape[1] == 1 and edge.shape[1] == 1 and self.in_channels == 2:
                x = torch.cat([x, edge], dim=1)

        # Encoder path
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        attentions = []

        # Decoder path with attention-gated skip connections
        if return_attentions:
            u1, a1 = self.u1(d8, d7, return_attention=True)
            u2, a2 = self.u2(u1, d6, return_attention=True)
            u3, a3 = self.u3(u2, d5, return_attention=True)
            u4, a4 = self.u4(u3, d4, return_attention=True)
            u5, a5 = self.u5(u4, d3, return_attention=True)
            u6, a6 = self.u6(u5, d2, return_attention=True)
            u7, a7 = self.u7(u6, d1, return_attention=True)
            attentions = [a1, a2, a3, a4, a5, a6, a7]
        else:
            u1 = self.u1(d8, d7)
            u2 = self.u2(u1, d6)
            u3 = self.u3(u2, d5)
            u4 = self.u4(u3, d4)
            u5 = self.u5(u4, d3)
            u6 = self.u6(u5, d2)
            u7 = self.u7(u6, d1)

        out = self.final(u7)

        if return_attentions:
            return out, attentions
        return out
