"""
src/attention_gate.py – Additive Spatial Attention Gate for Gated U-Net.

Reference:
  Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
  Medical Image Computing and Computer Assisted Intervention (MICCAI) / MIDL 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Additive Spatial Attention Gate.

    The gate receives:
      1. x: Encoder feature map of shape (B, F_x, H, W)
      2. g: Decoder gating signal of shape (B, F_g, H_g, W_g)

    Processing steps:
      - W_x(x): 1x1 conv + BatchNorm projecting x to inter_channels.
      - W_g(g): 1x1 conv + BatchNorm projecting g to inter_channels.
      - Element-wise addition and ReLU: f = ReLU(W_x(x) + W_g(g))
      - psi(f): 1x1 conv + Sigmoid producing attention map alpha in [0, 1] of shape (B, 1, H, W)
      - F_gated = alpha * x

    The output preserves the exact encoder feature dimensions (B, F_x, H, W).
    """

    def __init__(self, in_channels_x: int, in_channels_g: int, inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(in_channels_x // 2, 16)

        self.in_channels_x = in_channels_x
        self.in_channels_g = in_channels_g
        self.inter_channels = inter_channels

        # Encoder feature transform
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        # Decoder gating signal transform
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        # Linear projection to single attention weight map
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor, return_attention: bool = False):
        """
        Forward pass of Attention Gate.

        Args:
          x: Encoder feature map (B, F_x, H, W)
          g: Decoder gating signal (B, F_g, H_g, W_g)
          return_attention: If True, returns tuple (F_gated, alpha)

        Returns:
          F_gated (B, F_x, H, W), and optionally alpha (B, 1, H, W)
        """
        # Ensure spatial resolution of gating signal matches encoder feature map
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode="bilinear", align_corners=True)

        w_x = self.W_x(x)
        w_g = self.W_g(g)
        f = self.relu(w_x + w_g)
        alpha = self.psi(f)

        gated_x = alpha * x

        if return_attention:
            return gated_x, alpha
        return gated_x


if __name__ == "__main__":
    # Unit test
    B, H, W = 2, 32, 32
    enc_channels = 256
    gate_channels = 256
    gate = AttentionGate(enc_channels, gate_channels)
    x = torch.randn(B, enc_channels, H, W)
    g = torch.randn(B, gate_channels, H, W)
    out, attn = gate(x, g, return_attention=True)
    print(f"Input x: {x.shape}, g: {g.shape}")
    print(f"Output gated_x: {out.shape}, Attention map: {attn.shape}")
    print(f"Attention range: [{attn.min().item():.4f}, {attn.max().item():.4f}]")
    assert out.shape == x.shape, "Output shape must match input x!"
    assert (attn >= 0.0).all() and (attn <= 1.0).all(), "Attention map must be in [0, 1]!"
    print("AttentionGate unit test passed successfully!")
