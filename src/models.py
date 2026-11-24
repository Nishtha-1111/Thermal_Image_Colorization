"""
models.py – Pix2Pix Generator (U-Net 256) + PatchGAN Discriminator
            with edge-aware loss and VGG perceptual loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ============================================================
# Building Blocks
# ============================================================

class ConvNormReLU(nn.Module):
    """Conv → Norm → (optional) ReLU/LeakyReLU block."""
    def __init__(self, in_c, out_c, kernel=4, stride=2, padding=1,
                 norm="batch", activation="leaky", bias=False):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=bias)]
        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_c))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_c, affine=False))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetDown(nn.Module):
    """Encoder block: Conv → Norm → LeakyReLU (stride=2 downsamples)."""
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block: TransposeConv → Norm → ReLU with skip connection."""
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


# ============================================================
# Generator – U-Net 256
# ============================================================

class UNetGenerator(nn.Module):
    """
    Input:  (B, 2, H, W)  – 1-ch IR concatenated with 1-ch edge map
    Output: (B, 3, H, W)  – synthetic visible image in [-1, 1]
    """

    def __init__(self, in_channels=2, out_channels=3, ngf=64):
        super().__init__()

        # Encoder (8 downsampling layers for 256→1)
        self.d1 = nn.Sequential(nn.Conv2d(in_channels, ngf, 4, 2, 1),
                                 nn.LeakyReLU(0.2, inplace=True))   # 128
        self.d2 = UNetDown(ngf,       ngf * 2)   # 64
        self.d3 = UNetDown(ngf * 2,   ngf * 4)   # 32
        self.d4 = UNetDown(ngf * 4,   ngf * 8)   # 16
        self.d5 = UNetDown(ngf * 8,   ngf * 8)   # 8
        self.d6 = UNetDown(ngf * 8,   ngf * 8)   # 4
        self.d7 = UNetDown(ngf * 8,   ngf * 8)   # 2
        self.d8 = UNetDown(ngf * 8,   ngf * 8, normalize=False)  # 1 (bottleneck)

        # Decoder (each in_c = decoder_out + skip)
        self.u1 = UNetUp(ngf * 8,     ngf * 8, dropout=True)   # 2  → concat → ngf*16
        self.u2 = UNetUp(ngf * 16,    ngf * 8, dropout=True)   # 4  → concat → ngf*16
        self.u3 = UNetUp(ngf * 16,    ngf * 8, dropout=True)   # 8  → concat → ngf*16
        self.u4 = UNetUp(ngf * 16,    ngf * 8)                 # 16 → concat → ngf*16
        self.u5 = UNetUp(ngf * 16,    ngf * 4)                 # 32 → concat → ngf*8
        self.u6 = UNetUp(ngf * 8,     ngf * 2)                 # 64 → concat → ngf*4
        self.u7 = UNetUp(ngf * 4,     ngf)                     # 128→ concat → ngf*2

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        # Decoder
        u1 = self.u1(d8, d7)
        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)
        return self.final(u7)


# ============================================================
# Discriminator – PatchGAN 70×70
# ============================================================

class PatchGANDiscriminator(nn.Module):
    """
    Input:  (B, in_c, H, W)   – concatenation of IR+edge (cond) + fake/real RGB
    Output: (B, 1, H', W')    – patch-level realness score
    """

    def __init__(self, in_channels=5, ndf=64):
        # in_channels = 1 (IR) + 1 (edge) + 3 (RGB) = 5
        super().__init__()
        self.model = nn.Sequential(
            # C64 (no norm on first layer)
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # C128
            ConvNormReLU(ndf,     ndf * 2, norm="batch"),
            # C256
            ConvNormReLU(ndf * 2, ndf * 4, norm="batch"),
            # C512 (stride=1)
            ConvNormReLU(ndf * 4, ndf * 8, stride=1, norm="batch"),
            # Output patch
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Loss Functions
# ============================================================

class GANLoss(nn.Module):
    """Flexible GAN loss: lsgan (MSE) or vanilla (BCE)."""

    def __init__(self, mode="lsgan", device="cpu"):
        super().__init__()
        self.mode = mode
        self.real_label = torch.tensor(1.0, device=device)
        self.fake_label = torch.tensor(0.0, device=device)
        if mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target(self, pred, is_real):
        label = self.real_label if is_real else self.fake_label
        return label.expand_as(pred)

    def forward(self, pred, is_real):
        return self.loss(pred, self.get_target(pred, is_real))


class EdgePreservingLoss(nn.Module):
    """
    Penalizes differences in edge magnitude between generated and target images.
    Uses Sobel gradients to measure edge sharpness.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels (applied per channel)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edge_magnitude(self, img):
        # img: (B, C, H, W) in [-1,1], convert to grayscale
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, fake, real):
        edge_fake = self._edge_magnitude(fake)
        edge_real = self._edge_magnitude(real)
        return F.l1_loss(edge_fake, edge_real)


class VGGPerceptualLoss(nn.Module):
    """
    Feature-level perceptual loss using VGG-16 relu2_2 and relu3_3.
    Input images should be in [-1, 1]; internally normalized to ImageNet stats.
    """

    def __init__(self, device="cpu"):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:9])    # relu2_2
        self.slice2 = nn.Sequential(*list(vgg.children())[9:16])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.to(device)
        # ImageNet normalization
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _normalize(self, x):
        x = (x + 1.0) / 2.0           # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        fake_n = self._normalize(fake)
        real_n = self._normalize(real)
        f1_fake = self.slice1(fake_n)
        f1_real = self.slice1(real_n)
        f2_fake = self.slice2(f1_fake)
        f2_real = self.slice2(f1_real)
        return F.l1_loss(f1_fake, f1_real) + F.l1_loss(f2_fake, f2_real)


# ============================================================
# Weight initialization
# ============================================================

def init_weights(net, init_type="normal", gain=0.02):
    def _init(m):
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            if init_type == "normal":
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif "BatchNorm2d" in cls:
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.constant_(m.bias, 0.0)
    net.apply(_init)
    return net
