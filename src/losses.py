"""
src/losses.py – Loss functions for Thermal-to-Visible colorization GANs.

Losses included:
  1. GANLoss: LSGAN (MSE) or Vanilla (BCE) for Generator and Discriminator
  2. L1Loss: Pixel-level L1 distance comparing generated image to visible ground truth
  3. SobelLoss (EdgePreservingLoss): L1 distance on Sobel gradient magnitudes
  4. VGGPerceptualLoss: Multi-scale perceptual loss on frozen VGG16 features (relu2_2, relu3_3)
  5. CompositeGeneratorLoss: Unified objective with configurable use_sobel_loss toggle:
       L_G = L_GAN + 100 * L_L1 + 10 * L_Sobel + 5 * L_VGG  (when use_sobel_loss=True)
       L_G = L_GAN + 100 * L_L1 + 5 * L_VGG                 (when use_sobel_loss=False)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ============================================================
# 1. GAN Loss (LSGAN / Vanilla)
# ============================================================

class GANLoss(nn.Module):
    """
    Adversarial GAN loss for both Discriminator and Generator.
    Defaults to LSGAN (Least Squares GAN, MSE loss against target labels 1.0/0.0).
    """

    def __init__(self, mode: str = "lsgan", device: torch.device = "cpu"):
        super().__init__()
        self.mode = mode
        self.device = device
        self.real_label = torch.tensor(1.0, device=device)
        self.fake_label = torch.tensor(0.0, device=device)

        if mode == "lsgan":
            self.loss_fn = nn.MSELoss()
        elif mode == "vanilla":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode '{mode}' is not implemented.")

    def get_target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        label = self.real_label if is_real else self.fake_label
        return label.expand_as(pred)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        """
        Calculates loss:
          - Discriminator on real: pred=D(real), is_real=True  -> MSE(pred, 1.0)
          - Discriminator on fake: pred=D(fake), is_real=False -> MSE(pred, 0.0)
          - Generator objective:   pred=D(fake), is_real=True  -> MSE(pred, 1.0)
        """
        target = self.get_target(pred, is_real)
        return self.loss_fn(pred, target)


# ============================================================
# 2. Pixel L1 Loss
# ============================================================

class L1ReconstructionLoss(nn.Module):
    """
    Direct pixel-level L1 reconstruction loss between generated visible
    RGB image and target visible ground truth.
    Both images expected in [-1, 1].
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(fake, real)


# ============================================================
# 3. Sobel Edge-Preserving Loss
# ============================================================

class SobelLoss(nn.Module):
    """
    Penalizes differences in edge gradient magnitudes between generated
    and target images using standard 3x3 Sobel kernels.
    """

    def __init__(self):
        super().__init__()
        # Horizontal and vertical Sobel convolution kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _gradient_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        # Convert RGB in [-1, 1] to luminance grayscale
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        mag_fake = self._gradient_magnitude(fake)
        mag_real = self._gradient_magnitude(real)
        return F.l1_loss(mag_fake, mag_real)


# Backward-compatible alias
EdgePreservingLoss = SobelLoss


# ============================================================
# 4. VGG-16 Perceptual Loss
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual feature distance using frozen ImageNet pre-trained VGG-16.
    Extracts features at relu2_2 (layer 9) and relu3_3 (layer 16).
    Input images in [-1, 1] are normalized to ImageNet mean and std.
    """

    def __init__(self, device: torch.device = "cpu"):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:9])    # up to relu2_2
        self.slice2 = nn.Sequential(*list(vgg.children())[9:16])  # up to relu3_3

        for p in self.parameters():
            p.requires_grad = False

        self.to(device)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from [-1, 1] to [0, 1], then normalize by ImageNet statistics
        x_01 = (x + 1.0) / 2.0
        return (x_01 - self.mean) / self.std

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_n = self._normalize(fake)
        real_n = self._normalize(real)

        f1_fake = self.slice1(fake_n)
        f1_real = self.slice1(real_n)
        f2_fake = self.slice2(f1_fake)
        f2_real = self.slice2(f1_real)

        loss_relu2_2 = F.l1_loss(f1_fake, f1_real)
        loss_relu3_3 = F.l1_loss(f2_fake, f2_real)
        return loss_relu2_2 + loss_relu3_3


# ============================================================
# 5. Composite Generator Loss
# ============================================================

class CompositeGeneratorLoss(nn.Module):
    """
    Composite generator loss manager:
      L_G = L_GAN + 100 * L_L1 + (10 * L_Sobel if use_sobel_loss else 0) + 5 * L_VGG
    """

    def __init__(
        self,
        use_sobel_loss: bool = True,
        lambda_gan: float = 1.0,
        lambda_l1: float = 100.0,
        lambda_sobel: float = 10.0,
        lambda_vgg: float = 5.0,
        gan_mode: str = "lsgan",
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.use_sobel_loss = use_sobel_loss
        self.lambda_gan = lambda_gan
        self.lambda_l1 = lambda_l1
        self.lambda_sobel = lambda_sobel
        self.lambda_vgg = lambda_vgg

        self.criterion_gan = GANLoss(mode=gan_mode, device=device)
        self.criterion_l1 = L1ReconstructionLoss()
        self.criterion_sobel = SobelLoss().to(device)
        self.criterion_vgg = VGGPerceptualLoss(device=device)

    def forward(
        self,
        pred_fake: torch.Tensor,
        fake_vis: torch.Tensor,
        real_vis: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite generator loss.

        Returns:
          total_loss (torch.Tensor): Backpropagatable loss tensor
          loss_dict (dict): Numerical breakdown for logging
        """
        loss_gan = self.criterion_gan(pred_fake, True)
        loss_l1 = self.criterion_l1(fake_vis, real_vis)
        loss_vgg = self.criterion_vgg(fake_vis, real_vis)

        total_loss = (
            self.lambda_gan * loss_gan
            + self.lambda_l1 * loss_l1
            + self.lambda_vgg * loss_vgg
        )

        loss_dict = {
            "loss_gan": loss_gan.item(),
            "loss_l1": loss_l1.item(),
            "loss_vgg": loss_vgg.item(),
            "loss_sobel": 0.0,
        }

        if self.use_sobel_loss:
            loss_sobel = self.criterion_sobel(fake_vis, real_vis)
            total_loss = total_loss + self.lambda_sobel * loss_sobel
            loss_dict["loss_sobel"] = loss_sobel.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing losses.py on device: {device}")

    B = 2
    fake_img = torch.randn(B, 3, 256, 256, device=device).clamp(-1.0, 1.0)
    real_img = torch.randn(B, 3, 256, 256, device=device).clamp(-1.0, 1.0)
    pred_disc = torch.randn(B, 1, 30, 30, device=device)

    # 1. Test with use_sobel_loss = True
    comp_loss_on = CompositeGeneratorLoss(use_sobel_loss=True, device=device)
    loss_on, dict_on = comp_loss_on(pred_disc, fake_img, real_img)
    print("\n[With Sobel Loss (use_sobel_loss=True)]")
    for k, v in dict_on.items():
        print(f"  {k:12s}: {v:.6f}")
    assert dict_on["loss_sobel"] > 0.0

    # 2. Test with use_sobel_loss = False
    comp_loss_off = CompositeGeneratorLoss(use_sobel_loss=False, device=device)
    loss_off, dict_off = comp_loss_off(pred_disc, fake_img, real_img)
    print("\n[Without Sobel Loss (use_sobel_loss=False)]")
    for k, v in dict_off.items():
        print(f"  {k:12s}: {v:.6f}")
    assert dict_off["loss_sobel"] == 0.0

    # Verify difference is strictly 10 * Sobel
    expected_diff = 10.0 * dict_on["loss_sobel"]
    actual_diff = dict_on["loss_total"] - dict_off["loss_total"]
    diff_error = abs(actual_diff - expected_diff)
    print(f"\nDifference (ON - OFF): {actual_diff:.6f} (Expected: {expected_diff:.6f}, Err: {diff_error:.2e})")
    assert diff_error < 1e-4

    print("\nAll losses and use_sobel_loss toggling verified successfully!")
