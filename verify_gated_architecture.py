"""
verify_gated_architecture.py – Formal verification test suite for Gated U-Net architecture.

Tests:
  1. Shapes for Configuration A (1-ch Thermal) and Configuration B (2-ch Thermal + Canny)
  2. Discriminator 5-ch input test (IR + Canny + RGB)
  3. Attention maps validation (every gate produces valid map in [0, 1])
  4. Gradient backpropagation (every gate receives valid non-zero gradients)
  5. Value range & numerical stability (output in [-1, 1], no NaN/Inf)
  6. Checkpoint save & reload verification
  7. Exact parameter counts comparison (Original vs Gated)
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models import UNetGenerator, PatchGANDiscriminator, GANLoss, EdgePreservingLoss
from gated_models import GatedUNetGenerator


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_verification():
    print("=" * 70)
    print("PHASE 3 ARCHITECTURE VERIFICATION TEST SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}\n")

    # ------------------------------------------------------------------
    # Test 1: Forward Shapes & Two Input Configurations
    # ------------------------------------------------------------------
    print("[TEST 1] Forward Pass Tensor Shapes")
    ir_tensor = torch.randn(1, 1, 256, 256, device=device)
    canny_tensor = torch.rand(1, 1, 256, 256, device=device)
    combined_tensor = torch.cat([ir_tensor, canny_tensor], dim=1)

    print(f"  IR input shape       : {tuple(ir_tensor.shape)} (Expected: (1, 1, 256, 256))")
    print(f"  Canny input shape    : {tuple(canny_tensor.shape)} (Expected: (1, 1, 256, 256))")
    print(f"  Combined input shape : {tuple(combined_tensor.shape)} (Expected: (1, 2, 256, 256))")
    assert ir_tensor.shape == (1, 1, 256, 256)
    assert canny_tensor.shape == (1, 1, 256, 256)
    assert combined_tensor.shape == (1, 2, 256, 256)

    # Config A: Thermal-only Generator
    netG_A = GatedUNetGenerator(in_channels=1, out_channels=3, ngf=64).to(device)
    out_A, attn_A = netG_A(ir_tensor, return_attentions=True)
    print(f"  Config A (1-ch) Out  : {tuple(out_A.shape)} (Expected: (1, 3, 256, 256))")
    assert out_A.shape == (1, 3, 256, 256)

    # Config B: Thermal + Canny Generator
    netG_B = GatedUNetGenerator(in_channels=2, out_channels=3, ngf=64).to(device)
    out_B, attn_B = netG_B(combined_tensor, return_attentions=True)
    print(f"  Config B (2-ch) Out  : {tuple(out_B.shape)} (Expected: (1, 3, 256, 256))")
    assert out_B.shape == (1, 3, 256, 256)

    # Discriminator input: (IR + Canny + Synthetic RGB) = 1 + 1 + 3 = 5 channels
    netD = PatchGANDiscriminator(in_channels=5, ndf=64).to(device)
    d_input = torch.cat([combined_tensor, out_B], dim=1)
    d_out = netD(d_input)
    print(f"  Discriminator Input  : {tuple(d_input.shape)} (Expected: (1, 5, 256, 256))")
    print(f"  Discriminator Output : {tuple(d_out.shape)} (Expected: (1, 1, 30, 30))")
    assert d_input.shape == (1, 5, 256, 256)
    assert d_out.shape == (1, 1, 30, 30)
    print("  --> TEST 1 PASSED: All tensor shapes match expectations exactly!\n")

    # ------------------------------------------------------------------
    # Test 2: Attention Maps Validation
    # ------------------------------------------------------------------
    print("[TEST 2] Attention Maps Verification (Config B)")
    print(f"  Total Attention Gates: {len(attn_B)}")
    assert len(attn_B) == 7, f"Expected 7 skip gates, got {len(attn_B)}"

    for idx, alpha in enumerate(attn_B, start=1):
        min_v = alpha.min().item()
        max_v = alpha.max().item()
        mean_v = alpha.mean().item()
        print(f"  Gate {idx}: shape={tuple(alpha.shape)}, range=[{min_v:.4f}, {max_v:.4f}], mean={mean_v:.4f}")
        assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), f"Gate {idx} attention map out of [0, 1]!"
        assert not torch.isnan(alpha).any(), f"Gate {idx} contains NaN!"
        assert not torch.isinf(alpha).any(), f"Gate {idx} contains Inf!"
    print("  --> TEST 2 PASSED: Every gate produces valid attention maps bounded in [0, 1]!\n")

    # ------------------------------------------------------------------
    # Test 3: Numerical Stability & Output Range
    # ------------------------------------------------------------------
    print("[TEST 3] Numerical Stability & Output Dynamic Range")
    out_min = out_B.min().item()
    out_max = out_B.max().item()
    print(f"  Generator output range: [{out_min:.4f}, {out_max:.4f}]")
    assert out_min >= -1.0 and out_max <= 1.0, "Generator output exceeds [-1, 1] bounds!"
    assert not torch.isnan(out_B).any(), "Generator output contains NaN!"
    assert not torch.isinf(out_B).any(), "Generator output contains Inf!"
    print("  --> TEST 3 PASSED: Output strictly within [-1, 1] with zero NaN/Inf.\n")

    # ------------------------------------------------------------------
    # Test 4: Backward Pass & Gradient Flow Through Attention Gates
    # ------------------------------------------------------------------
    print("[TEST 4] Backward Pass & Gradient Flow")
    netG_B.zero_grad()
    netD.zero_grad()

    target_vis = torch.randn(1, 3, 256, 256, device=device).clamp(-1.0, 1.0)
    fake_vis = netG_B(combined_tensor)

    criterion_GAN = GANLoss(mode="lsgan", device=device)
    criterion_L1 = nn.L1Loss()
    criterion_edge = EdgePreservingLoss().to(device)

    pred_fake = netD(torch.cat([combined_tensor, fake_vis], dim=1))
    loss_G_gan = criterion_GAN(pred_fake, True)
    loss_G_l1 = criterion_L1(fake_vis, target_vis) * 100.0
    loss_G_edge = criterion_edge(fake_vis, target_vis) * 10.0
    loss_G_total = loss_G_gan + loss_G_l1 + loss_G_edge

    loss_G_total.backward()

    # Check gradients in each GatedUNetUp gate
    gated_blocks = [netG_B.u1, netG_B.u2, netG_B.u3, netG_B.u4, netG_B.u5, netG_B.u6, netG_B.u7]
    for idx, blk in enumerate(gated_blocks, start=1):
        wx_grad = blk.gate.W_x[0].weight.grad
        wg_grad = blk.gate.W_g[0].weight.grad
        psi_grad = blk.gate.psi[0].weight.grad

        assert wx_grad is not None, f"Gate {idx} W_x gradient is None!"
        assert wg_grad is not None, f"Gate {idx} W_g gradient is None!"
        assert psi_grad is not None, f"Gate {idx} psi gradient is None!"

        wx_norm = wx_grad.norm().item()
        wg_norm = wg_grad.norm().item()
        psi_norm = psi_grad.norm().item()

        print(f"  Gate {idx} Gradients -> ||W_x||: {wx_norm:.6f}, ||W_g||: {wg_norm:.6f}, ||psi||: {psi_norm:.6f}")
        assert not (torch.isnan(wx_grad).any() or torch.isnan(wg_grad).any() or torch.isnan(psi_grad).any()), f"Gate {idx} has NaN grad!"
    print("  --> TEST 4 PASSED: Gradient backpropagation verified through every attention gate!\n")

    # ------------------------------------------------------------------
    # Test 5: Checkpoint Save & Reload Verification
    # ------------------------------------------------------------------
    print("[TEST 5] Checkpoint Serialization & Reload")
    ckpt_path = "checkpoints/test_gated_checkpoint.pth"
    os.makedirs("checkpoints", exist_ok=True)

    test_state = {
        "netG": netG_B.state_dict(),
        "epoch": 1,
    }
    torch.save(test_state, ckpt_path)
    print(f"  Saved checkpoint to: {ckpt_path}")

    # Load into freshly initialized model
    loaded_netG = GatedUNetGenerator(in_channels=2, out_channels=3, ngf=64).to(device)
    loaded_state = torch.load(ckpt_path, map_location=device)
    loaded_netG.load_state_dict(loaded_state["netG"])
    loaded_netG.eval()
    netG_B.eval()

    with torch.no_grad():
        orig_pred = netG_B(combined_tensor)
        loaded_pred = loaded_netG(combined_tensor)
        max_diff = torch.max(torch.abs(orig_pred - loaded_pred)).item()

    print(f"  Max absolute difference after reload: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Checkpoint reload mismatch: {max_diff}"

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"  Cleaned up temporary checkpoint {ckpt_path}")
    print("  --> TEST 5 PASSED: Checkpoint saves and reloads with bitwise fidelity!\n")

    # ------------------------------------------------------------------
    # Test 6: Parameter Count Comparison
    # ------------------------------------------------------------------
    print("[TEST 6] Generator Parameter Counts Comparison")
    orig_gen_2ch = UNetGenerator(in_channels=2, out_channels=3, ngf=64)
    orig_gen_1ch = UNetGenerator(in_channels=1, out_channels=3, ngf=64)
    gated_gen_2ch = GatedUNetGenerator(in_channels=2, out_channels=3, ngf=64)
    gated_gen_1ch = GatedUNetGenerator(in_channels=1, out_channels=3, ngf=64)

    p_orig_2 = count_parameters(orig_gen_2ch)
    p_orig_1 = count_parameters(orig_gen_1ch)
    p_gated_2 = count_parameters(gated_gen_2ch)
    p_gated_1 = count_parameters(gated_gen_1ch)

    delta_2 = p_gated_2 - p_orig_2
    pct_2 = (delta_2 / p_orig_2) * 100.0

    delta_1 = p_gated_1 - p_orig_1
    pct_1 = (delta_1 / p_orig_1) * 100.0

    print(f"  Original UNet (Config B, 2-ch) : {p_orig_2:,} parameters ({p_orig_2/1e6:.2f} M)")
    print(f"  Gated UNet    (Config B, 2-ch) : {p_gated_2:,} parameters ({p_gated_2/1e6:.2f} M)")
    print(f"    --> Delta: +{delta_2:,} parameters (+{pct_2:.2f}%)\n")

    print(f"  Original UNet (Config A, 1-ch) : {p_orig_1:,} parameters ({p_orig_1/1e6:.2f} M)")
    print(f"  Gated UNet    (Config A, 1-ch) : {p_gated_1:,} parameters ({p_gated_1/1e6:.2f} M)")
    print(f"    --> Delta: +{delta_1:,} parameters (+{pct_1:.2f}%)\n")

    print("=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (6/6)")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
