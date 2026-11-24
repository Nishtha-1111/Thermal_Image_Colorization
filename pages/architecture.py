"""pages/architecture.py — Model architecture visualization."""
import streamlit as st
from components.metric_cards import section_title, info_box


def render(settings: dict):
    cfg = settings["cfg"]
    ngf = cfg.get("model", {}).get("ngf", 64)
    ndf = cfg.get("model", {}).get("ndf", 64)
    norm = cfg.get("model", {}).get("norm", "batch")

    st.markdown("""
<div class="hero-banner">
  <div class="hero-title">Model Architecture</div>
  <div class="hero-sub">
    Technical specifications of the Edge-Guided Pix2Pix network —
    Generator, PatchGAN Discriminator, and objective loss formulations.
  </div>
</div>""", unsafe_allow_html=True)

    # ── High-level overview ──────────────────────────────────────
    section_title("System Dataflow")

    col_diag, col_desc = st.columns([1.4, 1])
    with col_diag:
        st.markdown(f"""
<div class="arch-block">
  Thermal IR Image  (1024 × 1280)
           │
  ┌────────▼────────┐
  │  Resize & Norm  │  → {settings['img_size']}×{settings['img_size']}, range [-1, 1]
  └────────┬────────┘
           │
  ┌────────▼──────────────┐
  │  Canny Edge Extract   │  → Gaussian blur → Canny(50,150) → Dilate
  │  compute_edge_map()   │  → float32 edge map  [0, 1]
  └────────┬──────────────┘
           │
           ├──── IR tensor  (1,1,H,W) [-1,1]
           └──── Edge tensor (1,1,H,W) [0,1]
                       │
           ┌───────────▼──────────────────┐
           │  torch.cat([IR, Edge], dim=1) │  → (1, 2, H, W)
           └───────────┬──────────────────┘
                       │
  ┌────────────────────▼──────────────────────┐
  │         U-Net Generator (256)              │
  │  Encoder: 8 × DownBlock (stride-2 conv)   │
  │  Bottleneck: 1×1 spatial  ({ngf*8} ch)        │
  │  Decoder: 8 × UpBlock (transposed conv)   │
  │  Skip connections: d_i → u_(8-i)          │
  │  Dropout on top 3 decoder blocks (p=0.5)  │
  │  Final: Tanh → (1, 3, H, W) in [-1, 1]   │
  └────────────────────┬──────────────────────┘
                       │
           ┌───────────▼──────────┐
           │  Generated RGB image │  → rescale [0,255] uint8
           └───────────┬──────────┘
                       │
  ┌────────────────────▼──────────────────────────┐
  │             Training Losses                    │
  │  • LSGAN Adversarial   ×1                     │
  │  • L1 Pixel            ×{int(cfg.get('training',{}).get('lambda_L1',100))}                    │
  │  • Sobel Edge-Preserving ×{int(cfg.get('training',{}).get('lambda_edge',10))}                  │
  │  • VGG Perceptual      ×{int(cfg.get('training',{}).get('lambda_perceptual',5))}                     │
  └───────────────────────────────────────────────┘
</div>""", unsafe_allow_html=True)

    with col_desc:
        st.markdown("#### Key Design Choices")
        st.markdown("""
**Why 2-channel input?**
Standard Pix2Pix receives only the IR image (1 ch).
Adding the Canny edge map (1 ch) gives the generator
explicit structural guidance — it "knows" where
pedestrian boundaries are before colorization.

**Why edge loss?**
The Sobel-gradient loss penalises blurring of edges
in the generated image, encouraging preservation of
pedestrian silhouettes.

**Why LSGAN?**
Least-Squares GAN is more stable than vanilla BCE
GAN and reduces mode collapse.

**Why VGG loss?**
Pixel-level L1 alone produces blurry outputs.
VGG perceptual loss matches high-level feature
distributions, improving texture quality.
""")

    # ── Generator detail ─────────────────────────────────────────
    st.markdown("---")
    section_title("Generator — U-Net 256")
    info_box(
        f"<b>UNetGenerator</b> — in_channels=2, out_channels=3, ngf={ngf}, norm={norm}. "
        "Skip connections from every encoder layer to the symmetric decoder layer."
    )

    gen_cols = st.columns(2)
    with gen_cols[0]:
        st.markdown("**Encoder (8 downsampling blocks)**")
        enc_data = [
            ("d1", f"Conv(2→{ngf},  4×4 s2)",       f"{ngf}",   "128²", "No norm, LeakyReLU(0.2)"),
            ("d2", f"Down({ngf}→{ngf*2})",            f"{ngf*2}", "64²",  "BatchNorm, LeakyReLU(0.2)"),
            ("d3", f"Down({ngf*2}→{ngf*4})",          f"{ngf*4}", "32²",  "BatchNorm, LeakyReLU(0.2)"),
            ("d4", f"Down({ngf*4}→{ngf*8})",          f"{ngf*8}", "16²",  "BatchNorm, LeakyReLU(0.2)"),
            ("d5", f"Down({ngf*8}→{ngf*8})",          f"{ngf*8}", "8²",   "BatchNorm, LeakyReLU(0.2)"),
            ("d6", f"Down({ngf*8}→{ngf*8})",          f"{ngf*8}", "4²",   "BatchNorm, LeakyReLU(0.2)"),
            ("d7", f"Down({ngf*8}→{ngf*8})",          f"{ngf*8}", "2²",   "BatchNorm, LeakyReLU(0.2)"),
            ("d8", f"Down({ngf*8}→{ngf*8})",          f"{ngf*8}", "1²",   "No norm — bottleneck"),
        ]
        st.markdown("| Layer | Operation | Channels | Spatial | Note |")
        st.markdown("|---|---|---|---|---|")
        for row in enc_data:
            st.markdown(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")

    with gen_cols[1]:
        st.markdown("**Decoder (8 upsampling blocks with skip connections)**")
        dec_data = [
            ("u1", f"Up({ngf*8}→{ngf*8})",             f"{ngf*16}", "2²",   "Dropout(0.5) + skip d7"),
            ("u2", f"Up({ngf*16}→{ngf*8})",            f"{ngf*16}", "4²",   "Dropout(0.5) + skip d6"),
            ("u3", f"Up({ngf*16}→{ngf*8})",            f"{ngf*16}", "8²",   "Dropout(0.5) + skip d5"),
            ("u4", f"Up({ngf*16}→{ngf*8})",            f"{ngf*16}", "16²",  "skip d4"),
            ("u5", f"Up({ngf*16}→{ngf*4})",            f"{ngf*8}",  "32²",  "skip d3"),
            ("u6", f"Up({ngf*8}→{ngf*2})",             f"{ngf*4}",  "64²",  "skip d2"),
            ("u7", f"Up({ngf*4}→{ngf})",               f"{ngf*2}",  "128²", "skip d1"),
            ("out",f"ConvT({ngf*2}→3, 4×4 s2) + Tanh","3",          "256²", "RGB output [-1,1]"),
        ]
        st.markdown("| Layer | Operation | Out Ch | Spatial | Note |")
        st.markdown("|---|---|---|---|---|")
        for row in dec_data:
            st.markdown(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")

    # ── Discriminator detail ─────────────────────────────────────
    st.markdown("---")
    section_title("Discriminator — PatchGAN 70×70")
    info_box(
        f"<b>PatchGANDiscriminator</b> — in_channels=5 (IR+Edge+RGB), ndf={ndf}. "
        "Classifies overlapping 70×70 patches as real/fake, not the entire image. "
        "Conditioning: the discriminator sees [IR | Edge | RGB] so it evaluates "
        "realism given the input."
    )
    disc_data = [
        (f"Conv(5→{ndf}, 4×4 s2)",      "No norm",      f"{ndf}",   "128²"),
        (f"Conv({ndf}→{ndf*2}, 4×4 s2)","BatchNorm",    f"{ndf*2}", "64²"),
        (f"Conv({ndf*2}→{ndf*4}, 4×4 s2)","BatchNorm",  f"{ndf*4}", "32²"),
        (f"Conv({ndf*4}→{ndf*8}, 4×4 s1)","BatchNorm",  f"{ndf*8}", "31²"),
        (f"Conv({ndf*8}→1,  4×4 s1)",    "None",         "1",         "30²  patch score"),
    ]
    st.markdown("| Layer | Norm | Channels | Spatial |")
    st.markdown("|---|---|---|---|")
    for row in disc_data:
        st.markdown(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} |")

    # ── Loss functions detail ────────────────────────────────────
    st.markdown("---")
    section_title("Objective Loss Formulations")

    l1, l2 = st.columns(2)
    with l1:
        st.markdown("""
**GAN Loss (LSGAN)**
```
L_GAN = MSE(D(IR,Edge,fake), 1)   # generator
      + 0.5*(MSE(D(real),1) + MSE(D(fake),0))  # discriminator
```
*Least-Squares GAN for training stability.*

**L1 Pixel Loss (×100)**
```
L_L1 = mean(|fake - real|)
```
*Global pixel-level fidelity. High weight → sharp outputs.*
""")
    with l2:
        st.markdown("""
**Sobel Edge-Preserving Loss (×10)**
```python
gray = 0.299*R + 0.587*G + 0.114*B
Gx = Sobel_x(gray);  Gy = Sobel_y(gray)
edge_mag = sqrt(Gx² + Gy² + ε)
L_edge = L1(edge_mag(fake), edge_mag(real))
```
*Penalises blurring of edges. Custom to this project.*

**VGG Perceptual Loss (×5)**
```
L_vgg = L1(VGG.relu2_2(fake), VGG.relu2_2(real))
      + L1(VGG.relu3_3(fake), VGG.relu3_3(real))
```
*Feature-level similarity. Improves texture quality.*
""")

    st.markdown("""
---
**Total Generator Loss:**

$$\\mathcal{L}_G = \\mathcal{L}_{GAN} + 100 \\cdot \\mathcal{L}_{L1} + 10 \\cdot \\mathcal{L}_{Edge} + 5 \\cdot \\mathcal{L}_{VGG}$$
""")

    # ── Parameter count ──────────────────────────────────────────
    st.markdown("---")
    section_title("Model Parameter Counts")
    try:
        import sys, torch
        from pathlib import Path
        sys.path.insert(0, str(Path("src")))
        from models import UNetGenerator, PatchGANDiscriminator
        G = UNetGenerator(2, 3, ngf)
        D = PatchGANDiscriminator(5, ndf)
        g_params = sum(p.numel() for p in G.parameters()) / 1e6
        d_params = sum(p.numel() for p in D.parameters()) / 1e6
        total    = g_params + d_params
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Generator Params",     f"{g_params:.2f} M")
        pc2.metric("Discriminator Params", f"{d_params:.2f} M")
        pc3.metric("Total Params",         f"{total:.2f} M")
    except Exception as e:
        st.info(f"Could not count parameters: {e}")
