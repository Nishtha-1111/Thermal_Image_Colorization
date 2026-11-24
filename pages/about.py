"""pages/about.py — Project description and research context."""
import streamlit as st
from components.metric_cards import section_title, info_box


def render(settings: dict):
    cfg = settings["cfg"]

    st.markdown("""
<div class="hero-banner">
  <div class="hero-title">Project Overview & Background</div>
  <div class="hero-sub">
    Research context, architectural rationale, dataset specifications, and experimental roadmap.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Problem statement ────────────────────────────────────────
    section_title("Problem Formulation")
    st.markdown("""
Thermal infrared cameras capture scenes based on **radiant heat emission** rather than reflected
ambient light. This makes thermal imaging essential in night surveillance, automotive driver
assistance systems (ADAS), and search-and-rescue operations where visible spectrum sensors fail.

However, raw thermal data presents key perceptual bottlenecks:
- **Monochromatic representations:** Intensity reflects temperature variations rather than surface chromaticity.
- **Absence of chromatic cues:** Distinguishing fine semantic boundaries (e.g. clothing vs background) is difficult for human operators.
- **Interpretability degradation:** Low spatial texture hampers rapid pedestrian and obstacle localization.

**Objective:** Map single-channel thermal infrared images into visually interpretable 3-channel RGB imagery
while explicitly enforcing **pedestrian silhouette and structural boundary fidelity**.
""")

    # ── Proposed approach ────────────────────────────────────────
    section_title("Technical Approach")
    info_box(
        "<b>Key Insight:</b> Thermal radiance gradients correlate strongly with physical object silhouettes. "
        "Extracting explicit edge boundaries via Canny filters and injecting them as an auxiliary input channel "
        "provides a structural inductive bias, guiding the generator to preserve object boundaries without "
        "having to infer edge topologies from scratch."
    )
    st.markdown("""
**Core Contributions over Standard Pix2Pix:**

1. **Dual-Channel Conditioned Input:** Concatenates normalized thermal IR (Channel 0) with a morphologically dilated Canny edge map (Channel 1), providing dual radiometric and structural cues.
2. **Sobel Gradient Consistency Loss:** Introduces a custom differential operator loss that penalizes boundary blurring by matching first-order horizontal and vertical spatial derivatives.
3. **Structurally-Aware Evaluation:** Benchmarks pedestrian boundary preservation against ground-truth Pascal VOC annotations using an edge-saliency IoU heuristic.
""")

    # ── Research objectives ──────────────────────────────────────
    section_title("Experimental Objectives")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
- Enhance visual interpretability of nocturnal thermal scenes
- Enforce pedestrian silhouette and high-frequency edge retention
- Synthesize coherent 3-channel visible approximations
- Benchmark against classical histogram equalization baselines
""")
    with cols[1]:
        st.markdown("""
- Quantify pixel-level reconstruction fidelity (PSNR, SSIM)
- Quantify edge preservation across modalities (Sobel F1)
- Verify structural proxy correlation with bounding box ground truth
- Establish baseline metrics for future object detection downstream tasks
""")

    # ── Dataset ──────────────────────────────────────────────────
    st.markdown("---")
    section_title("LLVIP Dataset Specifications")
    st.markdown("""
**LLVIP: Low-Light Visible-Infrared Paired Dataset**
([Jia et al., 2021](https://arxiv.org/abs/2108.10831))

A benchmark dataset of strictly spatiotemporally registered visible and thermal infrared night street scenes captured via co-aligned sensor rigs.
""")
    d1, d2, d3 = st.columns(3)
    d1.markdown("""
| Split | Pairs |
|---|---|
| Training | 12,025 |
| Testing  |  3,463 |
| **Total** | **15,488** |
""")
    d2.markdown("""
| Dimension | Specification |
|---|---|
| Raw Resolution | 1024 × 1280 px |
| Model Resolution | 256 × 256 px |
| Annotations | Pascal VOC XML |
| Primary Class | Pedestrian |
""")
    d3.markdown("""
**Dataset Properties:**
- Rigid co-axial mounting minimizing parallax
- Complex night illumination scenarios
- High-density urban pedestrian scenes
- Strictly synchronous acquisition
""")

    # ── Methodology ──────────────────────────────────────────────
    st.markdown("---")
    section_title("Model & Optimization Protocol")
    train_cfg = cfg.get("training", {})
    st.markdown(f"""
| Component | Hyperparameter / Setting |
|---|---|
| Generator Architecture | U-Net 256 (8 DownBlocks, 8 UpBlocks, skip connections) |
| Discriminator Architecture | PatchGAN 70×70 (Markovian receptive field) |
| Adversarial Loss Formulation | Least Squares GAN (LSGAN) |
| Optimizer | Adam (β₁ = 0.5, β₂ = 0.999) |
| Learning Rate | {train_cfg.get('lr_G', 0.0002)} (Generator), {train_cfg.get('lr_D', 0.0002)} (Discriminator) |
| Batch Size | {train_cfg.get('batch_size', 8)} |
| Training Epochs | {train_cfg.get('num_epochs', 100)} (Linear decay from epoch {train_cfg.get('decay_epoch', 50)}) |
| L1 Pixel Loss Weight | ×{int(train_cfg.get('lambda_L1', 100))} |
| Sobel Edge Loss Weight | ×{int(train_cfg.get('lambda_edge', 10))} |
| VGG Perceptual Loss Weight | ×{int(train_cfg.get('lambda_perceptual', 5))} |
| Data Augmentation | Random horizontal flipping |
| Layer Normalization | {cfg.get('model',{}).get('norm','batch').capitalize()} Normalization |
""")

    # ── Limitations ───────────────────────────────────────────────
    st.markdown("---")
    section_title("Limitations & Analysis Scope")
    st.markdown("""
- **Ill-Posed Chromatic Synthesis:** Mapping single-channel radiation to 3-channel visible color is mathematically ill-posed; generated colors reflect statistical training priors rather than true daytime reflectance.
- **Resolution Downsampling:** Downsampling $1024 \\times 1280$ images to $256 \\times 256$ for GAN training limits extreme distant pedestrian resolution.
- **Structural Proxy vs Trained Detector:** The edge-saliency evaluation metric provides an objective measure of boundary sharpness, but does not replace inference from a full detector such as YOLOv8.
""")

    # ── Future work ───────────────────────────────────────────────
    section_title("Research Roadmap")
    st.markdown("""
| Priority | Focus Area | Proposed Approach |
|---|---|---|
| High | Downstream Object Detection | Benchmark mAP on colorized imagery using fine-tuned YOLOv8 |
| High | Semantic Edge Weighting | Upweight loss selectively on pedestrian bounding box regions |
| Medium | High-Resolution Synthesis | 512×512 synthesis with multi-scale discriminator architectures |
| Medium | Perceptual Feature Shifts | Contrastive DINO / CLIP perceptual embeddings for cross-modal loss |
| Low | Frequency-Domain Supervision | Wavelet or Fourier transform losses for fine edge preservation |
""")

    # ── Citation ─────────────────────────────────────────────────
    st.markdown("---")
    section_title("Citations & References")
    st.code("""
@misc{jia2021llvip,
  title   = {LLVIP: A Visible-infrared Paired Dataset for Low-light Vision},
  author  = {Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  year    = {2021},
  url     = {https://arxiv.org/abs/2108.10831}
}

@inproceedings{isola2017pix2pix,
  title     = {Image-to-Image Translation with Conditional Adversarial Networks},
  author    = {Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A.},
  booktitle = {CVPR},
  year      = {2017}
}

@inproceedings{mao2017lsgan,
  title   = {Least Squares Generative Adversarial Networks},
  author  = {Mao, Xudong et al.},
  booktitle = {ICCV},
  year    = {2017}
}
""", language="bibtex")
