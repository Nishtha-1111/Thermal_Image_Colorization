# Thermal Image Colorization using Gated U-Net

A PyTorch deep learning framework for translating nighttime thermal infrared imagery into realistic full-color visible (RGB) images in real time.

---

## Project Overview

In low-light and nighttime environments, standard optical sensors fail, whereas long-wave infrared (thermal) cameras clearly detect human subjects via emitted body heat. However, thermal sensors produce single-channel intensity maps lacking natural color and chromatic detail.

This project implements a generative deep learning pipeline to reconstruct three-channel RGB imagery from single-channel thermal inputs:

```
[Thermal Input (1-channel)] ---> [Gated U-Net Generator] ---> [Colorized RGB Output (3-channel)]
```

---

## Motivation and Methodology

- **The Problem:** Standard U-Net architectures connect the encoder directly to the decoder via unweighted skip connections. In thermal-to-visible translation, this indiscriminately propagates thermal background noise (such as residual ground heat and warm engines) into the synthesis path, causing spatial blur and color bleed artifacts around pedestrians.
- **The Solution:** We introduce **Spatial Attention Gates** into the skip connections. These gates utilize high-level decoder features as gating signals to selectively suppress irrelevant thermal background clutter and highlight salient pedestrian silhouettes prior to feature concatenation.
- **Supervision:** The network is trained with a composite structural loss combining pixel-level L1 loss, Multi-Scale SSIM (MS-SSIM), Sobel gradient edge loss, and color distribution regularization.

---

## Key Benchmark Results

The architecture was evaluated on nighttime pedestrian benchmarks:

1. **LLVIP Dataset (3,463 withheld test images):**
   - **+6.70% improvement in structural similarity (SSIM)** compared to standard Pix2Pix.
   - **+3.68% sharper pedestrian boundaries (Edge F1 score)**.
   - Requires only **2.1% parameter overhead** (55.5M vs 54.4M).
   - Achieves real-time throughput of **~24 FPS** on an NVIDIA RTX 3050 GPU.

2. **KAIST Multispectral Dataset (1,000 zero-shot test images):**
   - Evaluated without fine-tuning (zero-shot transfer).
   - Achieved an **+8.98% improvement in SSIM** over the baseline model, demonstrating generalization across different sensors and optical characteristics.

---

## Project Structure

```text
├── configs/                  # Experiment configuration files (YAML)
│   ├── E1_baseline.yaml      # Baseline U-Net (L1 loss)
│   ├── E2_sobel.yaml         # Baseline U-Net + Sobel edge loss
│   ├── E3_gated.yaml         # Gated U-Net (L1 loss)
│   └── E4_proposed.yaml      # Proposed Gated U-Net + Multi-Scale Structural loss
├── datasets/                 # Dataset split indices
│   └── llvip_pairs.csv
├── results/                  # Benchmark evaluation summaries
│   ├── llvip_evaluation_summary.json
│   └── kaist_zero_shot_summary.json
├── src/                      # Core implementation
│   ├── attention_gate.py     # Additive Spatial Attention Gate module
│   ├── dataset.py            # Dataset loading and transforms
│   ├── evaluate_kaist.py     # KAIST zero-shot evaluation pipeline
│   ├── evaluate_models.py    # LLVIP evaluation pipeline (PSNR, SSIM, Edge F1)
│   ├── gated_models.py       # Proposed Gated U-Net generator
│   ├── losses.py             # Multi-scale structural and edge losses
│   ├── models.py             # Baseline U-Net and PatchGAN discriminator
│   └── train_experiment.py   # Training script with validation and checkpointing
├── requirements.txt          # Dependencies
└── README.md
```

---

## Installation and Usage

### 1. Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Evaluate Models on LLVIP

Evaluate the models on the 3,463 withheld LLVIP test images to report PSNR, SSIM, Edge F1, and latency:

```bash
python src/evaluate_models.py
```

### 3. Zero-Shot Evaluation on KAIST

Evaluate cross-dataset generalization on KAIST:

```bash
python src/evaluate_kaist.py --kaist_dir KAIST
```

### 4. Training from Scratch

To train the proposed Gated U-Net model:

```bash
python src/train_experiment.py --exp E4
```

To resume training from an existing checkpoint:

```bash
python src/train_experiment.py --exp E4 --resume
```
