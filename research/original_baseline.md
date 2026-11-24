# Original Baseline Model Documentation

This document records the exact configuration, architecture, training protocol, and evaluation baseline for the original edge-guided Pix2Pix thermal-to-visible colorization network on the LLVIP dataset. This serves as the benchmark reference before any architectural modifications (e.g., Gated U-Net or attention mechanisms) are introduced.

---

## 1. Model Properties

| Property | Existing Model Specification |
| :--- | :--- |
| **Generator Architecture** | U-Net (`unet_256`, 8 encoder downsampling stages, 7 decoder upsampling stages with skip connections) |
| **Generator Input** | 2 channels (`[B, 2, 256, 256]`: Channel 0 = 1-ch IR thermal image, Channel 1 = 1-ch edge map) |
| **Generator Output** | 3 channels (`[B, 3, 256, 256]`: Synthetic visible RGB image normalized to $[-1, 1]$ via `Tanh`) |
| **Discriminator Architecture** | 70 × 70 PatchGAN (`patch70`, 4 Conv layers with LeakyReLU and BatchNorm) |
| **Discriminator Input** | 5 channels (`[B, 5, 256, 256]`: Condition 2 channels [IR + Edge] + 3 channels [Real or Fake RGB]) |
| **GAN Loss** | LSGAN (Least Squares GAN, MSE loss against target labels 1.0 real / 0.0 fake) |
| **L1 Weight ($\lambda_{L1}$)** | 100.0 |
| **Sobel Edge Weight ($\lambda_{\text{edge}}$)** | 10.0 |
| **VGG Perceptual Weight ($\lambda_{\text{perceptual}}$)** | 5.0 |
| **Base Filters** | 64 (`ngf = 64` for generator, `ndf = 64` for discriminator) |
| **Image Resolution** | 256 × 256 pixels |
| **Training Epochs** | 100 epochs (linear learning rate decay starting from epoch 50) |

---

## 2. Detailed Training Configuration & Loss Formulation

### 2.1 Optimization Parameters
- **Optimizer**: Adam ($\beta_1 = 0.5, \beta_2 = 0.999$) for both Generator and Discriminator
- **Learning Rate**: $\eta_G = 0.0002$, $\eta_D = 0.0002$
- **LR Scheduling**: Constant for epochs 1–50; linearly decayed to 0 over epochs 51–100
- **Batch Size**: 8
- **Normalization**: Batch Normalization (`BatchNorm2d`)
- **Dropout**: Dropout ($p = 0.5$) applied in the first three decoder blocks (`u1`, `u2`, `u3`)

### 2.2 Objective Function
The composite loss used during generator training is:
$$\mathcal{L}_G = \mathcal{L}_{\text{LSGAN}}(G) + \lambda_{L1} \mathcal{L}_{L1}(G(x), y) + \lambda_{\text{edge}} \mathcal{L}_{\text{Sobel}}(G(x), y) + \lambda_{\text{perceptual}} \mathcal{L}_{\text{VGG}}(G(x), y)$$

Where:
1. **$\mathcal{L}_{\text{LSGAN}}$**:
   $$\mathcal{L}_{\text{LSGAN}} = \mathbb{E}_{x}\left[(D(x, G(x)) - 1)^2\right]$$
2. **$\mathcal{L}_{L1}$**:
   $$\mathcal{L}_{L1} = \mathbb{E}_{x, y}\left[\|y - G(x)\|_1\right]$$
3. **$\mathcal{L}_{\text{Sobel}}$ (Edge-Preserving Loss)**:
   Measures the L1 difference between the gradient magnitude maps of the synthesized and target visible images:
   $$\nabla I = \sqrt{(I * K_x)^2 + (I * K_y)^2 + 10^{-8}}$$
   $$\mathcal{L}_{\text{Sobel}} = \mathbb{E}_{x, y}\left[\|\nabla G(x) - \nabla y\|_1\right]$$
   where $K_x, K_y$ are standard 3×3 Sobel differentiation kernels applied across luminance.
4. **$\mathcal{L}_{\text{VGG}}$ (Perceptual Loss)**:
   L1 feature distance computed at `relu2_2` and `relu3_3` layers of an ImageNet pre-trained VGG-16 backbone.

---

## 3. Dataset Specification

- **Dataset**: LLVIP (Low-Light Visible-Infrared Pairs)
- **Train split**: 12,025 aligned infrared-visible image pairs (`LLVIP/infrared/train`, `LLVIP/visible/train`)
- **Test split**: 3,463 aligned infrared-visible image pairs (`LLVIP/infrared/test`, `LLVIP/visible/test`)
- **Native Resolution**: 1024 × 1280
- **Training/Testing Resolution**: Resized to 256 × 256

---

## 4. Evaluation Protocol

All quantitative evaluations follow the official script `phase2_eval.py` executed across the entire LLVIP test set ($N = 3,463$ image pairs):

1. **Resolution & Interpolation**:
   - Both predicted and ground-truth visible images are resized to $(256, 256)$ using `cv2.INTER_AREA`.
2. **Color Space**:
   - Converted to single-channel 8-bit grayscale via `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`.
3. **Peak Signal-to-Noise Ratio (PSNR)**:
   - Evaluated on float64 values over dynamic range $[0, 255]$:
     $$\text{MSE} = \frac{1}{HW}\sum_{i=1}^H \sum_{j=1}^W (I_{\text{pred}}(i,j) - I_{\text{gt}}(i,j))^2$$
     $$\text{PSNR} = 10 \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$$
4. **Structural Similarity Index (SSIM)**:
   - Evaluated using `skimage.metrics.structural_similarity` on grayscale float64 images with `data_range=255`.
5. **Macro Canny Edge F1 Score**:
   - Gaussian blur applied with kernel size $(3, 3)$ and $\sigma = 0$.
   - Canny edge detection computed with thresholds $(50, 150)$.
   - Morphological dilation applied using an elliptical structuring element $(3, 3)$, 1 iteration.
   - True Positives ($\text{TP} = |E_{\text{pred}} \cap E_{\text{gt}}|$), False Positives ($\text{FP} = |E_{\text{pred}} \setminus E_{\text{gt}}|$), False Negatives ($\text{FN} = |E_{\text{gt}} \setminus E_{\text{pred}}|$).
   - Precision: $P = \frac{\text{TP}}{\text{TP} + \text{FP} + 10^{-8}}$
   - Recall: $R = \frac{\text{TP}}{\text{TP} + \text{FN} + 10^{-8}}$
   - Image F1: $F_1 = \frac{2 \cdot P \cdot R}{P + R + 10^{-8}}$
   - Macro average aggregated across all $N = 3,463$ images.

---

## 5. Original Evaluation Results

The baseline evaluation results on the full LLVIP test set ($N = 3,463$ images) are:

| Method | Test Samples ($N$) | PSNR (dB) | SSIM | Macro Canny Edge F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Original Edge-Guided Pix2Pix (Baseline)** | **3,463** | **13.8644** | **0.2659** | **0.2359** |
| *Traditional Enhanced (CLAHE)* | 3,463 | 11.5290 | 0.2026 | 0.3171 |
| *Traditional Edge-Overlay* | 3,463 | 11.4007 | 0.2002 | 0.3197 |

*Summary results stored in `results/evaluation_final/summary_metrics.json` and `results/eval/metrics_pix2pix.json`.*

---

## 6. Preserved Baseline Assets & Backup Locations

1. **Git Branch**:
   - `original-pix2pix` (clean commit preserving the baseline state)
2. **Thermal_Research Backup Structure**:
   - `Thermal_Research/original_pix2pix/src/` — Models, dataset loader, training, evaluation, inference
   - `Thermal_Research/original_pix2pix/checkpoints/best_model.pth` — 100-epoch trained model checkpoint (686.3 MB)
   - `Thermal_Research/original_pix2pix/results/` — Baseline evaluation JSON metrics and CSV logs
   - `Thermal_Research/original_pix2pix/app.py`, `pages/`, `components/`, `services/` — Complete Streamlit interactive demo
   - `Thermal_Research/original_pix2pix/phase2_eval.py` & `verify_claims.py` — Evaluation harness and verification scripts
