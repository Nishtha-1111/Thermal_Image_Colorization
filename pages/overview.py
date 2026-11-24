"""pages/overview.py — Project overview dashboard."""
import os, glob
import streamlit as st
from PIL import Image
import cv2

from components.metric_cards import (
    metric_row, section_title, info_box, pipeline_step, status_pill
)
from services.model_loader import list_checkpoints, checkpoint_info, device_summary
from services.preprocessing import bgr_to_pil


def render(settings: dict):
    cfg = settings["cfg"]
    ckpt_files = settings["ckpt_files"]

    # ── Hero ────────────────────────────────────────────────────
    st.markdown("""
<div class="hero-banner">
  <div style="margin-bottom:0.6rem">
    <span class="hero-badge">Research Project</span>
    <span class="hero-badge">Pix2Pix GAN</span>
    <span class="hero-badge">LLVIP Dataset</span>
  </div>
  <div class="hero-title">Edge-Guided Thermal Image Colorization</div>
  <div class="hero-sub">
    An intelligent infrared-to-visible image translation system that combines thermal
    information with Canny edge maps to generate visually informative RGB images —
    emphasizing pedestrian boundaries through a custom Sobel-gradient loss.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Key metrics ──────────────────────────────────────────────
    section_title("Project Statistics")
    n_epochs = cfg.get("training", {}).get("num_epochs", 100)
    ngf      = cfg.get("model", {}).get("ngf", 64)
    metric_row([
        {"label": "Dataset",         "value": "LLVIP",   "unit": "Visible–IR pairs"},
        {"label": "Training Pairs",  "value": "12,025",  "unit": "images"},
        {"label": "Test Pairs",      "value": "3,463",   "unit": "images"},
        {"label": "Epochs Trained",  "value": n_epochs,  "unit": "epochs"},
    ])
    metric_row([
        {"label": "Generator Filters", "value": ngf,              "unit": "ngf base"},
        {"label": "Input Channels",    "value": "2",              "unit": "IR + Edge"},
        {"label": "Output Channels",   "value": "3",              "unit": "RGB"},
        {"label": "Checkpoints",       "value": len(ckpt_files),  "unit": "saved"},
    ])

    # ── System status ────────────────────────────────────────────
    section_title("System Status")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Model Checkpoint**")
        best = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
        if os.path.exists(best):
            info = checkpoint_info(best)
            status_pill(f"best_model.pth ({info['size_mb']} MB)", "ok")
        elif ckpt_files:
            info = checkpoint_info(ckpt_files[-1])
            status_pill(f"{os.path.basename(ckpt_files[-1])} ({info['size_mb']} MB)", "warn")
        else:
            status_pill("No checkpoint found", "err")

    with col2:
        st.markdown("**Compute Device**")
        dev = device_summary()
        if dev["available"]:
            status_pill(f"CUDA: {dev['name']} ({dev['vram_gb']} GB)", "ok")
        else:
            status_pill("Device: CPU", "warn")

    with col3:
        st.markdown("**Evaluation Artifacts**")
        eval_dir  = cfg["paths"]["results_eval"]
        pix_json  = os.path.join(eval_dir, "metrics_pix2pix.json")
        trad_json = os.path.join(eval_dir, "metrics_traditional.json")
        if os.path.exists(pix_json) and os.path.exists(trad_json):
            status_pill("Full benchmarks logged", "ok")
        elif os.path.exists(pix_json):
            status_pill("Pix2Pix evaluated", "warn")
        else:
            status_pill("Pending evaluation run", "warn")

    # ── Pipeline workflow ────────────────────────────────────────
    st.markdown("---")
    section_title("End-to-End Pipeline")

    p_cols = st.columns(7)
    steps = [
        ("01", "Thermal IR"),
        ("02", "Denoise & CLAHE"),
        ("03", "Canny Edge"),
        ("04", "Concat [IR | Edge]"),
        ("05", "U-Net Generator"),
        ("06", "Generated RGB"),
        ("07", "Evaluation"),
    ]
    for col, (step_num, label) in zip(p_cols, steps):
        col.markdown(
            f"<div style='text-align:center;padding:0.7rem 0.5rem;"
            f"background:#0F1D30;border:1px solid #1E3A5F;"
            f"border-radius:8px;'>"
            f"<div style='font-size:0.8rem;font-weight:700;color:#00D4B4;letter-spacing:0.05em'>{step_num}</div>"
            f"<div style='font-size:0.75rem;color:#E2E8F0;margin-top:4px;font-weight:500'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Visual Pipeline Preview (Real Sample) ───────────────────
    preview_ir_path = os.path.join(cfg["dataset"].get("test_ir", "LLVIP/infrared/test"), "190001.jpg")
    preview_gen_path = os.path.join(cfg["paths"].get("results_pix2pix", "results/pix2pix"), "test_gen", "190001.jpg")

    if os.path.exists(preview_ir_path):
        ir_sample = cv2.imread(preview_ir_path)
        if ir_sample is not None:
            gray = cv2.cvtColor(ir_sample, cv2.COLOR_BGR2GRAY) if ir_sample.ndim == 3 else ir_sample
            gray_res = cv2.resize(gray, (256, 256))
            blurred = cv2.GaussianBlur(gray_res, (3, 3), 0)
            canny = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edge_map = cv2.dilate(canny, kernel, iterations=1)

            st.markdown("<div style='margin-top:1.2rem; margin-bottom:0.6rem; font-weight:600; font-size:0.88rem; color:#8BA5C2;'>Sample Pipeline Output (Test Image 190001)</div>", unsafe_allow_html=True)
            v_cols = st.columns(3)
            with v_cols[0]:
                st.image(gray_res, caption="Input Thermal IR", width='stretch')
            with v_cols[1]:
                st.image(edge_map, caption="Extracted Canny Edge Map", width='stretch')
            with v_cols[2]:
                if os.path.exists(preview_gen_path):
                    gen_img = cv2.imread(preview_gen_path)
                    st.image(bgr_to_pil(gen_img), caption="Pix2Pix Generated Output", width='stretch')
                else:
                    st.image(gray_res, caption="Generated Output (Ready)", width='stretch')

    # ── Training summary (from config) ──────────────────────────
    st.markdown("---")
    section_title("Training Configuration")
    train_cfg = cfg.get("training", {})
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Optimization**")
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| LR (G) | `{train_cfg.get('lr_G', 0.0002)}` |
| LR (D) | `{train_cfg.get('lr_D', 0.0002)}` |
| β₁ | `{train_cfg.get('beta1', 0.5)}` |
| β₂ | `{train_cfg.get('beta2', 0.999)}` |
| Batch size | `{train_cfg.get('batch_size', 8)}` |
""")

    with col_b:
        st.markdown("**Loss Weights**")
        st.markdown(f"""
| Loss | Weight |
|---|---|
| GAN (LSGAN) | ×1 |
| L1 Pixel | ×{int(train_cfg.get('lambda_L1', 100))} |
| Sobel Edge | ×{int(train_cfg.get('lambda_edge', 10))} |
| VGG Perceptual | ×{int(train_cfg.get('lambda_perceptual', 5))} |
""")

    with col_c:
        st.markdown("**Schedule**")
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Epochs | `{n_epochs}` |
| LR decay start | Epoch `{train_cfg.get('decay_epoch', 50)}` |
| Save interval | `{train_cfg.get('save_every', 10)}` epochs |
| Resolution | `256 × 256` |
| Normalization | `{cfg.get('model',{}).get('norm','batch')}` |
""")

    # ── Sample generated images ──────────────────────────────────
    section_title("Training Progression Samples")
    pix_dir  = cfg["paths"]["results_pix2pix"]
    samples  = sorted(glob.glob(os.path.join(pix_dir, "epoch*.jpg")))
    # Pick a spread across training epochs
    picks = []
    if samples:
        idxs = [0, len(samples)//5, 2*len(samples)//5,
                3*len(samples)//5, 4*len(samples)//5, len(samples)-1]
        picks = [samples[i] for i in idxs if i < len(samples)]

    if picks:
        info_box(
            "Visual check during training: <b>IR input | Edge map | Generated RGB | Ground-truth visible</b> "
            "— sampled from Epoch 1 through Epoch 100."
        )
        s_cols = st.columns(len(picks))
        for col, path in zip(s_cols, picks):
            bgr = cv2.imread(path)
            if bgr is not None:
                name = os.path.basename(path)
                ep   = name.split("_")[0].replace("epoch", "Ep ")
                col.image(bgr_to_pil(bgr), caption=ep, width='stretch')
    else:
        st.info("No sample images found in `results/pix2pix/`.")

    # ── Traditional results preview ──────────────────────────────
    section_title("Traditional Enhancement Baselines")
    trad_dir = cfg["paths"]["results_traditional"]
    trad_enh = sorted(glob.glob(os.path.join(trad_dir, "*_enhanced.jpg")))[:6]
    if trad_enh:
        t_cols = st.columns(min(6, len(trad_enh)))
        for col, path in zip(t_cols, trad_enh):
            bgr = cv2.imread(path)
            if bgr is not None:
                stem = os.path.basename(path).replace("_enhanced.jpg", "")
                col.image(bgr_to_pil(bgr), caption=stem, width='stretch')
    else:
        st.info("No traditional results found in `results/traditional/`.")
