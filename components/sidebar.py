"""components/sidebar.py — Shared sidebar rendered on every page."""
import os, glob
import streamlit as st
import torch

from services.model_loader import load_config, list_checkpoints, device_summary


def render_sidebar() -> dict:
    """
    Render the shared sidebar.
    Returns a dict of all user-selected settings.
    """
    with st.sidebar:
        # Identity and page navigation are rendered in app.py

        # ── Load config ──────────────────────────────────────
        try:
            cfg = load_config("configs/config.yaml")
        except Exception:
            cfg = {
                "model": {"ngf": 64},
                "paths": {
                    "checkpoints": "checkpoints",
                    "results_eval": "results/eval",
                    "results_traditional": "results/traditional",
                    "results_pix2pix": "results/pix2pix",
                    "logs": "logs",
                },
                "dataset": {
                    "image_size": [256, 256],
                    "train_ir": "LLVIP/infrared/train",
                    "train_vis": "LLVIP/visible/train",
                    "test_ir": "LLVIP/infrared/test",
                    "test_vis": "LLVIP/visible/test",
                    "annotations": "LLVIP/Annotations",
                },
                "training": {"lambda_L1": 100.0, "lambda_edge": 10.0,
                             "lambda_perceptual": 5.0, "num_epochs": 100},
            }

        ckpt_dir   = cfg["paths"]["checkpoints"]
        ckpt_files = list_checkpoints(ckpt_dir)

        # ── Model checkpoint ─────────────────────────────────
        st.markdown("#### Model Checkpoint")
        ckpt_names = ["(none — enhancement only)"] + \
                     [os.path.basename(c) for c in ckpt_files]
        ckpt_choice = st.selectbox("Checkpoint", ckpt_names,
                                   index=min(len(ckpt_names)-1, len(ckpt_names)-1),
                                   label_visibility="collapsed")

        # Device pill
        dev = device_summary()
        if dev["available"]:
            st.markdown(
                f"<span class='status-pill status-ok'>GPU: {dev['name']}"
                f" · {dev['vram_gb']} GB</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='status-pill status-warn'>Device: CPU</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Image settings ───────────────────────────────────
        st.markdown("#### Input Resolution")
        img_size = st.select_slider(
            "Model input size", options=[128, 256, 512], value=256
        )

        # ── Enhancement settings ─────────────────────────────
        st.markdown("#### Preprocessing Parameters")
        clahe_clip = st.slider("CLAHE clip limit", 1.0, 8.0, 2.0, 0.5)
        denoise_h  = st.slider("Denoise strength", 0, 30, 10, 1)

        CMAPS_NAMES = ["Inferno", "Jet", "Plasma", "Hot", "Winter", "Bone"]
        cmap_name   = st.selectbox("Colormap", CMAPS_NAMES)

        # ── Edge settings ────────────────────────────────────
        st.markdown("#### Edge Extraction")
        edge_low  = st.slider("Canny low",  10, 100, 50, 5)
        edge_high = st.slider("Canny high", 50, 250, 150, 10)

        # ── Visual options ───────────────────────────────────
        st.markdown("#### Overlays")
        show_boxes = st.checkbox("Pedestrian detection boxes", value=True)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem;color:#64748B;text-align:center'>"
            "LLVIP · Pix2Pix GAN · 100 Epochs<br>"
            "<a href='https://arxiv.org/abs/2108.10831' "
            "style='color:#3B82F6'>arXiv:2108.10831</a>"
            "</div>",
            unsafe_allow_html=True,
        )

    return {
        "cfg":         cfg,
        "ckpt_choice": ckpt_choice,
        "ckpt_files":  ckpt_files,
        "img_size":    img_size,
        "clahe_clip":  clahe_clip,
        "denoise_h":   denoise_h,
        "cmap_name":   cmap_name,
        "edge_low":    edge_low,
        "edge_high":   edge_high,
        "show_boxes":  show_boxes,
    }
