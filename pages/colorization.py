"""pages/colorization.py — Main image colorization workspace."""
import os, glob, tempfile
import streamlit as st
import cv2, numpy as np
from PIL import Image

from components.metric_cards import section_title, info_box, warn_box, success_box, error_box
from components.image_viewer  import show_image, draw_pedestrian_boxes, encode_download
from services.preprocessing   import (uploaded_to_bgr, preprocess_for_model,
                                       apply_colormap, bgr_to_pil, gray_to_pil)
from services.inference        import run_inference, encode_png
from services.model_loader     import load_generator

CMAPS = {
    "Inferno": cv2.COLORMAP_INFERNO, "Jet": cv2.COLORMAP_JET,
    "Plasma":  cv2.COLORMAP_PLASMA,  "Hot": cv2.COLORMAP_HOT,
    "Winter":  cv2.COLORMAP_WINTER,  "Bone": cv2.COLORMAP_BONE,
}


def render(settings: dict):
    cfg        = settings["cfg"]
    ckpt_choice= settings["ckpt_choice"]
    img_size   = settings["img_size"]
    clahe_clip = settings["clahe_clip"]
    denoise_h  = settings["denoise_h"]
    cmap_name  = settings["cmap_name"]
    show_boxes = settings["show_boxes"]

    st.markdown("""
<div class="hero-banner">
  <div class="hero-title">Image Colorization Pipeline</div>
  <div class="hero-sub">
    Interactive execution of the edge-guided translation pipeline — from raw thermal IR
    to generated RGB output with intermediate structural representations.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Input mode ───────────────────────────────────────────────
    section_title("Image Input")
    input_mode = st.radio("Source", ["Upload image", "Select from dataset"],
                          horizontal=True, label_visibility="collapsed")

    bgr_ir = None

    if input_mode == "Upload image":
        up = st.file_uploader("Upload Thermal IR Image (JPG / PNG)",
                               type=["jpg", "jpeg", "png"])
        if up:
            bgr_ir = uploaded_to_bgr(up)
            h, w   = bgr_ir.shape[:2]
            size_kb = round(up.size / 1024, 1)
            success_box(
                f"Loaded: <b>{up.name}</b> &nbsp;|&nbsp; {w} × {h} px &nbsp;|&nbsp; {size_kb} KB"
            )
    else:
        ir_dir = cfg["dataset"].get("test_ir", "LLVIP/infrared/test")
        all_ir = sorted(glob.glob(os.path.join(ir_dir, "*.jpg")) +
                        glob.glob(os.path.join(ir_dir, "*.png")))
        if not all_ir:
            warn_box("No images found in the test IR directory.")
        else:
            names = [os.path.basename(p) for p in all_ir]
            choice = st.selectbox("Select image", names)
            path   = all_ir[names.index(choice)]
            bgr_ir = cv2.imread(path)
            # Thumbnail strip
            strip = all_ir[:8]
            s_cols = st.columns(len(strip))
            for col, p in zip(s_cols, strip):
                b = cv2.imread(p)
                if b is not None:
                    thumb = cv2.resize(b, (80, 80))
                    brd   = "3px solid #00D4B4" if p == path else "2px solid #1E3A5F"
                    col.markdown(
                        f"<div style='border:{brd};border-radius:6px;overflow:hidden'>",
                        unsafe_allow_html=True,
                    )
                    col.image(bgr_to_pil(thumb), width='stretch')
                    col.markdown("</div>", unsafe_allow_html=True)

    if bgr_ir is None:
        warn_box("Upload or select a thermal IR image above to begin.")
        return

    # ═══ Run preprocessing ══════════════════════════════════════
    with st.spinner("Running preprocessing pipeline…"):
        proc = preprocess_for_model(bgr_ir, size=img_size,
                                    denoise_h=denoise_h,
                                    clahe_clip=clahe_clip)

    st.markdown("---")

    # ── Stage 1: Original ────────────────────────────────────────
    section_title("Stage 1 — Original Thermal IR")
    info_box(
        "Raw thermal infrared image (single-channel intensity). Resolution: "
        f"<b>{bgr_ir.shape[1]} × {bgr_ir.shape[0]} px</b> → resized to "
        f"<b>{img_size} × {img_size}</b> for model input."
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        show_image(proc["gray_full"], caption="Full-resolution grayscale")
    with c2:
        st.markdown("**Image Statistics**")
        g = proc["gray_full"]
        st.markdown(f"""
| Property | Value |
|---|---|
| Width × Height | `{g.shape[1]} × {g.shape[0]}` |
| Mean intensity | `{g.mean():.1f}` |
| Std deviation  | `{g.std():.1f}` |
| Min / Max      | `{g.min()}` / `{g.max()}` |
| Dtype          | `{g.dtype}` |
""")

    st.markdown("---")

    # ── Stage 2: Preprocessing ───────────────────────────────────
    section_title("Stage 2 — Preprocessing")
    info_box(
        "Preprocessing sequence: <b>FastNLM Denoising</b> "
        f"(h={denoise_h}) → <b>CLAHE</b> (clip={clahe_clip}, grid=8×8) → "
        f"<b>Resize</b> to {img_size}×{img_size} → Normalization to <b>[-1, 1]</b>"
    )
    p1, p2, p3 = st.columns(3)
    p1.image(gray_to_pil(proc["gray_rs"]),  caption="Resized",   width='stretch')
    p2.image(gray_to_pil(proc["denoised"]), caption="Denoised",  width='stretch')
    p3.image(gray_to_pil(proc["clahe"]),    caption="CLAHE Enhanced", width='stretch')

    with st.expander("Pixel statistics after preprocessing"):
        sc1, sc2 = st.columns(2)
        sc1.markdown(f"""
| | Before | After CLAHE |
|---|---|---|
| Mean  | `{proc['gray_rs'].mean():.1f}` | `{proc['clahe'].mean():.1f}` |
| Std   | `{proc['gray_rs'].std():.1f}`  | `{proc['clahe'].std():.1f}` |
| Min   | `{proc['gray_rs'].min()}`      | `{proc['clahe'].min()}` |
| Max   | `{proc['gray_rs'].max()}`      | `{proc['clahe'].max()}` |
""")
        sc2.markdown("""
**Contrast Limited Adaptive Histogram Equalization (CLAHE):**
Enhances local contrast across thermal gradients without amplifying high-frequency noise,
providing clearer edge definitions for the structural channel.
""")

    st.markdown("---")

    # ── Stage 3: Edge extraction ─────────────────────────────────
    section_title("Stage 3 — Canny Edge Extraction")
    edge_info = proc["edge_info"]
    info_box(
        f"Gaussian blur (3×3) → Canny (thresholds: {settings.get('edge_low', 50)}, {settings.get('edge_high', 150)}) → "
        f"Ellipse dilation (3×3, 1 iter). "
        f"Edge density: <b>{edge_info['density_pct']}%</b>. "
        "Formulates the second input channel for conditional generation."
    )
    e1, e2, e3, e4 = st.columns(4)
    e1.image(gray_to_pil(proc["clahe"]),            caption="CLAHE IR",               width='stretch')
    e2.image(gray_to_pil(edge_info["raw_canny"]),   caption="Canny Edges",            width='stretch')
    e3.image(gray_to_pil(edge_info["dilated"]),     caption="Dilated Structural Map", width='stretch')
    e4.image(bgr_to_pil(edge_info["overlay"]),      caption="Edge Overlay on IR",     width='stretch')

    with st.expander("Edge extraction parameters"):
        st.markdown(f"""
| Step | Parameter | Value |
|---|---|---|
| Gaussian blur   | Kernel        | `(3, 3)` |
| Canny threshold | Low           | `50` |
| Canny threshold | High          | `150` |
| Dilation        | Kernel        | Ellipse `(3, 3)` |
| Dilation        | Iterations    | `1` |
| Output Range    | Tensor        | `[0, 1]` float32 |
""")

    st.markdown("---")

    # ── Stage 4: Combined input ──────────────────────────────────
    section_title("Stage 4 — 2-Channel Conditioned Input")
    info_box(
        "Tensor assembly: <b>Channel 0</b> (normalized thermal intensity [-1, 1]) + "
        "<b>Channel 1</b> (structural edge map [0, 1]). "
        "Provides dual structural and radiometric priors to the U-Net encoder."
    )
    i1, i2, i3 = st.columns(3)
    i1.image(gray_to_pil(proc["gray_rs"]),         caption="Channel 0: Thermal IR",      width='stretch')
    i2.image(gray_to_pil(edge_info["dilated"]),    caption="Channel 1: Canny Edge Map",  width='stretch')
    # Pseudo-colour blend to visualize both channels together
    blend_r = proc["gray_rs"].astype(np.float32) / 255.0
    blend_g = edge_info["dilated"].astype(np.float32) / 255.0
    pseudo  = np.stack([blend_g, blend_r, (blend_r * 0.5 + blend_g * 0.5)], axis=-1)
    pseudo  = (pseudo * 255).clip(0, 255).astype(np.uint8)
    i3.image(Image.fromarray(pseudo),             caption="Structural Overlay Visualization", width='stretch')

    st.markdown("---")

    # ── Stage 5: Pix2Pix inference ───────────────────────────────
    section_title("Stage 5 — Generator Inference")

    if ckpt_choice == "(none — enhancement only)":
        warn_box("No checkpoint selected. Choose a model checkpoint in the sidebar to run inference.")
        fake_bgr = None
    else:
        ckpt_path = os.path.join(cfg["paths"]["checkpoints"], ckpt_choice)
        try:
            with st.spinner("Running inference…"):
                G, device = load_generator(ckpt_path, cfg["model"]["ngf"])
                fake_bgr  = run_inference(G, device,
                                          proc["ir_tensor"],
                                          proc["edge_tensor"])
            success_box(f"Inference complete — Checkpoint: <b>{ckpt_choice}</b> | Device: <b>{device}</b>")
        except Exception as e:
            error_box(f"Inference error: {e}")
            fake_bgr = None

    # ── Results display ──────────────────────────────────────────
    if fake_bgr is not None:
        trad_bgr = apply_colormap(proc["clahe"], CMAPS[cmap_name])

        r1, r2, r3 = st.columns(3)
        r1.image(gray_to_pil(proc["gray_rs"]),    caption="Input Thermal IR",          width='stretch')
        r2.image(bgr_to_pil(trad_bgr),            caption=f"Baseline CLAHE ({cmap_name})", width='stretch')
        r3.image(bgr_to_pil(fake_bgr),            caption="Pix2Pix Generated Output",  width='stretch')

        # Pedestrian detection boxes
        if show_boxes:
            st.markdown("---")
            section_title("Pedestrian Candidate Localization (Edge Saliency)")
            info_box(
                "Saliency proxy: Connected vertical edge components (aspect ratio h > w, area ≥ 800 px²) "
                "isolated as pedestrian candidates to benchmark structural boundary clarity."
            )
            b1, b2 = st.columns(2)
            trad_ann, t_cnt = draw_pedestrian_boxes(trad_bgr)
            fake_ann, f_cnt = draw_pedestrian_boxes(fake_bgr)
            b1.image(bgr_to_pil(trad_ann), caption=f"Baseline — {t_cnt} candidates localized", width='stretch')
            b2.image(bgr_to_pil(fake_ann), caption=f"Pix2Pix — {f_cnt} candidates localized", width='stretch')

        # Download buttons
        st.markdown("---")
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button("Download Generated Output",
                            encode_download(fake_bgr),
                            "generated_rgb.png", "image/png")
        dl2.download_button("Download Baseline Image",
                            encode_download(trad_bgr),
                            "traditional.png", "image/png")
        # Side-by-side grid
        grid = np.concatenate([
            cv2.cvtColor(proc["gray_rs"], cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(edge_info["dilated"], cv2.COLOR_GRAY2BGR),
            trad_bgr, fake_bgr
        ], axis=1)
        dl3.download_button("Download Comparison Strip",
                            encode_download(grid),
                            "comparison_grid.png", "image/png")
    else:
        # Still show traditional
        trad_bgr = apply_colormap(proc["clahe"], CMAPS[cmap_name])
        st.markdown("**Baseline enhancement output:**")
        show_image(trad_bgr, caption=f"CLAHE + {cmap_name} Colormap")
        st.download_button("Download Baseline Image",
                           encode_download(trad_bgr),
                           "traditional.png", "image/png")
