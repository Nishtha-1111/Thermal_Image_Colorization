"""pages/comparison.py — Side-by-side visual comparison workspace."""
import os, glob
import streamlit as st
import cv2, numpy as np
from PIL import Image

from components.metric_cards import section_title, info_box, warn_box, success_box
from components.image_viewer  import bgr_to_pil, gray_to_pil, draw_pedestrian_boxes, encode_download
from components.charts        import metric_comparison_bar
from services.preprocessing   import (uploaded_to_bgr, preprocess_for_model,
                                       apply_colormap, bgr_to_pil as _b2p)
from services.inference        import run_inference
from services.model_loader     import load_generator
from services.evaluation_service import compute_all_metrics

CMAPS = {
    "Inferno": cv2.COLORMAP_INFERNO, "Jet": cv2.COLORMAP_JET,
    "Plasma":  cv2.COLORMAP_PLASMA,  "Hot": cv2.COLORMAP_HOT,
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
  <div class="hero-title">Visual & Quantitative Comparison</div>
  <div class="hero-sub">
    Side-by-side inspection across original thermal IR, classical enhancement, and
    Pix2Pix synthesis, with objective metrics computed against aligned ground-truth visible pairs.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────
    section_title("Input Pair Selection")
    input_mode = st.radio("Source", ["Upload files", "Select from dataset"],
                          horizontal=True, label_visibility="collapsed")

    bgr_ir = None
    bgr_gt = None

    if input_mode == "Upload files":
        c1, c2 = st.columns(2)
        with c1:
            up_ir = st.file_uploader("Thermal IR image", type=["jpg","jpeg","png"], key="cmp_ir")
            if up_ir:
                bgr_ir = uploaded_to_bgr(up_ir)
        with c2:
            up_gt = st.file_uploader("Ground-truth visible (optional)", type=["jpg","jpeg","png"], key="cmp_gt")
            if up_gt:
                bgr_gt = uploaded_to_bgr(up_gt)

    else:
        ir_dir  = cfg["dataset"].get("test_ir",  "LLVIP/infrared/test")
        vis_dir = cfg["dataset"].get("test_vis",  "LLVIP/visible/test")
        all_ir  = sorted(glob.glob(os.path.join(ir_dir,  "*.jpg")))
        if not all_ir:
            warn_box("No images found in test IR directory.")
        else:
            names  = [os.path.basename(p) for p in all_ir]
            choice = st.selectbox("Select image", names)
            path   = all_ir[names.index(choice)]
            bgr_ir = cv2.imread(path)
            stem   = os.path.splitext(os.path.basename(path))[0]
            for ext in (".jpg", ".png"):
                vp = os.path.join(vis_dir, stem + ext)
                if os.path.exists(vp):
                    bgr_gt = cv2.imread(vp)
                    success_box(f"Ground-truth visible loaded: <b>{stem}{ext}</b>")
                    break
            # Also check for pre-computed traditional result
            trad_path = os.path.join(cfg["paths"]["results_traditional"], stem + "_enhanced.jpg")
            if os.path.exists(trad_path):
                info_box(f"Pre-computed traditional result found: <b>{stem}_enhanced.jpg</b>")

    if bgr_ir is None:
        warn_box("Select or upload a thermal IR image to begin comparison.")
        return

    # ── Preprocess + generate ────────────────────────────────────
    with st.spinner("Processing…"):
        proc     = preprocess_for_model(bgr_ir, size=img_size,
                                         denoise_h=denoise_h,
                                         clahe_clip=clahe_clip)
        trad_bgr = apply_colormap(proc["clahe"], CMAPS.get(cmap_name, cv2.COLORMAP_INFERNO))

    fake_bgr = None
    if ckpt_choice != "(none — enhancement only)":
        ckpt_path = os.path.join(cfg["paths"]["checkpoints"], ckpt_choice)
        try:
            with st.spinner("Running Pix2Pix…"):
                G, device = load_generator(ckpt_path, cfg["model"]["ngf"])
                fake_bgr  = run_inference(G, device, proc["ir_tensor"], proc["edge_tensor"])
        except Exception as e:
            st.error(f"Model error: {e}")

    bgr_gt_rs = None
    if bgr_gt is not None:
        bgr_gt_rs = cv2.resize(bgr_gt, (img_size, img_size))

    # ── Build panel dict ─────────────────────────────────────────
    panels = {
        "Thermal IR Input":          cv2.cvtColor(proc["gray_rs"],    cv2.COLOR_GRAY2BGR),
        "CLAHE Enhanced":            cv2.cvtColor(proc["clahe"],      cv2.COLOR_GRAY2BGR),
        "Canny Edge Map":            cv2.cvtColor(proc["edge_info"]["dilated"], cv2.COLOR_GRAY2BGR),
        f"Baseline ({cmap_name})":   trad_bgr,
    }
    if fake_bgr is not None:
        panels["Pix2Pix Generated"] = fake_bgr
    if bgr_gt_rs is not None:
        panels["Ground Truth Visible"] = bgr_gt_rs

    # ── Full comparison strip ────────────────────────────────────
    st.markdown("---")
    section_title("Multi-Modal Comparison")

    n = len(panels)
    cols = st.columns(n)
    for col, (name, img) in zip(cols, panels.items()):
        disp = img
        if show_boxes and "Edge" not in name:
            disp, cnt = draw_pedestrian_boxes(img)
            col.image(bgr_to_pil(disp), caption=name.replace("\n", " "),
                      width='stretch')
            col.caption(f"{cnt} candidate regions")
        else:
            col.image(bgr_to_pil(img), caption=name.replace("\n", " "),
                      width='stretch')

    # ── Tabbed detail view ───────────────────────────────────────
    st.markdown("---")
    section_title("Detailed Pairwise Analysis")
    tab_names = ["IR vs Baseline", "IR vs Pix2Pix", "Baseline vs Pix2Pix",
                 "Edge Structural Maps", "Reconstruction vs Ground Truth"]
    tabs = st.tabs(tab_names)

    def side_by_side(img_a, label_a, img_b, label_b):
        ca, cb = st.columns(2)
        ca.image(bgr_to_pil(img_a), caption=label_a, width='stretch')
        cb.image(bgr_to_pil(img_b), caption=label_b, width='stretch')

    ir_bgr = cv2.cvtColor(proc["gray_rs"], cv2.COLOR_GRAY2BGR)

    with tabs[0]:
        side_by_side(ir_bgr, "Original Thermal IR", trad_bgr, f"Baseline ({cmap_name})")
    with tabs[1]:
        if fake_bgr is not None:
            side_by_side(ir_bgr, "Original Thermal IR", fake_bgr, "Pix2Pix Generated RGB")
        else:
            warn_box("Select a checkpoint in the sidebar to view model output.")
    with tabs[2]:
        if fake_bgr is not None:
            side_by_side(trad_bgr, f"Baseline ({cmap_name})", fake_bgr, "Pix2Pix Generated RGB")
        else:
            warn_box("Select a checkpoint in the sidebar to view model output.")
    with tabs[3]:
        edge_dil = proc["edge_info"]["dilated"]
        edge_raw = proc["edge_info"]["raw_canny"]
        e1, e2, e3 = st.columns(3)
        e1.image(gray_to_pil(edge_raw),          caption="Raw Canny Edges",           width='stretch')
        e2.image(gray_to_pil(edge_dil),          caption="Dilated Structural Tensor", width='stretch')
        e3.image(bgr_to_pil(proc["edge_info"]["overlay"]), caption="Edge Overlay", width='stretch')
    with tabs[4]:
        if bgr_gt_rs is not None:
            g_cols = st.columns(min(4, len(panels)))
            for col, (name, img) in zip(g_cols, panels.items()):
                col.image(bgr_to_pil(img), caption=name.replace("\n", " "),
                          width='stretch')
        else:
            warn_box("Upload or load a ground-truth visible image to compute comparisons.")

    # ── Quantitative metrics ─────────────────────────────────────
    if bgr_gt_rs is not None:
        st.markdown("---")
        section_title("Objective Metrics vs Ground Truth")

        methods_eval = {"Baseline": trad_bgr}
        if fake_bgr is not None:
            methods_eval["Pix2Pix"] = fake_bgr

        all_m = {}
        for mname, mimg in methods_eval.items():
            all_m[mname] = compute_all_metrics(mimg, bgr_gt_rs)

        # Metric cards
        for mname, metrics in all_m.items():
            st.markdown(f"**{mname} Evaluation**")
            m_cols = st.columns(len(metrics))
            for col, (key, val) in zip(m_cols, metrics.items()):
                better = "higher is better" if key in ("PSNR", "SSIM", "Edge F1") else "lower is better"
                disp   = f"{val:.4f}" if val is not None else "N/A"
                col.metric(f"{key} ({better})", disp)

        # Bar charts
        if len(all_m) > 1:
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                metric_comparison_bar(all_m, "PSNR", "PSNR Comparison (dB)")
            with bc2:
                metric_comparison_bar(all_m, "SSIM", "SSIM Comparison")
            with bc3:
                metric_comparison_bar(all_m, "Edge F1", "Edge F1 Comparison")

        # Detailed table
        with st.expander("Full metrics summary"):
            import pandas as pd
            rows = []
            for mname, metrics in all_m.items():
                row = {"Method": mname}
                row.update({k: (f"{v:.4f}" if v else "N/A") for k, v in metrics.items()})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index("Method"), width='stretch')

    # ── Downloads ────────────────────────────────────────────────
    st.markdown("---")
    section_title("Export Outputs")
    dl_cols = st.columns(3)
    dl_cols[0].download_button("Export Baseline PNG", encode_download(trad_bgr),
                                "traditional.png", "image/png")
    if fake_bgr is not None:
        dl_cols[1].download_button("Export Pix2Pix PNG", encode_download(fake_bgr),
                                    "pix2pix.png", "image/png")
        # Comparison grid
        grid_imgs = [ir_bgr, trad_bgr, fake_bgr]
        if bgr_gt_rs is not None:
            grid_imgs.append(bgr_gt_rs)
        grid = np.concatenate(grid_imgs, axis=1)
        dl_cols[2].download_button("Export Comparison Strip", encode_download(grid),
                                    "comparison_grid.png", "image/png")
