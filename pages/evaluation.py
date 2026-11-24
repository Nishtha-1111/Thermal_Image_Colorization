"""pages/evaluation.py — Quantitative evaluation dashboard."""
import os, json
import streamlit as st
import pandas as pd

from components.metric_cards  import section_title, info_box, warn_box
from components.charts        import (loss_curves, batch_loss_curves,
                                       metric_comparison_bar, metrics_radar,
                                       psnr_ssim_scatter, histogram_metric)
from services.evaluation_service import (load_eval_results, parse_train_log,
                                          sample_batch_losses)


def render(settings: dict):
    cfg = settings["cfg"]

    st.markdown("""
<div class="hero-banner">
  <div class="hero-title">Evaluation & Benchmarks</div>
  <div class="hero-sub">
    Quantitative validation across the LLVIP test set (3,463 pairs) —
    reconstruction fidelity, structural edge consistency, and training convergence.
  </div>
</div>""", unsafe_allow_html=True)

    eval_dir = cfg["paths"]["results_eval"]
    log_path = os.path.join(cfg["paths"]["logs"], "train.log")

    # ── Load saved results ───────────────────────────────────────
    results = load_eval_results(eval_dir)

    # ── Summary metric cards ─────────────────────────────────────
    section_title("Test-Set Performance Summary")

    if not results:
        warn_box(
            "No evaluation results found. Run:<br>"
            "<code>python src/evaluate.py --config configs/config.yaml "
            "--mode pix2pix --infer</code><br>"
            "<code>python src/evaluate.py --config configs/config.yaml "
            "--mode traditional</code>"
        )
    else:
        for mode_key, label in [("pix2pix", "Pix2Pix (Edge-Guided GAN)"),
                                 ("traditional", "Classical Baseline (CLAHE)")]:
            if mode_key not in results:
                continue
            s = results[mode_key]["summary"]
            st.markdown(f"#### {label}")
            m_cols = st.columns(5)
            m_cols[0].metric("Evaluated Pairs", s.get("n_images", "—"))
            m_cols[1].metric("Mean PSNR",        f"{s.get('mean_psnr', 0):.4f} dB")
            m_cols[2].metric("Mean SSIM",        f"{s.get('mean_ssim', 0):.4f}")
            # support both old key (mean_recall) and new key (mean_ped_recall)
            ped_recall = s.get("mean_ped_recall", s.get("mean_recall", 0))
            m_cols[3].metric("Mean Ped. Recall", f"{ped_recall:.4f}")
            m_cols[4].metric("Mode",             s.get("mode", "—").capitalize())

        # Honest disclaimer
        info_box(
            "<b>Evaluation Methodology:</b> PSNR and SSIM measure pixel and structural similarity to "
            "ground-truth visible images. The pedestrian recall metric uses an edge-saliency "
            "heuristic against annotated VOC bounding boxes as a <b>structural boundary proxy</b>, "
            "providing an objective indicator of silhouette preservation."
        )

    # ── Method comparison ────────────────────────────────────────
    if len(results) >= 2:
        st.markdown("---")
        section_title("Comparative Analysis")

        summaries = {
            label: results[key]["summary"]
            for key, label in [("pix2pix", "Pix2Pix"), ("traditional", "Baseline")]
            if key in results
        }
        # Reformat for charts
        chart_data = {
            m: {"PSNR": s["mean_psnr"], "SSIM": s["mean_ssim"],
                "Ped Recall": s.get("mean_ped_recall", s.get("mean_recall", 0))}
            for m, s in summaries.items()
        }
        bc1, bc2, bc3 = st.columns(3)
        with bc1: metric_comparison_bar(chart_data, "PSNR", "Mean PSNR (dB)")
        with bc2: metric_comparison_bar(chart_data, "SSIM", "Mean SSIM")
        with bc3: metric_comparison_bar(chart_data, "Ped Recall", "Mean Pedestrian Recall")

        metrics_radar(chart_data)

    # ── Per-image distributions ──────────────────────────────────
    if results:
        st.markdown("---")
        section_title("Per-Image Metric Distributions")

        tab_p2p, tab_trad = st.tabs(["Pix2Pix", "Baseline"])

        for tab, key in [(tab_p2p, "pix2pix"), (tab_trad, "traditional")]:
            with tab:
                if key not in results:
                    st.info(f"No {key} results found.")
                    continue
                per = results[key].get("per_image", [])
                if not per:
                    st.info("No per-image data.")
                    continue

                h1, h2 = st.columns(2)
                with h1: histogram_metric(per, "psnr", "PSNR (dB)")
                with h2: histogram_metric(per, "ssim", "SSIM")
                psnr_ssim_scatter(per, f"— {key.capitalize()}")

                # Top / bottom 10
                df = pd.DataFrame(per)
                # normalise recall column name (old: recall, new: ped_recall)
                if "ped_recall" in df.columns and "recall" not in df.columns:
                    df["recall"] = df["ped_recall"]
                display_cols = [c for c in ["filename", "psnr", "ssim", "recall"] if c in df.columns]
                col_t, col_b = st.columns(2)
                with col_t:
                    st.markdown("**Top 10 Pairs by PSNR**")
                    st.dataframe(
                        df.nlargest(10, "psnr")[display_cols],
                        width='stretch',
                    )
                with col_b:
                    st.markdown("**Bottom 10 Pairs by PSNR**")
                    st.dataframe(
                        df.nsmallest(10, "psnr")[display_cols],
                        width='stretch',
                    )

                with st.expander(f"Tabular results ({key}) — first 200"):
                    st.dataframe(df.head(200), width='stretch')

    # ── Training loss curves ─────────────────────────────────────
    st.markdown("---")
    section_title("Training Loss Convergence")

    log_data = parse_train_log(log_path)
    if not log_data:
        warn_box(
            f"Could not parse training log at <code>{log_path}</code>. "
            "Ensure training has generated logs."
        )
    else:
        st.markdown(f"**Training Log:** {len(log_data.get('epochs', []))} epochs recorded")

        tab_epoch, tab_batch = st.tabs(["Epoch Averages", "Batch-Level Trajectories"])
        with tab_epoch:
            loss_curves(log_data)
        with tab_batch:
            sampled = sample_batch_losses(log_data.get("batch_rows", []))
            batch_loss_curves(sampled)

        # Epoch summary table
        with st.expander("Epoch loss logs"):
            epoch_rows = [
                {"Epoch": e, "Avg G Loss": g, "Avg D Loss": d}
                for e, g, d in zip(
                    log_data["epochs"], log_data["avg_G"], log_data["avg_D"]
                )
            ]
            st.dataframe(pd.DataFrame(epoch_rows), width='stretch')

    # ── Metric definitions ───────────────────────────────────────
    st.markdown("---")
    section_title("Metric Formulations")
    st.markdown("""
| Metric | Full Name | Range | Target |
|---|---|---|---|
| PSNR | Peak Signal-to-Noise Ratio | 0 – ∞ dB | Higher is better |
| SSIM | Structural Similarity Index | 0 – 1 | Higher is better |
| Ped. Recall | Edge-saliency recall vs VOC annotations (IoU ≥ 0.3) | 0 – 1 | Higher is better |
| Edge F1 | Canny-edge harmonic mean (precision/recall) | 0 – 1 | Higher is better |
| LPIPS | Learned Perceptual Image Patch Similarity | 0 – 1 | Lower is better |

> **Implementation:** PSNR and SSIM are computed using `scikit-image`. LPIPS is computed via pretrained AlexNet features when `lpips` is loaded.
""")
