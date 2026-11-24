"""components/charts.py — Plotly chart helpers for the dashboard."""
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import streamlit as st

# Shared dark theme tokens
_BG    = "#0A0F1E"
_PAPER = "#111827"
_GRID  = "#1E3A5F"
_TEXT  = "#94A3B8"
_TEAL  = "#00D4B4"
_CYAN  = "#38BDF8"
_BLUE  = "#3B82F6"
_AMBER = "#F59E0B"
_PINK  = "#F472B6"

_LAYOUT = dict(
    plot_bgcolor=_PAPER,
    paper_bgcolor=_BG,
    font=dict(color=_TEXT, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=_GRID),
)


def loss_curves(log_data: dict):
    """Plot per-epoch avg G and D loss curves."""
    if not HAS_PLOTLY or not log_data:
        st.info("No training log data available.")
        return
    epochs = log_data["epochs"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=log_data["avg_G"],
                             name="Generator Loss", line=dict(color=_TEAL, width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=log_data["avg_D"],
                             name="Discriminator Loss", line=dict(color=_AMBER, width=2)))
    fig.update_layout(**_LAYOUT,
                      title="Per-Epoch Average Loss",
                      xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, width='stretch')


def batch_loss_curves(sampled: dict):
    """Plot per-batch G / L1 / Edge / VGG losses."""
    if not HAS_PLOTLY or not sampled:
        st.info("No batch loss data available.")
        return
    xs = sampled["x"]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Generator (G)", "L1 Loss",
                                        "Edge Loss", "VGG Perceptual"),
                        vertical_spacing=0.14)
    pairs = [("G", _TEAL, 1, 1), ("L1", _CYAN, 1, 2),
             ("Edge", _AMBER, 2, 1), ("VGG", _PINK, 2, 2)]
    for key, color, row, col in pairs:
        fig.add_trace(
            go.Scatter(x=xs, y=sampled[key], name=key,
                       line=dict(color=color, width=1.5), showlegend=False),
            row=row, col=col,
        )
    fig.update_layout(**_LAYOUT, title="Batch-Level Loss Curves (sampled)",
                      height=480)
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].gridcolor = _GRID
            fig.layout[ax].zerolinecolor = _GRID
            fig.layout[ax].color = _TEXT
    fig.update_annotations(font_color=_TEXT)
    st.plotly_chart(fig, width='stretch')


def metric_comparison_bar(metrics_dict: dict, metric_key: str,
                           title: str = "", higher_better: bool = True):
    """
    metrics_dict: {"Method A": {"PSNR": 13.4, ...}, "Method B": {...}}
    """
    if not HAS_PLOTLY:
        return
    methods = list(metrics_dict.keys())
    vals    = [metrics_dict[m].get(metric_key) for m in methods]
    colors  = [_TEAL if i == vals.index(max(v for v in vals if v))
               else _BLUE for i, v in enumerate(vals)]
    fig = go.Figure(go.Bar(x=methods, y=vals,
                           marker_color=colors,
                           text=[f"{v:.4f}" if v else "N/A" for v in vals],
                           textposition="outside"))
    fig.update_layout(**_LAYOUT, title=title or metric_key,
                      yaxis_title=metric_key, height=320)
    st.plotly_chart(fig, width='stretch')


def metrics_radar(metrics_dict: dict):
    """Spider / radar chart of normalized metrics across methods."""
    if not HAS_PLOTLY:
        return
    all_keys = ["PSNR", "SSIM", "Edge F1"]
    methods  = list(metrics_dict.keys())
    # Normalize each metric to [0,1] across methods
    raw = {k: [metrics_dict[m].get(k, 0) or 0 for m in methods]
           for k in all_keys}
    normed = {}
    for k, vals in raw.items():
        mx = max(vals) or 1
        normed[k] = [v / mx for v in vals]

    colors = [_TEAL, _CYAN, _AMBER, _PINK]
    fig    = go.Figure()
    for i, method in enumerate(methods):
        r_vals = [normed[k][i] for k in all_keys] + [normed[all_keys[0]][i]]
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=all_keys + [all_keys[0]],
            fill="toself", name=method,
            line_color=colors[i % len(colors)],
            opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=_PAPER,
            radialaxis=dict(visible=True, gridcolor=_GRID, color=_TEXT),
            angularaxis=dict(gridcolor=_GRID, color=_TEXT),
        ),
        **{k: v for k, v in _LAYOUT.items() if k not in ("xaxis", "yaxis")},
        title="Normalised Metric Radar",
        height=380,
    )
    st.plotly_chart(fig, width='stretch')


def psnr_ssim_scatter(per_image: list[dict], method_label: str = ""):
    """Scatter PSNR vs SSIM per image."""
    if not HAS_PLOTLY or not per_image:
        return
    psnrs = [r.get("psnr", 0) for r in per_image[:500]]
    ssims = [r.get("ssim", 0) for r in per_image[:500]]
    fnames = [r.get("filename", str(i)) for i, r in enumerate(per_image[:500])]
    fig = go.Figure(go.Scatter(
        x=psnrs, y=ssims, mode="markers",
        marker=dict(color=_TEAL, size=4, opacity=0.6),
        text=fnames, hovertemplate="<b>%{text}</b><br>PSNR: %{x:.2f}<br>SSIM: %{y:.4f}",
    ))
    fig.update_layout(**_LAYOUT, title=f"PSNR vs SSIM {method_label}",
                      xaxis_title="PSNR (dB)", yaxis_title="SSIM", height=380)
    st.plotly_chart(fig, width='stretch')


def histogram_metric(per_image: list[dict], key: str = "psnr", label: str = "PSNR"):
    if not HAS_PLOTLY or not per_image:
        return
    vals = [r.get(key, 0) for r in per_image]
    fig  = go.Figure(go.Histogram(x=vals, nbinsx=40,
                                  marker_color=_CYAN, opacity=0.8))
    fig.update_layout(**_LAYOUT, title=f"{label} Distribution",
                      xaxis_title=label, yaxis_title="Count", height=300)
    st.plotly_chart(fig, width='stretch')
