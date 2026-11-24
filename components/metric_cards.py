"""components/metric_cards.py — Reusable metric display helpers."""
import streamlit as st


def metric_card(label: str, value, unit: str = "", cols=None):
    """Render a single styled metric card via HTML."""
    val_str = str(value) if value is not None else "—"
    html = f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{val_str}</div>
      <div class="metric-unit">{unit}</div>
    </div>"""
    if cols:
        cols.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def metric_row(items: list[dict]):
    """
    Render a row of metric cards.
    items: [{"label": str, "value": any, "unit": str}, ...]
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric_card(
                item.get("label", ""),
                item.get("value", "—"),
                item.get("unit", ""),
            )


def status_pill(label: str, status: str = "ok"):
    """status: 'ok' | 'warn' | 'err'"""
    cls = {"ok": "status-ok", "warn": "status-warn", "err": "status-err"}.get(status, "status-warn")
    bullet = {"ok": "●", "warn": "▲", "err": "■"}.get(status, "●")
    st.markdown(
        f"<span class='status-pill {cls}'><span style='font-size:0.7em;vertical-align:middle;margin-right:4px;'>{bullet}</span>{label}</span>",
        unsafe_allow_html=True,
    )


def pipeline_step(num: int, title: str, description: str):
    st.markdown(f"""
    <div class="pipeline-step">
      <div class="pipeline-step-num">Step {num}</div>
      <div class="pipeline-step-title">{title}</div>
      <div class="pipeline-step-desc">{description}</div>
    </div>""", unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f"<div class='section-title'>{text}</div>",
                unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)


def warn_box(text: str):
    st.markdown(f"<div class='warn-box'>{text}</div>", unsafe_allow_html=True)


def success_box(text: str):
    st.markdown(f"<div class='success-box'>{text}</div>", unsafe_allow_html=True)


def error_box(text: str):
    st.markdown(f"<div class='error-box'>{text}</div>", unsafe_allow_html=True)
