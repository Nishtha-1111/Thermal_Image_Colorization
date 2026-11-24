"""components/image_viewer.py — Image display helpers."""
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from services.preprocessing import bgr_to_pil, gray_to_pil


def show_image(img, caption: str = "", stretch: bool = True):
    """Accept BGR ndarray, gray ndarray, or PIL Image."""
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            pil = gray_to_pil(img)
        else:
            pil = bgr_to_pil(img)
    else:
        pil = img
    st.image(pil, caption=caption, width='stretch' if stretch else 'content')


def image_panel(img, title: str, caption: str = ""):
    """Render an image inside a styled card."""
    if isinstance(img, np.ndarray):
        pil = gray_to_pil(img) if img.ndim == 2 else bgr_to_pil(img)
    else:
        pil = img
    st.markdown(f"""
    <div class="img-panel">
      <div class="img-panel-title">{title}</div>
    </div>""", unsafe_allow_html=True)
    st.image(pil, caption=caption, width='stretch')


def comparison_strip(images: list[dict], n_cols: int = None):
    """
    images: [{"img": ndarray_or_pil, "title": str, "caption": str}, ...]
    """
    n = len(images)
    cols = st.columns(n_cols or n)
    for col, item in zip(cols, images):
        with col:
            image_panel(item["img"], item.get("title", ""), item.get("caption", ""))


def draw_pedestrian_boxes(bgr: np.ndarray,
                           min_area: int = 800) -> tuple[np.ndarray, int]:
    """
    Draw edge-saliency pedestrian bounding boxes on a copy of bgr.
    Returns (annotated_bgr, num_boxes).
    """
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    dil    = cv2.dilate(edges, kernel, iterations=3)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out    = bgr.copy()
    count  = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area and h > w:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 150), 2)
            count += 1
    return out, count


def encode_download(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()
