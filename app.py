"""
app.py — Edge-Guided Thermal Image Colorization Research Dashboard
Run:  streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

# ── Ensure src/ on path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LLVIP — Thermal-to-Visible Colorization",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR  (settings shared across all pages)
# ═══════════════════════════════════════════════════════════════
from components.sidebar import render_sidebar

PAGES = {
    "Overview":       "overview",
    "Colorization":   "colorization",
    "Comparison":     "comparison",
    "Architecture":   "architecture",
    "Evaluation":     "evaluation",
    "About":          "about",
}

# Navigation radio injected into sidebar before settings
with st.sidebar:
    st.markdown("""
<div style="
  background:linear-gradient(135deg,#0D2137,#0A1628);
  border:1px solid #1E3A5F; border-radius:10px;
  padding:0.9rem 1rem; margin-bottom:1rem; text-align:left;">
  <div style="font-size:0.95rem;font-weight:700;
    color:#F0F6FF; letter-spacing:0.01em;">
    LLVIP Colorization
  </div>
  <div style="font-size:0.75rem;color:#8BA5C2;margin-top:3px;">
    Edge-Guided Pix2Pix GAN
  </div>
</div>""", unsafe_allow_html=True)

    page_choice = st.radio(
        "navigate",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

settings = render_sidebar()

# ═══════════════════════════════════════════════════════════════
# PAGE DISPATCH
# ═══════════════════════════════════════════════════════════════
page_key = PAGES[page_choice]

if page_key == "overview":
    from pages.overview     import render
elif page_key == "colorization":
    from pages.colorization import render
elif page_key == "comparison":
    from pages.comparison   import render
elif page_key == "architecture":
    from pages.architecture import render
elif page_key == "evaluation":
    from pages.evaluation   import render
else:
    from pages.about        import render

render(settings)
