"""
services/model_loader.py
Cached model loading — loads the UNet generator once per session.
"""
import os, sys, glob
from pathlib import Path

import torch
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@st.cache_resource(show_spinner=False)
def load_generator(ckpt_path: str, ngf: int = 64):
    """
    Load UNetGenerator from checkpoint. Cached for the session lifetime.
    Returns (model, device) or raises on failure.
    """
    from models import UNetGenerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = UNetGenerator(in_channels=2, out_channels=3, ngf=ngf).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(state["netG"])
    G.eval()
    return G, device


@st.cache_resource(show_spinner=False)
def load_config(cfg_path: str = "configs/config.yaml"):
    import yaml
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def list_checkpoints(ckpt_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    return files


def checkpoint_info(ckpt_path: str) -> dict:
    """Return epoch number and file size from a checkpoint path."""
    if not os.path.exists(ckpt_path):
        return {"epoch": None, "size_mb": None}
    size_mb = os.path.getsize(ckpt_path) / (1024 ** 2)
    name = os.path.basename(ckpt_path)
    epoch = None
    if "epoch" in name:
        try:
            epoch = int(name.split("epoch")[1].split(".")[0])
        except Exception:
            pass
    return {"epoch": epoch, "size_mb": round(size_mb, 1)}


def device_summary() -> dict:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "name": props.name,
            "vram_gb": round(props.total_memory / 1024 ** 3, 1),
        }
    return {"available": False, "name": "CPU", "vram_gb": None}
