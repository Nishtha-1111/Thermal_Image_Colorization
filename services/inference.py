"""services/inference.py — Pix2Pix inference pipeline."""
import sys
from pathlib import Path
import cv2, numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_inference(G, device, ir_tensor: torch.Tensor,
                  edge_tensor: torch.Tensor) -> np.ndarray:
    """
    Run the trained generator.
    ir_tensor   : (1,1,H,W) float32 in [-1,1]
    edge_tensor : (1,1,H,W) float32 in [ 0,1]
    Returns     : uint8 BGR numpy (H,W,3)
    """
    ir_t   = ir_tensor.to(device)
    edge_t = edge_tensor.to(device)
    gen_in = torch.cat([ir_t, edge_t], dim=1)   # (1,2,H,W)
    with torch.no_grad():
        fake = G(gen_in)[0].cpu().numpy()        # (3,H,W) in [-1,1]
    img = ((fake + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    img = img.transpose(1, 2, 0)                 # H,W,3  RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def encode_png(bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", bgr)
    return buf.tobytes()
