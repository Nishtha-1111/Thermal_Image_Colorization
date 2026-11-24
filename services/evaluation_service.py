"""services/evaluation_service.py — Metrics and training-log parsing."""
import os, re, json, glob
import numpy as np

# ── optional deps ───────────────────────────────────────────────
try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity   as _ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

import cv2


# ── Image quality metrics ───────────────────────────────────────

def compute_psnr(gen_bgr: np.ndarray, gt_bgr: np.ndarray) -> float | None:
    if not HAS_SKIMAGE:
        return None
    g = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    t = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    return round(_psnr(t, g, data_range=1.0), 3)


def compute_ssim(gen_bgr: np.ndarray, gt_bgr: np.ndarray) -> float | None:
    if not HAS_SKIMAGE:
        return None
    g = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    t = cv2.cvtColor(gt_bgr,  cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    return round(float(_ssim(t, g, data_range=1.0)), 4)


def compute_edge_f1(pred_bgr: np.ndarray,
                    real_bgr: np.ndarray) -> dict:
    p_g = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY)
    r_g = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)
    p_e = cv2.Canny(cv2.GaussianBlur(p_g, (3, 3), 0), 50, 150) > 0
    r_e = cv2.Canny(cv2.GaussianBlur(r_g, (3, 3), 0), 50, 150) > 0
    tp   = (p_e & r_e).sum()
    prec = float(tp / (p_e.sum() + 1e-8))
    rec  = float(tp / (r_e.sum() + 1e-8))
    f1   = float(2 * prec * rec / (prec + rec + 1e-8))
    return {"f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)}


def compute_all_metrics(gen_bgr: np.ndarray,
                         gt_bgr: np.ndarray) -> dict:
    h, w = gt_bgr.shape[:2]
    gen_rs = cv2.resize(gen_bgr, (w, h))
    return {
        "PSNR":      compute_psnr(gen_rs, gt_bgr),
        "SSIM":      compute_ssim(gen_rs, gt_bgr),
        "Edge F1":   compute_edge_f1(gen_rs, gt_bgr)["f1"],
        "Edge Prec": compute_edge_f1(gen_rs, gt_bgr)["precision"],
        "Edge Rec":  compute_edge_f1(gen_rs, gt_bgr)["recall"],
    }


# ── Load saved eval JSON results ────────────────────────────────

def load_eval_results(eval_dir: str) -> dict:
    """Load metrics_pix2pix.json and metrics_traditional.json if present."""
    out = {}
    for mode in ["pix2pix", "traditional"]:
        p = os.path.join(eval_dir, f"metrics_{mode}.json")
        if os.path.exists(p):
            with open(p) as f:
                out[mode] = json.load(f)
    return out


# ── Parse training log → per-epoch loss curves ──────────────────

def parse_train_log(log_path: str) -> dict:
    """
    Parse logs/train.log to extract per-epoch average G / D losses
    and per-batch loss values for plotting.
    Returns dict with keys: epochs, avg_G, avg_D, batches (sampled).
    """
    if not os.path.exists(log_path):
        return {}

    epoch_data: dict[int, dict] = {}   # epoch → {G_sum, D_sum, count}
    batch_rows = []   # (epoch, batch, D, G, L1, Edge, VGG)

    epoch_re = re.compile(
        r"Epoch \[(\d+)/\d+\] Batch \[(\d+)/\d+\] "
        r"D: ([\d.]+)\s+G: ([\d.]+)\s+L1: ([\d.]+)\s+Edge: ([\d.]+)\s+VGG: ([\d.]+)"
    )
    end_re = re.compile(
        r"==> Epoch (\d+)/\d+ done.*avg G: ([\d.]+)\s+avg D: ([\d.]+)"
    )

    with open(log_path, errors="ignore") as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                ep, bat = int(m.group(1)), int(m.group(2))
                D, G, L1, Edge, VGG = (float(m.group(i)) for i in range(3, 8))
                batch_rows.append((ep, bat, D, G, L1, Edge, VGG))
                continue
            m2 = end_re.search(line)
            if m2:
                ep = int(m2.group(1))
                epoch_data[ep] = {
                    "avg_G": float(m2.group(2)),
                    "avg_D": float(m2.group(3)),
                }

    if not epoch_data:
        return {}

    epochs = sorted(epoch_data)
    return {
        "epochs":  epochs,
        "avg_G":   [epoch_data[e]["avg_G"] for e in epochs],
        "avg_D":   [epoch_data[e]["avg_D"] for e in epochs],
        "batch_rows": batch_rows,   # raw (ep, bat, D, G, L1, Edge, VGG)
    }


def sample_batch_losses(batch_rows: list, max_points: int = 500) -> dict:
    """Downsample batch_rows for plotting."""
    if not batch_rows:
        return {}
    step = max(1, len(batch_rows) // max_points)
    rows = batch_rows[::step]
    xs   = list(range(len(rows)))
    return {
        "x":    xs,
        "D":    [r[2] for r in rows],
        "G":    [r[3] for r in rows],
        "L1":   [r[4] for r in rows],
        "Edge": [r[5] for r in rows],
        "VGG":  [r[6] for r in rows],
    }
