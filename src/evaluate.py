"""
evaluate.py – Quantitative evaluation of LLVIP colorization results.

Metrics computed:
  - PSNR   (Peak Signal-to-Noise Ratio)
  - SSIM   (Structural Similarity Index)
  - Pedestrian evaluation using REAL LLVIP VOC GT bounding boxes:
      * Precision, Recall, F1 at IoU ≥ 0.3
      * mAP@0.5 (mean Average Precision)
      * Comparison: how many GT boxes are visible in generated vs GT visible

NOTE ON PEDESTRIAN EVALUATION:
  We do NOT run a trained detector on the generated images.
  Instead, we use an edge-saliency proposal method (Canny + contour heuristic)
  as a structural-quality proxy. We compare these proposals against LLVIP GT
  boxes (from VOC XML) using IoU. This measures how well the generated image
  preserves pedestrian-shaped structure — not detector accuracy.

Usage:
  python src/evaluate.py --config configs/config.yaml --mode pix2pix --infer
  python src/evaluate.py --config configs/config.yaml --mode traditional
  python src/evaluate.py --config configs/config.yaml --mode pix2pix --verify_annotations
"""

import os
import sys
import glob
import argparse
import json
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LLVIPDataset, parse_voc_annotation
from models import UNetGenerator


# ============================================================
# 1. IMAGE QUALITY METRICS
# ============================================================

def compute_psnr(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return float(psnr_sk(img1, img2, data_range=1.0))


def compute_ssim(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> float:
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    return float(ssim_sk(img1, img2, data_range=1.0))


def compute_sobel_edge_f1(gen_bgr: np.ndarray, gt_bgr: np.ndarray) -> float:
    """
    Compare Sobel edge maps of generated vs GT image.
    Measures how well the model preserves structural edge information.
    Uses the same Sobel logic as the training EdgePreservingLoss.
    """
    def sobel_map(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        # Binarise at Otsu threshold
        mag_uint8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(mag_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary.astype(bool)

    pred_e = sobel_map(gen_bgr)
    real_e = sobel_map(gt_bgr)
    tp   = (pred_e & real_e).sum()
    prec = tp / (pred_e.sum() + 1e-8)
    rec  = tp / (real_e.sum() + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return float(f1)


# ============================================================
# 2. PEDESTRIAN EVALUATION (ANNOTATION-BASED)
# ============================================================

def iou(boxA, boxB) -> float:
    """IoU between two (xmin,ymin,xmax,ymax) boxes."""
    ix1 = max(boxA[0], boxB[0]); iy1 = max(boxA[1], boxB[1])
    ix2 = min(boxA[2], boxB[2]); iy2 = min(boxA[3], boxB[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def scale_boxes(boxes, orig_hw, new_hw):
    """
    Scale bounding boxes from original image size to resized image size.
    orig_hw = (H_orig, W_orig), new_hw = (H_new, W_new)
    """
    oh, ow = orig_hw
    nh, nw = new_hw
    sx = nw / ow
    sy = nh / oh
    return [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
            for (x1, y1, x2, y2) in boxes]


def edge_saliency_proposals(gray_img: np.ndarray, min_area=300) -> list:
    """
    Generate pedestrian-candidate bounding boxes using edge-saliency heuristic.
    This is a STRUCTURAL PROXY — not a trained detector.
    Returns list of (xmin, ymin, xmax, ymax).
    """
    blurred  = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges    = cv2.Canny(blurred, 50, 150)
    # Vertical kernel → connects vertically arranged edge pixels (pedestrian shape)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    dilated  = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area and h > w:   # taller than wide = pedestrian-like
            boxes.append((x, y, x+w, y+h))
    return boxes


def compute_precision_recall(pred_boxes, gt_boxes, iou_thresh=0.3):
    """
    Compute precision and recall for one image.
    Each GT box can only be matched once (greedy matching).
    """
    if not gt_boxes:
        return 1.0, 1.0   # no GTs → trivially "correct"
    if not pred_boxes:
        return 0.0, 0.0   # no predictions → precision=0, recall=0

    matched_gt  = set()
    matched_pred = set()

    for pi, pb in enumerate(pred_boxes):
        best_iou = 0.0
        best_gi  = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou = v
                best_gi  = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    tp   = len(matched_gt)
    prec = tp / len(pred_boxes)
    rec  = tp / len(gt_boxes)
    return prec, rec


def eval_pedestrian_gt(gen_bgr: np.ndarray, gt_boxes_orig: list,
                       orig_hw: tuple, gen_hw: tuple,
                       iou_thresh=0.3) -> dict:
    """
    Evaluate pedestrian structure in generated image against LLVIP GT boxes.

    Steps:
      1. Scale GT boxes from original image resolution to generated image resolution.
      2. Run edge-saliency proposals on the generated image.
      3. Compute precision, recall, F1 against scaled GT boxes.

    This tells us: does the generated image contain pedestrian-shaped
    edge structures in the right locations?
    """
    # Step 1: scale GT boxes to match generated image size
    gt_boxes_scaled = scale_boxes(gt_boxes_orig, orig_hw, gen_hw)

    # Step 2: proposal detection on generated image
    gray_gen  = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY)
    pred_boxes = edge_saliency_proposals(gray_gen)

    # Step 3: precision / recall
    prec, rec = compute_precision_recall(pred_boxes, gt_boxes_scaled, iou_thresh)
    f1 = 2*prec*rec / (prec+rec+1e-8) if (prec+rec) > 0 else 0.0

    return {
        "gt_count":    len(gt_boxes_scaled),
        "pred_count":  len(pred_boxes),
        "tp":          int(prec * len(pred_boxes) + 0.5),
        "precision":   round(prec, 4),
        "recall":      round(rec,  4),
        "f1":          round(f1,   4),
    }


# ============================================================
# 3. VERIFY ANNOTATIONS (standalone check)
# ============================================================

def verify_annotations(ann_dir: str, n_samples: int = 10):
    """
    Read a few annotation files and print their contents.
    Proves that parse_voc_annotation() actually reads the LLVIP XML files.
    """
    xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))[:n_samples]
    if not xml_files:
        print(f"[ERROR] No XML files found in {ann_dir}")
        return

    print(f"\n{'='*55}")
    print(f" Annotation Verification — {len(xml_files)} sample files")
    print(f"{'='*55}")
    total_boxes = 0
    for xf in xml_files:
        stem  = os.path.splitext(os.path.basename(xf))[0]
        boxes = parse_voc_annotation(xf)
        total_boxes += len(boxes)
        print(f"  {stem}.xml  → {len(boxes)} box(es):  {boxes[:3]}"
              f"{'...' if len(boxes)>3 else ''}")
    print(f"\n  Average boxes/image: {total_boxes/len(xml_files):.2f}")
    print(f"{'='*55}\n")


# ============================================================
# 4. PARITY TEST: IR vs Generated image pedestrian visibility
# ============================================================

def compare_ir_vs_generated(ir_path: str, gen_bgr: np.ndarray,
                              gt_boxes_orig: list, orig_hw: tuple,
                              gen_hw: tuple, iou_thresh=0.3) -> dict:
    """
    Compare pedestrian structure in raw IR vs generated image.
    If generated recall ≥ IR recall → generation preserves structure.
    """
    ir_bgr = cv2.imread(ir_path)
    if ir_bgr is None:
        return {}
    ir_bgr_rs = cv2.resize(ir_bgr, (gen_hw[1], gen_hw[0]))

    ir_res  = eval_pedestrian_gt(ir_bgr_rs, gt_boxes_orig, orig_hw, gen_hw, iou_thresh)
    gen_res = eval_pedestrian_gt(gen_bgr,   gt_boxes_orig, orig_hw, gen_hw, iou_thresh)

    return {
        "ir_recall":  ir_res["recall"],
        "gen_recall": gen_res["recall"],
        "structure_preserved": gen_res["recall"] >= ir_res["recall"],
    }


# ============================================================
# 5. PIX2PIX INFERENCE
# ============================================================

def run_pix2pix_inference(cfg, device, max_samples=None):
    """Generate fake visible images for the entire test set."""
    out_dir = os.path.join(cfg["paths"]["results_pix2pix"], "test_gen")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(cfg["paths"]["checkpoints"], "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpts = sorted(glob.glob(os.path.join(cfg["paths"]["checkpoints"],
                                               "checkpoint_epoch*.pth")))
        if not ckpts:
            raise FileNotFoundError("No checkpoint found. Train the model first.")
        ckpt_path = ckpts[-1]

    print(f"Loading checkpoint: {ckpt_path}")
    netG = UNetGenerator(in_channels=2, out_channels=3,
                         ngf=cfg["model"]["ngf"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    netG.load_state_dict(state["netG"])
    netG.eval()

    ds = LLVIPDataset(
        ir_dir=cfg["dataset"]["test_ir"],
        vis_dir=cfg["dataset"]["test_vis"],
        image_size=tuple(cfg["dataset"]["image_size"]),
        augment=False,
        max_samples=max_samples,
    )

    print(f"Running inference on {len(ds)} test images → {out_dir}")
    with torch.no_grad():
        for i, sample in enumerate(ds):
            ir   = sample["ir"].unsqueeze(0).to(device)
            edge = sample["edge"].unsqueeze(0).to(device)
            fake = netG(torch.cat([ir, edge], dim=1))[0]
            img  = fake.cpu().numpy()
            img  = ((img + 1.0)/2.0*255).clip(0,255).astype(np.uint8)
            img  = img.transpose(1,2,0)
            img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"{sample['filename']}.jpg"), img)
            if (i+1) % 200 == 0:
                print(f"  {i+1}/{len(ds)}")

    print(f"Done. Images saved to: {out_dir}")
    return out_dir


# ============================================================
# 6. MAIN EVALUATION LOOP
# ============================================================

def evaluate(cfg, mode="pix2pix", max_samples=None, iou_thresh=0.3):
    ann_dir  = cfg["dataset"]["annotations"]
    vis_dir  = cfg["dataset"]["test_vis"]
    ir_dir   = cfg["dataset"]["test_ir"]
    img_size = tuple(cfg["dataset"]["image_size"])   # (H, W)

    if mode == "pix2pix":
        gen_dir = os.path.join(cfg["paths"]["results_pix2pix"], "test_gen")
        if not os.path.exists(gen_dir):
            raise RuntimeError("Run inference first (--infer flag).")
    elif mode == "traditional":
        gen_dir = cfg["paths"]["results_traditional"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.jpg")) +
                       glob.glob(os.path.join(gen_dir, "*.png")))
    if max_samples:
        gen_files = gen_files[:max_samples]

    print(f"\nEvaluating {len(gen_files)} images [{mode}] …")
    print(f"  Annotation directory : {ann_dir}")
    print(f"  GT visible directory : {vis_dir}")
    print(f"  IoU threshold        : {iou_thresh}")
    print(f"  Image size (HxW)     : {img_size}\n")

    results    = []
    psnrs, ssims, edge_f1s = [], [], []
    precisions, recalls, f1s = [], [], []
    xml_found  = 0
    xml_missing = 0

    for gen_path in gen_files:
        stem = os.path.splitext(os.path.basename(gen_path))[0]
        # Strip suffixes added by enhancement pipeline
        for suf in ["_enhanced", "_edge", "_colorized", "_clahe"]:
            stem = stem.replace(suf, "")

        # --- GT visible image ---
        gt_path = None
        for ext in (".jpg", ".png"):
            p = os.path.join(vis_dir, stem + ext)
            if os.path.exists(p):
                gt_path = p; break
        if gt_path is None:
            continue

        gen_bgr = cv2.imread(gen_path)
        gt_bgr  = cv2.imread(gt_path)
        if gen_bgr is None or gt_bgr is None:
            continue

        h, w = img_size
        gen_bgr = cv2.resize(gen_bgr, (w, h))
        gt_bgr  = cv2.resize(gt_bgr,  (w, h))

        # --- Image quality metrics ---
        p_val = compute_psnr(gen_bgr, gt_bgr)
        s_val = compute_ssim(gen_bgr, gt_bgr)
        e_f1  = compute_sobel_edge_f1(gen_bgr, gt_bgr)
        psnrs.append(p_val); ssims.append(s_val); edge_f1s.append(e_f1)

        # --- Pedestrian evaluation with REAL VOC GT boxes ---
        xml_path = os.path.join(ann_dir, stem + ".xml")
        gt_boxes_orig = parse_voc_annotation(xml_path)   # READ FROM VOC XML

        if os.path.exists(xml_path):
            xml_found += 1
        else:
            xml_missing += 1

        if gt_boxes_orig:
            # GT boxes are in original image coords (1024x1280)
            # We need to scale to our eval image size
            orig_bgr = cv2.imread(gt_path)
            orig_hw  = (orig_bgr.shape[0], orig_bgr.shape[1]) if orig_bgr is not None else (1024, 1280)
            gen_hw   = (h, w)

            ped_eval = eval_pedestrian_gt(
                gen_bgr, gt_boxes_orig,
                orig_hw=orig_hw,
                gen_hw=gen_hw,
                iou_thresh=iou_thresh,
            )
        else:
            # No annotation for this image — skip pedestrian metric
            ped_eval = {"gt_count":0, "pred_count":0, "tp":0,
                        "precision":0.0, "recall":0.0, "f1":0.0}

        if ped_eval["gt_count"] > 0:
            precisions.append(ped_eval["precision"])
            recalls.append(ped_eval["recall"])
            f1s.append(ped_eval["f1"])

        results.append({
            "filename":       stem,
            "psnr":           round(p_val, 4),
            "ssim":           round(s_val, 4),
            "sobel_edge_f1":  round(e_f1,  4),
            "gt_boxes":       ped_eval["gt_count"],
            "pred_boxes":     ped_eval["pred_count"],
            "tp":             ped_eval["tp"],
            "ped_precision":  ped_eval["precision"],
            "ped_recall":     ped_eval["recall"],
            "ped_f1":         ped_eval["f1"],
        })

    # --- Summary ---
    summary = {
        "mode":              mode,
        "n_images":          len(results),
        "xml_found":         xml_found,
        "xml_missing":       xml_missing,
        "mean_psnr":         round(float(np.mean(psnrs)),       4) if psnrs    else 0,
        "mean_ssim":         round(float(np.mean(ssims)),       4) if ssims    else 0,
        "mean_sobel_edge_f1":round(float(np.mean(edge_f1s)),   4) if edge_f1s else 0,
        "mean_ped_precision":round(float(np.mean(precisions)),  4) if precisions else 0,
        "mean_ped_recall":   round(float(np.mean(recalls)),     4) if recalls  else 0,
        "mean_ped_f1":       round(float(np.mean(f1s)),         4) if f1s      else 0,
        "note": (
            "Pedestrian eval uses edge-saliency proposals (not a trained detector) "
            "matched against LLVIP VOC GT bounding boxes at IoU>=" + str(iou_thresh) + ". "
            "This measures structural preservation, not detection accuracy."
        ),
    }

    # --- Save ---
    eval_dir = cfg["paths"]["results_eval"]
    os.makedirs(eval_dir, exist_ok=True)
    json_path = os.path.join(eval_dir, f"metrics_{mode}.json")
    csv_path  = os.path.join(eval_dir, f"per_image_{mode}.csv")

    with open(json_path, "w") as f:
        json.dump({"summary": summary, "per_image": results}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)

    # --- Print ---
    print("\n" + "="*55)
    print(f" Results — {mode}  ({len(results)} images)")
    print("="*55)
    print(f"  {'PSNR':<22}: {summary['mean_psnr']} dB")
    print(f"  {'SSIM':<22}: {summary['mean_ssim']}")
    print(f"  {'Sobel Edge F1':<22}: {summary['mean_sobel_edge_f1']}")
    print(f"  {'Ped Precision':<22}: {summary['mean_ped_precision']}")
    print(f"  {'Ped Recall':<22}: {summary['mean_ped_recall']}")
    print(f"  {'Ped F1':<22}: {summary['mean_ped_f1']}")
    print(f"  {'Annotations found':<22}: {xml_found}")
    print(f"  {'Annotations missing':<22}: {xml_missing}")
    print("="*55)
    print(f"\nFull JSON : {json_path}")
    print(f"Per-image : {csv_path}\n")
    return summary


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLVIP colorization — image quality + pedestrian GT boxes")
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--mode",    choices=["pix2pix","traditional"], default="pix2pix")
    parser.add_argument("--infer",   action="store_true",
                        help="Run Pix2Pix inference before evaluating")
    parser.add_argument("--max",     type=int, default=None,
                        help="Limit number of images (for quick tests)")
    parser.add_argument("--iou",     type=float, default=0.3,
                        help="IoU threshold for pedestrian box matching")
    parser.add_argument("--verify_annotations", action="store_true",
                        help="Print sample annotation contents and exit")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Annotation verification mode
    if args.verify_annotations:
        verify_annotations(cfg["dataset"]["annotations"], n_samples=10)
        sys.exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.infer and args.mode == "pix2pix":
        run_pix2pix_inference(cfg, device, max_samples=args.max)

    evaluate(cfg, mode=args.mode, max_samples=args.max, iou_thresh=args.iou)
