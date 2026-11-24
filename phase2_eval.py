"""
Phase 2 Evaluation — final version
Writes progress to both stdout and a progress file for monitoring.
"""
import os, shutil, json, csv, math, sys, time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

ROOT         = Path(r"C:\Users\Nishtha\Desktop\LLVIP")
TRAD_SRC     = ROOT / "results" / "traditional"
PIX_SRC      = ROOT / "results" / "pix2pix" / "test_gen"
GT_DIR       = ROOT / "LLVIP" / "visible" / "test"
EVAL_ROOT    = ROOT / "results" / "evaluation_final"
TRAD_ENH_DST = EVAL_ROOT / "traditional_enhanced"
TRAD_EDG_DST = EVAL_ROOT / "traditional_edge"
PIX_DST      = EVAL_ROOT / "pix2pix"
OUT_CSV      = EVAL_ROOT / "per_image_metrics.csv"
OUT_JSON     = EVAL_ROOT / "summary_metrics.json"
PROGRESS_LOG = EVAL_ROOT / "progress.txt"
SIZE         = (256, 256)

def log(msg):
    print(msg)
    with open(PROGRESS_LOG, "a") as f:
        f.write(msg + "\n")

# ── Step 1: Organise folders ──────────────────────────────────────────────────
for d in [TRAD_ENH_DST, TRAD_EDG_DST, PIX_DST]:
    d.mkdir(parents=True, exist_ok=True)

PROGRESS_LOG.write_text("")   # clear
log("=" * 60)
log("STEP 1 — Organising evaluation_final folders")
log("=" * 60)

for f in TRAD_SRC.glob("*.jpg"):
    if "_enhanced" in f.stem:
        dst = TRAD_ENH_DST / f.name
        if not dst.exists(): shutil.copy2(f, dst)
    elif "_edge" in f.stem:
        dst = TRAD_EDG_DST / f.name
        if not dst.exists(): shutil.copy2(f, dst)

for f in PIX_SRC.glob("*.jpg"):
    dst = PIX_DST / f.name
    if not dst.exists(): shutil.copy2(f, dst)

for label, d in [("traditional_enhanced", TRAD_ENH_DST),
                  ("traditional_edge",     TRAD_EDG_DST),
                  ("pix2pix",              PIX_DST)]:
    log(f"  {label}: {len(list(d.glob('*.jpg')))} files")

# ── Helpers ───────────────────────────────────────────────────────────────────
gt_map = {f.stem: f for f in GT_DIR.iterdir()}
log(f"\nGround-truth images: {len(gt_map)}")

def compute(pred_path, gt_path):
    p = cv2.imread(str(pred_path))
    g = cv2.imread(str(gt_path))
    if p is None or g is None:
        return None
    pg = cv2.cvtColor(cv2.resize(p, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gg = cv2.cvtColor(cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    # PSNR
    mse = np.mean((pg.astype(np.float64) - gg.astype(np.float64))**2)
    psnr_v = 100.0 if mse == 0 else 10*math.log10(255.0**2/mse)
    # SSIM
    ssim_v = float(ssim_fn(pg, gg, data_range=255))
    # Edge F1
    def edges(x):
        b = cv2.GaussianBlur(x, (3,3), 0)
        e = cv2.Canny(b, 50, 150)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        return cv2.dilate(e, k, iterations=1) > 0
    pe, ge_ = edges(pg), edges(gg)
    tp = int((pe & ge_).sum()); fp = int((pe & ~ge_).sum()); fn = int((~pe & ge_).sum())
    pr = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
    f1 = float(2*pr*rc/(pr+rc+1e-8))
    return psnr_v, ssim_v, f1

# ── Step 2: Compute metrics ───────────────────────────────────────────────────
log("\n" + "=" * 60)
log("STEP 2 — Computing metrics")
log("=" * 60)

METHODS = {
    "traditional_enhanced": TRAD_ENH_DST,
    "traditional_edge":     TRAD_EDG_DST,
    "pix2pix":              PIX_DST,
}

rows = []
summaries = {}

for method_name, method_dir in METHODS.items():
    log(f"\n  [{method_name}]")
    files = sorted(method_dir.glob("*.jpg"))
    log(f"  {len(files)} files found")
    t0 = time.time()
    ps, ss, fs = [], [], []
    skipped = 0
    for i, f in enumerate(files, 1):
        base = f.stem.replace("_enhanced","").replace("_edge","")
        if base not in gt_map:
            skipped += 1; continue
        result = compute(f, gt_map[base])
        if result is None:
            skipped += 1; continue
        p, s, f1 = result
        ps.append(p); ss.append(s); fs.append(f1)
        rows.append({"method": method_name, "image_id": base,
                     "psnr": round(p,4), "ssim": round(s,4), "edge_f1": round(f1,4)})
        if i % 500 == 0 or i == len(files):
            elapsed = time.time() - t0
            log(f"  {i}/{len(files)}  psnr={np.mean(ps):.2f}  ssim={np.mean(ss):.4f}  f1={np.mean(fs):.4f}  ({elapsed:.0f}s)")

    n = len(ps)
    summaries[method_name] = {
        "n": n, "skipped": skipped,
        "psnr":    round(float(np.mean(ps)) if n else 0, 4),
        "ssim":    round(float(np.mean(ss)) if n else 0, 4),
        "edge_f1": round(float(np.mean(fs)) if n else 0, 4),
    }
    v = summaries[method_name]
    log(f"  DONE  n={v['n']} skipped={v['skipped']}  PSNR={v['psnr']}  SSIM={v['ssim']}  EdgeF1={v['edge_f1']}")

# ── Save outputs ──────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["method","image_id","psnr","ssim","edge_f1"])
    w.writeheader(); w.writerows(rows)

with open(OUT_JSON, "w") as fh:
    json.dump(summaries, fh, indent=2)

# ── Final table ───────────────────────────────────────────────────────────────
log("\n" + "=" * 60)
log("FINAL COMPARISON TABLE")
log("=" * 60)
log(f"{'Method':<26} {'PSNR(dB)':>10} {'SSIM':>8} {'Edge-F1':>10}")
log("-" * 60)
for m, v in summaries.items():
    log(f"{m:<26} {v['psnr']:>10.4f} {v['ssim']:>8.4f} {v['edge_f1']:>10.4f}")
log("=" * 60)
log(f"\nCSV: {OUT_CSV}")
log(f"JSON: {OUT_JSON}")
log("COMPLETE")
