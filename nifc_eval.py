"""
nifc_eval.py — Real-world accuracy check against NIFC observed fire perimeters.

Three-way comparison at the NIFC perimeter timestamp (2020-09-02 13:00 UTC):
  1. WRF-SFIRE simulated FIRE_AREA  vs NIFC observed  → simulation accuracy
  2. Model predicted fire mask       vs NIFC observed  → end-to-end accuracy
  3. Model predicted                 vs WRF-SFIRE      → model vs simulator (IoU=0.2112 baseline)

Usage:
  python nifc_eval.py
  python nifc_eval.py --ckpt unet_ms_gpu_retrain_best.pt --out_dir nifc_eval_output/

Env overrides:
  NIFC_CKPT      path to model checkpoint (default: unet_ms_gpu_retrain_best.pt)
  NIFC_OUT_DIR   output directory         (default: nifc_eval_output)
  NIFC_FIRE_THR  fire mask threshold      (default: 0.1)
"""

import os, sys, json, datetime, argparse, urllib.request, urllib.parse
import numpy as np
import netCDF4 as nc
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from shapely.geometry import shape, Point

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

WRF_DIR    = Path("/scratch/wdt/sfire/new_wrfxpy/wrfxpy/wksp/testing-NAM218_evans_canyon/wrf")
SHARD_DIR  = Path("/home/abazan/wrfout_sandbox/vit_dataset_evans_tplus1/shards")

# NIFC WFIGS target: Evans Canyon — polygon timestamp is 2020-09-02 20:00 UTC
NIFC_FIRE_ID   = "2020-WASES-000565"
NIFC_TIMESTAMP = datetime.datetime(2020, 9, 2, 20, 0, 0, tzinfo=datetime.timezone.utc)

# WRF file matching the NIFC perimeter timestamp
WRF_TARGET    = WRF_DIR / "wrfout_d03_2020-09-02_20:00:00"
TARGET_H = TARGET_W = 99       # model grid


# ─────────────────────────────────────────────────────────────────────────────
# 1. Download NIFC perimeter polygon
# ─────────────────────────────────────────────────────────────────────────────

def download_nifc_polygon(fire_id: str, cache_path: Path):
    """Download Evans Canyon perimeter from WFIGS API. Returns shapely geometry."""
    if cache_path.exists():
        print(f"  [nifc] using cached perimeter: {cache_path}")
        with open(cache_path) as f:
            geojson = json.load(f)
    else:
        print(f"  [nifc] downloading perimeter for {fire_id} …")
        base = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
                "/WFIGS_Interagency_Perimeters/FeatureServer/0/query")
        params = urllib.parse.urlencode({
            "where": f"attr_UniqueFireIdentifier = '{fire_id}'",
            "outFields": "poly_IncidentName,poly_PolygonDateTime,poly_GISAcres",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": 50,
            "orderByFields": "poly_PolygonDateTime ASC",
        })
        with urllib.request.urlopen(f"{base}?{params}", timeout=20) as r:
            geojson = json.loads(r.read())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(geojson, f)
        print(f"  [nifc] cached to {cache_path}")

    feats = geojson.get("features", [])
    if not feats:
        raise RuntimeError(f"No NIFC features found for fire_id={fire_id}")

    # Use the snapshot with largest area (most complete perimeter)
    best = max(feats, key=lambda x: x["properties"].get("poly_GISAcres") or 0)
    ts   = best["properties"].get("poly_PolygonDateTime")
    dt   = datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc) if ts else None
    acres = best["properties"].get("poly_GISAcres", 0)
    name  = best["properties"].get("poly_IncidentName", "?")
    print(f"  [nifc] {name} | {acres:,.0f} acres | {dt}")
    print(f"  [nifc] {len(feats)} snapshot(s) found, using largest")
    return shape(best["geometry"]), dt, acres


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load WRF grid and fire state
# ─────────────────────────────────────────────────────────────────────────────

def load_wrf_grid(wrf_file: Path, target_h=99, target_w=99):
    """Return lat/lon arrays resized to target grid (same resize as dataset builder)."""
    ds = nc.Dataset(str(wrf_file))
    xlat = ds.variables["XLAT"][0]   # (H, W)
    xlon = ds.variables["XLONG"][0]
    ds.close()
    # same resize factor used in build_pnw_dataset.py
    zy = target_h / xlat.shape[0]
    zx = target_w / xlat.shape[1]
    xlat_r = zoom(xlat, (zy, zx), order=1).astype(np.float32)
    xlon_r = zoom(xlon, (zy, zx), order=1).astype(np.float32)
    return xlat_r, xlon_r


def load_wrf_fire_area(wrf_file: Path, target_h=99, target_w=99, thr=0.0001):
    """Read FIRE_AREA from wrfout, resize to target grid, binarize."""
    ds = nc.Dataset(str(wrf_file))
    fa = ds.variables["FIRE_AREA"][0]   # (H_fire, W_fire) — fire sub-grid
    ds.close()
    # FIRE_AREA is on the fire sub-grid (sr_x × sr_y finer than mass grid).
    # Coarsen to mass grid first via block mean, then resize to target.
    # Evans Canyon: domain 123×123, fire grid likely 615×615 (sr=5)
    mass_h, mass_w = 123, 123
    if fa.shape[0] > mass_h:
        sr = fa.shape[0] // mass_h
        # reshape and mean
        fa = fa[:mass_h * sr, :mass_w * sr]
        fa = fa.reshape(mass_h, sr, mass_w, sr).mean(axis=(1, 3))
    zy = target_h / fa.shape[0]
    zx = target_w / fa.shape[1]
    fa_r = zoom(fa, (zy, zx), order=1).astype(np.float32)
    return (fa_r > thr).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rasterize NIFC polygon onto WRF grid
# ─────────────────────────────────────────────────────────────────────────────

def rasterize_polygon(polygon, xlat: np.ndarray, xlon: np.ndarray) -> np.ndarray:
    """
    Point-in-polygon rasterization of a WGS84 polygon onto a curvilinear WRF grid.
    Returns binary mask (H, W) matching xlat/xlon shape.
    """
    H, W = xlat.shape
    mask = np.zeros((H, W), dtype=np.float32)
    # vectorised bounding-box pre-filter
    minx, miny, maxx, maxy = polygon.bounds
    for i in range(H):
        for j in range(W):
            lat, lon = float(xlat[i, j]), float(xlon[i, j])
            if miny <= lat <= maxy and minx <= lon <= maxx:
                if polygon.contains(Point(lon, lat)):
                    mask[i, j] = 1.0
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 4. Load model and run one-step prediction
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device):
    """Load UNet, inferring in_c / out_c directly from the checkpoint weights."""
    import train_unet as tu

    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = raw.get("model", raw.get("ema_model", raw))

    # in_c: first depthwise conv has shape [in_c, 1, kH, kW]
    dw_key = next(k for k in state if k.endswith(".dw.weight"))
    in_c = state[dw_key].shape[0]

    # out_c: final head conv has shape [out_c, base, 1, 1]
    head_key = next(k for k in reversed(list(state)) if "head" in k and "weight" in k)
    out_c = state[head_key].shape[0]

    model = tu.UNet(in_c=in_c, out_c=out_c,
                    base=tu.BASE_CHANNELS).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  [model] loaded {ckpt_path.name}  in_c={in_c}  out_c={out_c}")
    return model


def normalize_batch(X: torch.Tensor) -> torch.Tensor:
    """Per-sample Z-score normalization — must match train_unet.normalize_batch exactly."""
    mean = X.mean(dim=(0, 2, 3), keepdim=True)
    std  = X.std(dim=(0, 2, 3),  keepdim=True) + 1e-6
    return (X - mean) / std


def predict_from_shard(model, shard_path: Path, device, fire_thr=0.1):
    """Run model on a single shard (with per-sample norm). Returns (binary, prob, gt)."""
    s = np.load(str(shard_path))
    X = torch.from_numpy(s["X"]).unsqueeze(0).to(device)   # (1, C, H, W)
    X = normalize_batch(X)                                  # same as training
    with torch.no_grad():
        pred = model(X)                                     # (1, out_c, H, W)
    pred_fire = torch.sigmoid(pred[0, 0]).cpu().numpy()     # (H, W)
    gt_fire   = s["Y"][0]                                   # (H, W)
    return (pred_fire > fire_thr).astype(np.float32), pred_fire, gt_fire


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute IoU, Dice, precision, recall for binary masks."""
    p = pred.astype(bool)
    t = target.astype(bool)
    inter = (p & t).sum()
    union = (p | t).sum()
    iou   = inter / union if union > 0 else 0.0
    dice  = 2 * inter / (p.sum() + t.sum()) if (p.sum() + t.sum()) > 0 else 0.0
    prec  = inter / p.sum() if p.sum() > 0 else 0.0
    rec   = inter / t.sum() if t.sum() > 0 else 0.0
    return {"iou": iou, "dice": dice, "precision": prec, "recall": rec,
            "pred_cells": int(p.sum()), "obs_cells": int(t.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_plot(
    nifc_mask, wrf_mask, model_pred_prob, model_mask,
    xlat, xlon, metrics_wrf, metrics_model, nifc_acres,
    save_path: Path, nifc_dt
):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    ext = [xlon.min(), xlon.max(), xlat.min(), xlat.max()]

    def _show(ax, data, title, cmap="hot_r", vmax=1):
        im = ax.imshow(data, extent=ext, origin="lower", cmap=cmap,
                       vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, fraction=0.046)

    _show(axes[0], nifc_mask, f"NIFC Observed\n{nifc_dt:%Y-%m-%d %H:%M} UTC\n{nifc_acres:,.0f} acres")
    _show(axes[1], wrf_mask,
          f"WRF-SFIRE Simulated\nvs NIFC  IoU={metrics_wrf['iou']:.3f}  Dice={metrics_wrf['dice']:.3f}")
    _show(axes[2], model_pred_prob,
          "Model Predicted (prob)", cmap="RdYlGn_r")
    _show(axes[3], model_mask,
          f"Model Predicted (binary)\nvs NIFC  IoU={metrics_model['iou']:.3f}  Dice={metrics_model['dice']:.3f}")

    # overlay NIFC contour on model panel
    for ax in axes[1:]:
        ax.contour(np.flip(nifc_mask, axis=0),
                   levels=[0.5], colors="cyan", linewidths=1.5,
                   extent=ext)
    axes[1].contour(np.flip(nifc_mask, axis=0), levels=[0.5],
                    colors="cyan", linewidths=1.5, extent=ext)

    fig.suptitle("Evans Canyon 2020 — NIFC vs WRF-SFIRE vs UNet Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {save_path}")


def make_diff_plot(nifc_mask, model_mask, xlat, xlon, save_path: Path):
    """False-positive / false-negative breakdown map."""
    tp = ((model_mask == 1) & (nifc_mask == 1)).astype(float)
    fp = ((model_mask == 1) & (nifc_mask == 0)).astype(float)
    fn = ((model_mask == 0) & (nifc_mask == 1)).astype(float)

    rgb = np.zeros((*nifc_mask.shape, 3))
    rgb[..., 0] = fp          # red = false positive
    rgb[..., 1] = tp          # green = true positive
    rgb[..., 2] = fn * 0.7    # blue = false negative

    ext = [xlon.min(), xlon.max(), xlat.min(), xlat.max()]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(rgb, extent=ext, origin="lower", aspect="auto")
    ax.set_title("Model vs NIFC  —  TP=green  FP=red  FN=blue", fontsize=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="green", label="True Positive"),
                        Patch(color="red",   label="False Positive"),
                        Patch(color=(0,0,.7), label="False Negative")],
              loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default=os.environ.get("NIFC_CKPT",    "unet_ms_gpu_retrain_best.pt"))
    parser.add_argument("--out_dir", default=os.environ.get("NIFC_OUT_DIR", "nifc_eval_output"))
    parser.add_argument("--fire_thr",type=float, default=float(os.environ.get("NIFC_FIRE_THR", "0.1")))
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    ckpt     = Path(args.ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"NIFC Evaluation — Evans Canyon 2020")
    print(f"  checkpoint : {ckpt}")
    print(f"  output dir : {out_dir}")
    print(f"  device     : {device}")
    print(f"{'='*60}\n")

    # ── 1. NIFC perimeter ─────────────────────────────────────────────────────
    polygon, nifc_dt, nifc_acres = download_nifc_polygon(
        NIFC_FIRE_ID, cache_path=out_dir / "nifc_evans_canyon.geojson"
    )

    # ── 2. WRF grid (99×99) ───────────────────────────────────────────────────
    print(f"  [wrf] loading grid from {WRF_TARGET.name}")
    xlat, xlon = load_wrf_grid(WRF_TARGET, TARGET_H, TARGET_W)
    print(f"  [wrf] grid {xlat.shape}  lat [{xlat.min():.3f},{xlat.max():.3f}]  "
          f"lon [{xlon.min():.3f},{xlon.max():.3f}]")

    # ── 3. Rasterize NIFC polygon ─────────────────────────────────────────────
    nifc_cache = out_dir / "nifc_mask_99x99.npy"
    if nifc_cache.exists():
        print("  [nifc] loading cached rasterized mask")
        nifc_mask = np.load(str(nifc_cache))
    else:
        print(f"  [nifc] rasterizing polygon onto {TARGET_H}×{TARGET_W} grid …")
        nifc_mask = rasterize_polygon(polygon, xlat, xlon)
        np.save(str(nifc_cache), nifc_mask)
    print(f"  [nifc] mask  fire cells={nifc_mask.sum():.0f}/{nifc_mask.size}  "
          f"({100*nifc_mask.mean():.1f}% coverage)")

    if nifc_mask.sum() == 0:
        print("\n  WARNING: NIFC polygon does not overlap WRF domain!")
        print("  Polygon bounds:", polygon.bounds)
        print("  WRF domain:    lat", xlat.min(), "-", xlat.max(),
              "  lon", xlon.min(), "-", xlon.max())
        sys.exit(1)

    # ── 4. WRF-SFIRE simulated fire area at target time ───────────────────────
    print(f"  [wrf] reading FIRE_AREA at {WRF_TARGET.name}")
    wrf_mask = load_wrf_fire_area(WRF_TARGET, TARGET_H, TARGET_W)
    print(f"  [wrf] fire cells={wrf_mask.sum():.0f}/{wrf_mask.size}  "
          f"({100*wrf_mask.mean():.1f}% coverage)")

    # ── 5. Model prediction ───────────────────────────────────────────────────
    # Identify the shard at the target timestamp by time-based index:
    # Sim starts 2020-08-31 00:00 UTC, 15-min outputs, NIFC target = 2020-09-02 20:00 UTC.
    # Target step = (NIFC_TIMESTAMP - sim_start) / 15min = 272.
    # With 334 valid shards from 337 timesteps, estimated shard ≈ round(272 * 334/337).
    SIM_START      = datetime.datetime(2020, 8, 31, 0, 0, tzinfo=datetime.timezone.utc)
    SIM_INTERVAL   = datetime.timedelta(minutes=15)
    SIM_TOTAL_STEPS = 337

    target_step    = int((NIFC_TIMESTAMP - SIM_START) / SIM_INTERVAL)
    shards         = sorted(SHARD_DIR.glob("*.npz"))
    n_shards       = len(shards)
    est_shard_idx  = round(target_step * n_shards / SIM_TOTAL_STEPS)
    est_shard_idx  = max(0, min(n_shards - 1, est_shard_idx))

    # Verify by checking Y channel IoU vs wrf_mask in a ±10 shard window
    FIRE_CH = 4
    window = range(max(0, est_shard_idx - 10), min(n_shards, est_shard_idx + 11))
    best_idx, best_iou = est_shard_idx, -1.0
    for idx in window:
        s = np.load(str(shards[idx]))
        shard_fire = (s["Y"][0] > args.fire_thr).astype(float)
        inter = (shard_fire * wrf_mask).sum()
        union = np.clip(shard_fire + wrf_mask, 0, 1).sum()
        iou = float(inter / union) if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_idx = iou, idx

    best_shard = shards[best_idx]
    print(f"  [shard] target step={target_step}/{SIM_TOTAL_STEPS}  "
          f"est_idx={est_shard_idx}  best={best_idx}  IoU_Y_vs_WRF={best_iou:.3f}")
    print(f"  [shard] using {best_shard.name}")

    os.environ["OUT_C"] = "3"
    model = load_model(ckpt, device)
    model_mask, model_prob, gt_mask = predict_from_shard(model, best_shard, device, args.fire_thr)

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    m_wrf   = compute_metrics(wrf_mask,   nifc_mask)
    m_model = compute_metrics(model_mask, nifc_mask)
    m_vs_wrf = compute_metrics(model_mask, wrf_mask)

    print(f"\n{'─'*55}")
    print(f"  Comparison vs NIFC observed ({nifc_dt:%Y-%m-%d %H:%M} UTC)")
    print(f"{'─'*55}")
    print(f"  {'':30s}  {'IoU':>6}  {'Dice':>6}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'WRF-SFIRE sim vs NIFC':30s}  {m_wrf['iou']:6.3f}  {m_wrf['dice']:6.3f}  "
          f"{m_wrf['precision']:6.3f}  {m_wrf['recall']:6.3f}")
    print(f"  {'Model predicted vs NIFC':30s}  {m_model['iou']:6.3f}  {m_model['dice']:6.3f}  "
          f"{m_model['precision']:6.3f}  {m_model['recall']:6.3f}")
    print(f"  {'Model vs WRF-SFIRE':30s}  {m_vs_wrf['iou']:6.3f}  {m_vs_wrf['dice']:6.3f}  "
          f"{m_vs_wrf['precision']:6.3f}  {m_vs_wrf['recall']:6.3f}")
    print(f"{'─'*55}")
    print(f"  NIFC observed area:  {nifc_acres:,.0f} acres  ({nifc_mask.sum():.0f} grid cells)")
    print(f"  WRF-SFIRE area:      {wrf_mask.sum():.0f} grid cells")
    print(f"  Model predicted:     {model_mask.sum():.0f} grid cells")

    # ── 7. Save results ───────────────────────────────────────────────────────
    results = {
        "fire": "Evans Canyon 2020", "fire_id": NIFC_FIRE_ID,
        "nifc_timestamp": str(nifc_dt), "nifc_acres": nifc_acres,
        "wrf_target_file": WRF_TARGET.name,
        "best_shard": best_shard.name, "shard_match_iou": best_iou,
        "checkpoint": str(ckpt),
        "wrf_vs_nifc": m_wrf, "model_vs_nifc": m_model, "model_vs_wrf": m_vs_wrf,
    }
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [results] saved → {results_path}")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    make_comparison_plot(
        nifc_mask, wrf_mask, model_prob, model_mask, xlat, xlon,
        m_wrf, m_model, nifc_acres,
        save_path=out_dir / "comparison_4panel.png",
        nifc_dt=nifc_dt,
    )
    make_diff_plot(
        nifc_mask, model_mask, xlat, xlon,
        save_path=out_dir / "model_vs_nifc_diff.png",
    )
    make_diff_plot(
        nifc_mask, wrf_mask, xlat, xlon,
        save_path=out_dir / "wrf_vs_nifc_diff.png",
    )

    print(f"\n  Done. Results in {out_dir}/")
    return results


if __name__ == "__main__":
    main()
