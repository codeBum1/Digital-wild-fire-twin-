#!/usr/bin/env python3
"""
smoke_viz.py – Smoke / PM2.5 prediction visualisation

Runs the trained UNet on held-out test shards and produces side-by-side
figures showing the fire perimeter prediction alongside the auxiliary
regression outputs: Rate of Spread (ROS), Flame Length, and PM2.5 proxy.

Usage
-----
    # Default: loads unet_v2_best.pt, processes all test shards
    python smoke_viz.py

    # Override checkpoint / dataset / output dir via env vars
    LOAD_CKPT_PATH=unet_ms_gpu_best.pt  \\
    SHARDS_DIR=vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards \\
    OUT_DIR=smoke_viz_out \\
    python smoke_viz.py

    # Only visualise a specific shard index
    SAMPLE_IDX=5 python smoke_viz.py

Env vars
--------
    LOAD_CKPT_PATH   checkpoint to load (falls back to unet_v2_best.pt)
    SHARDS_DIR       NPZ shard directory
    OUT_DIR          output directory  (default: smoke_viz_out/)
    SAMPLE_IDX       if set, only process that one shard index
    FIRE_THR         fire binarisation threshold  (default: 0.1)
    MAKE_GIF         if 1, stitch per-sample PNGs into an animated GIF
    BASE_CHANNELS    UNet base width  (default: 64)
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F

from wrf_vit_dataset import WrfVitShardDataset

# ── re-use architecture + helpers from train_unet ────────────────────────────
# Import only what we need; avoids triggering the full training main().
from train_unet import (
    UNet,
    normalize_batch,
    BASE_CHANNELS,
    DROPOUT,
    DROP_PATH_RATE,
    FIRE_THR,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def _env(k, default, cast=str):
    return cast(os.environ.get(k, str(default)))

SHARDS_DIR     = _env("SHARDS_DIR",
    "/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards")
LOAD_CKPT_PATH = _env("LOAD_CKPT_PATH", "/home/abazan/wrfout_sandbox/unet_v2_best.pt")
OUT_DIR        = _env("OUT_DIR",        "smoke_viz_out")
SAMPLE_IDX     = _env("SAMPLE_IDX",     "",             str)   # "" = all
MAKE_GIF       = _env("MAKE_GIF",       "0",            int)

# Channel names for Y[:,1:] (order must match dataset build config)
AUX_CHANNEL_LABELS = ["Rate of Spread (ROS)", "Flame Length", "PM2.5 proxy"]

# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, in_c: int, out_c: int, device: torch.device):
    model = UNet(in_c=in_c, out_c=out_c, base=BASE_CHANNELS,
                 dropout=DROPOUT, drop_path_rate=DROP_PATH_RATE).to(device)
    if not Path(ckpt_path).exists():
        print(f"[warn] checkpoint not found: {ckpt_path}  — using random weights")
        return model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # Strip DDP prefix if present
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] missing keys: {missing[:4]}{'...' if len(missing)>4 else ''}")
    if unexpected:
        print(f"  [ckpt] unexpected keys: {unexpected[:4]}{'...' if len(unexpected)>4 else ''}")
    print(f"  [ckpt] loaded from {ckpt_path}")
    return model


@torch.no_grad()
def predict(model, X: torch.Tensor, device: torch.device):
    """Return raw prediction tensor (out_c, H, W) and fire probability (H, W)."""
    Xb = X.unsqueeze(0).to(device)
    Xn = normalize_batch(Xb)
    pred = model(Xn).squeeze(0).cpu()   # (out_c, H, W)
    fire_prob = torch.sigmoid(pred[0]).numpy()
    return pred.numpy(), fire_prob


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample figure
# ─────────────────────────────────────────────────────────────────────────────

def save_smoke_figure(
    X:         np.ndarray,   # (C, H, W)
    Y:         np.ndarray,   # (F, H, W)
    pred_raw:  np.ndarray,   # (out_c, H, W)
    fire_prob: np.ndarray,   # (H, W)
    out_path:  str,
    sample_idx: int = 0,
) -> None:
    """
    Multi-panel figure:
    Row 1: fire state (input) | fire prob (pred) | true fire mask
    Row 2: aux channel predictions vs targets  (one col per aux channel)
    """
    fire_thr   = FIRE_THR
    fire_pred  = fire_prob > fire_thr
    true_fire  = Y[0] > fire_thr
    n_aux      = pred_raw.shape[0] - 1    # number of auxiliary output channels
    n_aux_vis  = min(n_aux, 3)            # cap at 3 (ROS, FLAME_LENGTH, PM2.5)

    ncols = max(3, n_aux_vis * 2)
    fig   = plt.figure(figsize=(5 * ncols, 10), facecolor="#1a1a2e")
    gs    = gridspec.GridSpec(2, ncols, figure=fig, hspace=0.35, wspace=0.3)

    imkw_fire = dict(origin="lower", cmap="hot",    vmin=0, vmax=1, interpolation="bilinear")
    imkw_bin  = dict(origin="lower", cmap="Greys_r",vmin=0, vmax=1, interpolation="nearest")

    def _styled_ax(ax, data, cmap_kw, title, clabel=""):
        ax.set_facecolor("#0d0d1a")
        im = ax.imshow(data, **cmap_kw)
        ax.set_title(title, color="white", fontsize=10, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7)
        if clabel:
            cb.set_label(clabel, color="white", fontsize=8)
        return ax

    # Row 1: fire channels
    ax_fire_in  = fig.add_subplot(gs[0, 0])
    ax_fire_pred = fig.add_subplot(gs[0, 1])
    ax_fire_true = fig.add_subplot(gs[0, 2])

    _styled_ax(ax_fire_in,   X[-1],               imkw_fire, "Fire state t (input)",       "fire fraction")
    _styled_ax(ax_fire_pred, fire_prob,            imkw_fire, "Fire prob (predicted)",       "P(fire)")
    _styled_ax(ax_fire_true, true_fire.astype(np.float32), imkw_bin, "True fire mask t+K",  "")

    # True/predicted perimeter contours on fire panels
    for ax in (ax_fire_pred, ax_fire_true):
        ax.contour(fire_prob,  levels=[fire_thr], colors=["cyan"], linewidths=[1.2])
        ax.contour(true_fire.astype(np.float32), levels=[0.5], colors=["lime"], linewidths=[1.2])
    ax_fire_in.contour(X[-1], levels=[fire_thr], colors=["red"], linewidths=[1.2])

    # IoU annotation
    tp = int((fire_pred & true_fire).sum())
    un = int((fire_pred | true_fire).sum())
    iou = tp / un if un > 0 else 0.0
    ax_fire_pred.set_title(f"Fire prob (pred)  IoU={iou:.3f}", color="white", fontsize=10)

    # Row 2: auxiliary channels (pred vs true)
    aux_cmaps = ["plasma", "inferno", "YlOrRd"]
    for ch in range(n_aux_vis):
        col_pred = ch * 2
        col_true = ch * 2 + 1
        if col_pred >= ncols or col_true >= ncols:
            break

        label = AUX_CHANNEL_LABELS[ch] if ch < len(AUX_CHANNEL_LABELS) else f"Aux ch {ch+1}"
        pred_ch = pred_raw[ch + 1]
        true_ch = Y[ch + 1] if ch + 1 < Y.shape[0] else np.zeros_like(pred_ch)

        vmax = max(float(pred_ch.max()), float(true_ch.max()), 1e-6)
        kw   = dict(origin="lower", cmap=aux_cmaps[ch % len(aux_cmaps)],
                    vmin=0, vmax=vmax, interpolation="bilinear")

        ax_p = fig.add_subplot(gs[1, col_pred])
        ax_t = fig.add_subplot(gs[1, col_true])

        mae = float(np.abs(pred_ch - true_ch).mean())
        _styled_ax(ax_p, pred_ch, kw, f"{label}\n(pred)  MAE={mae:.4f}", "")
        _styled_ax(ax_t, true_ch, kw, f"{label}\n(true / proxy)",        "")

    fig.suptitle(f"Smoke/PM2.5 Channel Predictions   sample={sample_idx}",
                 color="white", fontsize=13, y=1.01)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color="cyan", lw=2, label=f"Pred fire front (p>{fire_thr})"),
        Line2D([0],[0], color="lime", lw=2, label="True fire mask"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GIF helper
# ─────────────────────────────────────────────────────────────────────────────

def make_gif(frame_paths: list, out_path: str, fps: int = 2) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("  [gif] Pillow not available; skipping GIF")
        return
    imgs = [Image.open(p) for p in frame_paths]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"  [gif] → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = WrfVitShardDataset(SHARDS_DIR)
    X0, Y0 = ds[0]
    in_c, out_c = int(X0.shape[0]), int(Y0.shape[0])
    print(f"Dataset: {len(ds)} shards  in_c={in_c}  out_c={out_c}")

    model = load_model(LOAD_CKPT_PATH, in_c, out_c, device)
    model.eval()

    # Which samples to process?
    if SAMPLE_IDX.strip():
        indices = [int(SAMPLE_IDX)]
    else:
        indices = list(range(len(ds)))

    frame_paths = []
    for i in indices:
        X, Y = ds[i]
        pred_raw, fire_prob = predict(model, X, device)
        out_path = str(out_dir / f"smoke_sample_{i:04d}.png")
        save_smoke_figure(
            X.numpy(), Y.numpy(), pred_raw, fire_prob,
            out_path=out_path, sample_idx=i,
        )
        frame_paths.append(out_path)

        # Print quick stats
        n_aux = pred_raw.shape[0] - 1
        aux_info = ""
        if n_aux > 0 and Y.shape[0] > 1:
            for ch in range(min(n_aux, 3)):
                lbl = AUX_CHANNEL_LABELS[ch] if ch < len(AUX_CHANNEL_LABELS) else f"aux{ch}"
                mae = float(np.abs(pred_raw[ch+1] - Y[ch+1].numpy()).mean())
                aux_info += f"  {lbl.split()[0]}_mae={mae:.4f}"
        fire_iou = float(
            ((fire_prob > FIRE_THR) & (Y[0].numpy() > FIRE_THR)).sum() /
            max(((fire_prob > FIRE_THR) | (Y[0].numpy() > FIRE_THR)).sum(), 1)
        )
        print(f"  sample {i:4d}  fire_iou={fire_iou:.3f}{aux_info}")

    if MAKE_GIF and len(frame_paths) > 1:
        make_gif(frame_paths, str(out_dir / "smoke_animation.gif"))

    print(f"\n✓ Smoke viz complete. Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
