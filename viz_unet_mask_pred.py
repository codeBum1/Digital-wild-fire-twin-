import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from wrf_vit_dataset import WrfVitShardDataset

SHARDS_DIR = os.environ.get("SHARDS_DIR", "/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards")
CKPT_PATH  = os.environ.get("CKPT_PATH",  "/home/abazan/wrfout_sandbox/unet_ckpt.pt")

FIRE_THR   = float(os.environ.get("FIRE_THR", "0.1"))   # true mask threshold on yf intensity
PB_THR     = float(os.environ.get("PB_THR", "0.8"))     # predicted prob -> mask threshold
SAMPLE_IDX = int(os.environ.get("SAMPLE_IDX", "0"))
OUT_PATH   = os.environ.get("OUT_PATH",  f"viz_pred_{SAMPLE_IDX}.png")

# -------- Model (must match train_unet.py) --------
class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.enc1 = Block(in_c, 64)
        self.enc2 = Block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = Block(128, 64)
        self.out  = nn.Conv2d(64, out_c, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1)
        x3 = self.enc2(x2)
        x4 = F.interpolate(x3, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x5 = self.dec1(x4)
        return self.out(x5)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = WrfVitShardDataset(SHARDS_DIR)
    n = len(ds)
    if not (0 <= SAMPLE_IDX < n):
        raise SystemExit(f"SAMPLE_IDX={SAMPLE_IDX} out of range [0..{n-1}]")

    # Load shard meta (optional: printed on figure)
    shard_files = sorted(glob.glob(os.path.join(SHARDS_DIR, "*.npz")))
    npz_path = shard_files[SAMPLE_IDX] if SAMPLE_IDX < len(shard_files) else None
    meta_str = None
    if npz_path is not None:
        z = np.load(npz_path, allow_pickle=True)
        if "meta" in z.files:
            meta_str = str(z["meta"])

    # Get sample
    X, Y = ds[SAMPLE_IDX]       # X:(C,H,W), Y:(F,H,W)
    C, H, W = X.shape
    out_c = Y.shape[0]

    # Normalize like training (per-sample)
    Xb = X.unsqueeze(0).to(device)
    mean = Xb.mean(dim=(0,2,3), keepdim=True)
    std  = Xb.std(dim=(0,2,3), keepdim=True) + 1e-6
    Xb_n = (Xb - mean) / std

    # Load model weights
    model = UNet(in_c=C, out_c=out_c).to(device)
    sd = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        pred = model(Xb_n).cpu().squeeze(0)  # (F,H,W)
        pf_logits = pred[0]                  # (H,W)
        pf_prob = torch.sigmoid(pf_logits).numpy()

    # Inputs/labels
    xf = X[-1].numpy()            # fire(t) input channel (your last channel)
    yf = Y[0].numpy()             # fire(t+K) intensity label (channel 0)
    y_true = (yf > FIRE_THR)
    y_pred = (pf_prob > PB_THR)

    # Metrics on this sample
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    iou = float(inter / union) if union > 0 else float("nan")
    prob_mae = float(np.mean(np.abs(pf_prob - y_true.astype(np.float32))))
    pred_frac = float(y_pred.mean())
    true_frac = float(y_true.mean())

    # ---- Plot (pixel-space “map style”) ----
    fig = plt.figure(figsize=(14, 10))
    title = (
        f"UNet Fire Mask Prediction (sample {SAMPLE_IDX})\n"
        f"FIRE_THR={FIRE_THR} | PB_THR={PB_THR} | IoU={iou:.4f} | ProbMAE={prob_mae:.4f} | "
        f"pred_frac={pred_frac:.4f} true_frac={true_frac:.4f}"
    )
    fig.suptitle(title, fontsize=13)

    def show(ax, img, t, cmap="viridis", alpha=1.0):
        h = ax.imshow(img, origin="lower", cmap=cmap, alpha=alpha)
        ax.set_title(t)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        plt.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

    # Panel 1: Input fire(t)
    ax1 = plt.subplot(2,2,1)
    show(ax1, xf, "Input: fire(t) intensity (X[-1])", cmap="inferno")

    # Panel 2: Ground truth mask over input
    ax2 = plt.subplot(2,2,2)
    show(ax2, xf, f"Ground Truth mask (Y[0] > {FIRE_THR}) over fire(t)", cmap="inferno")
    ax2.imshow(y_true.astype(float), origin="lower", cmap="Blues", alpha=0.35)

    # Panel 3: Pred probability
    ax3 = plt.subplot(2,2,3)
    show(ax3, pf_prob, "Prediction: probability map p(fire at t+K)", cmap="magma")

    # Panel 4: Contours (pred vs true)
    ax4 = plt.subplot(2,2,4)
    show(ax4, xf, "Contours on fire(t): True (cyan) vs Pred (lime)", cmap="inferno")
    ax4.contour(y_true.astype(float), levels=[0.5], colors=["cyan"], linewidths=2)
    ax4.contour(y_pred.astype(float), levels=[0.5], colors=["lime"], linewidths=2)
    ax4.legend(
        [plt.Line2D([0],[0], color="cyan", lw=2),
         plt.Line2D([0],[0], color="lime", lw=2)],
        [f"True mask (Y[0]>{FIRE_THR})", f"Pred mask (p>{PB_THR})"],
        loc="lower right"
    )

    # Add meta text (if present)
    if meta_str:
        fig.text(0.01, 0.01, f"meta: {meta_str[:200]}{'...' if len(meta_str) > 200 else ''}",
                 fontsize=9, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    plt.savefig(OUT_PATH, dpi=200)
    print("Saved:", OUT_PATH)
    if npz_path:
        print("NPZ:", npz_path)

if __name__ == "__main__":
    main()
