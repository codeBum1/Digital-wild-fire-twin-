import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wrf_vit_dataset import WrfVitShardDataset
from train_unet import UNet

# -------- Config --------
SHARDS_DIR = "/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards"
CKPT_PATH = "/home/abazan/wrfout_sandbox/unet_ckpt_k3_mask50_skip.pt"
OUT_DIR = "/home/abazan/wrfout_sandbox/viz_unet_tplus3_mask50_skip"

N_SAMPLES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIRE_THR = 0.1   # threshold for true fire mask
PB_THR = 0.5     # threshold for predicted probability -> predicted mask


def robust_percentile(arr, lo=1.0, hi=99.0):
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0.0, 1.0
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = WrfVitShardDataset(SHARDS_DIR)
    print("Dataset len:", len(ds))

    X0, Y0 = ds[0]
    model = UNet(in_c=X0.shape[0], out_c=Y0.shape[0])
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu", weights_only=True))
    model.to(DEVICE).eval()

    idxs = np.linspace(0, len(ds) - 1, N_SAMPLES).astype(int).tolist()

    for idx in idxs:
        X, Y = ds[idx]  # X:(C,H,W), Y:(F,H,W)

        xf = X[-1].numpy()                      # FIRE(t)
        yf = Y[0].numpy()                       # FIRE(t+3) absolute field
        yb = (yf > FIRE_THR).astype(np.float32) # true fire mask

        Xb = X.unsqueeze(0).to(DEVICE)          # (1,C,H,W)

        # same normalization as training
        mean = Xb.mean(dim=(0, 2, 3), keepdim=True)
        std = Xb.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        Xb = (Xb - mean) / std

        with torch.no_grad():
            pred = model(Xb).cpu().squeeze(0)   # (F,H,W)

        pf_logits = pred[0].numpy()

        # confidence / probability of fire
        pf_prob = 1.0 / (1.0 + np.exp(-pf_logits))

        # thresholded predicted fire mask
        pb = (pf_prob > PB_THR).astype(np.float32)

        # binary error mask (1 where predicted mask != true mask)
        err_mask = np.abs(pb - yb)

        fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(
            f"Sample {idx}: K=3 Fire Mask Forecast | pred_thr={PB_THR} | true_thr={FIRE_THR}",
            fontsize=14
        )

        # Panel 1: input FIRE(t)
        vmin0, vmax0 = robust_percentile(xf, 1, 99)
        if vmax0 <= vmin0:
            vmin0, vmax0 = 0.0, max(1e-6, float(xf.max()))
        ax[0].imshow(xf, vmin=vmin0, vmax=vmax0, cmap="hot")
        ax[0].set_title("Input FIRE(t)")
        ax[0].axis("off")

        # Panel 2: true mask
        ax[1].imshow(yb, vmin=0.0, vmax=1.0, cmap="gray")
        ax[1].set_title(f"True Mask > {FIRE_THR}")
        ax[1].axis("off")

        # Panel 3: predicted probability / confidence
        im2 = ax[2].imshow(pf_prob, vmin=0.0, vmax=1.0, cmap="viridis")
        ax[2].set_title("Pred Fire Confidence")
        ax[2].axis("off")
        cbar = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("P(fire)", rotation=90)

        # Panel 4: thresholded predicted mask
        ax[3].imshow(pb, vmin=0.0, vmax=1.0, cmap="gray")
        ax[3].set_title(f"Pred Mask > {PB_THR}")
        ax[3].axis("off")

        # Panel 5: binary error mask
        ax[4].imshow(err_mask, vmin=0.0, vmax=1.0, cmap="magma")
        ax[4].set_title("Error Mask")
        ax[4].axis("off")

        out = os.path.join(OUT_DIR, f"sample_{idx:05d}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", out)

    print("Done. Images in:", OUT_DIR)


if __name__ == "__main__":
    main()
