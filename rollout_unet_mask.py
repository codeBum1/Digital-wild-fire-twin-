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
OUT_DIR = "/home/abazan/wrfout_sandbox/rollout_k3_mask50_skip"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIRE_THR = 0.1
PB_THR = 0.8

# choose one sample to rollout
SAMPLE_IDX = 22

# how many rollout steps
N_STEPS = 6


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normalize_sample(Xb):
    mean = Xb.mean(dim=(0, 2, 3), keepdim=True)
    std = Xb.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    return (Xb - mean) / std


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = WrfVitShardDataset(SHARDS_DIR)
    print("Dataset len:", len(ds))

    X0, Y0 = ds[0]
    model = UNet(in_c=X0.shape[0], out_c=Y0.shape[0])
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu", weights_only=True))
    model.to(DEVICE).eval()

    X, Y = ds[SAMPLE_IDX]
    X_roll = X.clone()  # this will be updated each step

    # store initial fire
    fire_prev = X_roll[-1].numpy()

    for step in range(N_STEPS):
        Xb = X_roll.unsqueeze(0).to(DEVICE)
        Xb_n = normalize_sample(Xb)

        with torch.no_grad():
            pred = model(Xb_n).cpu().squeeze(0)

        fire_logits = pred[0].numpy()
        fire_prob = sigmoid(fire_logits)
        fire_mask = (fire_prob > PB_THR).astype(np.float32)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Sample {SAMPLE_IDX} rollout step {step+1}", fontsize=14)

        ax[0].imshow(fire_prev, cmap="hot", vmin=0, vmax=1)
        ax[0].set_title("Input fire")
        ax[0].axis("off")

        im = ax[1].imshow(fire_prob, cmap="viridis", vmin=0, vmax=1)
        ax[1].set_title("Pred fire confidence")
        ax[1].axis("off")
        cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("P(fire)", rotation=90)

        ax[2].imshow(fire_mask, cmap="gray", vmin=0, vmax=1)
        ax[2].set_title(f"Pred mask > {PB_THR}")
        ax[2].axis("off")

        out = os.path.join(OUT_DIR, f"rollout_step_{step+1:02d}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", out)

        # feed prediction back into the fire channel
        X_roll[-1] = torch.from_numpy(fire_mask).float()
        fire_prev = fire_mask.copy()

    print("Done. Rollout images in:", OUT_DIR)


if __name__ == "__main__":
    main()
