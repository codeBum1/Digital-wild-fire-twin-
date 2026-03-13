import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from wrf_vit_dataset import WrfVitShardDataset
from train_unet import UNet

# ---- paths ----
DATASET_PATH = "/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus1/shards"
CKPT_PATH    = "/home/abazan/wrfout_sandbox/unet_ckpt.pt"
OUT_GIF      = "/home/abazan/wrfout_sandbox/unet_overlay_fire_area.gif"

# ---- dataset channel convention (your X has 10 channels) ----
# X channels you built earlier:
# 0:T2 1:Q2 2:PSFC 3:U10 4:V10 5:HGT 6:U_mass(k) 7:V_mass(k) 8:T(k) 9:QVAPOR(k)
HGT_CH = 5

# Y channels:
# 0:FIRE_AREA(t+1) 1:ROS(t+1) 2:FLAME_LENGTH(t+1)
FIRE_CH = 0

# GIF speed
DURATION_MS = 300  # 0.3s per frame

def to_uint8(img2d):
    """Robust normalize -> uint8 (handles constant fields)."""
    a = np.asarray(img2d, dtype=np.float32)
    mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
    if mx - mn < 1e-8:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - mn) / (mx - mn)
    a = np.clip(a, 0, 1)
    return (255 * a).astype(np.uint8)

def fig_to_pil(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    return Image.fromarray(buf)

def main():
    print("Loading dataset:", DATASET_PATH)
    ds = WrfVitShardDataset(DATASET_PATH)
    print("Dataset len:", len(ds))

    print("Loading model:", CKPT_PATH)
    model = UNet()
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model.eval()

    frames = []

    for i in range(len(ds)):
        X, Y = ds[i]  # X: (10,99,99) Y: (3,99,99)

        with torch.no_grad():
            pred = model(X.unsqueeze(0))[0]  # (3,99,99)

        # ---- background "map": terrain ----
        hgt = X[HGT_CH].numpy()
        hgt_u8 = to_uint8(hgt)

        # ---- fire fields ----
        gt_fire   = Y[FIRE_CH].numpy()
        pred_fire = pred[FIRE_CH].numpy()
        err_fire  = np.abs(pred_fire - gt_fire)

        # optional: turn fire into a clearer mask
        # (FIRE_AREA often is 0..1-ish; but keep robust)
        gt_u8   = to_uint8(gt_fire)
        pred_u8 = to_uint8(pred_fire)
        err_u8  = to_uint8(err_fire)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        # Panel 1: GT overlay
        ax[0].imshow(hgt_u8, cmap="gray")
        ax[0].imshow(gt_u8, cmap="inferno", alpha=0.65)
        ax[0].set_title(f"GT FIRE_AREA (t+1)  idx={i}")

        # Panel 2: Pred overlay
        ax[1].imshow(hgt_u8, cmap="gray")
        ax[1].imshow(pred_u8, cmap="inferno", alpha=0.65)
        ax[1].set_title("Pred FIRE_AREA (t+1)")

        # Panel 3: Error overlay
        ax[2].imshow(hgt_u8, cmap="gray")
        ax[2].imshow(err_u8, cmap="magma", alpha=0.65)
        ax[2].set_title("|Pred - GT|")

        for a in ax:
            a.axis("off")

        frames.append(fig_to_pil(fig))
        plt.close(fig)

        if (i + 1) % 10 == 0 or i == len(ds) - 1:
            print(f"frame {i+1}/{len(ds)}")

    print("Writing GIF:", OUT_GIF)
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=False,
    )
    print("DONE:", OUT_GIF)

if __name__ == "__main__":
    main()
