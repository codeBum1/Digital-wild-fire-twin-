import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from wrf_vit_dataset import WrfVitShardDataset
from train_unet import UNet


DATASET_PATH = "/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus1/shards"
CKPT_PATH    = "/home/abazan/wrfout_sandbox/unet_ckpt.pt"
OUT_GIF      = "/home/abazan/wrfout_sandbox/unet_fire_area.gif"


device = "cpu"


def main():

    print("Loading dataset...")
    ds = WrfVitShardDataset(DATASET_PATH)

    print("Loading model...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    frames = []

    for i in range(len(ds)):

        X, Y = ds[i]

        with torch.no_grad():

            pred = model(X.unsqueeze(0).to(device))[0].cpu()

        gt   = Y[0].numpy()
        pr   = pred[0].numpy()
        err  = np.abs(pr - gt)

        fig, ax = plt.subplots(1,3, figsize=(9,3))

        ax[0].imshow(gt)
        ax[0].set_title("GT")

        ax[1].imshow(pr)
        ax[1].set_title("Pred")

        ax[2].imshow(err)
        ax[2].set_title("Error")

        for a in ax:
            a.axis("off")

        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        frames.append(frame)

        plt.close(fig)

        print("frame", i, "done")

    print("Writing GIF...")

    imageio.mimsave(OUT_GIF, frames, duration=0.5)

    print("DONE:", OUT_GIF)


if __name__ == "__main__":
    main()
