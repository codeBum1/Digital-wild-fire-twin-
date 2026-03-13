import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# geo basemap support
import rasterio
import cartopy.crs as ccrs

from train_unet import UNet, SHARDS_DIR, CKPT_PATH, FIRE_THR
from wrf_vit_dataset import WrfVitShardDataset

# ---------------- Cinematic controls ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

PB_THR = float(os.environ.get("PB_THR", "0.7"))   # threshold for prediction mask
FPS    = int(os.environ.get("FPS", "6"))          # smoother than 2
DPI    = int(os.environ.get("DPI", "160"))        # higher = sharper frames
OUT    = os.environ.get("OUT_GIF", "/home/abazan/wrfout_sandbox/wildfire_cinematic.gif")

# overlay | triptych
MODE = os.environ.get("MODE", "overlay")

# Basemap options
USE_BASEMAP = int(os.environ.get("USE_BASEMAP", "1"))
BASEMAP_TIF = os.environ.get("BASEMAP_TIF", "/home/abazan/wrfout_sandbox/basemaps/basemap.tif")

# Optional: limit frames to avoid huge runs (0 = all)
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))

# ---------------- Helpers ----------------
def iou(pb, yb):
    union = (pb | yb).sum()
    inter = (pb & yb).sum()
    return (inter / union) if union > 0 else 0.0


def load_basemap(tif_path):
    """
    Load a GeoTIFF basemap and return:
      bg: (H,W,3) array
      extent: [left, right, bottom, top] in CRS coordinates
      crs: rasterio CRS object (may be None if missing)
    """
    with rasterio.open(tif_path) as src:
        if src.count >= 3:
            bg = src.read([1, 2, 3])
            bg = np.transpose(bg, (1, 2, 0))
        else:
            bg = src.read(1)
            bg = np.stack([bg, bg, bg], axis=-1)

        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs = src.crs
    return bg, extent, crs


def choose_projection_for_basemap(rasterio_crs):
    """
    If GeoTIFF is EPSG:4326, PlateCarree is correct.
    Otherwise we fall back to PlateCarree for styling.
    """
    try:
        epsg = rasterio_crs.to_epsg() if rasterio_crs is not None else None
    except Exception:
        epsg = None

    if epsg == 4326:
        return ccrs.PlateCarree(), ccrs.PlateCarree()
    else:
        return ccrs.PlateCarree(), ccrs.PlateCarree()


def fig_to_frame(fig):
    """
    Convert matplotlib figure -> RGB uint8 image without deprecated tostring_rgb.
    Uses the RGBA buffer then drops alpha.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _style_axis_dark(ax):
    ax.set_facecolor("black")
    ax.axis("off")


def render_overlay(xf, yf, pf_prob, pb, yb, i, n, iou_val, basemap=None):
    fig = plt.figure(figsize=(12.5, 7.0), dpi=DPI)

    if basemap is not None:
        bg, extent, proj_ax, proj_img = basemap
        ax = plt.axes(projection=proj_ax)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.imshow(bg, extent=extent, transform=proj_img, alpha=1.0)

        ax.imshow(
            pf_prob, extent=extent, transform=proj_img,
            cmap="inferno", alpha=0.75, vmin=0.0, vmax=1.0
        )

        ax.contour(pb.astype(np.float32), levels=[0.5], extent=extent, transform=proj_img,
                   colors="deepskyblue", linewidths=3.0, alpha=0.28)
        ax.contour(pb.astype(np.float32), levels=[0.5], extent=extent, transform=proj_img,
                   colors="cyan", linewidths=1.6, alpha=0.95)
        ax.contour(yb.astype(np.float32), levels=[0.5], extent=extent, transform=proj_img,
                   colors="lime", linewidths=1.7, alpha=0.95)

        try:
            ax.gridlines(draw_labels=False, linewidth=0.6, alpha=0.25, color="white")
        except Exception:
            pass

        ax.axis("off")

    else:
        ax = plt.gca()
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.imshow(xf, cmap="inferno", alpha=0.45)
        ax.imshow(pf_prob, cmap="inferno", alpha=0.85, vmin=0.0, vmax=1.0)

        ax.contour(pb, colors="deepskyblue", linewidths=3.0, alpha=0.30)
        ax.contour(pb, colors="cyan", linewidths=1.6, alpha=0.95)
        ax.contour(yb, colors="lime", linewidths=1.7, alpha=0.95)
        ax.axis("off")

    ax.text(
        0.02, 0.98,
        "Wildfire Growth Prediction (AI)",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=18, fontweight="bold",
        color="white"
    )
    ax.text(
        0.02, 0.92,
        f"Frame {i}/{n-1}   IoU={iou_val:.3f}   PB_THR={PB_THR:.2f}   FIRE_THR={FIRE_THR:.2f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=11,
        color="white",
        alpha=0.90
    )
    ax.text(
        0.02, 0.05,
        "cyan = predicted fire boundary    lime = true fire boundary",
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=10,
        color="white",
        alpha=0.80
    )
    return fig


def render_triptych(xf, yf, pf_prob, pb, yb, i, n, iou_val, basemap=None):
    fig = plt.figure(figsize=(18, 6.2), dpi=DPI)
    fig.patch.set_facecolor("black")

    if basemap is not None:
        bg, extent, proj_ax, proj_img = basemap

        ax1 = plt.subplot(1, 3, 1, projection=proj_ax)
        ax2 = plt.subplot(1, 3, 2, projection=proj_ax)
        ax3 = plt.subplot(1, 3, 3, projection=proj_ax)

        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("black")
            ax.imshow(bg, extent=extent, transform=proj_img, alpha=1.0)
            try:
                ax.gridlines(draw_labels=False, linewidth=0.6, alpha=0.22, color="white")
            except Exception:
                pass
            ax.axis("off")

        xb = (xf > FIRE_THR)
        ax1.imshow(xf, extent=extent, transform=proj_img, cmap="inferno", alpha=0.65)
        ax1.contour(xb.astype(np.float32), levels=[0.5], extent=extent, transform=proj_img,
                    colors="white", linewidths=1.2, alpha=0.75)
        ax1.set_title("CURRENT FIRE (t)", color="white", fontsize=13, fontweight="bold", pad=10)

        ax2.imshow(yf, extent=extent, transform=proj_img, cmap="inferno", alpha=0.65)
        ax2.contour(yb.astype(np.float32), levels=[0.5], extent=extent, transform=proj_img,
                    colors="lime", linewidths=1.8, alpha=0.95)
        ax2.set_title("TRUE FIRE (t+K)", color="white", fontsize=13, fontweight="bold", pad=10)

        ax3.imshow(pf_prob, extent=extent, transform=proj_img, cmap="inferno", alpha=0.80, vmin=0, vmax=1)
        ax3.contour(pb.astype(np.float32), levels=[0.5], extent=extent, transform=proj_img,
                    colors="cyan", linewidths=1.9, alpha=0.95)
        ax3.contour(yb.astype(np.float32), levels=[0.5], extent=extent, transform=proj_img,
                    colors="lime", linewidths=1.6, alpha=0.80)
        ax3.set_title("PREDICTED FIRE (prob)", color="white", fontsize=13, fontweight="bold", pad=10)

    else:
        axs = fig.subplots(1, 3)
        for ax in axs:
            _style_axis_dark(ax)

        axs[0].imshow(xf, cmap="inferno", alpha=0.95)
        axs[0].set_title("CURRENT FIRE (t)", color="white", fontsize=12, fontweight="bold")

        axs[1].imshow(yf, cmap="inferno", alpha=0.95)
        axs[1].contour(yb, colors="lime", linewidths=1.8, alpha=0.95)
        axs[1].set_title("TRUE FIRE (t+K)", color="white", fontsize=12, fontweight="bold")

        axs[2].imshow(pf_prob, cmap="inferno", alpha=0.95, vmin=0.0, vmax=1.0)
        axs[2].contour(pb, colors="cyan", linewidths=1.9, alpha=0.95)
        axs[2].contour(yb, colors="lime", linewidths=1.6, alpha=0.80)
        axs[2].set_title("PREDICTED FIRE (prob)", color="white", fontsize=12, fontweight="bold")

    fig.text(
        0.5, 0.02,
        f"IoU = {iou_val:.3f}   |   PB_THR = {PB_THR:.2f}   |   FIRE_THR = {FIRE_THR:.2f}",
        ha="center",
        va="bottom",
        color="white",
        fontsize=12,
        alpha=0.9
    )
    return fig


# ---------------- Load dataset + model ----------------
ds = WrfVitShardDataset(SHARDS_DIR)

X0, Y0 = ds[0]
model = UNet(X0.shape[0], Y0.shape[0]).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

# ---------------- Load basemap (optional) ----------------
basemap = None
if USE_BASEMAP:
    if not os.path.exists(BASEMAP_TIF):
        print("WARNING: USE_BASEMAP=1 but GeoTIFF not found:", BASEMAP_TIF)
        print("         Falling back to non-basemap visualization.")
    else:
        bg, extent, raster_crs = load_basemap(BASEMAP_TIF)
        proj_ax, proj_img = choose_projection_for_basemap(raster_crs)
        basemap = (bg, extent, proj_ax, proj_img)
        print("Loaded basemap:", BASEMAP_TIF, "extent:", extent, "crs:", raster_crs)

# ---------------- Write GIF incrementally (no OOM) ----------------
n = len(ds)
if MAX_FRAMES and MAX_FRAMES > 0:
    n = min(n, MAX_FRAMES)

with imageio.get_writer(OUT, mode="I", fps=FPS) as writer:
    for i in range(n):
        X, Y = ds[i]

        xf = X[-1].numpy()
        yf = Y[0].numpy()

        Xb = X.unsqueeze(0).to(device)
        mean = Xb.mean(dim=(0, 2, 3), keepdim=True)
        std  = Xb.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        Xb   = (Xb - mean) / std

        with torch.no_grad():
            logits = model(Xb).squeeze(0)[0]
            pf_prob = torch.sigmoid(logits).detach().cpu().numpy()

        pb = (pf_prob > PB_THR)
        yb = (yf > FIRE_THR)
        iou_val = float(iou(pb, yb))

        if MODE == "triptych":
            fig = render_triptych(xf, yf, pf_prob, pb, yb, i, n, iou_val, basemap=basemap)
        else:
            fig = render_overlay(xf, yf, pf_prob, pb, yb, i, n, iou_val, basemap=basemap)

        writer.append_data(fig_to_frame(fig))

print("Saved:", OUT)
