#!/usr/bin/env python3
"""
Human Danger Rating System – WRF-SFIRE wildfire spread predictions.

Post-processing module that combines fire perimeter probability predictions with:
  - Population density  (Census/GPW raster, WRF land-use proxy, or osmnx fallback)
  - Road accessibility  (OpenStreetMap via osmnx, precomputed raster, or LU proxy)
  - Terrain escape routes (slope analysis from WRF HGT elevation)

Usage
-----
# From rollout output (single .npy step):
    python danger_rating.py \\
        --fire_prob unet_rollout_step3.npy \\
        --wrf_file  sandbox_wrfout.nc \\
        --out_dir   danger_output/

# Multi-step (directory of step_*.npy):
    python danger_rating.py \\
        --fire_prob_dir unet_rollout_output/ \\
        --wrf_file sandbox_wrfout.nc \\
        --make_gif

# Override via env vars (consistent with train_unet.py style):
    FIRE_PROB_PATH=step3.npy WRF_FILE=sandbox_wrfout.nc W_POP=0.6 python danger_rating.py

Outputs (in OUT_DIR)
--------------------
    danger_score.npy          raw danger score (H, W) float32
    danger_map.png            4-panel visualization
    danger_animation.gif      multi-step evolution (if --make_gif)
    danger_summary.json       per-cell stats and alert counts
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found – terrain & road layers will be approximate")

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import box as shapely_box
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DangerConfig:
    # ── inputs ────────────────────────────────────────────────────────────────
    fire_prob_path: str   = os.environ.get("FIRE_PROB_PATH",   "")
    fire_prob_dir:  str   = os.environ.get("FIRE_PROB_DIR",    "")
    wrf_file:       str   = os.environ.get("WRF_FILE",         "sandbox_wrfout.nc")
    pop_raster:     str   = os.environ.get("POP_RASTER",       "")
    road_raster:    str   = os.environ.get("ROAD_RASTER",      "")
    out_dir:        str   = os.environ.get("OUT_DIR",          "danger_output")

    # ── layer weights (normalised inside compute_danger_score) ────────────────
    w_pop:     float = float(os.environ.get("W_POP",     "0.50"))  # population at risk
    w_road:    float = float(os.environ.get("W_ROAD",    "0.30"))  # evacuation difficulty
    w_terrain: float = float(os.environ.get("W_TERRAIN", "0.20"))  # terrain escape risk

    # ── terrain parameters ────────────────────────────────────────────────────
    grid_dx_km:       float = float(os.environ.get("GRID_DX_KM",       "1.0"))
    max_escape_dist_km: float = float(os.environ.get("MAX_ESCAPE_DIST_KM", "5.0"))
    slope_thr_pct:    float = float(os.environ.get("SLOPE_THR_PCT",    "30.0"))

    # ── road parameters ───────────────────────────────────────────────────────
    use_osmnx:      bool  = os.environ.get("USE_OSMNX",      "0") == "1"
    road_buffer_km: float = float(os.environ.get("ROAD_BUFFER_KM", "2.0"))

    # ── display ───────────────────────────────────────────────────────────────
    danger_thr: float = float(os.environ.get("DANGER_THR", "0.60"))
    fire_thr:   float = float(os.environ.get("FIRE_THR",   "0.30"))
    make_gif:   bool  = os.environ.get("MAKE_GIF", "0") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# WRF coordinate / field extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_wrf_fields(wrf_file: str) -> dict:
    """
    Load geographic and land-use fields from a WRF NetCDF output.

    Returns dict with:
        xlat  (H, W)  – latitude per cell (degrees N)
        xlong (H, W)  – longitude per cell (degrees E)
        hgt   (H, W)  – elevation (m)
        lu    (H, W)  – NLCD land-use index (int); zeros if unavailable
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required to load WRF files")

    ds = xr.open_dataset(wrf_file)

    def _squeeze(da):
        arr = da.values
        while arr.ndim > 2:
            arr = arr[0]
        return arr.astype(np.float32)

    xlat  = _squeeze(ds["XLAT"])
    xlong = _squeeze(ds["XLONG"])
    hgt   = _squeeze(ds["HGT"]) if "HGT" in ds else np.zeros_like(xlat)

    lu = np.zeros_like(xlat, dtype=np.int32)
    for key in ("LU_INDEX", "IVGTYP", "ISLTYP"):
        if key in ds:
            lu = _squeeze(ds[key]).astype(np.int32)
            break

    ds.close()
    return {"xlat": xlat, "xlong": xlong, "hgt": hgt, "lu": lu}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 – Terrain escape risk
# ─────────────────────────────────────────────────────────────────────────────

def compute_terrain_risk(
    hgt:                np.ndarray,
    grid_dx_km:         float = 1.0,
    slope_thr_pct:      float = 30.0,
    max_escape_dist_km: float = 5.0,
) -> np.ndarray:
    """
    Per-cell terrain escape risk in [0, 1].

    Logic:
    - Compute slope (% grade) from elevation gradient.
    - Cells with slope < slope_thr_pct are passable "escape corridors".
    - For every other cell, distance to the nearest escape corridor is
      converted to a 0–1 risk score (capped at max_escape_dist_km).
    - Steep slope also adds a direct penalty (hard-to-cross terrain).

    A value of 1 means high terrain risk (steep, far from escape routes).
    """
    dx_m = grid_dx_km * 1000.0

    # Slope magnitude in % grade (rise / run × 100)
    grad_y, grad_x = np.gradient(hgt.astype(np.float32), dx_m, dx_m)
    slope_pct = np.sqrt(grad_x**2 + grad_y**2) * 100.0  # m/m → percent

    # Escape corridor mask: low-slope, passable terrain
    escape_mask = slope_pct < slope_thr_pct   # True = passable

    # Distance (km) from each cell to nearest escape corridor
    if HAS_SCIPY and escape_mask.any():
        # distance_transform_edt returns distances in pixels; convert to km
        dist_cells = distance_transform_edt(~escape_mask)
        dist_km    = dist_cells * grid_dx_km
    else:
        # Fallback: crude binary – steep cells get distance = max
        dist_km = np.where(~escape_mask, max_escape_dist_km, 0.0).astype(np.float32)

    escape_dist_risk = np.clip(dist_km / max_escape_dist_km, 0.0, 1.0)

    # Slope penalty: cells that are inherently steep get extra risk
    slope_norm = np.clip(slope_pct / (slope_thr_pct * 3), 0.0, 1.0)

    # Combine: 70% escape-distance risk, 30% slope severity
    terrain_risk = 0.70 * escape_dist_risk + 0.30 * slope_norm
    return terrain_risk.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 – Population density
# ─────────────────────────────────────────────────────────────────────────────

# NLCD land-use codes for developed/residential land (WRF default NLCD scheme)
_DEVELOPED_LU_CODES = frozenset([3, 4, 5, 6])   # Open space → High intensity developed


def _lu_population_proxy(lu: np.ndarray) -> np.ndarray:
    """
    Map WRF NLCD land-use index to a population density proxy in [0, 1].

    NLCD developed classes (3-6) correspond to suburban → urban intensity.
    Returns a smoothed proxy – actual population rasters are preferred.
    """
    # Intensity increases with development class
    pop = np.zeros(lu.shape, dtype=np.float32)
    pop[lu == 3] = 0.2   # Open space (parks, sparse suburban)
    pop[lu == 4] = 0.5   # Low intensity (residential, single-family)
    pop[lu == 5] = 0.8   # Medium intensity (dense residential)
    pop[lu == 6] = 1.0   # High intensity (commercial, dense urban)

    if HAS_SCIPY:
        # Smooth to account for nearby population
        pop = gaussian_filter(pop, sigma=2.0)

    # Re-normalise after smoothing
    if pop.max() > 0:
        pop = pop / pop.max()

    return pop


def _load_raster_to_grid(raster_path: str, xlat: np.ndarray, xlong: np.ndarray) -> np.ndarray:
    """
    Bilinearly resample a GeoTIFF raster onto the WRF grid (xlat, xlong).
    Returns a (H, W) float32 array normalised to [0, 1].
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to load external rasters")

    with rasterio.open(raster_path) as src:
        # Sample at each WRF grid cell's lat/lon using rasterio's sample iterator
        coords = list(zip(xlong.ravel(), xlat.ravel()))
        vals   = np.array(list(src.sample(coords)), dtype=np.float32)[:, 0]
        grid   = vals.reshape(xlat.shape)

    # Clip negatives (nodata), log-scale population counts
    grid = np.clip(grid, 0, None)
    if grid.max() > 0:
        grid = np.log1p(grid) / np.log1p(grid.max())

    return grid.astype(np.float32)


def compute_population_layer(
    xlat:       np.ndarray,
    xlong:      np.ndarray,
    lu:         np.ndarray,
    pop_raster: str = "",
) -> np.ndarray:
    """
    Population density layer in [0, 1].

    Priority:
    1. External raster (GPW, LandScan, or Census block-group raster)
    2. WRF land-use proxy (NLCD developed classes)
    """
    if pop_raster and Path(pop_raster).exists():
        print(f"  [pop] loading raster: {pop_raster}")
        return _load_raster_to_grid(pop_raster, xlat, xlong)

    print("  [pop] using WRF land-use proxy (no raster provided)")
    return _lu_population_proxy(lu)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 – Road / evacuation accessibility
# ─────────────────────────────────────────────────────────────────────────────

def _osmnx_road_raster(
    xlat:          np.ndarray,
    xlong:         np.ndarray,
    road_buffer_km: float = 2.0,
) -> np.ndarray:
    """
    Download OSM road network for the WRF domain bounding box and rasterise
    it onto the 99×99 grid.  Returns inverse-distance-to-road in [0, 1]
    (1 = far from roads = high evacuation difficulty).
    """
    lat_min, lat_max = float(xlat.min()), float(xlat.max())
    lon_min, lon_max = float(xlong.min()), float(xlong.max())

    print(f"  [road] downloading OSM roads for [{lat_min:.3f},{lat_max:.3f}] × [{lon_min:.3f},{lon_max:.3f}]")
    G = ox.graph_from_bbox(
        lat_max, lat_min, lon_max, lon_min,
        network_type="drive",
        simplify=True,
    )
    edges = ox.graph_to_gdfs(G, nodes=False)[["geometry"]]

    H, W = xlat.shape
    road_mask = np.zeros((H, W), dtype=bool)

    for geom in edges.geometry:
        # For each road segment, mark grid cells it passes through
        for coord in geom.coords:
            lon_r, lat_r = coord[0], coord[1]
            # Nearest grid cell (linear interpolation)
            col = int(round((lon_r - lon_min) / (lon_max - lon_min) * (W - 1)))
            row = int(round((lat_r - lat_min) / (lat_max - lat_min) * (H - 1)))
            if 0 <= row < H and 0 <= col < W:
                road_mask[row, col] = True

    return road_mask


def _lu_road_proxy(lu: np.ndarray) -> np.ndarray:
    """
    Approximate road presence from WRF land-use: developed areas (3-6) tend
    to have roads.  Returns a boolean road-like mask.
    """
    return np.isin(lu, [3, 4, 5, 6])


def compute_road_accessibility(
    xlat:          np.ndarray,
    xlong:         np.ndarray,
    lu:            np.ndarray,
    road_raster:   str   = "",
    use_osmnx:     bool  = False,
    grid_dx_km:    float = 1.0,
    road_buffer_km: float = 2.0,
) -> np.ndarray:
    """
    Road inaccessibility in [0, 1].
    1 = far from roads (hard to evacuate).
    0 = on or near a road.

    Priority:
    1. Precomputed road raster
    2. Live OSM download (if use_osmnx=True and osmnx installed)
    3. WRF land-use proxy
    """
    road_mask: np.ndarray | None = None

    if road_raster and Path(road_raster).exists():
        print(f"  [road] loading raster: {road_raster}")
        road_presence = _load_raster_to_grid(road_raster, xlat, xlong)
        road_mask     = road_presence > 0.1
    elif use_osmnx and HAS_OSMNX:
        try:
            road_mask = _osmnx_road_raster(xlat, xlong, road_buffer_km)
        except Exception as e:
            print(f"  [road] osmnx failed ({e}), falling back to LU proxy")

    if road_mask is None:
        print("  [road] using WRF land-use proxy (no road raster/OSM)")
        road_mask = _lu_road_proxy(lu)

    # Distance-to-road in cells → km → normalised
    if HAS_SCIPY:
        dist_cells = distance_transform_edt(~road_mask)
    else:
        dist_cells = (~road_mask).astype(np.float32) * road_buffer_km / grid_dx_km

    dist_km             = dist_cells * grid_dx_km
    inaccessibility     = np.clip(dist_km / road_buffer_km, 0.0, 1.0)
    return inaccessibility.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Composite danger score
# ─────────────────────────────────────────────────────────────────────────────

def compute_danger_score(
    fire_prob:    np.ndarray,
    pop_layer:    np.ndarray,
    road_layer:   np.ndarray,
    terrain_layer: np.ndarray,
    w_pop:        float = 0.50,
    w_road:       float = 0.30,
    w_terrain:    float = 0.20,
) -> np.ndarray:
    """
    Composite danger score in [0, 1].

    danger = fire_prob × (w_pop × pop + w_road × road_inaccessibility + w_terrain × terrain_risk)

    Interpretation:
    - High danger = fire is likely AND (many people at risk OR hard to evacuate OR rugged terrain)
    - A cell with zero fire probability always scores 0
    """
    w_sum = w_pop + w_road + w_terrain
    vulnerability = (
        w_pop     * pop_layer     +
        w_road    * road_layer    +
        w_terrain * terrain_layer
    ) / w_sum                         # normalise weights

    danger = fire_prob * vulnerability
    # Re-normalise to [0, 1] in case of floating-point drift
    if danger.max() > 0:
        danger = danger / danger.max()

    return danger.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

_DANGER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "danger",
    [(0.05, 0.05, 0.05, 0.0),   # transparent black (no danger)
     (0.95, 0.95, 0.10, 0.5),   # yellow (moderate)
     (1.00, 0.40, 0.00, 0.75),  # orange
     (0.90, 0.00, 0.00, 0.95)], # red (extreme danger)
)


def visualize_danger(
    fire_prob:     np.ndarray,
    pop_layer:     np.ndarray,
    road_layer:    np.ndarray,
    terrain_layer: np.ndarray,
    danger_score:  np.ndarray,
    xlat:          np.ndarray,
    xlong:         np.ndarray,
    out_path:      str,
    step:          int  = 0,
    fire_thr:      float = 0.30,
    danger_thr:    float = 0.60,
) -> None:
    """
    4-panel figure:
    [TL] Fire probability | [TR] Danger score
    [BL] Population layer | [BR] Road accessibility (inverted: high = inaccessible)
    """
    lon_min, lon_max = float(xlong.min()), float(xlong.max())
    lat_min, lat_max = float(xlat.min()),  float(xlat.max())
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#1a1a2e")
    title_kw = dict(color="white", fontsize=11, fontweight="bold", pad=8)
    label_kw = dict(fraction=0.046, pad=0.04, shrink=0.85)

    def _ax(ax, data, cmap, vmin, vmax, title, clabel):
        ax.set_facecolor("#0d0d1a")
        im = ax.imshow(
            data, origin="lower", extent=extent,
            cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )
        # Fire perimeter contour on every panel
        ax.contour(
            xlong, xlat, fire_prob,
            levels=[fire_thr], colors=["cyan"], linewidths=[1.0],
        )
        if np.any(danger_score >= danger_thr):
            ax.contour(
                xlong, xlat, danger_score,
                levels=[danger_thr], colors=["red"], linewidths=[1.5],
            )
        ax.set_title(title, **title_kw)
        ax.set_xlabel("Longitude", color="white", fontsize=8)
        ax.set_ylabel("Latitude",  color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        cb = fig.colorbar(im, ax=ax, **label_kw)
        cb.set_label(clabel, color="white", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        return ax

    _ax(axes[0, 0], fire_prob,     "hot",           0, 1, "Fire Probability",           "P(fire)")
    _ax(axes[0, 1], danger_score,  _DANGER_CMAP,    0, 1, "Human Danger Score",         "Danger [0–1]")
    _ax(axes[1, 0], pop_layer,     "YlOrRd",        0, 1, "Population Density Proxy",   "Relative density")
    _ax(axes[1, 1], 1 - road_layer,"RdPu",          0, 1, "Road Accessibility",         "Access [0=isolated]")

    # Legend
    legend_patches = [
        Patch(edgecolor="cyan",  facecolor="none", linewidth=1.5, label=f"Fire front (p>{fire_thr:.2f})"),
        Patch(edgecolor="red",   facecolor="none", linewidth=1.5, label=f"High danger (>{danger_thr:.2f})"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               facecolor="#1a1a2e", edgecolor="#444",
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, 0.01))

    step_label = f"  –  Step {step}" if step > 0 else ""
    fig.suptitle(f"WRF-SFIRE Human Danger Rating{step_label}", color="white", fontsize=14, y=1.01)

    # Stats annotation
    n_high   = int(np.sum(danger_score >= danger_thr))
    mean_d   = float(danger_score[fire_prob > fire_thr].mean()) if fire_prob.max() > fire_thr else 0.0
    stats_txt = (
        f"High-danger cells: {n_high} ({100*n_high/(danger_score.size):.1f}%)\n"
        f"Mean danger in fire zone: {mean_d:.3f}"
    )
    fig.text(0.01, 0.99, stats_txt, transform=fig.transFigure,
             color="white", fontsize=8, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#333366", alpha=0.7))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] saved → {out_path}")


def make_danger_gif(frame_paths: list[str], out_path: str, fps: int = 2) -> None:
    """Stitch individual frame PNGs into an animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("  [gif] Pillow not available; skipping GIF generation")
        return

    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"  [gif] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def danger_summary(
    danger_score:  np.ndarray,
    fire_prob:     np.ndarray,
    xlat:          np.ndarray,
    xlong:         np.ndarray,
    danger_thr:    float = 0.60,
    fire_thr:      float = 0.30,
) -> dict:
    """
    Return a JSON-serialisable summary dict for logging / downstream use.
    """
    fire_mask  = fire_prob >= fire_thr
    high_danger = danger_score >= danger_thr

    # Bounding box of high-danger zone
    rows, cols = np.where(high_danger)
    if len(rows):
        bbox = {
            "lat_min": float(xlat[rows, cols].min()),
            "lat_max": float(xlat[rows, cols].max()),
            "lon_min": float(xlong[rows, cols].min()),
            "lon_max": float(xlong[rows, cols].max()),
        }
    else:
        bbox = {}

    return {
        "n_fire_cells":        int(fire_mask.sum()),
        "n_high_danger_cells": int(high_danger.sum()),
        "frac_high_danger":    float(high_danger.mean()),
        "max_danger":          float(danger_score.max()),
        "mean_danger_in_fire": float(danger_score[fire_mask].mean()) if fire_mask.any() else 0.0,
        "high_danger_bbox":    bbox,
        "danger_threshold":    danger_thr,
        "fire_threshold":      fire_thr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_danger_rating(
    fire_prob: np.ndarray,
    wrf_fields: dict,
    cfg: DangerConfig,
    step: int = 0,
    out_dir: Path | None = None,
) -> dict:
    """
    Full danger-rating pipeline for one fire-probability snapshot.

    Parameters
    ----------
    fire_prob  : (H, W) float32, values in [0, 1]
    wrf_fields : dict from load_wrf_fields()
    cfg        : DangerConfig
    step       : rollout step index (for filenames)
    out_dir    : output directory (overrides cfg.out_dir if provided)

    Returns
    -------
    dict with keys: danger_score, pop_layer, road_layer, terrain_layer, summary
    """
    out_dir = Path(out_dir or cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xlat  = wrf_fields["xlat"]
    xlong = wrf_fields["xlong"]
    hgt   = wrf_fields["hgt"]
    lu    = wrf_fields["lu"]

    print(f"\n── Danger Rating  step={step} ──────────────────────────────────")

    print("  [terrain] computing slope / escape-route risk …")
    terrain_layer = compute_terrain_risk(
        hgt,
        grid_dx_km         = cfg.grid_dx_km,
        slope_thr_pct      = cfg.slope_thr_pct,
        max_escape_dist_km = cfg.max_escape_dist_km,
    )

    print("  [pop] computing population layer …")
    pop_layer = compute_population_layer(
        xlat, xlong, lu,
        pop_raster = cfg.pop_raster,
    )

    print("  [road] computing road accessibility …")
    road_layer = compute_road_accessibility(
        xlat, xlong, lu,
        road_raster    = cfg.road_raster,
        use_osmnx      = cfg.use_osmnx,
        grid_dx_km     = cfg.grid_dx_km,
        road_buffer_km = cfg.road_buffer_km,
    )

    print("  [danger] computing composite score …")
    danger_score = compute_danger_score(
        fire_prob, pop_layer, road_layer, terrain_layer,
        w_pop     = cfg.w_pop,
        w_road    = cfg.w_road,
        w_terrain = cfg.w_terrain,
    )

    # ── save artefacts ───────────────────────────────────────────────────────
    step_tag = f"_step{step:02d}" if step > 0 else ""
    np.save(str(out_dir / f"danger_score{step_tag}.npy"), danger_score)

    viz_path = str(out_dir / f"danger_map{step_tag}.png")
    visualize_danger(
        fire_prob, pop_layer, road_layer, terrain_layer, danger_score,
        xlat, xlong, out_path=viz_path, step=step,
        fire_thr=cfg.fire_thr, danger_thr=cfg.danger_thr,
    )

    summary = danger_summary(
        danger_score, fire_prob, xlat, xlong,
        danger_thr=cfg.danger_thr, fire_thr=cfg.fire_thr,
    )
    with open(out_dir / f"danger_summary{step_tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"  [done] max_danger={summary['max_danger']:.3f}  "
        f"high_danger_cells={summary['n_high_danger_cells']}  "
        f"({100*summary['frac_high_danger']:.1f}%)"
    )

    return dict(
        danger_score   = danger_score,
        pop_layer      = pop_layer,
        road_layer     = road_layer,
        terrain_layer  = terrain_layer,
        summary        = summary,
        viz_path       = viz_path,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fire_prob",     default="", help="Path to a .npy fire probability array (H×W)")
    p.add_argument("--fire_prob_dir", default="", help="Directory of step_*.npy files for multi-step")
    p.add_argument("--wrf_file",      default="", help="WRF NetCDF file for coordinates / terrain")
    p.add_argument("--pop_raster",    default="", help="Optional population density GeoTIFF")
    p.add_argument("--road_raster",   default="", help="Optional road network GeoTIFF")
    p.add_argument("--out_dir",       default="", help="Output directory")
    p.add_argument("--make_gif",      action="store_true", help="Stitch multi-step PNGs into animated GIF")
    p.add_argument("--use_osmnx",     action="store_true", help="Download road network from OSM via osmnx")
    p.add_argument("--w_pop",         type=float, default=None)
    p.add_argument("--w_road",        type=float, default=None)
    p.add_argument("--w_terrain",     type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = DangerConfig()

    # CLI args override env vars
    if args.fire_prob:     cfg.fire_prob_path = args.fire_prob
    if args.fire_prob_dir: cfg.fire_prob_dir  = args.fire_prob_dir
    if args.wrf_file:      cfg.wrf_file       = args.wrf_file
    if args.pop_raster:    cfg.pop_raster     = args.pop_raster
    if args.road_raster:   cfg.road_raster    = args.road_raster
    if args.out_dir:       cfg.out_dir        = args.out_dir
    if args.make_gif:      cfg.make_gif       = True
    if args.use_osmnx:     cfg.use_osmnx      = True
    if args.w_pop     is not None: cfg.w_pop     = args.w_pop
    if args.w_road    is not None: cfg.w_road    = args.w_road
    if args.w_terrain is not None: cfg.w_terrain = args.w_terrain

    # ── load WRF fields ──────────────────────────────────────────────────────
    if not cfg.wrf_file or not Path(cfg.wrf_file).exists():
        print(f"ERROR: WRF file not found: '{cfg.wrf_file}'")
        sys.exit(1)

    print(f"Loading WRF fields from: {cfg.wrf_file}")
    wrf_fields = load_wrf_fields(cfg.wrf_file)
    print(f"  grid: {wrf_fields['xlat'].shape}  "
          f"lat=[{wrf_fields['xlat'].min():.3f}, {wrf_fields['xlat'].max():.3f}]  "
          f"lon=[{wrf_fields['xlong'].min():.3f}, {wrf_fields['xlong'].max():.3f}]")

    # ── collect fire probability arrays ─────────────────────────────────────
    fire_steps: list[tuple[int, np.ndarray]] = []

    if cfg.fire_prob_dir and Path(cfg.fire_prob_dir).is_dir():
        step_files = sorted(Path(cfg.fire_prob_dir).glob("step_*.npy"))
        if not step_files:
            step_files = sorted(Path(cfg.fire_prob_dir).glob("*.npy"))
        for i, fp in enumerate(step_files):
            fire_steps.append((i + 1, np.load(str(fp)).astype(np.float32)))
        print(f"Found {len(fire_steps)} step files in {cfg.fire_prob_dir}")

    elif cfg.fire_prob_path and Path(cfg.fire_prob_path).exists():
        arr = np.load(cfg.fire_prob_path).astype(np.float32)
        fire_steps = [(0, arr)]
        print(f"Loaded fire_prob from {cfg.fire_prob_path}: shape={arr.shape}")

    else:
        # Synthetic demo: random fire front in the centre of the domain
        print("WARNING: no fire_prob provided — generating synthetic demo array")
        H, W      = wrf_fields["xlat"].shape
        cx, cy    = H // 2, W // 2
        ys, xs    = np.mgrid[:H, :W]
        r         = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        fire_prob = np.clip(1.0 - r / (min(H, W) * 0.25), 0, 1).astype(np.float32)
        fire_steps = [(0, fire_prob)]

    # ── run per-step ─────────────────────────────────────────────────────────
    out_dir    = Path(cfg.out_dir)
    frame_paths: list[str] = []

    for step, fire_prob in fire_steps:
        result = run_danger_rating(fire_prob, wrf_fields, cfg, step=step, out_dir=out_dir)
        frame_paths.append(result["viz_path"])

    # ── optional GIF ─────────────────────────────────────────────────────────
    if cfg.make_gif and len(frame_paths) > 1:
        make_danger_gif(frame_paths, str(out_dir / "danger_animation.gif"))

    print(f"\n✓ Danger rating complete. Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
