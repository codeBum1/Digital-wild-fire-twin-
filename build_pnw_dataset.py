#!/usr/bin/env python3
"""
build_pnw_dataset.py — Production dataset builder for PNW WRF-SFIRE simulations.

Design principles
-----------------
1. Simulation-boundary-aware pairing — never pairs timesteps across simulations.
2. Fire-presence QC — configurable minimum fire-coverage threshold filters
   "boring" (pre-ignition) pairs.
3. Variable & shape validation — every file checked before any data is extracted.
4. Full provenance — every shard records simulation name, file pair paths,
   timestep indices, and per-field statistics.
5. QC report — JSON manifest + human-readable summary written alongside data.
6. Vectorized downsampling — numpy-native max-pool, no Python loops over H/W.

Usage
-----
    python build_pnw_dataset.py

Environment variable overrides
-------------------------------
    PNW_RAW_DIR       Path to directory containing per-simulation subdirs
                      (default: pnw_sfire_raw)
    PNW_OUT_DIR       Output directory for shards + reports
                      (default: vit_dataset_pnw_tplus1)
    VIT_PAIR_K        Timesteps ahead for Y pairing (default: 1)
    VIT_FIRE_THR      Min FIRE_AREA max in Y frame to accept pair (default: 1e-4)
    VIT_ENGINE        xarray engine: "netcdf4" or "h5netcdf" (default: netcdf4)
    VIT_K_LEVELS      Comma-separated vertical levels to extract, e.g. "0,5,10"
                      (default: 0)
    VIT_INCLUDE_FUEL_FRAC    "1" to add FUEL_FRAC as extra X channel
    VIT_INCLUDE_PM25_PROXY   "1" to add PM2.5 proxy as extra Y channel
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    raw_dir: str = "pnw_sfire_raw"
    out_dir: str = "vit_dataset_pnw_tplus1"
    engine: str = "netcdf4"

    # Which WRF variables to extract
    vars_2d: List[str] = field(default_factory=lambda: [
        "T2", "Q2", "PSFC", "U10", "V10", "HGT"
    ])
    vars_3d: List[str] = field(default_factory=lambda: ["T", "QVAPOR"])
    fire_vars: List[str] = field(default_factory=lambda: [
        "FIRE_AREA", "ROS", "FLAME_LENGTH"
    ])
    k_levels: List[int] = field(default_factory=lambda: [0])

    pair_k: int = 1           # Y comes from timestep t + pair_k
    fire_thr: float = 1e-4    # min FIRE_AREA max in Y frame to keep pair
    domain: str = "d01"       # WRF domain to use (e.g. "d01", "d03")
    include_fuel_frac: bool = False
    include_pm25_proxy: bool = False
    pm25_dt_h: float = 0.5    # half-hour WRF output interval

    # Target spatial resolution — resize all channels to this after extraction.
    # Set to match the original sandbox training grid (99x99) so PNW shards are
    # compatible with the trained UNet.  Set to 0 to disable (use native grid).
    target_h: int = 99
    target_w: int = 99


def load_config() -> Config:
    cfg = Config()
    cfg.raw_dir  = os.environ.get("PNW_RAW_DIR",  cfg.raw_dir)
    cfg.out_dir  = os.environ.get("PNW_OUT_DIR",  cfg.out_dir)
    cfg.engine   = os.environ.get("VIT_ENGINE",   cfg.engine)
    cfg.pair_k   = int(os.environ.get("VIT_PAIR_K", str(cfg.pair_k)))
    cfg.fire_thr = float(os.environ.get("VIT_FIRE_THR", str(cfg.fire_thr)))

    k_env = os.environ.get("VIT_K_LEVELS")
    if k_env:
        cfg.k_levels = [int(x) for x in k_env.split(",") if x.strip()]

    if os.environ.get("VIT_INCLUDE_FUEL_FRAC", "0") == "1":
        cfg.include_fuel_frac = True
    if os.environ.get("VIT_INCLUDE_PM25_PROXY", "0") == "1":
        cfg.include_pm25_proxy = True

    th = os.environ.get("PNW_TARGET_H")
    tw = os.environ.get("PNW_TARGET_W")
    if th:
        cfg.target_h = int(th)
    if tw:
        cfg.target_w = int(tw)

    cfg.domain = os.environ.get("VIT_DOMAIN", cfg.domain)

    fv_env = os.environ.get("VIT_FIRE_VARS")
    if fv_env:
        cfg.fire_vars = [v.strip() for v in fv_env.split(",") if v.strip()]

    return cfg


# ---------------------------------------------------------------------------
# Fast vectorised max-pool: fire subgrid → atmospheric grid
# ---------------------------------------------------------------------------

def maxpool_to_atm(fire: np.ndarray, Ha: int, Wa: int) -> np.ndarray:
    """
    Max-pool fire array from (Hf, Wf) to (Ha, Wa) using floor-division binning.
    Fully vectorised — no Python loops over spatial dimensions.

    Uses np.maximum.at scatter which is O(Hf*Wf), ~50-100× faster than the
    nested-loop approach for typical 500→99 reductions.

    Parameters
    ----------
    fire : (Hf, Wf) float32
    Ha, Wa : target atmospheric grid shape

    Returns
    -------
    out : (Ha, Wa) float32
    """
    Hf, Wf = fire.shape
    y_bins = np.arange(Hf, dtype=np.int32) * Ha // Hf   # (Hf,) in [0, Ha)
    x_bins = np.arange(Wf, dtype=np.int32) * Wa // Wf   # (Wf,) in [0, Wa)
    yy, xx = np.meshgrid(y_bins, x_bins, indexing="ij")  # (Hf, Wf)
    linear  = yy.ravel() * Wa + xx.ravel()               # (Hf*Wf,)

    out_flat = np.full(Ha * Wa, -1e9, dtype=np.float32)
    np.maximum.at(out_flat, linear, fire.ravel().astype(np.float32))
    out = out_flat.reshape(Ha, Wa)
    out[out < 0] = 0.0    # cells with no source fire pixel stay 0
    return out


def downsample_fire_var(ds: xr.Dataset, varname: str, Ha: int, Wa: int) -> np.ndarray:
    """Extract fire variable (Time=1, Hf, Wf) and max-pool to (Ha, Wa)."""
    arr = ds[varname].values  # (1, Hf, Wf)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(f"{varname}: expected (1, Hf, Wf), got {arr.shape}")
    return maxpool_to_atm(arr[0].astype(np.float32), Ha, Wa)


def resize_channel(arr: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
    """
    Bilinear resize a (H, W) float32 array to (Ht, Wt).
    Uses pure numpy / scipy — no PIL/cv2 dependency.
    """
    from scipy.ndimage import zoom
    if arr.shape == (Ht, Wt):
        return arr
    zy = Ht / arr.shape[0]
    zx = Wt / arr.shape[1]
    return zoom(arr.astype(np.float32), (zy, zx), order=1).astype(np.float32)


def resize_tensor(tensor: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
    """Resize (C, H, W) tensor to (C, Ht, Wt) channel-by-channel."""
    if tensor.shape[1] == Ht and tensor.shape[2] == Wt:
        return tensor
    return np.stack([resize_channel(tensor[c], Ht, Wt) for c in range(tensor.shape[0])], axis=0)


# ---------------------------------------------------------------------------
# Variable validation
# ---------------------------------------------------------------------------

REQUIRED_DIMS_2D   = {"Time", "south_north", "west_east"}
REQUIRED_DIMS_3D   = {"Time", "bottom_top", "south_north", "west_east"}
REQUIRED_DIMS_FIRE = {"Time"}   # fire vars have fire-subgrid dims; just check Time exists


class QCError(Exception):
    """Raised when a file fails a hard QC check."""


def validate_file(ds: xr.Dataset, cfg: Config) -> Tuple[int, int]:
    """
    Check all required variables exist and have valid shapes.
    Returns (Ha, Wa) — atmospheric grid size.
    Raises QCError on failure.
    """
    if "Time" not in ds.sizes:
        raise QCError("Missing Time dimension")
    if ds.sizes["Time"] != 1:
        raise QCError(f"Expected Time=1 per file, got {ds.sizes['Time']}")

    for v in cfg.vars_2d:
        if v not in ds:
            raise QCError(f"Missing 2D var: {v}")
        if set(ds[v].dims) != REQUIRED_DIMS_2D:
            raise QCError(f"{v} dims {ds[v].dims} != expected {REQUIRED_DIMS_2D}")

    for v in cfg.vars_3d:
        if v not in ds:
            raise QCError(f"Missing 3D var: {v}")
        if set(ds[v].dims) != REQUIRED_DIMS_3D:
            raise QCError(f"{v} dims {ds[v].dims} != expected {REQUIRED_DIMS_3D}")
        for k in cfg.k_levels:
            if k >= ds[v].sizes.get("bottom_top", 0):
                raise QCError(f"{v} bottom_top={ds[v].sizes['bottom_top']} < k={k}")

    for v in cfg.fire_vars:
        if v not in ds:
            raise QCError(f"Missing fire var: {v}")
        if "Time" not in ds[v].dims:
            raise QCError(f"{v} missing Time dim")

    if "south_north" not in ds.sizes or "west_east" not in ds.sizes:
        raise QCError("Missing south_north or west_east")

    Ha = ds.sizes["south_north"]
    Wa = ds.sizes["west_east"]
    return Ha, Wa


def check_values(arr: np.ndarray, name: str, allow_nan: bool = False) -> List[str]:
    """Return list of warnings if array has NaN/Inf or suspicious values."""
    warnings = []
    if np.any(np.isnan(arr)):
        warnings.append(f"{name}: contains NaN")
    if np.any(np.isinf(arr)):
        warnings.append(f"{name}: contains Inf")
    if not allow_nan and arr.size > 0:
        if np.abs(arr).max() > 1e10:
            warnings.append(f"{name}: extreme values (max abs={np.abs(arr).max():.2e})")
    return warnings


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_X(ds: xr.Dataset, cfg: Config, Ha: int, Wa: int, Ht: int, Wt: int) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build X tensor (C, Ht, Wt) from atmospheric + fire(t) variables.
    Atm channels are extracted at native (Ha, Wa) then bilinear-resized to (Ht, Wt).
    Fire channels are max-pooled directly from the fire subgrid to (Ht, Wt).
    Returns (X, channel_names, warnings).
    """
    channels: List[np.ndarray] = []
    names: List[str] = []
    warnings: List[str] = []

    # 2D atmosphere — extract at native grid, resize to target
    for v in cfg.vars_2d:
        arr = ds[v].values[0].astype(np.float32)   # (Ha, Wa)
        warnings += check_values(arr, v)
        channels.append(resize_channel(arr, Ht, Wt))
        names.append(v)

    # 3D atmosphere at each vertical level
    for v in cfg.vars_3d:
        da = ds[v]
        for k in cfg.k_levels:
            arr = da.isel(bottom_top=k).values[0].astype(np.float32)
            warnings += check_values(arr, f"{v}_k{k}")
            channels.append(resize_channel(arr, Ht, Wt))
            names.append(f"{v}_k{k}")

    # Fire state at time t — max-pool fire subgrid directly to target size
    fa = downsample_fire_var(ds, "FIRE_AREA", Ht, Wt)
    warnings += check_values(fa, "FIRE_AREA_t")
    channels.append(fa)
    names.append("FIRE_AREA_t")

    # Optional: fuel fraction
    if cfg.include_fuel_frac:
        if "FUEL_FRAC" in ds:
            ff = downsample_fire_var(ds, "FUEL_FRAC", Ht, Wt)
            warnings += check_values(ff, "FUEL_FRAC")
            channels.append(ff)
            names.append("FUEL_FRAC")
        else:
            warnings.append("FUEL_FRAC requested but not in dataset — skipping channel")

    X = np.stack(channels, axis=0)
    return X, names, warnings


def extract_Y(ds: xr.Dataset, cfg: Config, Ha: int, Wa: int, Ht: int, Wt: int) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build Y tensor (F, Ht, Wt) from fire target variables.
    Fire vars are max-pooled directly from the fire subgrid to (Ht, Wt).
    Returns (Y, channel_names, warnings).
    """
    channels: List[np.ndarray] = []
    names: List[str] = []
    warnings: List[str] = []

    for v in cfg.fire_vars:
        arr = downsample_fire_var(ds, v, Ht, Wt)
        warnings += check_values(arr, v)
        channels.append(arr)
        names.append(v)

    # Optional: PM2.5 proxy
    if cfg.include_pm25_proxy:
        try:
            from scipy.ndimage import gaussian_filter

            fire_area  = channels[0]
            fuel_burnt = np.zeros_like(fire_area)
            if "FUEL_FRAC_BURNT" in ds:
                fuel_burnt = downsample_fire_var(ds, "FUEL_FRAC_BURNT", Ht, Wt)

            u10 = ds["U10"].values[0].astype(np.float32) if "U10" in ds else np.zeros((Ha, Wa), dtype=np.float32)
            v10 = ds["V10"].values[0].astype(np.float32) if "V10" in ds else np.zeros((Ha, Wa), dtype=np.float32)

            emission = fire_area * fuel_burnt
            mean_wind = float(np.sqrt(np.mean(u10**2 + v10**2)))
            sigma_cells = np.clip(mean_wind * cfg.pm25_dt_h * 3600.0 / 1000.0, 0.3, 8.0)
            pm25 = gaussian_filter(emission, sigma=sigma_cells)
            if pm25.max() > 0:
                pm25 /= pm25.max()

            channels.append(pm25.astype(np.float32))
            names.append("PM25_proxy")
        except Exception as e:
            warnings.append(f"PM2.5 proxy failed: {e}")

    Y = np.stack(channels, axis=0)
    return Y, names, warnings


# ---------------------------------------------------------------------------
# Per-simulation builder
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    sim_name: str
    n_files: int
    n_pairs_attempted: int
    n_pairs_written: int
    n_skipped_no_fire: int
    n_skipped_error: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def build_sim(
    sim_dir: Path,
    cfg: Config,
    shard_dir: Path,
    global_shard_idx: int,
    manifest: List[Dict],
) -> Tuple[int, SimResult]:
    """
    Process one simulation directory.
    Returns (updated_global_shard_idx, SimResult).
    """
    sim_name = sim_dir.name
    result = SimResult(sim_name=sim_name, n_files=0, n_pairs_attempted=0,
                       n_pairs_written=0, n_skipped_no_fire=0, n_skipped_error=0)

    # Collect and sort wrfout files within this simulation
    files = sorted(sim_dir.glob(f"wrfout_{cfg.domain}_*"))
    result.n_files = len(files)

    if len(files) < cfg.pair_k + 1:
        result.errors.append(f"Only {len(files)} files — cannot form pairs with K={cfg.pair_k}")
        return global_shard_idx, result

    pairs = [(files[i], files[i + cfg.pair_k]) for i in range(len(files) - cfg.pair_k)]
    result.n_pairs_attempted = len(pairs)

    for t_path, tpK_path in pairs:
        try:
            ds_t   = xr.open_dataset(str(t_path),   engine=cfg.engine)
            ds_tpK = xr.open_dataset(str(tpK_path), engine=cfg.engine)

            # Validate both files
            Ha, Wa = validate_file(ds_t,   cfg)
            _,  _  = validate_file(ds_tpK, cfg)

            # Fire-presence QC: check FIRE_AREA max in Y frame
            fa_tpK_raw = ds_tpK["FIRE_AREA"].values[0].astype(np.float32)
            fa_max = float(fa_tpK_raw.max())

            if fa_max < cfg.fire_thr:
                result.n_skipped_no_fire += 1
                ds_t.close(); ds_tpK.close()
                continue

            # Resolve target grid size
            Ht = cfg.target_h if cfg.target_h > 0 else Ha
            Wt = cfg.target_w if cfg.target_w > 0 else Wa

            # Extract features
            X, x_names, x_warn = extract_X(ds_t,   cfg, Ha, Wa, Ht, Wt)
            Y, y_names, y_warn = extract_Y(ds_tpK, cfg, Ha, Wa, Ht, Wt)

            ds_t.close(); ds_tpK.close()

            all_warnings = x_warn + y_warn
            if all_warnings:
                result.warnings.extend([f"shard {global_shard_idx}: {w}" for w in all_warnings])

            # Compute per-shard statistics for QC manifest
            fire_area_Y = Y[0]   # FIRE_AREA at t+K
            fa_t_ds     = downsample_fire_var(
                xr.open_dataset(str(t_path), engine=cfg.engine), "FIRE_AREA", Ht, Wt
            )
            fire_frac_t    = float((fa_t_ds > cfg.fire_thr).mean())
            fire_frac_tpK  = float((fire_area_Y > cfg.fire_thr).mean())

            shard_path = shard_dir / f"{global_shard_idx:06d}.npz"
            np.savez_compressed(
                str(shard_path),
                X=X,
                Y=Y,
            )

            manifest.append({
                "shard_idx":    global_shard_idx,
                "shard_file":   shard_path.name,
                "sim":          sim_name,
                "t_file":       t_path.name,
                "tpK_file":     tpK_path.name,
                "pair_k":       cfg.pair_k,
                "Ha_native":    Ha,
                "Wa_native":    Wa,
                "Ha":           Ht,
                "Wa":           Wt,
                "C":            int(X.shape[0]),
                "F":            int(Y.shape[0]),
                "x_channels":  x_names,
                "y_channels":  y_names,
                "fire_frac_t":  round(fire_frac_t,  6),
                "fire_frac_tpK": round(fire_frac_tpK, 6),
                "X_min":        round(float(X.min()), 6),
                "X_max":        round(float(X.max()), 6),
                "Y_min":        round(float(Y.min()), 6),
                "Y_max":        round(float(Y.max()), 6),
                "warnings":     all_warnings,
            })

            global_shard_idx   += 1
            result.n_pairs_written += 1

        except QCError as e:
            result.n_skipped_error += 1
            result.errors.append(f"{t_path.name}: QCError: {e}")
        except Exception as e:
            result.n_skipped_error += 1
            result.errors.append(f"{t_path.name}: {type(e).__name__}: {e}")
            # Log traceback for debugging
            tb = traceback.format_exc()
            result.errors.append(tb.strip().split("\n")[-1])

    return global_shard_idx, result


# ---------------------------------------------------------------------------
# QC report writer
# ---------------------------------------------------------------------------

def write_qc_report(
    out_dir: Path,
    sim_results: List[SimResult],
    manifest: List[Dict],
    cfg: Config,
) -> None:
    total_pairs      = sum(r.n_pairs_written  for r in sim_results)
    total_skipped_nf = sum(r.n_skipped_no_fire for r in sim_results)
    total_skipped_e  = sum(r.n_skipped_error  for r in sim_results)
    total_warnings   = sum(len(r.warnings)     for r in sim_results)
    total_errors     = sum(len(r.errors)       for r in sim_results)

    summary = {
        "config": asdict(cfg),
        "totals": {
            "simulations":        len(sim_results),
            "shards_written":     total_pairs,
            "skipped_no_fire":    total_skipped_nf,
            "skipped_error":      total_skipped_e,
            "warnings":           total_warnings,
            "error_log_entries":  total_errors,
        },
        "per_sim": [asdict(r) for r in sim_results],
    }

    with open(out_dir / "qc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Human-readable report
    lines = [
        "=" * 70,
        "PNW WRF-SFIRE Dataset QC Report",
        "=" * 70,
        f"  Simulations processed : {len(sim_results)}",
        f"  Total shards written  : {total_pairs}",
        f"  Skipped (no fire)     : {total_skipped_nf}",
        f"  Skipped (errors)      : {total_skipped_e}",
        f"  Warnings              : {total_warnings}",
        "",
        "Per-simulation breakdown:",
        f"  {'Simulation':<45} {'Files':>5} {'OK':>6} {'NoFire':>7} {'Err':>5}",
        "  " + "-" * 65,
    ]
    for r in sim_results:
        lines.append(
            f"  {r.sim_name:<45} {r.n_files:>5} {r.n_pairs_written:>6} "
            f"{r.n_skipped_no_fire:>7} {r.n_skipped_error:>5}"
        )

    errors_section = [r for r in sim_results if r.errors]
    if errors_section:
        lines += ["", "Errors (first 3 per sim):"]
        for r in errors_section:
            lines.append(f"  [{r.sim_name}]")
            for e in r.errors[:3]:
                lines.append(f"    {e}")
    else:
        lines += ["", "No errors."]

    lines += ["", "=" * 70]
    report_text = "\n".join(lines)
    print(report_text)

    with open(out_dir / "qc_report.txt", "w") as f:
        f.write(report_text + "\n")
    print(f"\nQC report written to {out_dir / 'qc_report.txt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()

    raw_dir  = Path(cfg.raw_dir)
    out_dir  = Path(cfg.out_dir)
    shard_dir = out_dir / "shards"

    if not raw_dir.exists():
        print(f"ERROR: PNW_RAW_DIR '{raw_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Discover simulation directories (anything that contains wrfout files)
    sim_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and any(d.glob(f"wrfout_{cfg.domain}_*"))
    ])

    if not sim_dirs:
        print(f"ERROR: No simulation directories found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(exist_ok=True)

    print(f"Found {len(sim_dirs)} simulation directories in {raw_dir}")
    print(f"Output → {out_dir}")
    print(f"Config: domain={cfg.domain}, pair_k={cfg.pair_k}, fire_thr={cfg.fire_thr}, "
          f"engine={cfg.engine}, k_levels={cfg.k_levels}")
    print(f"        fire_vars={cfg.fire_vars}")
    print(f"        target_grid={cfg.target_h}x{cfg.target_w} "
          f"(0=native), fuel_frac={cfg.include_fuel_frac}, pm25_proxy={cfg.include_pm25_proxy}")
    print()

    global_shard_idx = 0
    manifest: List[Dict] = []
    sim_results: List[SimResult] = []

    for i, sim_dir in enumerate(sim_dirs):
        print(f"[{i+1:>3}/{len(sim_dirs)}] {sim_dir.name} ...", flush=True)
        global_shard_idx, result = build_sim(
            sim_dir, cfg, shard_dir, global_shard_idx, manifest
        )
        sim_results.append(result)
        print(f"         → {result.n_pairs_written} shards "
              f"({result.n_skipped_no_fire} no-fire, {result.n_skipped_error} errors)")

    if not manifest:
        print("\nERROR: No shards produced. Check errors above.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal shards: {global_shard_idx}")
    write_qc_report(out_dir, sim_results, manifest, cfg)


if __name__ == "__main__":
    main()
