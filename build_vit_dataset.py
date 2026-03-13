#!/usr/bin/env python3
"""
Build a ViT-ready dataset from a sequence of WRF-SFIRE wrfout NetCDF/HDF5 files.

Outputs:
- Zarr dataset: dataset_vit.zarr (recommended)
or
- NPZ shards: shards/00000.npz, ...

Key features:
- destagger U/V -> U_mass, V_mass on (Time, bottom_top, south_north, west_east)
- choose vertical level(s)
- downsample fire 500x500 -> 99x99 (prototype)
- robust missing-var handling + shape validation
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import xarray as xr


# ----------------------------
# Config
# ----------------------------

@dataclass
class BuildConfig:
    # Variables we want as 2D channels (must be (Time, south_north, west_east))
    vars_2d: List[str]

    # Variables we want as 3D channels (must be (Time, bottom_top, south_north, west_east))
    vars_3d: List[str]

    # Fire vars on subgrid (Time, south_north_subgrid, west_east_subgrid)
    fire_vars: List[str]

    k_levels: List[int]

    out_dir: str
    out_format: str  # "zarr" or "npz_shards"

    # defaults BELOW
    downsample_fire_to_atm: bool = True
    read_engine: str = "netcdf4"
    assume_time_len_1: bool = True
    zarr_chunk_n: int = 16
    pair_k: int = 1  # how many timesteps ahead to pair (1 = consecutive)

DEFAULT_CONFIG = BuildConfig(
    vars_2d=["T2", "Q2", "PSFC", "U10", "V10", "HGT"],
    vars_3d=["T", "QVAPOR"],
    fire_vars=["FIRE_AREA", "ROS", "FLAME_LENGTH"],
    k_levels=[0],  # start with 0; you can change to [0,5,10] later
    out_dir="dataset_vit_out",
    out_format="zarr",  # or "npz_shards"
    downsample_fire_to_atm=True,
    read_engine="netcdf4",
    assume_time_len_1=True,
    zarr_chunk_n=16,
)


# ----------------------------
# Helpers
# ----------------------------

def _assert_has(ds: xr.Dataset, names: List[str], label: str) -> None:
    missing = [n for n in names if n not in ds.variables]
    if missing:
        raise KeyError(f"Missing {label} variables: {missing}")


def destagger_u_to_mass(ds: xr.Dataset) -> xr.DataArray:
    """
    U: (Time, bottom_top, south_north, west_east_stag=100)
    -> U_mass: (Time, bottom_top, south_north, west_east=99)
    """
    U = ds["U"]
    if "west_east_stag" not in U.dims:
        raise ValueError("U does not have west_east_stag dim; cannot destagger.")
    U_mass = 0.5 * (U.isel(west_east_stag=slice(0, -1)) + U.isel(west_east_stag=slice(1, None)))
    # Rename stag dim to mass dim name
    U_mass = U_mass.rename({"west_east_stag": "west_east"})
    # Assign coords if present
    if "west_east" in ds.coords or "west_east" in ds.variables:
        U_mass = U_mass.assign_coords(west_east=ds["west_east"])
    return U_mass


def destagger_v_to_mass(ds: xr.Dataset) -> xr.DataArray:
    """
    V: (Time, bottom_top, south_north_stag=100, west_east)
    -> V_mass: (Time, bottom_top, south_north=99, west_east)
    """
    V = ds["V"]
    if "south_north_stag" not in V.dims:
        raise ValueError("V does not have south_north_stag dim; cannot destagger.")
    V_mass = 0.5 * (V.isel(south_north_stag=slice(0, -1)) + V.isel(south_north_stag=slice(1, None)))
    V_mass = V_mass.rename({"south_north_stag": "south_north"})
    if "south_north" in ds.coords or "south_north" in ds.variables:
        V_mass = V_mass.assign_coords(south_north=ds["south_north"])
    return V_mass


def downsample_fire_to_atm_grid(ds: xr.Dataset, fire_da: xr.DataArray) -> xr.DataArray:
    """
    Downsample fire field (typically 500x500) to atmospheric grid (typically 99x99)
    using MAX pooling (preserves sparse ignition / thin fronts).
    Returns DataArray with dims (Time, south_north, west_east).
    """
    import numpy as np
    import xarray as xr

    fire = fire_da.values  # expected (Time, Hf, Wf)
    if fire.ndim != 3:
        raise ValueError(f"fire_da expected 3D (Time,Hf,Wf), got shape {fire.shape}")

    T, Hf, Wf = fire.shape
    Ha = int(ds.sizes["south_north"])
    Wa = int(ds.sizes["west_east"])

    # Bin edges mapping fire grid -> atm grid
    y_edges = np.linspace(0, Hf, Ha + 1).astype(int)
    x_edges = np.linspace(0, Wf, Wa + 1).astype(int)

    out = np.zeros((T, Ha, Wa), dtype=np.float32)

    for ti in range(T):
        f = fire[ti].astype(np.float32)
        for yi in range(Ha):
            y0, y1 = y_edges[yi], y_edges[yi + 1]
            if y1 <= y0:
                y1 = min(Hf, y0 + 1)
            for xi in range(Wa):
                x0, x1 = x_edges[xi], x_edges[xi + 1]
                if x1 <= x0:
                    x1 = min(Wf, x0 + 1)

                block = f[y0:y1, x0:x1]
                out[ti, yi, xi] = np.nanmax(block) if block.size else 0.0

    return xr.DataArray(
        out,
        dims=("Time", "south_north", "west_east"),
        coords={
            "Time": fire_da.coords.get("Time", np.arange(T)),
            "south_north": ds.coords.get("south_north", np.arange(Ha)),
            "west_east": ds.coords.get("west_east", np.arange(Wa)),
        },
        name=f"{fire_da.name}_99max",
    )

def load_one_file(path: str, engine: str) -> xr.Dataset:
    """
    Load a WRF wrfout file.
    engine: "netcdf4" or "h5netcdf"
    """
    return xr.open_dataset(path, engine=engine)


def extract_sample_tensors(
    ds: xr.Dataset,
    cfg: BuildConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Produce one training sample from one file:
      X: (C, H, W)
      Y: (F, H, W)  where F=len(fire_vars)
    with consistent H/W = (south_north, west_east)

    Also returns metadata dict for tracking.
    """
    # Ensure required base vars exist to compute U_mass/V_mass if requested
    need_u_mass = "U_mass" in cfg.vars_3d
    need_v_mass = "V_mass" in cfg.vars_3d
    if need_u_mass and "U_mass" not in ds.variables:
        _assert_has(ds, ["U"], "base (for U_mass)")
        ds = ds.assign(U_mass=destagger_u_to_mass(ds))
    if need_v_mass and "V_mass" not in ds.variables:
        _assert_has(ds, ["V"], "base (for V_mass)")
        ds = ds.assign(V_mass=destagger_v_to_mass(ds))

    # Validate 2D vars exist
    _assert_has(ds, cfg.vars_2d, "2D")

    # Validate 3D vars exist (after destagger inject)
    _assert_has(ds, cfg.vars_3d, "3D")

    # Validate fire vars exist
    _assert_has(ds, cfg.fire_vars, "FIRE")

    # Basic dims
    H = ds.sizes["south_north"]
    W = ds.sizes["west_east"]

    # Ensure Time exists
    if "Time" not in ds.sizes:
        raise ValueError("Dataset missing Time dimension.")

    if cfg.assume_time_len_1 and ds.sizes["Time"] != 1:
        raise ValueError(f"Expected Time=1 per file, but got Time={ds.sizes['Time']} for this file.")

    # ---- Build X channels ----
    X_channels: List[np.ndarray] = []

    # 2D vars: (Time, H, W) -> take time index 0 -> (H, W)
    for v in cfg.vars_2d:
        arr = ds[v].values
        # allow (Time,H,W)
        if arr.ndim != 3:
            raise ValueError(f"{v} expected 3D (Time,H,W), got shape {arr.shape}")
        X_channels.append(arr[0].astype(np.float32))

    # 3D vars at each selected k: (Time, Z, H, W) -> (H,W) per k
    for v in cfg.vars_3d:
        da = ds[v]
        if da.ndim != 4:
            raise ValueError(f"{v} expected 4D (Time,Z,H,W), got dims {da.dims} shape {da.shape}")
        for k in cfg.k_levels:
            if k < 0 or k >= da.sizes["bottom_top"]:
                raise IndexError(f"k={k} out of range for bottom_top={da.sizes['bottom_top']}")
            X_channels.append(da.isel(bottom_top=k).values[0].astype(np.float32))


    # ---- Add previous fire state as an input channel ----
    # This makes the task "propagate fire(t) -> fire(t+1)" instead of "ignite from atmosphere only".
    fire_in_name = "FIRE_AREA"
    if fire_in_name not in ds:
        raise KeyError(f"Expected {fire_in_name} in dataset to add as X input, but it was not found.")
    fire_in_da = ds[fire_in_name]

    if cfg.downsample_fire_to_atm:
        fire_in_99 = downsample_fire_to_atm_grid(ds, fire_in_da)
        X_channels.append(fire_in_99.values[0].astype(np.float32))
    else:
        X_channels.append(fire_in_da.values[0].astype(np.float32))

    X = np.stack(X_channels, axis=0)  # (C,H,W)

    # ---- Build Y channels ----
    Y_channels: List[np.ndarray] = []
    for fv in cfg.fire_vars:
        fire_da = ds[fv]
        # expected (Time, 500, 500)
        if cfg.downsample_fire_to_atm:
            fire_99 = downsample_fire_to_atm_grid(ds, fire_da)
            Y_channels.append(fire_99.values[0].astype(np.float32))
        else:
            # keep native fire grid
            Y_channels.append(fire_da.values[0].astype(np.float32))

    Y = np.stack(Y_channels, axis=0)  # (F,H,W) if downsampled else (F,500,500)

    meta = {
        "H": int(H),
        "W": int(W),
        "C": int(X.shape[0]),
        "F": int(Y.shape[0]),
        "k_levels": cfg.k_levels,
    }
    return X, Y, meta
# ----------------------------
# helper functions
# ----------------------------
def extract_X_only(ds: xr.Dataset, cfg: BuildConfig) -> np.ndarray:
    """Build ONLY X from one dataset."""
    _assert_has(ds, cfg.vars_2d, "2D")
    _assert_has(ds, cfg.vars_3d, "3D")
    _assert_has(ds, cfg.fire_vars, "FIRE")

    X_channels: List[np.ndarray] = []

    for v in cfg.vars_2d:
        arr = ds[v].values
        if arr.ndim != 3:
            raise ValueError(f"{v} expected 3D (Time,H,W), got shape {arr.shape}")
        X_channels.append(arr[0].astype(np.float32))

    for v in cfg.vars_3d:
        da = ds[v]
        if da.ndim != 4:
            raise ValueError(f"{v} expected 4D (Time,Z,H,W), got dims {da.dims} shape {da.shape}")
        for k in cfg.k_levels:
            if k < 0 or k >= da.sizes["bottom_top"]:
                raise IndexError(f"k={k} out of range for bottom_top={da.sizes['bottom_top']}")
            X_channels.append(da.isel(bottom_top=k).values[0].astype(np.float32))

    # fire(t) input channel (maxpool-downsampled)
    fire_in_da = ds["FIRE_AREA"]
    if cfg.downsample_fire_to_atm:
        fire_in_99 = downsample_fire_to_atm_grid(ds, fire_in_da)
        X_channels.append(fire_in_99.values[0].astype(np.float32))
    else:
        X_channels.append(fire_in_da.values[0].astype(np.float32))

    X = np.stack(X_channels, axis=0)
    return X


def extract_Y_only(ds: xr.Dataset, cfg: BuildConfig) -> np.ndarray:
    """Build ONLY Y from one dataset."""
    _assert_has(ds, cfg.fire_vars, "FIRE")

    Y_channels: List[np.ndarray] = []
    for fv in cfg.fire_vars:
        fire_da = ds[fv]
        if cfg.downsample_fire_to_atm:
            fire_99 = downsample_fire_to_atm_grid(ds, fire_da)
            Y_channels.append(fire_99.values[0].astype(np.float32))
        else:
            Y_channels.append(fire_da.values[0].astype(np.float32))

    Y = np.stack(Y_channels, axis=0)
    return Y
# ----------------------------
# Writers
# ----------------------------

def write_npz_shards(Xs: np.ndarray, Ys: np.ndarray, meta_list: List[Dict], out_dir: str) -> None:
    """
    Write one file per sample: shards/00000.npz etc.
    Xs: (N,C,H,W)
    Ys: (N,F,H,W)
    """
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    for i in range(Xs.shape[0]):
        p = os.path.join(shards_dir, f"{i:05d}.npz")
        np.savez_compressed(p, X=Xs[i], Y=Ys[i], meta=json.dumps(meta_list[i]))
    print(f"Wrote {Xs.shape[0]} shards to {shards_dir}")


def write_zarr(Xs: np.ndarray, Ys: np.ndarray, file_ids: List[str], out_dir: str, chunk_n: int) -> None:
    """
    Store as a single Zarr dataset:
      X: (N,C,H,W)
      Y: (N,F,H,W)
      file_id: (N,)
    """
    import zarr  # noqa: F401 (forces dependency check)

    os.makedirs(out_dir, exist_ok=True)
    zarr_path = os.path.join(out_dir, "dataset_vit.zarr")

    N, C, H, W = Xs.shape
    _, F, Hy, Wy = Ys.shape

    ds = xr.Dataset(
        {
            "X": (("sample", "channel", "y", "x"), Xs),
            "Y": (("sample", "fire_channel", "y", "x"), Ys),
            "file_id": (("sample",), np.array(file_ids, dtype=object)),
        },
        coords={
            "sample": np.arange(N, dtype=np.int32),
            "channel": np.arange(C, dtype=np.int32),
            "fire_channel": np.arange(F, dtype=np.int32),
            "y": np.arange(H, dtype=np.int32),
            "x": np.arange(W, dtype=np.int32),
        },
        attrs={
            "layout": "X=(N,C,H,W), Y=(N,F,H,W)",
        },
    )

    # Chunking (important for large N)
    chunk_spec = {"X": (chunk_n, C, H, W), "Y": (chunk_n, F, H, W), "file_id": (chunk_n,)}
    encoding = {k: {"chunks": v} for k, v in chunk_spec.items()}

    ds.to_zarr(zarr_path, mode="w", encoding=encoding)
    print(f"Wrote Zarr dataset to: {zarr_path}")


# ----------------------------
# Main build loop
# ----------------------------

def build_dataset(file_list: List[str], cfg: BuildConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    meta_list: List[Dict] = []
    file_ids: List[str] = []

    # Clean file list (strip + remove blanks)
    file_list = [p.strip() for p in file_list if p.strip()]

    print("=== Build config ===")
    print(cfg)
    print("====================")

    K = int(getattr(cfg, "pair_k", 1))
    N = len(file_list)
    print(f"Pairing with stride K={K}: N_files={N} -> N_pairs={max(0, N-K)}")

    # Pair with jump: (t file, t+K file)
    pairs = [(file_list[i], file_list[i + K]) for i in range(N - K)]

    for idx, (path_t, path_tpK) in enumerate(pairs):

        if not os.path.exists(path_t):
            print(f"[WARN] missing file: {path_t}", file=sys.stderr)
            continue
        if not os.path.exists(path_tpK):
            print(f"[WARN] missing file: {path_tpK}", file=sys.stderr)
            continue

        try:
            ds_t   = load_one_file(path_t, engine=cfg.read_engine)
            ds_tpK = load_one_file(path_tpK, engine=cfg.read_engine)

            # X from time t
            X = extract_X_only(ds_t, cfg)

            # Y from time t+K
            Y = extract_Y_only(ds_tpK, cfg)

            # close files
            ds_t.close()
            ds_tpK.close()

            meta = {
                "H": int(ds_t.sizes["south_north"]),
                "W": int(ds_t.sizes["west_east"]),
                "C": int(X.shape[0]),
                "F": int(Y.shape[0]),
                "k_levels": cfg.k_levels,
                "t_path": path_t,
                "tpK_path": path_tpK,
                "pair_k": K,
            }

            X_list.append(X)
            Y_list.append(Y)
            meta_list.append(meta)
            file_ids.append(path_t)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
                print(f"[OK] processed {idx+1}/{len(pairs)} pairs. Latest X={X.shape} Y={Y.shape}")

        except Exception as e:
            print(f"[ERROR] failed on pair {path_t} -> {path_tpK} (K={K}): {repr(e)}", file=sys.stderr)
            continue
    if not X_list:
        raise RuntimeError("No samples were produced. Check file paths, engine, or variable names.")

    Xs = np.stack(X_list, axis=0)  # (N,C,H,W)
    Ys = np.stack(Y_list, axis=0)  # (N,F,H,W)

    print("\n=== Final dataset shapes ===")
    print("X:", Xs.shape, "Y:", Ys.shape)
    print("Samples:", len(file_ids))
    print("===========================\n")

    # Save metadata
    meta_path = os.path.join(cfg.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "vars_2d": cfg.vars_2d,
                "vars_3d": cfg.vars_3d,
                "fire_vars": cfg.fire_vars,
                "k_levels": cfg.k_levels,
                "downsample_fire_to_atm": cfg.downsample_fire_to_atm,
                "file_ids": file_ids,
                "X_shape": list(Xs.shape),
                "Y_shape": list(Ys.shape),
                "layout": "X built from file_ids[i] (t), Y built from file_list[i+K] (t+K)",
                "pair_k": int(getattr(cfg, "pair_k", 1)),
            },
            f,
            indent=2,
        )
    print(f"Wrote meta: {meta_path}")

    # Write data
    if cfg.out_format == "npz_shards":
        write_npz_shards(Xs, Ys, meta_list, cfg.out_dir)
    elif cfg.out_format == "zarr":
        write_zarr(Xs, Ys, file_ids, cfg.out_dir, chunk_n=cfg.zarr_chunk_n)
    else:
        raise ValueError(f"Unknown out_format: {cfg.out_format}")

def read_file_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    """
    Usage examples:
      python build_vit_dataset.py /scratch/wdt/runs/wrf_sfire_palisades_run/wrfout_files.txt
    """
    if len(sys.argv) < 2:
        print("Usage: python build_vit_dataset.py /path/to/wrfout_files.txt", file=sys.stderr)
        sys.exit(1)

    list_path = sys.argv[1]
    files = read_file_list(list_path)

    cfg = DEFAULT_CONFIG

    # Optional: override output directory via env var
    out_dir = os.environ.get("VIT_OUT_DIR")
    if out_dir:
        cfg.out_dir = out_dir

    # Optional: choose format via env var ("zarr" or "npz_shards")
    out_fmt = os.environ.get("VIT_OUT_FMT")
    if out_fmt:
        cfg.out_format = out_fmt

    # Optional: choose vertical levels via env var "0,5,10"
    k_env = os.environ.get("VIT_K_LEVELS")
    if k_env:
        cfg.k_levels = [int(x) for x in k_env.split(",") if x.strip()]

    # Optional: engine override
    eng = os.environ.get("VIT_ENGINE")
    if eng:
        cfg.read_engine = eng

    cfg.pair_k = int(os.environ.get("VIT_PAIR_K", "1"))

    build_dataset(files, cfg)


