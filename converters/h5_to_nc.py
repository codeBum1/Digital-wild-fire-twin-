#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Sequence, Set, List

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class ConvertConfig:
    read_engine: str = "h5netcdf"
    write_engine: str = "netcdf4"
    decode_times: bool = False

    add_times_str: bool = True
    compress_level: int = 1          # 0-9
    chunk_target_mb: int = 16        # heuristic chunk sizing
    cast: str = "none"               # none|float32|float16

    include_vars: Optional[Set[str]] = None
    drop_vars: Optional[Set[str]] = None


# ---------- util ----------
def _split_csv(s: Optional[str]) -> Optional[Set[str]]:
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return set(items) if items else None

def _read_vars_file(path: str) -> Set[str]:
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line)
    return out

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def open_dataset(path: str, cfg: ConvertConfig) -> xr.Dataset:
    return xr.open_dataset(path, engine=cfg.read_engine, decode_times=cfg.decode_times)

def wrf_times_to_str(times_da: xr.DataArray) -> xr.DataArray:
    """
    Supports both:
      - WRF classic char array: Times(Time, DateStrLen) dtype S1
      - Collapsed string array: Times(Time,) dtype like |S19
    Returns Times_str(Time,) as unicode strings.
    """
    t = times_da.values
    t = np.asarray(t)

    # Case A: already (Time,) fixed-length bytes/str
    if t.ndim == 1:
        out = []
        for x in t:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(x.decode("utf-8").strip())
            else:
                out.append(str(x).strip())
        return xr.DataArray(np.array(out, dtype="U"), dims=("Time",), name="Times_str")

    # Case B: (Time, DateStrLen) char matrix
    if t.ndim == 2:
        rows = []
        for row in t:
            chars = []
            for c in row:
                if isinstance(c, (bytes, np.bytes_)):
                    chars.append(c.decode("utf-8"))
                else:
                    chars.append(str(c))
            rows.append("".join(chars).strip())
        return xr.DataArray(np.array(rows, dtype="U"), dims=("Time",), name="Times_str")

    raise ValueError(f"Times expected 1D or 2D. Got {t.shape}")

def add_times_str(ds: xr.Dataset) -> xr.Dataset:
    if "Times" in ds.variables and "Times_str" not in ds.variables:
        try:
            ds["Times_str"] = wrf_times_to_str(ds["Times"])
        except Exception as e:
            log(f"[WARN] Could not create Times_str: {e}")
    return ds

def select_vars(ds: xr.Dataset, cfg: ConvertConfig) -> xr.Dataset:
    # Include-list keeps only those variables + required coords/dims if present
    if cfg.include_vars is not None:
        keep = set(cfg.include_vars)

        # Always keep core dimension coords if they exist
        for k in ["Time", "Times", "Times_str", "south_north", "west_east", "bottom_top",
                  "south_north_stag", "west_east_stag", "bottom_top_stag",
                  "south_north_subgrid", "west_east_subgrid",
                  "XLAT", "XLONG", "FXLAT", "FXLONG"]:
            if k in ds.variables:
                keep.add(k)

        existing = [v for v in keep if v in ds.variables]
        ds = ds[existing]

    # Drop-list removes variables if present
    if cfg.drop_vars is not None:
        to_drop = [v for v in cfg.drop_vars if v in ds.variables]
        if to_drop:
            ds = ds.drop_vars(to_drop)

    return ds

def cast_dataset(ds: xr.Dataset, cast: str) -> xr.Dataset:
    cast = cast.lower()
    if cast == "none":
        return ds
    if cast not in {"float32", "float16"}:
        raise ValueError("cast must be one of: none, float32, float16")

    target = np.float32 if cast == "float32" else np.float16
    out = ds.copy()
    for v in out.data_vars:
        da = out[v]
        if np.issubdtype(da.dtype, np.floating):
            out[v] = da.astype(target)
    return out

def _guess_chunks(da: xr.DataArray, target_mb: int) -> Optional[tuple]:
    """
    Heuristic: choose chunks along leading dims so each chunk ~ target_mb.
    Only applies to 3D/4D+ arrays and only if sizes are large.
    """
    if da.ndim < 3:
        return None
    shape = da.shape
    itemsize = np.dtype(da.dtype).itemsize
    total_bytes = int(np.prod(shape)) * itemsize
    if total_bytes < target_mb * 1024 * 1024:
        return None

    # Start with full last 2 dims (spatial), chunk earlier dims
    chunks = list(shape)
    # If 4D+: reduce first dim, then second, etc.
    target_bytes = target_mb * 1024 * 1024
    chunk_bytes = int(np.prod(chunks)) * itemsize

    i = 0
    while chunk_bytes > target_bytes and i < len(chunks) - 2:
        chunks[i] = max(1, chunks[i] // 2)
        chunk_bytes = int(np.prod(chunks)) * itemsize
        i += 1

    return tuple(chunks)

def make_encoding(ds: xr.Dataset, cfg: ConvertConfig) -> Dict[str, Dict[str, Any]]:
    enc: Dict[str, Dict[str, Any]] = {}
    for v in ds.data_vars:
        da = ds[v]
        e: Dict[str, Any] = {}

        # Compression
        if cfg.compress_level and cfg.compress_level > 0 and da.size >= 10_000:
            e["zlib"] = True
            e["complevel"] = int(cfg.compress_level)

        # Chunking
        ch = _guess_chunks(da, cfg.chunk_target_mb)
        if ch is not None:
            e["chunksizes"] = ch

        if e:
            enc[v] = e
    return enc

def add_provenance(ds: xr.Dataset, in_path: str) -> xr.Dataset:
    out = ds.copy()
    out.attrs = dict(out.attrs)  # ensure mutable copy
    out.attrs["source_file"] = os.path.abspath(in_path)
    out.attrs["converted_utc"] = datetime.now(timezone.utc).isoformat()
    out.attrs["converter_host"] = socket.gethostname()
    out.attrs["converter_user"] = os.environ.get("USER", "unknown")
    out.attrs["converter_script"] = "h5_to_nc.py"
    return out

def write_netcdf(ds: xr.Dataset, out_path: str, cfg: ConvertConfig) -> None:
    encoding = make_encoding(ds, cfg)
    ds.to_netcdf(out_path, engine=cfg.write_engine, encoding=encoding)

def verify(out_path: str) -> None:
    ds = xr.open_dataset(out_path, decode_times=False)
    log(f"[VERIFY] wrote vars={len(ds.variables)} data_vars={len(ds.data_vars)} dims={dict(ds.sizes)}")
    # quick smoke check for SFIRE/WRF common vars if present
    for key in ["Times_str", "XLAT", "XLONG", "FXLAT", "FXLONG", "FIRE_AREA", "TIGN_G", "ROS", "U10", "V10", "T2", "PSFC"]:
        if key in ds.variables:
            log(f"[VERIFY] has {key} shape={ds[key].shape} dtype={ds[key].dtype}")
    ds.close()

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline-grade HDF5(.h5) -> NetCDF(.nc) converter for WRF/SFIRE-like files.")
    p.add_argument("--in_h5", required=True)
    p.add_argument("--out_nc", required=True)

    p.add_argument("--read_engine", default="h5netcdf")
    p.add_argument("--write_engine", default="netcdf4")

    p.add_argument("--compress", type=int, default=1, help="0-9 (0 disables)")
    p.add_argument("--chunk_mb", type=int, default=16, help="target chunk size in MB")
    p.add_argument("--cast", default="none", help="none|float32|float16")

    p.add_argument("--vars", default=None, help="comma-separated include list (keep only these vars)")
    p.add_argument("--vars_file", default=None, help="file with one var name per line (include list)")
    p.add_argument("--drop_vars", default=None, help="comma-separated drop list")

    p.add_argument("--no_times_str", action="store_true")
    p.add_argument("--inspect", action="store_true", help="print dataset summary then exit")
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    include = _split_csv(args.vars)
    if args.vars_file:
        include_from_file = _read_vars_file(args.vars_file)
        include = include_from_file if include is None else (include | include_from_file)

    cfg = ConvertConfig(
        read_engine=args.read_engine,
        write_engine=args.write_engine,
        decode_times=False,
        add_times_str=(not args.no_times_str),
        compress_level=int(args.compress),
        chunk_target_mb=int(args.chunk_mb),
        cast=args.cast,
        include_vars=include,
        drop_vars=_split_csv(args.drop_vars),
    )

    log(f"[INFO] reading {args.in_h5} (engine={cfg.read_engine})")
    ds = open_dataset(args.in_h5, cfg)

    if args.inspect:
        print(ds)
        ds.close()
        return 0

    if cfg.add_times_str:
        ds = add_times_str(ds)

    ds = select_vars(ds, cfg)
    ds = cast_dataset(ds, cfg.cast)
    ds = add_provenance(ds, args.in_h5)

    log(f"[INFO] writing {args.out_nc} (engine={cfg.write_engine}, compress={cfg.compress_level}, chunk_mb={cfg.chunk_target_mb}, cast={cfg.cast})")
    write_netcdf(ds, args.out_nc, cfg)
    ds.close()

    verify(args.out_nc)
    log("[DONE]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
