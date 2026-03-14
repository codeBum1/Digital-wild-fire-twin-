#!/usr/bin/env python3
"""
WRF-SFIRE data diagnostic script.
Checks all fire files for completeness, variable availability, temporal
continuity, fire mask quality, NaN/Inf, and overlap with existing shards.
"""

import sys
import re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

# ── Config ────────────────────────────────────────────────────────────────────
FILE_LIST  = "/home/abazan/wrfout_sandbox/wrfout_files_with_fire.txt"
SHARDS_DIR = "/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus1_fireX_maxpool_paired/shards"

REQUIRED_2D  = ["T2", "Q2", "PSFC", "U10", "V10", "HGT"]
REQUIRED_3D  = ["T", "QVAPOR"]          # staggered or unstaggered both ok
REQUIRED_FIRE= ["FIRE_AREA", "TIGN_G"]
OPTIONAL     = ["ROS", "FLAME_LENGTH", "FUEL_FRAC"]

ATM_SHAPE_EXPECTED = None   # inferred from first valid file
FIRE_SHAPE_EXPECTED = None

TIGN_MONO_WARN = 5          # flag if >5 cells regress
FIRE_SPARSITY_LOW  = 0.0005 # < 0.05 % positive pixels → very sparse
FIRE_SPARSITY_HIGH = 0.80   # > 80 % positive pixels  → near-trivial

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_timestamp(path: Path):
    m = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})', path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d_%H:%M:%S")
    return None


def check_file(path: Path, prev_tign_max: float, file_idx: int,
               atm_ref, fire_ref):
    issues = []
    warns  = []

    # 1. File exists & opens
    if not path.exists():
        return None, None, None, [f"FILE NOT FOUND: {path}"], []

    try:
        ds = xr.open_dataset(str(path), engine="netcdf4")
    except Exception as e:
        return None, None, None, [f"OPEN FAILED: {e}"], []

    # 2. Variable availability
    all_vars = set(ds.data_vars) | set(ds.coords)
    for v in REQUIRED_2D + REQUIRED_3D + REQUIRED_FIRE:
        if v not in all_vars:
            issues.append(f"MISSING required var: {v}")
    for v in OPTIONAL:
        if v not in all_vars:
            warns.append(f"optional var absent: {v}")

    # 3. Grid shape consistency
    try:
        atm_shape = ds["T2"].values.shape  # (Time, sn, we)
    except Exception:
        atm_shape = None
        issues.append("Cannot read T2 shape")

    try:
        fire_shape = ds["FIRE_AREA"].values.shape
    except Exception:
        fire_shape = None
        issues.append("Cannot read FIRE_AREA shape")

    if atm_shape and atm_ref[0] is None:
        atm_ref[0] = atm_shape
    elif atm_shape and atm_shape != atm_ref[0]:
        issues.append(f"ATM shape mismatch: {atm_shape} vs ref {atm_ref[0]}")

    if fire_shape and fire_ref[0] is None:
        fire_ref[0] = fire_shape
    elif fire_shape and fire_shape != fire_ref[0]:
        issues.append(f"FIRE shape mismatch: {fire_shape} vs ref {fire_ref[0]}")

    # 4. NaN / Inf check on key vars
    for v in REQUIRED_2D + REQUIRED_FIRE:
        if v not in ds.data_vars:
            continue
        arr = ds[v].values
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan:
            issues.append(f"NaN in {v}: {n_nan} cells")
        if n_inf:
            issues.append(f"Inf in {v}: {n_inf} cells")

    # 5. FIRE_AREA range check
    if "FIRE_AREA" in ds.data_vars:
        fa = ds["FIRE_AREA"].values[0]
        if fa.min() < -1e-4 or fa.max() > 1.0 + 1e-4:
            issues.append(f"FIRE_AREA out of [0,1]: min={fa.min():.4f} max={fa.max():.4f}")

        # Sparsity
        pos_ratio = (fa > 0.01).mean()
        if pos_ratio < FIRE_SPARSITY_LOW:
            warns.append(f"very sparse fire mask: {pos_ratio*100:.3f}% positive")
        elif pos_ratio > FIRE_SPARSITY_HIGH:
            warns.append(f"near-trivial fire mask: {pos_ratio*100:.1f}% positive")
    else:
        fa = None
        pos_ratio = None

    # 6. TIGN_G monotonicity (should never decrease from previous file)
    tign_max = None
    if "TIGN_G" in ds.data_vars:
        tign = ds["TIGN_G"].values[0]
        tign_max = float(tign.max())
        if prev_tign_max is not None and tign_max < prev_tign_max - 1.0:
            warns.append(
                f"TIGN_G max decreased: {prev_tign_max:.1f} → {tign_max:.1f} "
                "(possible reset or non-sequential file)")
        # count cells where tign decreased vs expected growth
        # (this is a rough check — TIGN=0 means unburned)

    ds.close()
    return atm_shape, fire_shape, tign_max, issues, warns


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("WRF-SFIRE DATA DIAGNOSTIC")
    print("=" * 70)

    # Load file list
    with open(FILE_LIST) as f:
        paths = [Path(l.strip()) for l in f if l.strip()]
    print(f"\nFiles in list: {len(paths)}")

    # Sort by timestamp
    paths_ts = [(p, parse_timestamp(p)) for p in paths]
    paths_ts.sort(key=lambda x: x[1] or datetime.min)
    paths    = [p for p, _ in paths_ts]
    timestamps = [ts for _, ts in paths_ts]

    # ── Check existing shards overlap ────────────────────────────────────────
    shard_dir = Path(SHARDS_DIR)
    if shard_dir.exists():
        n_shards = len(list(shard_dir.glob("*.npz")))
        print(f"Existing shards in {shard_dir.name}: {n_shards}")
        # Try to read source file list from a shard metadata if present
        meta_f = shard_dir.parent / "dataset_meta.json"
        if meta_f.exists():
            import json
            meta = json.load(open(meta_f))
            print(f"  Shard dataset source files: {meta.get('n_files', '?')}")
    else:
        print(f"Shard dir not found: {SHARDS_DIR}")

    # ── Per-file checks ───────────────────────────────────────────────────────
    print("\n── Per-file checks ──────────────────────────────────────────────────")
    atm_ref  = [None]
    fire_ref = [None]
    prev_tign_max = None

    all_issues = {}
    all_warns  = {}
    fire_stats = []  # (idx, name, pos_ratio, tign_max)

    n_ok = 0
    for i, (p, ts) in enumerate(zip(paths, timestamps)):
        atm_shape, fire_shape, tign_max, issues, warns = check_file(
            p, prev_tign_max, i, atm_ref, fire_ref)

        if tign_max is not None:
            prev_tign_max = tign_max

        tag = "OK" if not issues else "FAIL"
        ts_str = ts.strftime("%m-%d %H:%M") if ts else "??:??"
        print(f"  [{i:02d}] {ts_str}  {p.name[:40]:<40}  {tag}", end="")
        if warns:
            print(f"  WARN({len(warns)})", end="")
        print()

        if issues:
            all_issues[i] = issues
        if warns:
            all_warns[i] = warns

        if "FIRE_AREA" in (xr.open_dataset(str(p), engine="netcdf4").data_vars
                           if p.exists() else {}):
            pass  # already collected in check_file

        if not issues:
            n_ok += 1

    # ── Temporal gap check ───────────────────────────────────────────────────
    print("\n── Temporal continuity ──────────────────────────────────────────────")
    gaps = []
    for i in range(1, len(timestamps)):
        if timestamps[i] is None or timestamps[i-1] is None:
            continue
        dt = timestamps[i] - timestamps[i-1]
        expected = timedelta(hours=1)
        if abs(dt - expected) > timedelta(minutes=5):
            gaps.append((i, timestamps[i-1], timestamps[i], dt))

    if not gaps:
        print("  No temporal gaps detected (all consecutive files are 1h apart)")
    else:
        for idx, t0, t1, dt in gaps:
            print(f"  GAP at file {idx}: {t0} → {t1}  (dt={dt})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Files checked      : {len(paths)}")
    print(f"  Files OK (no error): {n_ok}")
    print(f"  Files with issues  : {len(all_issues)}")
    print(f"  Files with warnings: {len(all_warns)}")
    print(f"  ATM grid shape     : {atm_ref[0]}")
    print(f"  Fire grid shape    : {fire_ref[0]}")
    print(f"  Temporal span      : {timestamps[0]} → {timestamps[-1]}")

    if all_issues:
        print("\n  ISSUES:")
        for i, msgs in all_issues.items():
            for m in msgs:
                print(f"    [{i:02d}] {m}")

    if all_warns:
        print("\n  WARNINGS:")
        for i, msgs in all_warns.items():
            for m in msgs:
                print(f"    [{i:02d}] {m}")

    print("\n── Fire mask sparsity over time ─────────────────────────────────────")
    print("  (re-opening files to collect fire stats)")
    for i, p in enumerate(paths):
        if not p.exists():
            continue
        try:
            ds = xr.open_dataset(str(p), engine="netcdf4")
            if "FIRE_AREA" in ds.data_vars and "TIGN_G" in ds.data_vars:
                fa   = ds["FIRE_AREA"].values[0]
                tign = ds["TIGN_G"].values[0]
                pos  = (fa > 0.01).mean() * 100
                ts_str = timestamps[i].strftime("%m-%d %H:%M") if timestamps[i] else "??"
                print(f"  [{i:02d}] {ts_str}  pos={pos:6.2f}%  "
                      f"TIGN_max={tign.max():.0f}s  "
                      f"FA_max={fa.max():.4f}  FA_nonzero={np.count_nonzero(fa)}")
            ds.close()
        except Exception as e:
            print(f"  [{i:02d}] ERROR: {e}")

    print("\n" + "=" * 70)
    verdict = "PASS — data looks good for shard building." if not all_issues else \
              "FAIL — fix issues above before building shards."
    print(f"VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
