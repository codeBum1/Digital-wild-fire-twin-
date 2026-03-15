#!/bin/bash
#SBATCH --job-name=build_pnw_shards
#SBATCH --output=/home/abazan/wrfout_sandbox/build_pnw_shards.out
#SBATCH --error=/home/abazan/wrfout_sandbox/build_pnw_shards.err
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Build NPZ shards from 321 usable PNW WRF-SFIRE runs (300 full + 21 partial ≥48h).
# pnw_sfire_usable/ contains symlinks: SID -> /scratch/.../runs/BATCH/SID/wrf/
#
# PNW runs have FLAME_LENGTH (unlike Evans Canyon) so we include it.
# FUEL_FRAC and FUEL_FRAC_BURNT are also present — include for smoke channel.
# Domain is d01 (101x101 native), resized to 99x99 to match trained UNet.
#
# Expected output: ~25,000–35,000 shards (72h / 30min = 145 frames/sim,
#   ~100 valid pairs/sim after fire-presence QC, ×321 sims).

export PNW_RAW_DIR=/home/abazan/wrfout_sandbox/pnw_sfire_usable
export PNW_OUT_DIR=/home/abazan/wrfout_sandbox/vit_dataset_pnw_tplus1
export VIT_DOMAIN=d01
export VIT_FIRE_VARS="FIRE_AREA,ROS,FLAME_LENGTH"
export VIT_PAIR_K=1
export VIT_FIRE_THR=0.0001
export VIT_ENGINE=netcdf4
export PNW_TARGET_H=99
export PNW_TARGET_W=99

export VIT_INCLUDE_FUEL_FRAC=1
export VIT_INCLUDE_PM25_PROXY=1

echo "Building PNW shards from 321 runs at $(date)"
echo "Runs in PNW_RAW_DIR: $(ls $PNW_RAW_DIR | wc -l)"
python build_pnw_dataset.py
echo "Shards built: $(ls $PNW_OUT_DIR/shards/ 2>/dev/null | wc -l)"
echo "Done at $(date)"
