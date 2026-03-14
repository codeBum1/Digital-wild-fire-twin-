#!/bin/bash
#SBATCH --job-name=build_fire_shards
#SBATCH --output=/home/abazan/wrfout_sandbox/build_real_fire.out
#SBATCH --error=/home/abazan/wrfout_sandbox/build_real_fire.err
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Build NPZ shards from all 15 confirmed-fire real WRF-SFIRE runs.
# real_fire_raw/ contains symlinks: each points to a run's wrf/ dir
# which holds wrfout_d03_* files at 15-min intervals.
#
# FLAME_LENGTH is absent in all these runs — use FIRE_AREA + ROS only.

export PNW_RAW_DIR=/home/abazan/wrfout_sandbox/real_fire_raw
export PNW_OUT_DIR=/home/abazan/wrfout_sandbox/vit_dataset_real_fire_tplus1
export VIT_DOMAIN=d03
export VIT_FIRE_VARS="FIRE_AREA,ROS"
export VIT_PAIR_K=1
export VIT_FIRE_THR=0.0001
export VIT_ENGINE=netcdf4

# Resize to 99x99 to match existing UNet (builds_vit_dataset default)
export PNW_TARGET_H=99
export PNW_TARGET_W=99

echo "Starting real-fire corpus build at $(date)"
echo "Runs: $(ls $PNW_RAW_DIR | wc -l)"
python build_pnw_dataset.py
echo "Done at $(date)"
