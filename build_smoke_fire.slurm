#!/bin/bash
#SBATCH --job-name=build_smoke_shards
#SBATCH --output=/home/abazan/wrfout_sandbox/build_smoke_fire.out
#SBATCH --error=/home/abazan/wrfout_sandbox/build_smoke_fire.err
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Rebuild real-fire corpus shards with FUEL_FRAC (X channel) and PM2.5 proxy (Y channel).
# Produces in_c=10, out_c=3 shards for smoke/air-quality regression training.

export PNW_RAW_DIR=/home/abazan/wrfout_sandbox/real_fire_raw
export PNW_OUT_DIR=/home/abazan/wrfout_sandbox/vit_dataset_real_fire_smoke_tplus1
export VIT_DOMAIN=d03
export VIT_FIRE_VARS="FIRE_AREA,ROS"
export VIT_PAIR_K=1
export VIT_FIRE_THR=0.0001
export VIT_ENGINE=netcdf4
export PNW_TARGET_H=99
export PNW_TARGET_W=99

export VIT_INCLUDE_FUEL_FRAC=1
export VIT_INCLUDE_PM25_PROXY=1

echo "Starting smoke real-fire corpus build at $(date)"
echo "Runs: $(ls $PNW_RAW_DIR | wc -l)"
python build_pnw_dataset.py
echo "Shards built: $(ls $PNW_OUT_DIR/shards/ | wc -l)"
echo "Done at $(date)"
