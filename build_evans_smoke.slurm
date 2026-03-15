#!/bin/bash
#SBATCH --job-name=build_evans_smoke
#SBATCH --output=/home/abazan/wrfout_sandbox/build_evans_smoke.out
#SBATCH --error=/home/abazan/wrfout_sandbox/build_evans_smoke.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Rebuild Evans Canyon shards with FUEL_FRAC + PM2.5 proxy to match the smoke model.
# Needed so eval (eval_evans_smoke.slurm) uses compatible in_c=10 / out_c=3 shards.

export PNW_RAW_DIR=/scratch/wdt/sfire/new_wrfxpy/wrfxpy/wksp/testing-NAM218_evans_canyon
export PNW_OUT_DIR=/home/abazan/wrfout_sandbox/vit_dataset_evans_smoke_tplus1
export VIT_DOMAIN=d03
export VIT_FIRE_VARS="FIRE_AREA,ROS"
export VIT_PAIR_K=1
export VIT_FIRE_THR=0.0001
export VIT_ENGINE=netcdf4
export PNW_TARGET_H=99
export PNW_TARGET_W=99

export VIT_INCLUDE_FUEL_FRAC=1
export VIT_INCLUDE_PM25_PROXY=1

echo "Building Evans Canyon smoke shards at $(date)"
python build_pnw_dataset.py
echo "Shards built: $(ls $PNW_OUT_DIR/shards/ | wc -l)"
echo "Done at $(date)"
