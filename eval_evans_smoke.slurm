#!/bin/bash
#SBATCH --job-name=eval_evans_smoke
#SBATCH --output=/home/abazan/wrfout_sandbox/eval_evans_smoke.out
#SBATCH --error=/home/abazan/wrfout_sandbox/eval_evans_smoke.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Eval smoke model on Evans Canyon (smoke shards: in_c=10, out_c=3)
export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_evans_smoke_tplus1/shards
export LOAD_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_smoke_best.pt
export EVAL_ONLY=1

export FIRE_THR=0.1
export VIS_THR=0.40
export FIRE_CH_IDX=-2   # FUEL_FRAC is last X channel (in_c=10); FIRE_AREA is at -2
export ROLL_OUT_STEPS=5
export ROLL_OUT_N_SAMPLES=3
export BATCH_SIZE=8
export NUM_WORKERS=4
export SEED=42

# Danger rating
export RUN_DANGER_RATING=1
export DANGER_WRF_FILE=/scratch/wdt/sfire/new_wrfxpy/wrfxpy/wksp/testing-NAM218_evans_canyon/wrf/wrfout_d03_2020-08-31_00:00:00

echo "Evaluating smoke model on Evans Canyon at $(date)"
echo "Baseline (retrained, no smoke): iou=0.2112  dice=0.3194  mpd=2.76"
python train_unet.py |& tee eval_evans_smoke.log
echo "Done at $(date)"
