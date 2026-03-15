#!/bin/bash
#SBATCH --job-name=eval_evans_mx
#SBATCH --output=/home/abazan/wrfout_sandbox/eval_evans_mixed.out
#SBATCH --error=/home/abazan/wrfout_sandbox/eval_evans_mixed.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Eval mixed model on Evans Canyon only (original 334 shards, no FUEL_FRAC)
export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_evans_tplus1/shards
export LOAD_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_mixed_best.pt
export OUT_C=3
export EVAL_ONLY=1

export FIRE_THR=0.1
export VIS_THR=0.6
export ROLL_OUT_STEPS=5
export ROLL_OUT_N_SAMPLES=3
export BATCH_SIZE=8
export NUM_WORKERS=4
export SEED=42

export RUN_DANGER_RATING=1
export DANGER_WRF_FILE=/scratch/wdt/sfire/new_wrfxpy/wrfxpy/wksp/testing-NAM218_evans_canyon/wrf/wrfout_d03_2020-08-31_00:00:00

echo "Evaluating mixed model on Evans Canyon at $(date)"
echo "Baselines — zero-shot: iou=0.0715  retrained: iou=0.2112"
python train_unet.py |& tee eval_evans_mixed.log
echo "Done at $(date)"
