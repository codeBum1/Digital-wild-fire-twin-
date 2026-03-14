#!/bin/bash
#SBATCH --job-name=eval_evans
#SBATCH --output=/home/abazan/wrfout_sandbox/eval_evans.out
#SBATCH --error=/home/abazan/wrfout_sandbox/eval_evans.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Evaluate the sandbox-trained model on the Evans Canyon real fire dataset.
# This baselines cross-domain generalization before retraining on real fire data.

export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_evans_tplus1/shards
export LOAD_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_ms_gpu_best.pt

# Eval-only: skip training, run full evaluation + rollout visualization
export EVAL_ONLY=1

# Match train-time settings so normalization and thresholds are consistent
export FIRE_THR=0.1
export VIS_THR=0.6
export ROLL_OUT_STEPS=5
export ROLL_OUT_N_SAMPLES=3
export BATCH_SIZE=8
export NUM_WORKERS=4
export SEED=42

echo "Evaluating unet_ms_gpu_best.pt on Evans Canyon shards at $(date)"
python train_unet.py |& tee eval_evans.log
echo "Done at $(date)"
