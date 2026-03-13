#!/bin/bash
#SBATCH --job-name=unet_clean
#SBATCH --output=/home/abazan/wrfout_sandbox/unet_clean.out
#SBATCH --error=/home/abazan/wrfout_sandbox/unet_clean.err
#SBATCH --partition=short
#SBATCH --nodelist=orcaga06
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:4

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards

# Train from scratch — no warm-start
unset LOAD_CKPT_PATH
unset EVAL_ONLY
export CKPT_PATH=/home/abazan/wrfout_sandbox/unet_clean_last.pt
export BEST_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_clean_best.pt
export OUT_ROOT=/home/abazan/wrfout_sandbox/unet_clean_outputs

export EPOCHS=400
export BATCH_SIZE=16
export NUM_WORKERS=8
export LR=3e-4
export MIN_LR=1e-5
export WEIGHT_DECAY=1e-4
export GRAD_CLIP_NORM=1.0

export FIRE_THR=0.1
export AUTO_POS_WEIGHT=1
export DICE_W=0.30
export FOCAL_W=0.20

export TRAIN_ROLLOUT_STEPS_MAX=4
export ROLLOUT_CURRICULUM_EPOCHS=20
export STEP_LOSS_DECAY=0.90
export TEACHER_FORCING_START=1.0
export TEACHER_FORCING_END=0.15

export KEEP_BURNING=1
export SELFCOND_BLEND=0.90
export USE_FUTURE_EXOGENOUS=1
export USE_EMA=1
export EMA_DECAY=0.995

export ROLL_OUT_STEPS=5
export ROLL_OUT_N_SAMPLES=3
export VIS_THR=0.6
export SEED=42

# Honest evaluation config
export TEST_FRAC=0.20
export SPLIT_SEED=42

python -m py_compile /home/abazan/wrfout_sandbox/train_unet.py || exit 1
torchrun --nproc_per_node=4 /home/abazan/wrfout_sandbox/train_unet.py |& tee /home/abazan/wrfout_sandbox/train_unet_clean.log
