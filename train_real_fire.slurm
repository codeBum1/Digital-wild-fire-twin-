#!/bin/bash
#SBATCH --job-name=unet_real_fire
#SBATCH --output=/home/abazan/wrfout_sandbox/train_real_fire.out
#SBATCH --error=/home/abazan/wrfout_sandbox/train_real_fire.err
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Fine-tune on real-fire corpus (built from 18 WRF-SFIRE runs)
export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_real_fire_tplus1/shards

# Load sandbox checkpoint as starting point — fine-tune rather than train from scratch
export LOAD_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_ms_gpu_best.pt
export OUT_C=3

# Save to new names so baseline is preserved
export CKPT_PATH=/home/abazan/wrfout_sandbox/unet_real_fire_last.pt
export BEST_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_real_fire_best.pt

unset EVAL_ONLY

# Lower LR for fine-tuning
export EPOCHS=80
export BATCH_SIZE=8
export NUM_WORKERS=4
export LR=5e-5
export MIN_LR=5e-6
export WARMUP_EPOCHS=3
export LR_RESTART_EPOCHS=40
export WEIGHT_DECAY=1e-4
export GRAD_CLIP_NORM=1.0

export FIRE_THR=0.1
export AUTO_POS_WEIGHT=1
export DICE_W=0.30
export FOCAL_W=0.20

export TRAIN_ROLLOUT_STEPS_MAX=4
export ROLLOUT_CURRICULUM_EPOCHS=15
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

echo "Fine-tuning on real-fire corpus at $(date)"
python -m py_compile /home/abazan/wrfout_sandbox/train_unet.py || exit 1
python /home/abazan/wrfout_sandbox/train_unet.py |& tee /home/abazan/wrfout_sandbox/train_real_fire.log
echo "Done at $(date)"
