#!/bin/bash
#SBATCH --job-name=unet_smoke
#SBATCH --output=/home/abazan/wrfout_sandbox/train_smoke.out
#SBATCH --error=/home/abazan/wrfout_sandbox/train_smoke.err
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Train smoke model on real-fire corpus with FUEL_FRAC + PM2.5 proxy channels.
# in_c=10 and out_c=3 are inferred from shard shape; no OUT_C override needed.

export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_real_fire_smoke_tplus1/shards
export CKPT_PATH=/home/abazan/wrfout_sandbox/unet_smoke_last.pt
export BEST_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_smoke_best.pt
unset LOAD_CKPT_PATH
unset EVAL_ONLY

# Smoke auxiliary regression weight (0.1 = 10% of total loss)
export SMOKE_W=0.1

# Hyperparams — same as train_real_fire.slurm fine-tune settings
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

echo "Training smoke model at $(date)"
python -m py_compile /home/abazan/wrfout_sandbox/train_unet.py || exit 1
python /home/abazan/wrfout_sandbox/train_unet.py |& tee /home/abazan/wrfout_sandbox/train_smoke.log
echo "Done at $(date)"
