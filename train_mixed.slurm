#!/bin/bash
#SBATCH --job-name=unet_mixed
#SBATCH --output=/home/abazan/wrfout_sandbox/train_mixed.out
#SBATCH --error=/home/abazan/wrfout_sandbox/train_mixed.err
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

# Build a combined shard directory: real_fire (6794) + Evans Canyon (334) = ~7128 shards.
# Evans Canyon is included in training so the model sees real WA Cascade fire behavior.
# A 90/10 holdout is still applied; Evans Canyon shards will appear in both train and test
# splits at ~10% rate, allowing us to track Evans-specific improvement.

MIXED_DIR=/home/abazan/wrfout_sandbox/vit_dataset_mixed_tplus1/shards
mkdir -p "$MIXED_DIR"

# Symlink all real-fire shards
for f in /home/abazan/wrfout_sandbox/vit_dataset_real_fire_tplus1/shards/*.npz; do
    ln -sf "$f" "$MIXED_DIR/rf_$(basename $f)" 2>/dev/null || true
done

# Symlink all Evans Canyon shards with a distinct prefix so we can identify them
for f in /home/abazan/wrfout_sandbox/vit_dataset_evans_tplus1/shards/*.npz; do
    ln -sf "$f" "$MIXED_DIR/ev_$(basename $f)" 2>/dev/null || true
done

echo "Mixed shard count: $(ls $MIXED_DIR | wc -l)"

export SHARDS_DIR=$MIXED_DIR
export CKPT_PATH=/home/abazan/wrfout_sandbox/unet_mixed_last.pt
export BEST_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_mixed_best.pt
export OUT_C=3
unset LOAD_CKPT_PATH
unset EVAL_ONLY

export SMOKE_W=0.0   # no PM2.5 supervision in these shards (Y has 2 channels)

# Longer run: 3 cosine restarts every 50 epochs
export EPOCHS=150
export BATCH_SIZE=8
export NUM_WORKERS=4
export LR=3e-5
export MIN_LR=3e-6
export WARMUP_EPOCHS=3
export LR_RESTART_EPOCHS=50
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

echo "Training mixed model (real_fire + Evans Canyon) at $(date)"
python -m py_compile /home/abazan/wrfout_sandbox/train_unet.py || exit 1
python /home/abazan/wrfout_sandbox/train_unet.py |& tee /home/abazan/wrfout_sandbox/train_mixed.log
echo "Done at $(date)"
