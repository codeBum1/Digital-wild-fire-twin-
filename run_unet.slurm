#!/bin/bash
#SBATCH --job-name=unet_eval
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/abazan/wrfout_sandbox/unet_eval.out
#SBATCH --error=/home/abazan/wrfout_sandbox/unet_eval.err

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards
export LOAD_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_ms_gpu_best.pt
export EVAL_ONLY=1
export ROLL_OUT_STEPS=20

python -m py_compile /home/abazan/wrfout_sandbox/train_unet.py || exit 1
python /home/abazan/wrfout_sandbox/train_unet.py |& tee /home/abazan/wrfout_sandbox/train_unet_eval.log
