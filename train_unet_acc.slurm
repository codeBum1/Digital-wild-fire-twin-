#!/bin/bash
#SBATCH --job-name=unet_acc
#SBATCH --output=/home/abazan/wrfout_sandbox/unet_acc.out
#SBATCH --error=/home/abazan/wrfout_sandbox/unet_acc.err
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards

export CKPT_PATH=/home/abazan/wrfout_sandbox/unet_acc_last.pt
export BEST_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_acc_best.pt

unset LOAD_CKPT_PATH
unset EVAL_ONLY
unset VIS_THR

export RUN_TAG=acc_tuned_run1

python -m py_compile /home/abazan/wrfout_sandbox/train_unet.py || exit 1
python /home/abazan/wrfout_sandbox/train_unet.py |& tee /home/abazan/wrfout_sandbox/acc_tuned_run1.log
