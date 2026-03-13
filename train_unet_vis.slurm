#!/bin/bash
#SBATCH --job-name=unet_vis
#SBATCH --output=/home/abazan/wrfout_sandbox/unet_vis.out
#SBATCH --error=/home/abazan/wrfout_sandbox/unet_vis.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/venvs/wdt_ml/bin/activate
cd ~/wrfout_sandbox

export SHARDS_DIR=/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards
export LOAD_CKPT_PATH=/home/abazan/wrfout_sandbox/unet_v2_best.pt
export EVAL_ONLY=1
export VIS_THR=0.3

python -m py_compile /home/abazan/wrfout_sandbox/train_unet.py || exit 1
python /home/abazan/wrfout_sandbox/train_unet.py |& tee /home/abazan/wrfout_sandbox/train_unet_visonly.log
