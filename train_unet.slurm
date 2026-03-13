#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --output=unet_train.out
#SBATCH --error=unet_train.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

source ~/venvs/wdt_ml/bin/activate

python -m py_compile ~/wrfout_sandbox/train_unet.py
python ~/wrfout_sandbox/train_unet.py
