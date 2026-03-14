# CLAUDE.md – WRF Wildfire UNet Project

## Project Overview
Wildfire spread forecasting using a UNet model trained on WRF-SFIRE (Weather Research and Forecasting model with fire behavior) simulation output. The model predicts binary fire arrival masks from atmospheric and fire state inputs.

## Key Files
| File | Purpose |
|------|---------|
| `train_unet.py` | Main training script (UNet v2 with CBAM, SE blocks, Lovász loss, DDP) |
| `wrf_vit_dataset.py` | PyTorch Dataset that loads NPZ shards (X: inputs, Y: fire masks) |
| `build_vit_dataset.py` | Builds NPZ shard datasets from WRF NetCDF/HDF5 wrfout files |
| `build_pnw_dataset.py` | Multi-sim dataset builder for real WRF-SFIRE runs (supports `VIT_DOMAIN`, `VIT_FIRE_VARS` env vars) |
| `rollout_unet_mask.py` | Autoregressive rollout evaluation |
| `viz_*.py` | Visualization scripts (GIFs, overlays, predictions) |
| `converters/h5_to_nc.py` | Converts HDF5 wrfout files to NetCDF |

## Data
- Raw WRF output: `sandbox_wrfout.nc`, `sandbox_wrfout.h5`, `sandbox_wrfout_mlsubset.h5`
- File lists: `wrfout_files.txt`, `wrfout_files_with_fire.txt`, `wrfout_files_abs.txt`
- Shard datasets (NPZ): `vit_dataset_fireonly_tplus*/shards/`
  - Key datasets: `tplus1_fireX_maxpool_paired`, `tplus3_fireX_maxpool_paired`, `tplus6_fireX_maxpool_paired`
- **Evans Canyon fire** (2020-08-31 → 2020-09-03, real WRF-SFIRE, d03 15-min outputs):
  - Raw: `/scratch/wdt/sfire/new_wrfxpy/wrfxpy/wksp/testing-NAM218_evans_canyon/wrf/`
  - Shards: `vit_dataset_evans_tplus1/shards/` — **334 shards**, X=(9,99,99), Y=(2,99,99), 0 errors
  - Built with: `PNW_RAW_DIR=.../testing-NAM218_evans_canyon VIT_DOMAIN=d03 VIT_FIRE_VARS="FIRE_AREA,ROS" PNW_OUT_DIR=vit_dataset_evans_tplus1 python build_pnw_dataset.py`
  - Note: `FLAME_LENGTH` not present in these outputs — use `VIT_FIRE_VARS=FIRE_AREA,ROS`
- **PNW sims** (`pnw_sfire_raw/`, 111 sims 2015-2023): **all zero fire** — WRF-SFIRE ignition config issue in the pipeline, not a data builder issue

## Environment & Compute
- Python venv: `~/venvs/wdt_ml/bin/activate`
- Cluster: SLURM with GPU partition (`--partition=normal`, `--gres=gpu:1`)
- SLURM scripts: `train_unet_gpu.slurm`, `train_unet_acc.slurm`, `train_unet_vis.slurm`
- Run with: `sbatch train_unet_gpu.slurm` or `torchrun --nproc_per_node=N train_unet.py`

## Model Checkpoints
- `unet_ms_gpu_best.pt` / `unet_ms_gpu_last.pt` – latest full-training run
- `unet_ms_gpu_retrain_best.pt` / `unet_ms_gpu_retrain_last.pt` – retrain run
- `unet_acc_best.pt` / `unet_acc_last.pt` – accumulated/longer run

## Key Hyperparameters (env vars for train_unet.py)
- `SHARDS_DIR` – path to NPZ shards directory
- `CKPT_PATH`, `BEST_CKPT_PATH`, `LOAD_CKPT_PATH` – checkpoint paths
- `EPOCHS`, `BATCH_SIZE`, `LR`, `MIN_LR`, `WEIGHT_DECAY`
- `FIRE_THR` – fire mask binarization threshold (default 0.1)
- `AUTO_POS_WEIGHT`, `DICE_W`, `FOCAL_W` – loss weights
- `TRAIN_ROLLOUT_STEPS_MAX`, `ROLLOUT_CURRICULUM_EPOCHS` – curriculum settings
- `KEEP_BURNING`, `SELFCOND_BLEND`, `USE_FUTURE_EXOGENOUS`, `USE_EMA`
- `ROLL_OUT_STEPS`, `VIS_THR` – evaluation rollout settings
- `EVAL_ONLY=1` – skip training, run evaluation only

## Outputs
- Training logs: `train_unet_gpu.log`, `unet_gpu.out/err`
- Metrics: `metrics_log.csv` (written by training script)
- Visualizations: `unet_eval_viz/`, `unet_rollout_gifs/`, `unet_paper_viz/`, etc.
- WandB: enabled if `WANDB_PROJECT` env var is set

## Future Roadmap

### 1. Human Danger Rating System (~1-2 weeks) — do first
Post-processing layer on top of existing fire perimeter output. No model changes needed.
- Combine fire perimeter predictions with:
  - **Population density** (US Census)
  - **Road networks** (OpenStreetMap) — evacuation route accessibility
  - **Terrain escape routes** (USGS 3DEP elevation data)
- Output: per-cell human danger score overlaid on fire spread forecast
- Reuses `unet_l40s_best.pt` output directly

### 2. Smoke / Air Quality Prediction — infrastructure done; ground truth pending
- **[done]** `FUEL_FRAC` input channel: set `VIT_INCLUDE_FUEL_FRAC=1` when running `build_vit_dataset.py`
- **[done]** PM2.5 proxy Y channel: set `VIT_INCLUDE_PM25_PROXY=1`; physics surrogate from `FIRE_AREA × FUEL_FRAC_BURNT` + Gaussian wind dispersion
- **[done]** `SMOKE_W` env var in `train_unet.py` scales Huber regression loss on auxiliary channels
- **[done]** `smoke_viz.py` visualises fire + ROS + Flame Length + PM2.5 channels side-by-side
- **[next]** Rebuild shards with fuel+PM25 flags and retrain with `SMOKE_W=0.1`
- **[future]** Replace PM2.5 proxy with real WRF-Chem or HYSPLIT output when available

## Common Workflows
```bash
# Activate environment
source ~/venvs/wdt_ml/bin/activate

# Submit GPU training job
sbatch train_unet_gpu.slurm

# Build new dataset
python build_vit_dataset.py

# Run rollout evaluation
python rollout_unet_mask.py

# Visualize predictions as GIF
python viz_unet_pred_gif.py
```
