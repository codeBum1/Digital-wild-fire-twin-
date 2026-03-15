# Digital Wildfire Twin

Autoregressive wildfire spread forecasting using a deep UNet trained on WRF-SFIRE (Weather Research and Forecasting with fire behavior) simulation output. Given atmospheric state and current fire perimeter, the model predicts future fire arrival masks across multi-step rollouts.

---

## Overview

Operational fire behavior models (FARSITE, FlamMap, Phoenix) are physically grounded but slow — minutes to hours per run — and sensitive to fuel map accuracy. This project trains a neural surrogate on WRF-SFIRE simulation data to produce near-instant spread forecasts that incorporate wind, humidity, temperature, and fire state as inputs.

The model runs a 6-step autoregressive rollout in under a second on a single GPU.

---

## Architecture

- **UNet** with 4 encoder/decoder levels (base → 8× base channels)
- **CBAM** attention gates on every skip connection
- **Squeeze-and-Excitation (SE)** blocks inside every ResBlock
- **Depthwise separable convolutions** — fewer parameters, faster inference
- **Deep supervision** — auxiliary loss at each decoder level
- **EMA** model averaging for stable evaluation

### Training
- Multi-GPU DDP via `torchrun` (tested on 4× NVIDIA A30 and 4× L40S)
- Mixed precision (bfloat16 / float16)
- **Lovász-hinge** surrogate for IoU + Dice + Focal + boundary perimeter loss
- **Curriculum rollout** — starts at 1-step prediction, ramps to 6-step with exponential teacher-forcing decay
- Cosine LR schedule with warm restarts
- **Joint flip/rotation augmentation** applied to inputs and targets (8 ops, 4× effective data)
- **Held-out test split** (20%, `SPLIT_SEED=42`) — evaluation never sees training samples

---

## Results

**Simulation dataset** (31 train / 8 test samples, 400 epochs, 4× L40S GPUs):

| Metric | Value |
|--------|-------|
| Fire IoU | **0.956** |
| Spread IoU | **0.802** |
| Dice | 0.977 |
| Mean Perimeter Distance | 0.22 grid cells |
| MAE | 0.037 |
| Composite score | **0.911** |

> Numbers above are on the held-out test set — samples the model never trained on.

**Real-fire transfer (Evans Canyon 2020, WRF-SFIRE d03, 15-min outputs):**

| Model | Fire IoU | Notes |
|-------|----------|-------|
| Sandbox (zero-shot) | 0.0715 | No real-fire training data |
| Fine-tuned on real-fire corpus | **0.2112** | +196% vs zero-shot |

> Evans Canyon is a real 2020 WA Cascades wildfire with 334 evaluation shards. The fine-tuned model was trained on 19 WRF-SFIRE runs across California, Oregon, and Washington fires.

---

## Inputs / Outputs

**Input channels (per timestep):**
- U, V wind components (destaggered to mass grid)
- Temperature, humidity
- Fire presence mask (current perimeter)
- Additional WRF-SFIRE atmospheric fields

**Output:** Binary fire arrival mask at t+1 (and recursively for multi-step rollout)

---

## Repo Structure

```
train_unet.py              # Main training script (UNet v2, DDP, curriculum rollout)
wrf_vit_dataset.py         # PyTorch Dataset — NPZ shards, train/test split, augmentation
build_vit_dataset.py       # Build NPZ shard datasets from WRF NetCDF/HDF5 files
build_pnw_dataset.py       # Multi-sim dataset builder for real WRF-SFIRE runs
rollout_unet_mask.py       # Autoregressive rollout evaluation
danger_rating.py           # Human danger rating system (post-processing layer)
smoke_viz.py               # Smoke / PM2.5 auxiliary channel visualisation
viz_unet_pred_gif.py       # Animated rollout GIFs
viz_predictions.py         # Side-by-side prediction visualizations
viz_gif_overlay.py         # Fire perimeter overlay on WRF domain
converters/h5_to_nc.py     # Convert HDF5 wrfout files to NetCDF

# SLURM job scripts
train_unet_gpu.slurm       # Original sandbox training (A30, single GPU)
train_real_fire.slurm      # Fine-tune on real-fire corpus
train_mixed.slurm          # Mixed model: real-fire + Evans Canyon (150 epochs)
train_smoke.slurm          # Smoke model with PM2.5 auxiliary loss
eval_evans.slurm           # Eval on Evans Canyon real fire
eval_evans_retrained.slurm # Eval retrained model on Evans Canyon
eval_evans_mixed.slurm     # Eval mixed model on Evans Canyon
eval_evans_smoke.slurm     # Eval smoke model on Evans Canyon smoke shards
build_real_fire.slurm      # Build NPZ shards from 19 real-fire runs
build_pnw_shards.slurm     # Build NPZ shards from 321 PNW runs
build_smoke_fire.slurm     # Rebuild real-fire corpus with fuel + PM2.5 channels
build_evans_smoke.slurm    # Rebuild Evans Canyon shards with fuel + PM2.5 channels
collect_pnw_usable.sh      # Create symlinks to 321 usable PNW WRF-SFIRE runs
```

---

## Quick Start

```bash
# Activate environment
source ~/venvs/wdt_ml/bin/activate

# Build dataset from WRF output files
python build_vit_dataset.py

# Train (single node, 4 GPUs)
export SHARDS_DIR=/path/to/shards
export EPOCHS=400
export TEST_FRAC=0.20
export SPLIT_SEED=42
torchrun --nproc_per_node=4 train_unet.py

# Submit to SLURM cluster
sbatch train_unet_l40s.slurm

# Run rollout evaluation
python rollout_unet_mask.py

# Generate prediction GIFs
python viz_unet_pred_gif.py
```

**Key env var overrides for `train_unet.py`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SHARDS_DIR` | — | Path to NPZ shards directory |
| `EPOCHS` | 200 | Training epochs |
| `BATCH_SIZE` | 8 | Per-GPU batch size |
| `TEST_FRAC` | 0.20 | Fraction of data held out for test |
| `SPLIT_SEED` | 0 | RNG seed for reproducible train/test split |
| `LOAD_CKPT_PATH` | — | Resume from checkpoint |
| `EVAL_ONLY` | 0 | Skip training, run eval only |
| `FIRE_THR` | 0.1 | Fire mask binarization threshold |

---

## Human Danger Rating

`danger_rating.py` is a post-processing module that overlays fire spread predictions with human impact layers to produce a per-cell danger score.

**Layers (weighted composite):**
| Layer | Source | Weight |
|-------|--------|--------|
| Population density | WRF NLCD land-use proxy (or external GPW/Census raster) | 50% |
| Road inaccessibility | LU proxy / OSM via osmnx / precomputed raster | 30% |
| Terrain escape risk | Slope from WRF `HGT` elevation field | 20% |

**Formula:**
```
danger[i,j] = fire_prob[i,j] × (w_pop·pop + w_road·road_inacc + w_terrain·terrain_risk)
```

**Usage:**
```bash
# Single step
python danger_rating.py \
    --fire_prob unet_rollout_step3.npy \
    --wrf_file  sandbox_wrfout.nc \
    --out_dir   danger_output/

# Multi-step animated GIF
python danger_rating.py \
    --fire_prob_dir rollout_output/ \
    --wrf_file sandbox_wrfout.nc \
    --make_gif

# With external population raster (GPW or Census block-group GeoTIFF)
python danger_rating.py --fire_prob step3.npy --pop_raster gpw_v4_pop_density.tif

# With live OSM road download
python danger_rating.py --fire_prob step3.npy --use_osmnx
```

**Outputs** (in `--out_dir`):
- `danger_score.npy` — raw (H, W) float32 score
- `danger_map.png` — 4-panel: fire prob / danger score / population / road access
- `danger_summary.json` — high-danger cell count, bbox, mean score in fire zone
- `danger_animation.gif` — multi-step evolution (with `--make_gif`)

## Roadmap

- **[done] Human danger rating system** — `danger_rating.py` integrated into eval pipeline; produces per-cell danger score + animated GIFs combining fire probability with population, road access, and terrain risk layers
- **[done] Smoke / PM2.5 infrastructure** — fuel channel + PM2.5 proxy target in dataset builder; `SMOKE_W` weighted Huber loss; auxiliary regression metrics in eval; `smoke_viz.py` for visualisation. Rebuild shards with `VIT_INCLUDE_FUEL_FRAC=1 VIT_INCLUDE_PM25_PROXY=1` to activate. Swap proxy target for real WRF-Chem PM2.5 when available.
- **[done] Real fire validation** — 19 WRF-SFIRE runs (Bootleg, Caldor, CRAM, Palisades, Smokehouse Creek, Evans Canyon + others) → fine-tuned model achieves **IoU=0.2112** on Evans Canyon (up from 0.0715 zero-shot, +196%)
- **[in progress] Mixed model + smoke model** — training on combined 7128-shard corpus with cosine restarts; smoke model with PM2.5 auxiliary channel
- **[in progress] Expanded training data** — 321 PNW WRF-SFIRE runs staged and ready; 4 new fire configs created (Dixie 2021, McKinney 2022, Holiday Farm 2020, Flat 2023)

---

## Data

Training data is WRF-SFIRE simulation output (not included in repo due to size). Raw files are NetCDF/HDF5 wrfout format. See `wrfout_files_with_fire.txt` for the file list used and `build_vit_dataset.py` for the preprocessing pipeline.

---

## Dependencies

- PyTorch ≥ 2.0 (DDP, `torchrun`)
- xarray, netCDF4
- numpy, scipy
- matplotlib
- wandb (optional — set `WANDB_PROJECT` env var to enable)
