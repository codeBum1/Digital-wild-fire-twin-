"""
train_unet_v2.py  –  Wildfire spread forecasting UNet  (cluster-grade)

Improvements over train_unet_multistep_gpu.py
──────────────────────────────────────────────
Architecture
  • Deeper UNet: 4 encoder/decoder levels (base → 8×base)
  • CBAM attention gates on every skip connection
  • Squeeze-and-Excitation (SE) blocks inside every ResBlock
  • Separable depth-wise convolutions (faster + fewer params)
  • Deep supervision: auxiliary loss at each decoder level
  • Configurable base width  (default 64; try 96/128 on big GPUs)

Training
  • torch.distributed / DDP – single-node multi-GPU OR multi-node
  • Mixed precision (bfloat16 when available, else float16)
  • Gradient accumulation (effective batch = BATCH_SIZE × ACCUM_STEPS)
  • Warmup + Cosine-with-restarts LR schedule  (better than plain cosine)
  • Stochastic depth / drop-path regularisation
  • Label smoothing on BCE  (reduces overconfident predictions)
  • Boundary / perimeter loss term  (directly penalises edge mis-placement)
  • Lovász-hinge surrogate for IoU  (maximises the metric you care about)
  • Curriculum rollout with exponential teacher-forcing decay
  • Persistent DataLoader workers with pinned memory
  • Gradient clipping + NaN guard
  • EMA with warmup

Evaluation / checkpointing
  • Saves  best.pt  (by mean IoU) and  last.pt  every epoch
  • WandB logging  (set WANDB_PROJECT env var; skipped if wandb absent)
  • CSV metric log written locally  (metrics_log.csv)
  • Deterministic eval on rank-0 only

Cluster usage
  # Single node, 4 GPUs
  torchrun --nproc_per_node=4 train_unet_v2.py

  # Multi-node (SLURM example)
  srun torchrun --nnodes=$SLURM_NNODES \\
               --nproc_per_node=$SLURM_GPUS_ON_NODE \\
               --rdzv_id=$SLURM_JOB_ID \\
               --rdzv_backend=c10d \\
               --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
               train_unet_v2.py

All hyperparameters are overridable with environment variables.
"""

from wrf_vit_dataset import WrfVitShardDataset
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import csv
import copy
import math
import os
import random
import time
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────
#  Optional WandB
# ─────────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  Config  (all overridable via env vars)
# ─────────────────────────────────────────────────────────────
def _env(key, default, cast=str):
    return cast(os.environ.get(key, str(default)))


SHARDS_DIR = _env(
    "SHARDS_DIR",    "/home/abazan/wrfout_sandbox/vit_dataset_fireonly_tplus3_fireX_maxpool_paired/shards")
CKPT_PATH = _env(
    "CKPT_PATH",     "/home/abazan/wrfout_sandbox/unet_v2_last.pt")
BEST_CKPT_PATH = _env(
    "BEST_CKPT_PATH", "/home/abazan/wrfout_sandbox/unet_v2_best.pt")
OUT_ROOT = _env("OUT_ROOT",      "/home/abazan/wrfout_sandbox/unet_v2_outputs")

EPOCHS = _env("EPOCHS",                "200",  int)
BATCH_SIZE = _env("BATCH_SIZE",            "8",    int)   # per-GPU
# effective=BATCH_SIZE*ACCUM_STEPS
ACCUM_STEPS = _env("ACCUM_STEPS",           "2",    int)
NUM_WORKERS = _env("NUM_WORKERS",           "4",    int)
LR = _env("LR",                    "3e-4", float)
MIN_LR = _env("MIN_LR",                "5e-6", float)
WARMUP_EPOCHS = _env("WARMUP_EPOCHS",         "5",    int)
LR_RESTART_EPOCHS = _env("LR_RESTART_EPOCHS",     "50",
                         int)   # cosine restart period
WEIGHT_DECAY = _env("WEIGHT_DECAY",          "1e-4", float)
GRAD_CLIP_NORM = _env("GRAD_CLIP_NORM",        "1.0",  float)

BASE_CHANNELS = _env("BASE_CHANNELS",         "64",   int)
DROPOUT = _env("DROPOUT",               "0.10", float)
DROP_PATH_RATE = _env("DROP_PATH_RATE",        "0.10", float)
DEEP_SUPERVISION = _env("DEEP_SUPERVISION",      "1",    int)
DS_WEIGHT = _env("DS_WEIGHT",             "0.3",
                 float)  # deep-supervision weight

FIRE_THR = _env("FIRE_THR",              "0.1",  float)
AUTO_POS_WEIGHT = _env("AUTO_POS_WEIGHT",       "1",    int)
POS_WEIGHT = _env("POS_WEIGHT",            "3.5",  float)
DICE_W = _env("DICE_W",                "0.30", float)
FOCAL_W = _env("FOCAL_W",               "0.20", float)
LOVASZ_W = _env("LOVASZ_W",              "0.20", float)
BOUNDARY_W = _env("BOUNDARY_W",            "0.15", float)
LABEL_SMOOTH = _env("LABEL_SMOOTH",          "0.02", float)

TRAIN_ROLLOUT_STEPS_MAX = _env("TRAIN_ROLLOUT_STEPS_MAX", "5",   int)
ROLLOUT_CURRICULUM_EPOCHS = _env("ROLLOUT_CURRICULUM_EPOCHS", "20", int)
STEP_LOSS_DECAY = _env("STEP_LOSS_DECAY",       "0.95", float)
TEACHER_FORCING_START = _env("TEACHER_FORCING_START", "1.0",  float)
TEACHER_FORCING_END = _env("TEACHER_FORCING_END",   "0.05", float)
KEEP_BURNING = _env("KEEP_BURNING",          "1",    int)
SELFCOND_BLEND = _env("SELFCOND_BLEND",        "0.90", float)
USE_FUTURE_EXOGENOUS = _env("USE_FUTURE_EXOGENOUS",  "1",    int)

EMA_DECAY = _env("EMA_DECAY",             "0.9990", float)
USE_EMA = _env("USE_EMA",               "1",    int)
EMA_WARMUP_EPOCHS = _env("EMA_WARMUP_EPOCHS",     "5",   int)

EVAL_EVERY = _env("EVAL_EVERY",            "5",    int)
ROLL_OUT_STEPS = _env("ROLL_OUT_STEPS",        "6",    int)
ROLL_OUT_N_SAMPLES = _env("ROLL_OUT_N_SAMPLES",    "4",    int)
PB_THR_ENV = _env("PB_THR",                "")
VIS_THR_ENV = _env("VIS_THR",               "")
EVAL_ONLY = _env("EVAL_ONLY",             "0",    int)
LOAD_CKPT_PATH = _env("LOAD_CKPT_PATH",       "")
EVAL_NORMALIZE_PER_SAMPLE = _env("EVAL_NORMALIZE_PER_SAMPLE", "1", int)
RUN_TAG_ENV = _env("RUN_TAG",             "")
SEED = _env("SEED",                  "42",   int)
WANDB_PROJECT = _env("WANDB_PROJECT",         "")
TEST_FRAC = _env("TEST_FRAC",             "0.20", float)  # fraction held out for test (used when KFOLD_K<=1)
SPLIT_SEED = _env("SPLIT_SEED",            "0",    int)   # separate seed for reproducible split
KFOLD_K    = _env("KFOLD_K",              "0",    int)   # >1 enables k-fold CV
KFOLD_FOLD = _env("KFOLD_FOLD",           "0",    int)   # which fold is the test fold (0-indexed)

# Output sub-dirs  (created at runtime on rank-0)
VIZ_DIR = os.path.join(OUT_ROOT, "eval_viz")
GIF_DIR = os.path.join(OUT_ROOT, "eval_gifs")
PAPER_VIZ_DIR = os.path.join(OUT_ROOT, "paper_viz")
PROB_VIZ_DIR = os.path.join(OUT_ROOT, "prob_viz")
ROLL_OUT_GIF_DIR = os.path.join(OUT_ROOT, "rollout_gifs")
ROLL_OUT_PROB_GIF_DIR = os.path.join(OUT_ROOT, "rollout_prob_gifs")
CINEMATIC_DIR = os.path.join(OUT_ROOT, "cinematic_rollouts")
ROLL_OUT_PLOT_DIR = os.path.join(OUT_ROOT, "rollout_plots")
SPREAD_PLOT_DIR = os.path.join(OUT_ROOT, "spread_plots")
ARRIVAL_DIR = os.path.join(OUT_ROOT, "arrival_viz")
PERIM_GIF_DIR = os.path.join(OUT_ROOT, "perim_overlay_gifs")

ALL_OUTPUT_DIRS = [
    VIZ_DIR, GIF_DIR, PAPER_VIZ_DIR, PROB_VIZ_DIR,
    ROLL_OUT_GIF_DIR, ROLL_OUT_PROB_GIF_DIR, CINEMATIC_DIR,
    ROLL_OUT_PLOT_DIR, SPREAD_PLOT_DIR, ARRIVAL_DIR, PERIM_GIF_DIR,
]


def _default_run_tag():
    now = datetime.now()
    hour = now.strftime("%I").lstrip("0") or "0"
    return f"{now.month}_{now.day}_{now.year}_{hour}_{now.strftime('%M')}{now.strftime('%p')}"

RUN_TAG = RUN_TAG_ENV.strip() if str(RUN_TAG_ENV).strip() else _default_run_tag()

def with_run_tag(path):
    root, ext = os.path.splitext(path)
    return f"{root}_{RUN_TAG}{ext}"


# ─────────────────────────────────────────────────────────────
#  DDP helpers
# ─────────────────────────────────────────────────────────────


def setup_ddp():
    """Initialise DDP if torchrun launched us; otherwise single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def is_main(rank): return rank == 0


def all_reduce_mean(tensor, world_size):
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


# ─────────────────────────────────────────────────────────────
#  Reproducibility / performance
# ─────────────────────────────────────────────────────────────

def set_seed(seed, rank=0):
    seed = seed + rank  # different seed per GPU for data augmentation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


torch.backends.cudnn.benchmark = True
if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends.cudnn, "allow_tf32"):
    torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────────────────────
#  Losses
# ─────────────────────────────────────────────────────────────

def smooth_bce(logits, target, pos_weight, smooth=0.05):
    """BCE with label smoothing."""
    target_s = target * (1.0 - smooth) + 0.5 * smooth
    return F.binary_cross_entropy_with_logits(logits, target_s, pos_weight=pos_weight)


def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * target).sum(dim=(1, 2, 3))
    den = (probs + target).sum(dim=(1, 2, 3)) + eps
    return (1.0 - num / den).mean()


def focal_bce(logits, target, alpha=0.75, gamma=2.5):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t) ** gamma * bce).mean()


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Lovász-Hinge surrogate for the IoU loss.
    Reference: Berman et al. (2018).
    Works on (B,1,H,W) logits + binary labels.
    """
    logits = logits.squeeze(1)   # (B,H,W)
    labels = labels.squeeze(1).float()

    def _lovasz_hinge_flat(log, lab):
        n = lab.numel()
        if n == 0:
            return log.sum() * 0.0
        errors = (1.0 - lab * 2.0) * log          # errors in (-∞,+∞)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = lab[perm]
        grad = _lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)

    def _lovasz_grad(gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1.0 - gt_sorted).float().cumsum(0)
        iou = 1.0 - intersection / union
        p = len(gt_sorted)
        if p > 1:
            iou[1:p] = iou[1:p] - iou[0:p-1]
        return iou

    if per_image:
        losses = [_lovasz_hinge_flat(log.view(-1), lab.view(-1))
                  for log, lab in zip(logits, labels)]
        return torch.stack(losses).mean()
    else:
        return _lovasz_hinge_flat(logits.view(-1), labels.view(-1))


def boundary_loss(logits, target, dilation=2):
    """
    Penalises errors near the true fire perimeter using a distance-weighted mask.
    Boundary region determined by max-pooling the edge of the true mask.
    """
    probs = torch.sigmoid(logits)
    kernel = torch.ones(1, 1, 2 * dilation + 1, 2 * dilation + 1,
                        device=logits.device) / (2 * dilation + 1) ** 2
    dilated = F.conv2d(target, kernel, padding=dilation)
    boundary_mask = ((dilated > 0) & (dilated < 1)).float()
    if boundary_mask.sum() < 1:
        return logits.new_tensor(0.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * boundary_mask).sum() / (boundary_mask.sum() + 1e-6)


def combined_fire_loss(logits, fire_mask, pw):
    """Weighted sum of all fire loss terms."""
    loss = smooth_bce(logits, fire_mask, pw, smooth=LABEL_SMOOTH)
    loss += DICE_W * dice_loss(logits, fire_mask)
    loss += FOCAL_W * focal_bce(logits, fire_mask)
    loss += LOVASZ_W * lovasz_hinge(logits, fire_mask)
    loss += BOUNDARY_W * boundary_loss(logits, fire_mask)
    return loss


# ─────────────────────────────────────────────────────────────
#  Normalisation helper
# ─────────────────────────────────────────────────────────────

def normalize_batch(X):
    mean = X.mean(dim=(0, 2, 3), keepdim=True)
    std = X.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    return (X - mean) / std


# ─────────────────────────────────────────────────────────────
#  Architecture blocks
# ─────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth – drop entire samples from a residual branch."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype,
                            device=x.device).bernoulli_(keep) / keep
        return x * noise


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, r),
            nn.SiLU(),
            nn.Linear(r, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        # channel attention
        r = max(1, channels // reduction)
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_max = nn.AdaptiveMaxPool2d(1)
        self.ca_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, r),
            nn.SiLU(),
            nn.Linear(r, channels),
        )
        # spatial attention
        self.sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # channel
        avg = self.ca_mlp(self.ca_avg(x))
        mx = self.ca_mlp(self.ca_max(x))
        ca = torch.sigmoid(avg + mx).view(x.shape[0], x.shape[1], 1, 1)
        x = x * ca
        # spatial
        sa = torch.sigmoid(self.sa(torch.cat([x.mean(1, keepdim=True),
                                              x.max(1, keepdim=True)[0]], dim=1)))
        return x * sa


class DSConv(nn.Module):
    """Depthwise-separable convolution (cheaper than full conv2d)."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c,  3, stride=stride,
                            padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class ResBlock(nn.Module):
    """
    Residual block with:
      - depthwise-separable convolutions
      - GroupNorm
      - SE channel attention
      - DropPath stochastic depth
    """

    def __init__(self, in_c, out_c, dropout=0.0, drop_path=0.0):
        super().__init__()
        g = min(8, out_c)
        self.conv1 = DSConv(in_c, out_c)
        self.norm1 = nn.GroupNorm(g, out_c)
        self.conv2 = DSConv(out_c, out_c)
        self.norm2 = nn.GroupNorm(g, out_c)
        self.act = nn.SiLU()
        self.se = SEBlock(out_c)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.dp = DropPath(drop_path)
        self.skip = nn.Conv2d(
            in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        h = self.se(h)
        return self.act(self.dp(h) + residual)


class AttentionGate(nn.Module):
    """
    Additive attention gate for skip connections.
    Suppresses irrelevant activations in the skip path.
    """

    def __init__(self, g_c, x_c, inter_c):
        super().__init__()
        self.Wg = nn.Conv2d(g_c, inter_c, 1, bias=False)
        self.Wx = nn.Conv2d(x_c, inter_c, 1, bias=False)
        self.psi = nn.Conv2d(inter_c, 1, 1, bias=True)
        nn.init.zeros_(self.psi.bias)

    def forward(self, g, x):
        # g: gating signal (from decoder), x: skip from encoder
        g_up = F.interpolate(
            self.Wg(g), size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.silu(g_up + self.Wx(x))))
        return x * alpha


class UNet(nn.Module):
    """
    4-level UNet with:
      - ResBlock encoder / decoder
      - Attention gates on every skip
      - CBAM at bottleneck
      - Deep supervision heads at each decoder level
      - Configurable width
    """

    def __init__(self, in_c, out_c, base=64, dropout=0.10, drop_path_rate=0.10):
        super().__init__()
        b = base
        # ── Encoder ──────────────────────────────────────────
        dp = [drop_path_rate * i / 7 for i in range(8)]   # stagger drop path
        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResBlock(in_c, b,       dropout=0.0,     drop_path=dp[0])
        self.enc2 = ResBlock(b,    b*2,     dropout=0.0,     drop_path=dp[1])
        self.enc3 = ResBlock(b*2,  b*4,     dropout=dropout, drop_path=dp[2])
        self.enc4 = ResBlock(b*4,  b*8,     dropout=dropout, drop_path=dp[3])

        # ── Bottleneck ────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ResBlock(b*8, b*8, dropout=dropout, drop_path=dp[4]),
            CBAM(b*8),
        )

        # ── Decoder ──────────────────────────────────────────
        self.attn4 = AttentionGate(b*8, b*8, b*4)
        self.up4 = nn.Conv2d(b*8, b*4, 1, bias=False)
        # concat([up4(bn)=b*4, attn4(e4)=b*8]) -> b*12
        self.dec4 = ResBlock(b*12, b*4, dropout=dropout, drop_path=dp[5])

        self.attn3 = AttentionGate(b*4, b*4, b*2)
        self.up3 = nn.Conv2d(b*4, b*2, 1, bias=False)
        # concat([up3(d4)=b*2, attn3(e3)=b*4]) -> b*6
        self.dec3 = ResBlock(b*6, b*2, dropout=dropout, drop_path=dp[6])

        self.attn2 = AttentionGate(b*2, b*2, b)
        self.up2 = nn.Conv2d(b*2, b, 1, bias=False)
        # concat([up2(d3)=b, attn2(e2)=b*2]) -> b*3
        self.dec2 = ResBlock(b*3, b, dropout=0.0, drop_path=dp[7])

        self.attn1 = AttentionGate(b, b, b//2)
        self.up1 = nn.Conv2d(b, b, 1, bias=False)
        self.dec1 = ResBlock(b*2, b, dropout=0.0, drop_path=0.0)

        # ── Output + deep supervision heads ───────────────────
        self.out = nn.Conv2d(b, out_c, 1)
        self.ds4_head = nn.Conv2d(b*4, out_c, 1)
        self.ds3_head = nn.Conv2d(b*2, out_c, 1)
        self.ds2_head = nn.Conv2d(b,   out_c, 1)

    def forward(self, x, return_ds=False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        # Decoder level 4
        u4 = F.interpolate(
            self.up4(bn),  size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([u4, self.attn4(bn, e4)], dim=1))

        # Decoder level 3
        u3 = F.interpolate(
            self.up3(d4),  size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, self.attn3(d4, e3)], dim=1))

        # Decoder level 2
        u2 = F.interpolate(
            self.up2(d3),  size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, self.attn2(d3, e2)], dim=1))

        # Decoder level 1
        u1 = F.interpolate(
            self.up1(d2),  size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, self.attn1(d2, e1)], dim=1))

        out = self.out(d1)

        if return_ds:
            # Upsample deep supervision outputs to match output size
            h, w = out.shape[-2:]
            ds4 = F.interpolate(self.ds4_head(d4), size=(
                h, w), mode="bilinear", align_corners=False)
            ds3 = F.interpolate(self.ds3_head(d3), size=(
                h, w), mode="bilinear", align_corners=False)
            ds2 = F.interpolate(self.ds2_head(d2), size=(
                h, w), mode="bilinear", align_corners=False)
            return out, [ds4, ds3, ds2]
        return out


# ─────────────────────────────────────────────────────────────
#  EMA with warmup
# ─────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9975, warmup_epochs=10):
        self.decay = decay
        self.warmup_epochs = warmup_epochs
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    def _current_decay(self, epoch):
        # linear ramp from 0 → decay during warmup
        if self.warmup_epochs <= 0:
            return self.decay
        return self.decay * min(1.0, epoch / max(1, self.warmup_epochs))

    def update(self, model, epoch=0):
        d = self._current_decay(epoch)
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


# ─────────────────────────────────────────────────────────────
#  LR schedule: linear warmup + cosine-with-restarts
# ─────────────────────────────────────────────────────────────

def build_scheduler(optimizer, epochs, warmup=5, T_0=50, eta_min=5e-6):
    """
    Linear warmup for `warmup` epochs, then cosine-annealing-with-warm-restarts.
    """
    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(warmup + 1)
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(T_0, 1), T_mult=1, eta_min=eta_min
    )
    return warmup_sched, cosine_sched


# ─────────────────────────────────────────────────────────────
#  Mixed-precision autocast helper
# ─────────────────────────────────────────────────────────────

def get_autocast_dtype(device):
    if device.type != "cuda":
        return None
    # Prefer bfloat16 (A100/H100), fall back to float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@contextmanager
def maybe_autocast(device, dtype):
    if dtype is None:
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield


# ─────────────────────────────────────────────────────────────
#  Dataset helpers
# ─────────────────────────────────────────────────────────────

def estimate_pos_weight(ds, fire_thr=0.1, max_items=128):
    n = min(len(ds), max_items)
    fracs = []
    for i in range(n):
        _, Y = ds[i]
        fracs.append(float((Y[0].numpy() > fire_thr).mean()))
    pos_frac = max(float(np.mean(fracs)), 1e-4)
    pw = (1.0 - pos_frac) / pos_frac
    return float(np.clip(pw, 1.0, 25.0)), pos_frac


def stack_samples(ds, idxs, which="X"):
    out = []
    for i in idxs:
        X, Y = ds[i]
        out.append(X if which == "X" else Y)
    return torch.stack(out, dim=0)


# ─────────────────────────────────────────────────────────────
#  Teacher-forcing / rollout helpers
# ─────────────────────────────────────────────────────────────

def teacher_forcing_ratio(epoch):
    alpha = epoch / max(EPOCHS - 1, 1)
    return TEACHER_FORCING_START + alpha * (TEACHER_FORCING_END - TEACHER_FORCING_START)


def effective_rollout_steps(epoch, ds_len):
    max_s = max(1, min(TRAIN_ROLLOUT_STEPS_MAX, ds_len))
    return int(min(max_s, 1 + epoch // max(1, ROLLOUT_CURRICULUM_EPOCHS)))


def build_next_state(base_state, pred_fire_prob, true_fire_mask, tf_ratio):
    next_fire = tf_ratio * true_fire_mask + (1.0 - tf_ratio) * pred_fire_prob
    cur_fire = base_state[:, -1:, :, :]
    if KEEP_BURNING:
        next_fire = torch.maximum(next_fire, cur_fire)
    ns = base_state.clone()
    ns[:, -1:, :, :] = SELFCOND_BLEND * \
        next_fire + (1.0 - SELFCOND_BLEND) * cur_fire
    return ns


# ─────────────────────────────────────────────────────────────
#  Train one epoch
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, ds, device, optimizer, scaler, pw_value,
                    epoch, ema=None, rank=0, world_size=1, sampler=None):
    model.train()
    eff_steps = effective_rollout_steps(epoch, len(ds))
    tf_ratio = teacher_forcing_ratio(epoch)
    ac_dtype = get_autocast_dtype(device)

    valid_max = len(ds) - eff_steps
    start_idxs = list(range(valid_max + 1))
    if sampler is not None:
        # DDP: each rank gets a different slice
        random.shuffle(start_idxs)
        start_idxs = start_idxs[rank::world_size]
    else:
        random.shuffle(start_idxs)

    pw = torch.tensor(pw_value, device=device)
    epoch_loss = epoch_fire = epoch_other = 0.0
    nbatches = 0

    optimizer.zero_grad(set_to_none=True)

    for b0 in range(0, len(start_idxs), BATCH_SIZE):
        idxs = start_idxs[b0: b0 + BATCH_SIZE]
        state = stack_samples(ds, idxs, which="X").to(
            device, non_blocking=True)

        acc_loss = acc_fire = acc_other = 0.0
        is_accumulating = (b0 // BATCH_SIZE + 1) % ACCUM_STEPS != 0

        with maybe_autocast(device, ac_dtype):
            for step in range(eff_steps):
                Y = stack_samples(
                    ds, [i + step for i in idxs], which="Y").to(device, non_blocking=True)
                Xn = normalize_batch(state)

                if DEEP_SUPERVISION and model.training:
                    pred, ds_preds = model(Xn, return_ds=True)
                else:
                    pred = model(Xn)
                    ds_preds = []

                fire_logits = pred[:, 0:1, :, :]
                fire_mask = (Y[:, 0:1, :, :] > FIRE_THR).float()
                pred_other = pred[:, 1:, :, :]

                loss_fire = combined_fire_loss(fire_logits, fire_mask, pw)
                loss_other = (F.mse_loss(pred_other, Y[:, 1:, :, :])
                              if pred_other.numel() > 0
                              else fire_logits.new_tensor(0.0))

                # Deep supervision
                loss_ds = fire_logits.new_tensor(0.0)
                for ds_l in ds_preds:
                    loss_ds = loss_ds + DS_WEIGHT * \
                        combined_fire_loss(ds_l[:, 0:1], fire_mask, pw)

                sw = STEP_LOSS_DECAY ** step
                step_loss = sw * (loss_fire + loss_other + loss_ds)
                acc_loss = acc_loss + step_loss
                acc_fire = acc_fire + sw * loss_fire.detach()
                acc_other = acc_other + sw * loss_other.detach()

                if epoch == 0 and nbatches == 0 and step == 0 and rank == 0:
                    print(
                        f"[debug] pred {tuple(pred.shape)} Y {tuple(Y.shape)}")
                    print(
                        f"[debug] tf={tf_ratio:.3f} rollout_steps={eff_steps}")

                if step < eff_steps - 1:
                    if USE_FUTURE_EXOGENOUS:
                        next_base = stack_samples(
                            ds, [i + step + 1 for i in idxs], which="X").to(device, non_blocking=True)
                    else:
                        next_base = state.clone()
                    pfire = torch.sigmoid(fire_logits).detach()
                    if KEEP_BURNING:
                        pfire = torch.maximum(pfire, state[:, -1:].detach())
                    state = build_next_state(
                        next_base, pfire, fire_mask.detach(), tf_ratio)

        scaler.scale(acc_loss / ACCUM_STEPS).backward()

        if not is_accumulating or b0 + BATCH_SIZE >= len(start_idxs):
            scaler.unscale_(optimizer)
            # NaN/Inf grad guard + clipping (older PyTorch builds do not have get_total_norm)
            params = [p for p in model.parameters() if p.grad is not None]
            has_bad_grad = any(not torch.isfinite(p.grad).all() for p in params)
            if not has_bad_grad:
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
                scaler.step(optimizer)
            else:
                if rank == 0:
                    print(
                        f"[warn] NaN/Inf gradient at epoch {epoch}, batch {b0}. Skipping update.")
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model.module if world_size > 1 else model, epoch)

        epoch_loss += float(acc_loss.detach().item())
        epoch_fire += float(acc_fire.item()
                            if torch.is_tensor(acc_fire) else acc_fire)
        epoch_other += float(acc_other.item()
                             if torch.is_tensor(acc_other) else acc_other)
        nbatches += 1

    return {
        "loss":                   epoch_loss / max(nbatches, 1),
        "fire_loss":              epoch_fire / max(nbatches, 1),
        "other_loss":             epoch_other / max(nbatches, 1),
        "effective_rollout_steps": eff_steps,
        "teacher_forcing_ratio":   tf_ratio,
    }


# ─────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────

def predict_single(model, X, device):
    Xb = X.unsqueeze(0).to(device)
    Xn = normalize_batch(Xb) if EVAL_NORMALIZE_PER_SAMPLE else Xb
    with torch.no_grad():
        pred = model(Xn)
    pred = pred.cpu().squeeze(0)
    return pred, torch.sigmoid(pred[0]).numpy()


def compute_binary_metrics(pred_mask, true_mask):
    pred_mask = np.asarray(pred_mask).astype(bool)
    true_mask = np.asarray(true_mask).astype(bool)
    tp = int(np.logical_and(pred_mask, true_mask).sum())
    fp = int(np.logical_and(pred_mask, ~true_mask).sum())
    fn = int(np.logical_and(~pred_mask, true_mask).sum())
    union = tp + fp + fn
    return {
        "iou":       tp / union if union > 0 else 0.0,
        "dice":      2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0.0,
        "precision": tp / (tp+fp) if (tp+fp) > 0 else 0.0,
        "recall":    tp / (tp+fn) if (tp+fn) > 0 else 0.0,
    }


def compute_spread_metrics(prev_fire, pred_mask, true_mask):
    pf = np.asarray(prev_fire).astype(bool)
    pm = np.asarray(pred_mask).astype(bool)
    tm = np.asarray(true_mask).astype(bool)
    ts = tm & ~pf
    ps = pm & ~pf
    tp = int((ps & ts).sum())
    fp = int((ps & ~ts).sum())
    fn = int((~ps & ts).sum())
    union = tp + fp + fn
    return {
        "spread_iou":       tp / union if union > 0 else 0.0,
        "spread_precision": tp/(tp+fp) if (tp+fp) > 0 else 0.0,
        "spread_recall":    tp/(tp+fn) if (tp+fn) > 0 else 0.0,
    }


def evaluate_model(model, ds, device, fire_thr, save_thr_override=None):
    model.eval()
    pthrs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ious = {t: [] for t in pthrs}
    dice = {t: [] for t in pthrs}
    prec = {t: [] for t in pthrs}
    rec = {t: [] for t in pthrs}
    siou = {t: [] for t in pthrs}
    mpd = {t: [] for t in pthrs}
    hd = {t: [] for t in pthrs}
    maes = []

    with torch.no_grad():
        for i in range(len(ds)):
            X, Y = ds[i]
            xf = X[-1].numpy()
            yb = Y[0].numpy() > fire_thr
            _, pf_prob = predict_single(model, X, device)
            maes.append(float(np.abs(pf_prob - yb.astype(np.float32)).mean()))
            xf_fire = xf > fire_thr
            true_spread = yb & ~xf_fire

            for t in pthrs:
                pb = pf_prob > t
                bm = compute_binary_metrics(pb, yb)
                ps = pb & ~xf_fire
                su = int((true_spread | ps).sum())
                st = int((true_spread & ps).sum())
                m, h = perimeter_distance_metrics(pb, yb)
                ious[t].append(bm["iou"])
                dice[t].append(bm["dice"])
                prec[t].append(bm["precision"])
                rec[t].append(bm["recall"])
                siou[t].append(st/su if su > 0 else 0.0)
                if np.isfinite(m):
                    mpd[t].append(m)
                if np.isfinite(h):
                    hd[t].append(h)

    best_t = max(pthrs, key=lambda t: float(
        np.mean(ious[t])) if ious[t] else -1)
    save_t = float(save_thr_override) if str(
        save_thr_override).strip() else float(best_t)

    return {
        "prob_mae_mean":    float(np.mean(maes)) if maes else np.nan,
        "best_iou_mean":    float(np.mean(ious[best_t])) if ious[best_t] else np.nan,
        "spread_iou_at_best": float(np.mean(siou[best_t])) if siou[best_t] else np.nan,
        "dice_at_best":     float(np.mean(dice[best_t])) if dice[best_t] else np.nan,
        "recall_at_best":   float(np.mean(rec[best_t])) if rec[best_t] else np.nan,
        "mpd_at_best":      float(np.mean(mpd[best_t])) if mpd[best_t] else np.nan,
        "save_thr":         save_t,
        "ious_by_thr":      {t: float(np.mean(v)) if v else np.nan for t, v in ious.items()},
    }


# ─────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────

def extract_perimeter(mask):
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask.copy()
    # Pad before erosion so image-edge pixels are never falsely flagged as perimeter
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded  = binary_erosion(padded)[1:-1, 1:-1]
    perim   = np.logical_xor(mask, eroded)
    return perim if perim.any() else mask.copy()


def perimeter_distance_metrics(pred_mask, true_mask):
    pred_mask = np.asarray(pred_mask).astype(bool)
    true_mask = np.asarray(true_mask).astype(bool)
    if not pred_mask.any() and not true_mask.any():
        return 0.0, 0.0
    if not pred_mask.any() or not true_mask.any():
        return float("inf"), float("inf")
    pp = extract_perimeter(pred_mask)
    tp = extract_perimeter(true_mask)
    dt_t = distance_transform_edt(~tp)
    dt_p = distance_transform_edt(~pp)
    p2t = dt_t[pp]
    t2p = dt_p[tp]
    return float((p2t.mean() + t2p.mean()) / 2), float(max(p2t.max(), t2p.max()))


def compute_perimeter_error_map(pred_mask, true_mask):
    pred_mask = np.asarray(pred_mask).astype(bool)
    true_mask = np.asarray(true_mask).astype(bool)
    pp = extract_perimeter(pred_mask)
    tp = extract_perimeter(true_mask)
    err = np.zeros(pred_mask.shape, dtype=np.float32)
    if tp.any() and pp.any():
        err[pp] = distance_transform_edt(~tp)[pp]
    return err


def normalize_to_01(arr):
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def parse_vis_threshold(default_save_thr):
    s = str(VIS_THR_ENV).strip()
    if s == "":
        return float(default_save_thr)
    try:
        return float(s)
    except Exception:
        return float(default_save_thr)


def compute_checkpoint_score(stats):
    """
    Composite checkpoint score aligned with wildfire forecast usefulness.
    Higher is better.
    Emphasises IoU + spread IoU + Dice while mildly penalising perimeter drift.
    """
    iou = float(stats.get("best_iou_mean", np.nan))
    spread_iou = float(stats.get("spread_iou_at_best", np.nan))
    dice = float(stats.get("dice_at_best", np.nan))
    mpd = float(stats.get("mpd_at_best", np.nan))

    if not np.isfinite(iou):
        iou = -1.0
    if not np.isfinite(spread_iou):
        spread_iou = -1.0
    if not np.isfinite(dice):
        dice = -1.0
    if not np.isfinite(mpd):
        mpd = 999.0

    # Clamp perimeter penalty so a few hard cases do not dominate checkpoint selection.
    mpd_penalty = min(max(mpd, 0.0), 5.0) / 5.0
    return 0.50 * iou + 0.30 * spread_iou + 0.20 * dice - 0.05 * mpd_penalty


def resolve_eval_load_path():
    for cand in [str(LOAD_CKPT_PATH).strip(), BEST_CKPT_PATH, CKPT_PATH]:
        if cand and os.path.exists(cand):
            return cand
    return ""


def arrival_cmap_with_gray_bad():
    # Use matplotlib.colormaps (avoids deprecation warning from plt.cm.get_cmap)
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad(color="#111122")  # dark navy for never-burned background
    return cmap


def make_pred_overlay_rgb(base_fire, pred_mask):
    base_fire = np.asarray(base_fire, dtype=np.float32)
    pm = np.asarray(pred_mask).astype(bool)
    h, w = pm.shape
    rgb = np.full((h, w, 3), 0.38, dtype=np.float32)
    bf = normalize_to_01(base_fire) > 0
    if bf.any():
        rgb[bf] = [1.00, 0.95, 0.75]
    rgb[pm] = [0.92, 0.92, 0.92]
    pp = extract_perimeter(pm)
    rgb[pp] = [1.00, 0.12, 0.12]
    return np.clip(rgb, 0, 1)


# ─────────────────────────────────────────────────────────────
#  Autoregressive rollout
# ─────────────────────────────────────────────────────────────

def run_autoregressive_rollout(model, ds, start_idx, rollout_steps, thr, device):
    X0, _ = ds[start_idx]
    state = X0.clone().to(device)
    init_f = state[-1].cpu().numpy()
    pred_probs, pred_masks, true_masks = [], [], []

    for step in range(1, rollout_steps + 1):
        with torch.no_grad():
            xin = normalize_batch(state.unsqueeze(0)) if EVAL_NORMALIZE_PER_SAMPLE else state.unsqueeze(0)
            p = model(xin).squeeze(0)
        pf = torch.sigmoid(p[0]).cpu().numpy().astype(np.float32)
        pred_probs.append(pf)
        pred_masks.append((pf > thr).astype(bool))

        gt_idx = start_idx + step - 1
        if gt_idx < len(ds):
            _, Ygt = ds[gt_idx]
            true_masks.append((Ygt[0].numpy() > FIRE_THR).astype(bool))
        else:
            true_masks.append(None)

        if step < rollout_steps:
            nxt_idx = min(start_idx + step, len(ds) - 1)
            if USE_FUTURE_EXOGENOUS:
                nxt_X, _ = ds[nxt_idx]
                nxt_state = nxt_X.clone().to(device)
            else:
                nxt_state = state.clone()
            nxt_fire = torch.from_numpy(pf).to(device).unsqueeze(0)
            if KEEP_BURNING:
                nxt_fire = torch.maximum(nxt_fire, state[-1:])
            nxt_state[-1:] = nxt_fire
            state = nxt_state

    return init_f, pred_probs, pred_masks, true_masks


# ─────────────────────────────────────────────────────────────
#  Visualisation helpers  (rank-0 only)
# ─────────────────────────────────────────────────────────────

def make_paper_overlay_rgb(xf, pred_mask, true_mask, perim_err_map=None):
    xf = np.asarray(xf, dtype=np.float32)
    pm = np.asarray(pred_mask).astype(bool)
    tm = np.asarray(true_mask).astype(bool)
    h, w = pm.shape
    rgb = np.full((h, w, 3), 0.32, dtype=np.float32)
    rgb[pm] = 0.87
    xf_n = normalize_to_01(xf)
    cf = xf_n > 0
    if cf.any():
        rgb[cf, 0] = np.maximum(rgb[cf, 0], 1.00)
        rgb[cf, 1] = np.maximum(rgb[cf, 1], 0.95)
        rgb[cf, 2] = np.maximum(rgb[cf, 2], 0.75)
    if perim_err_map is None:
        perim_err_map = compute_perimeter_error_map(pm, tm)
    err_n = normalize_to_01(perim_err_map)
    if (err_n > 0).any():
        g = np.clip(err_n * 0.6, 0, 0.6)
        rgb[..., 0] += g * 0.05
        rgb[..., 1] += g * 0.12
        rgb[..., 2] += g * 0.30
    tp = extract_perimeter(tm)
    pp = extract_perimeter(pm)
    rgb[tp] = [0.10, 1.00, 0.10]
    rgb[pp] = [1.00, 0.10, 0.10]
    return np.clip(rgb, 0, 1)


def save_eval_figure(xf, yb, pb, out_path, title_prefix=""):
    yb = np.asarray(yb).astype(bool)
    pb = np.asarray(pb).astype(bool)
    xf = np.asarray(xf, dtype=np.float32)
    err = compute_perimeter_error_map(pb, yb)
    rgb = make_paper_overlay_rgb(xf, pb, yb, err)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    for ax, im, t in zip(axes,
                         [xf, yb.astype(np.float32), pb.astype(
                             np.float32), rgb, err],
                         ["Input FIRE(t)", "True Mask", "Pred Mask", "Perimeter Overlay", "Perim Error"]):
        ax.imshow(im, origin="lower", interpolation="nearest")
        ax.set_title(t)
        ax.axis("off")
    fig.colorbar(axes[4].images[0], ax=axes[4], fraction=0.046, pad=0.04)
    if title_prefix:
        fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_fire_growth_gif(xf, pb, yb, out_path, title_prefix=""):
    xf = np.asarray(xf, dtype=np.float32)
    frames = []
    for arr, t in [(xf, "Input FIRE(t)"), (pb.astype(np.float32), "Pred"), (yb.astype(np.float32), "True")]:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(arr, origin="lower", interpolation="nearest")
        ax.set_title(f"{title_prefix}\n{t}" if title_prefix else t)
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, duration=0.9, loop=0)


def save_paper_overlay_png(xf, pb, yb, out_path, title_prefix=""):
    rgb = make_paper_overlay_rgb(
        xf, pb, yb, compute_perimeter_error_map(pb, yb))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.plot([], [], color="lime", lw=2, label="True perimeter")
    ax.plot([], [], color="red",  lw=2, label="Pred perimeter")
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    if title_prefix:
        ax.set_title(title_prefix)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_probability_overlay_png(prob_map, pb, yb, out_path, title_prefix=""):
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(prob_map, origin="lower", vmin=0, vmax=1, cmap="magma", interpolation="nearest")
    tp = extract_perimeter(yb)
    pp = extract_perimeter(pb)
    yt, xt = np.where(tp)
    yp, xp = np.where(pp)
    if len(xt):
        ax.plot(xt, yt, color="lime", lw=2, label="True perimeter")
    if len(xp):
        ax.plot(xp, yp, color="cyan", lw=2, label="Pred perimeter")
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    if title_prefix:
        ax.set_title(title_prefix)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pred fire prob")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_true_pred_perimeter_overlay_gif(xf, pb, yb, out_path, title_prefix=""):
    frames = []
    for arr, t in [(np.asarray(xf, dtype=np.float32), "Current fire state"),
                   (np.asarray(pb).astype(np.float32), "Predicted fire mask")]:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(arr, origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title_prefix}\n{t}" if title_prefix else t)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    rgb = make_paper_overlay_rgb(
        xf, pb, yb, compute_perimeter_error_map(pb, yb))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.plot([], [], color="lime", lw=2, label="True")
    ax.plot([], [], color="red", lw=2, label="Pred")
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title_prefix}\nPerimeter overlay" if title_prefix else "Perimeter overlay")
    fig.tight_layout()
    fig.canvas.draw()
    ov = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    frames += [ov, ov]
    imageio.mimsave(out_path, frames, duration=[0.8, 0.8, 1.2, 1.2], loop=0)


def compute_arrival_time_map(initial_fire, pred_masks):
    init = np.asarray(initial_fire).astype(bool)
    arr = np.full(init.shape, -1, dtype=np.int32)
    arr[init] = 0
    reached = init.copy()
    for s, pm in enumerate(pred_masks, start=1):
        pm = np.asarray(pm).astype(bool)
        arr[pm & ~reached] = s
        reached |= pm
    return arr


def save_arrival_time_map(arrival_map, out_path, title_prefix=""):
    a = np.asarray(arrival_map)
    valid = a[a >= 0]
    m = np.ma.masked_where(a < 0, a)
    cmap = arrival_cmap_with_gray_bad()
    vmin = int(valid.min()) if len(valid) else 0
    vmax = int(valid.max()) if len(valid) else 1
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#111122")
    ax.set_facecolor("#111122")
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title_prefix or "Predicted fire arrival time", color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Arrival step")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_arrival_contours(arrival_map, out_path, title_prefix=""):
    a = np.asarray(arrival_map, dtype=np.float32)
    valid = a >= 0
    if not valid.any():
        return
    m = np.ma.masked_where(a < 0, a)
    cmap = arrival_cmap_with_gray_bad()
    vmin = float(a[valid].min())
    vmax = float(a[valid].max())
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#111122")
    ax.set_facecolor("#111122")
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.95, interpolation="nearest")
    ms = int(a[valid].max())
    if ms >= 1:
        ax.contour(a, levels=np.arange(1, ms + 1),
                   colors="white", linewidths=1.2, alpha=0.85, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title_prefix or "Fire arrival contours", color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Arrival step")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_rollout_binary_gif(initial_fire, pred_masks, out_path, title_prefix=""):
    frames = []
    for step, pm in enumerate(pred_masks):
        prev = initial_fire if step == 0 else pred_masks[step - 1].astype(
            np.float32)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(prev, origin="lower", interpolation="nearest")
        axes[0].set_title(f"Fire step {step}")
        axes[0].axis("off")
        axes[1].imshow(pm.astype(np.float32), origin="lower", interpolation="nearest")
        axes[1].set_title(f"Pred step {step+1}")
        axes[1].axis("off")
        if title_prefix:
            fig.suptitle(f"{title_prefix} rollout")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, duration=0.8, loop=0)


def save_rollout_probability_gif(initial_fire, pred_probs, pred_masks, out_path, title_prefix=""):
    frames = []
    for step, (pp, pm) in enumerate(zip(pred_probs, pred_masks), start=1):
        prev = initial_fire if step == 1 else pred_masks[step - 2].astype(
            np.float32)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(prev, origin="lower", interpolation="nearest")
        axes[0].set_title(f"Fire (step {step-1})")
        axes[0].axis("off")
        p_lo = float(np.percentile(pp, 2))
        p_hi = float(np.percentile(pp, 98))
        if p_hi - p_lo < 0.05:
            p_lo, p_hi = max(0.0, p_hi - 0.3), min(1.0, p_hi)
        im = axes[1].imshow(pp, origin="lower", vmin=p_lo, vmax=p_hi, cmap="magma", interpolation="nearest")
        yp, xp = np.where(extract_perimeter(pm))
        if len(xp):
            axes[1].plot(xp, yp, color="cyan", lw=2)
        axes[1].set_title(f"Pred fire probability (step {step})")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        if title_prefix:
            fig.suptitle(f"{title_prefix} prob rollout")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, duration=0.8, loop=0)


def save_cinematic_rollout_gif(initial_fire, pred_probs, pred_masks, out_path, title_prefix=""):
    frames = []
    for step, (pp, pm) in enumerate(zip(pred_probs, pred_masks), start=1):
        cur = initial_fire if step == 1 else pred_masks[step - 2].astype(
            np.float32)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
        axes[0].imshow(cur, origin="lower", interpolation="nearest")
        axes[0].set_title(f"Current fire state (step {step-1})")
        axes[0].axis("off")
        p_lo = float(np.percentile(pp, 2))
        p_hi = float(np.percentile(pp, 98))
        if p_hi - p_lo < 0.05:  # nearly flat – zoom in on the high end
            p_lo, p_hi = max(0.0, p_hi - 0.3), min(1.0, p_hi)
        im = axes[1].imshow(pp, origin="lower", vmin=p_lo, vmax=p_hi, cmap="magma", interpolation="nearest")
        yp, xp = np.where(extract_perimeter(pm))
        if len(xp):
            axes[1].plot(xp, yp, color="cyan", lw=1.5, alpha=0.9)
        axes[1].set_title(f"Pred fire probability (step {step})")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        ov = make_pred_overlay_rgb(cur, pm)
        axes[2].imshow(ov, origin="lower", interpolation="nearest")
        axes[2].set_title(f"Forecast overlay (step {step})\nred edge = predicted perimeter")
        axes[2].axis("off")
        if title_prefix:
            fig.suptitle(f"{title_prefix} cinematic")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, duration=0.8, loop=0)


def save_rollout_metric_plot(metric_series, out_path, title_prefix=""):
    steps = np.arange(1, len(metric_series) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, marker in [("iou", "o"), ("dice", "s"), ("mean_perim_dist", "^")]:
        ax.plot(steps, [m[key]
                for m in metric_series], marker=marker, label=key)
    ax.set_xlabel("Step")
    ax.set_ylabel("Metric")
    ax.set_title(title_prefix or "Rollout metrics")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def safe_nanmean(vals):
    v = [x for x in vals if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


# ─────────────────────────────────────────────────────────────
#  Metric CSV logger
# ─────────────────────────────────────────────────────────────

class MetricLogger:
    def __init__(self, path):
        self.path = path
        self._header_written = False

    def log(self, row: dict):
        write_header = not self._header_written and not Path(
            self.path).exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
                self._header_written = True
            w.writerow(row)


# ─────────────────────────────────────────────────────────────
#  Save all visualisations
# ─────────────────────────────────────────────────────────────

def save_all_visuals(model, ds, device, save_thr):
    save_thr = parse_vis_threshold(save_thr)
    with torch.no_grad():
        for i in range(min(6, len(ds))):
            X, Y = ds[i]
            xf = X[-1].numpy()
            yb = Y[0].numpy() > FIRE_THR
            _, pf_prob = predict_single(model, X, device)
            pb = pf_prob > save_thr

            save_eval_figure(xf, yb, pb,
                             with_run_tag(os.path.join(VIZ_DIR, f"eval_{i:03d}.png")),
                             title_prefix=f"sample={i} thr={save_thr}")
            save_fire_growth_gif(xf, pb, yb,
                                 with_run_tag(os.path.join(GIF_DIR, f"eval_{i:03d}.gif")),
                                 title_prefix=f"sample={i}")
            save_paper_overlay_png(xf, pb, yb,
                                   with_run_tag(os.path.join(
                                       PAPER_VIZ_DIR, f"paper_{i:03d}.png")),
                                   title_prefix=f"sample={i}")
            save_probability_overlay_png(pf_prob, pb, yb,
                                         with_run_tag(os.path.join(
                                             PROB_VIZ_DIR, f"prob_{i:03d}.png")),
                                         title_prefix=f"sample={i}")
            save_true_pred_perimeter_overlay_gif(xf, pb, yb,
                                                 with_run_tag(os.path.join(
                                                     PERIM_GIF_DIR, f"perim_{i:03d}.gif")),
                                                 title_prefix=f"sample={i}")
            print(f"  [viz] saved sample {i}")


def save_rollout_visuals(model, ds, device, save_thr):
    save_thr = parse_vis_threshold(save_thr)
    n = min(ROLL_OUT_N_SAMPLES, len(ds))
    metric_series_all = []
    spread_series_all = []

    with torch.no_grad():
        for i in range(n):
            init_f, probs, masks, trues = run_autoregressive_rollout(
                model, ds, i, ROLL_OUT_STEPS, save_thr, device)

            save_rollout_binary_gif(init_f, masks,
                                    with_run_tag(os.path.join(ROLL_OUT_GIF_DIR, f"rollout_{i:03d}.gif")), f"sample={i}")
            save_rollout_probability_gif(init_f, probs, masks,
                                         with_run_tag(os.path.join(ROLL_OUT_PROB_GIF_DIR, f"rollout_prob_{i:03d}.gif")), f"sample={i}")
            save_cinematic_rollout_gif(init_f, probs, masks,
                                       with_run_tag(os.path.join(CINEMATIC_DIR, f"cinematic_{i:03d}.gif")), f"sample={i}")

            arr_map = compute_arrival_time_map(init_f > FIRE_THR, masks)
            save_arrival_time_map(arr_map,
                                  with_run_tag(os.path.join(ARRIVAL_DIR, f"arrival_{i:03d}.png")), f"sample={i}")
            save_arrival_contours(arr_map,
                                  with_run_tag(os.path.join(ARRIVAL_DIR, f"contours_{i:03d}.png")), f"sample={i}")

            roll_m = []
            spread_m = []
            prev = init_f > FIRE_THR
            for step, (pm, ym) in enumerate(zip(masks, trues), start=1):
                if ym is None:
                    break
                bm = compute_binary_metrics(pm, ym)
                mpd, _ = perimeter_distance_metrics(pm, ym)
                sm = compute_spread_metrics(prev, pm, ym)
                roll_m.append({"iou": bm["iou"], "dice": bm["dice"],
                               "mean_perim_dist": mpd if np.isfinite(mpd) else np.nan})
                spread_m.append(sm)
                prev = pm.copy()

            if roll_m:
                metric_series_all.append(roll_m)
                save_rollout_metric_plot(roll_m,
                                         with_run_tag(os.path.join(ROLL_OUT_PLOT_DIR, f"rollout_metrics_{i:03d}.png")), f"sample={i}")
            if spread_m:
                spread_series_all.append(spread_m)
            print(f"  [rollout] saved sample {i}")

    # Mean across samples
    if metric_series_all:
        ml = max(len(x) for x in metric_series_all)
        mean_m = []
        for si in range(ml):
            vs = [seq[si] for seq in metric_series_all if si < len(seq)]
            mean_m.append({
                "iou":            safe_nanmean([v["iou"] for v in vs]),
                "dice":           safe_nanmean([v["dice"] for v in vs]),
                "mean_perim_dist": safe_nanmean([v["mean_perim_dist"] for v in vs]),
            })
        save_rollout_metric_plot(mean_m,
                                 with_run_tag(os.path.join(ROLL_OUT_PLOT_DIR,
                                              "rollout_metrics_mean.png")),
                                 title_prefix=f"Mean rollout thr={save_thr}")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    rank, world_size, local_rank, device = setup_ddp()
    set_seed(SEED, rank)
    main_proc = is_main(rank)

    if main_proc:
        print(f"Device: {device}  |  world_size={world_size}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        for d in ALL_OUTPUT_DIRS:
            os.makedirs(d, exist_ok=True)

    # ── Train / test split (random holdout or k-fold CV) ─────
    _full_ds = WrfVitShardDataset(SHARDS_DIR)
    n_total = len(_full_ds)
    all_idx = list(range(n_total))
    rng_split = np.random.default_rng(SPLIT_SEED)

    if KFOLD_K > 1:
        # K-fold: shuffle once with SPLIT_SEED, divide into K folds
        shuffled = all_idx.copy()
        rng_split.shuffle(shuffled)
        folds = [shuffled[i::KFOLD_K] for i in range(KFOLD_K)]
        test_idx  = sorted(folds[KFOLD_FOLD])
        train_idx = sorted(sum([folds[i] for i in range(KFOLD_K) if i != KFOLD_FOLD], []))
        split_meta = {"mode": "kfold", "k": KFOLD_K, "fold": KFOLD_FOLD,
                      "split_seed": SPLIT_SEED, "shards_dir": SHARDS_DIR}
        split_label = f"k={KFOLD_K}  fold={KFOLD_FOLD}/{KFOLD_K-1}"
    else:
        # Random holdout
        n_test = max(1, round(n_total * TEST_FRAC))
        test_idx  = sorted(rng_split.choice(all_idx, size=n_test, replace=False).tolist())
        train_idx = sorted(set(all_idx) - set(test_idx))
        split_meta = {"mode": "holdout", "test_frac": TEST_FRAC,
                      "split_seed": SPLIT_SEED, "shards_dir": SHARDS_DIR}
        split_label = f"TEST_FRAC={TEST_FRAC}  SPLIT_SEED={SPLIT_SEED}"

    n_train, n_test = len(train_idx), len(test_idx)
    ds      = WrfVitShardDataset(SHARDS_DIR, indices=train_idx, augment=True)   # training set
    test_ds = WrfVitShardDataset(SHARDS_DIR, indices=test_idx,  augment=False)  # held-out test set

    if main_proc:
        print(f"Dataset: {n_total} total  |  train={n_train}  test={n_test}  ({split_label})")
        print(f"Test indices: {test_idx}")
        split_path = os.path.join(OUT_ROOT, "train_test_split.json")
        os.makedirs(OUT_ROOT, exist_ok=True)
        import json
        with open(split_path, "w") as f:
            json.dump({"train": train_idx, "test": test_idx, **split_meta}, f, indent=2)
        print(f"Split saved to {split_path}")

    X0, Y0 = ds[0]
    in_c, out_c = X0.shape[0], Y0.shape[0]

    if main_proc:
        print(f"in_c={in_c}  out_c={out_c}  base={BASE_CHANNELS}")

    # Positive weight (computed on train set only)
    if AUTO_POS_WEIGHT:
        pw_val, pos_frac = estimate_pos_weight(ds, FIRE_THR)
        if main_proc:
            print(f"Auto pos_weight={pw_val:.2f}  (fire fraction={pos_frac:.4f})")
    else:
        pw_val = POS_WEIGHT

    # Model
    model = UNet(in_c=in_c, out_c=out_c, base=BASE_CHANNELS,
                 dropout=DROPOUT, drop_path_rate=DROP_PATH_RATE).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if main_proc:
        print(f"Trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup_sched, cosine_sched = build_scheduler(
        optimizer, EPOCHS, WARMUP_EPOCHS, LR_RESTART_EPOCHS, MIN_LR)

    ac_dtype = get_autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(ac_dtype == torch.float16))
    ema = EMA(model.module if world_size > 1 else model,
              EMA_DECAY, EMA_WARMUP_EPOCHS) if USE_EMA else None

    best_iou = -1.0
    best_score = -1e9
    best_state = None
    csv_path = with_run_tag(os.path.join(OUT_ROOT, "metrics_log.csv"))
    logger = MetricLogger(csv_path) if main_proc else None

    # WandB
    if main_proc and WANDB_AVAILABLE and WANDB_PROJECT:
        wandb.init(project=WANDB_PROJECT, config={
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "base_channels": BASE_CHANNELS, "drop_path_rate": DROP_PATH_RATE,
            "dice_w": DICE_W, "focal_w": FOCAL_W, "lovasz_w": LOVASZ_W,
            "boundary_w": BOUNDARY_W, "world_size": world_size,
            "train_rollout_steps_max": TRAIN_ROLLOUT_STEPS_MAX,
            "teacher_forcing_end": TEACHER_FORCING_END,
            "checkpoint_metric": "composite",
        })
        wandb.watch(model.module if world_size > 1 else model, log_freq=100)

    # Eval-only path
    if EVAL_ONLY:
        if main_proc:
            load_path = resolve_eval_load_path()
            if not load_path:
                raise FileNotFoundError(
                    f"No checkpoint found for eval-only. Checked LOAD_CKPT_PATH={LOAD_CKPT_PATH!r}, "
                    f"BEST_CKPT_PATH={BEST_CKPT_PATH!r}, CKPT_PATH={CKPT_PATH!r}"
                )
            state = torch.load(load_path, map_location=device)
            raw = model.module if world_size > 1 else model
            raw.load_state_dict(state, strict=True)
            raw.eval()
            print(f"Loaded checkpoint for eval-only run: {load_path}")

            final_eval = evaluate_model(
                raw, test_ds, device, FIRE_THR, save_thr_override=PB_THR_ENV)
            final_vis_thr = parse_vis_threshold(final_eval["save_thr"])
            print(
                f"[eval-only] iou={final_eval['best_iou_mean']:.4f}  "
                f"dice={final_eval['dice_at_best']:.4f}  "
                f"spread_iou={final_eval['spread_iou_at_best']:.4f}  "
                f"mpd={final_eval['mpd_at_best']:.2f}  "
                f"mae={final_eval['prob_mae_mean']:.4f}  "
                f"save_thr={final_eval['save_thr']}"
            )
            print(f"\nGenerating visualisations (thr={final_vis_thr}) ...")
            save_all_visuals(raw, test_ds, device, final_vis_thr)
            save_rollout_visuals(raw, test_ds, device, final_vis_thr)
            print("Done.")
            if WANDB_AVAILABLE and WANDB_PROJECT:
                wandb.finish()
        if world_size > 1:
            dist.destroy_process_group()
        return

    for epoch in range(EPOCHS):
        t0 = time.time()

        train_stats = train_one_epoch(
            model=model, ds=ds, device=device,
            optimizer=optimizer, scaler=scaler,
            pw_value=pw_val, epoch=epoch, ema=ema,
            rank=rank, world_size=world_size,
        )

        # LR schedule
        if epoch < WARMUP_EPOCHS:
            warmup_sched.step()
        else:
            cosine_sched.step(epoch - WARMUP_EPOCHS)

        cur_lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        if main_proc:
            print(
                f"Epoch {epoch:03d}/{EPOCHS}  "
                f"loss={train_stats['loss']:.5f}  fire={train_stats['fire_loss']:.5f}  "
                f"other={train_stats['other_loss']:.5f}  "
                f"steps={train_stats['effective_rollout_steps']}  "
                f"tf={train_stats['teacher_forcing_ratio']:.3f}  "
                f"lr={cur_lr:.2e}  t={dt:.1f}s"
            )

        if main_proc and (((epoch + 1) % EVAL_EVERY == 0) or (epoch == EPOCHS - 1)):
            raw_model = model.module if world_size > 1 else model
            eval_model = copy.deepcopy(raw_model).to(device)
            if ema is not None:
                ema.copy_to(eval_model)
            eval_model.eval()

            stats = evaluate_model(eval_model, test_ds, device, FIRE_THR,
                                   save_thr_override=PB_THR_ENV)
            score = stats["best_iou_mean"]
            ckpt_score = compute_checkpoint_score(stats)

            row = {"epoch": epoch, "lr": cur_lr, **train_stats, **stats,
                   "checkpoint_score": ckpt_score}
            logger.log(row)

            print(
                f"  [eval] iou={score:.4f}  dice={stats['dice_at_best']:.4f}  "
                f"spread_iou={stats['spread_iou_at_best']:.4f}  "
                f"mpd={stats['mpd_at_best']:.2f}  mae={stats['prob_mae_mean']:.4f}  "
                f"ckpt_score={ckpt_score:.4f}"
            )

            if WANDB_AVAILABLE and WANDB_PROJECT:
                wandb.log({"epoch": epoch, **train_stats,
                           "eval/iou": score, "eval/dice": stats["dice_at_best"],
                           "eval/spread_iou": stats["spread_iou_at_best"],
                           "eval/mpd": stats["mpd_at_best"],
                           "eval/checkpoint_score": ckpt_score,
                           "lr": cur_lr})

            if np.isfinite(score):
                best_iou = max(best_iou, score)

            if np.isfinite(ckpt_score) and ckpt_score > best_score:
                best_score = ckpt_score
                best_state = copy.deepcopy(eval_model.state_dict())
                torch.save(best_state, BEST_CKPT_PATH)
                print(
                    f"  [ckpt] NEW BEST  composite={best_score:.4f}  "
                    f"(iou={score:.4f}, spread_iou={stats['spread_iou_at_best']:.4f})  → {BEST_CKPT_PATH}")

            del eval_model

        # Save last checkpoint every epoch (on main proc)
        if main_proc:
            raw = model.module if world_size > 1 else model
            torch.save(raw.state_dict(), CKPT_PATH)

    # ── Final export ──────────────────────────────────────────
    if main_proc:
        raw = model.module if world_size > 1 else model
        if best_state is not None:
            raw.load_state_dict(best_state, strict=True)
            print("Loaded best checkpoint for final visualisation")
        elif os.path.exists(BEST_CKPT_PATH):
            state = torch.load(BEST_CKPT_PATH, map_location=device)
            raw.load_state_dict(state, strict=True)
            print("Loaded best checkpoint for final visualisation")
        elif ema is not None:
            ema.copy_to(raw)
            print("Loaded EMA weights for final visualisation")

        torch.save(raw.state_dict(), CKPT_PATH)

        final_eval = evaluate_model(
            raw, test_ds, device, FIRE_THR, save_thr_override=PB_THR_ENV)
        final_vis_thr = parse_vis_threshold(final_eval["save_thr"])
        print(f"\nGenerating visualisations (thr={final_vis_thr}) ...")
        save_all_visuals(raw, test_ds, device, final_vis_thr)
        save_rollout_visuals(raw, test_ds, device, final_vis_thr)
        print("Done.")

        if WANDB_AVAILABLE and WANDB_PROJECT:
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
