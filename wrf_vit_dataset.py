import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Augmentation with physically correct wind-vector handling
# ---------------------------------------------------------------------------
# Spatial transforms are applied to the full (C,H,W) X tensor and to Y.
# U10 and V10 are signed vector components, so a spatial flip/rotation must
# also update their *values* to reflect the new reference frame.
#
#   Transform       | new_U10 (eastward)  | new_V10 (northward)
#   ────────────────┼─────────────────────┼────────────────────
#   identity        | +U                  | +V
#   flip LR         | -U  (E↔W reversed)  | +V
#   flip UD         | +U                  | -V  (N↔S reversed)
#   rot90 CCW       | +V  (new E = old N) | -U
#   rot180          | -U                  | -V
#   rot270 CCW      | -V  (new E = old S) | +U
#   rot90 + flip LR | -V                  | -U
#   rot90 + flip UD | +V                  | +U
#
# "old" U/V values below are taken AFTER the spatial transform has already
# moved values to their new positions; only the direction must be corrected.

_SPATIAL_FNS = [
    lambda a: a,                                                  # identity
    lambda a: np.flip(a, axis=2),                                 # flip LR
    lambda a: np.flip(a, axis=1),                                 # flip UD
    lambda a: np.rot90(a, 1, (1, 2)),                             # rot 90 CCW
    lambda a: np.rot90(a, 2, (1, 2)),                             # rot 180
    lambda a: np.rot90(a, 3, (1, 2)),                             # rot 270 CCW
    lambda a: np.flip(np.rot90(a, 1, (1, 2)), axis=2),           # rot90 + flip LR
    lambda a: np.flip(np.rot90(a, 1, (1, 2)), axis=1),           # rot90 + flip UD
]

# (uv_swap, u_neg, v_neg) applied in that order to wind channels after spatial fn.
# uv_swap:  exchange x[u_ch] ↔ x[v_ch]
# u_neg:    negate x[u_ch]
# v_neg:    negate x[v_ch]
_WIND_CORRECTIONS = [
    (False, False, False),  # identity
    (False, True,  False),  # flip LR:   U → -U
    (False, False, True),   # flip UD:   V → -V
    (True,  False, True),   # rot90:     swap → (U'=V, V'=U), then V'→-U  → (V, -U)
    (False, True,  True),   # rot180:    U → -U, V → -V
    (True,  True,  False),  # rot270:    swap → (V, U), then U'→-V         → (-V, U)
    (True,  True,  True),   # rot90+LR:  swap, U→-V, V→-U
    (True,  False, False),  # rot90+UD:  swap only → (V, U)
]


def _apply_aug(x, y, op_idx, u_ch, v_ch):
    """Apply augmentation op_idx to (X, Y), correctly adjusting wind channels."""
    spatial = _SPATIAL_FNS[op_idx]
    uv_swap, u_neg, v_neg = _WIND_CORRECTIONS[op_idx]

    x = spatial(x)
    y = spatial(y)

    if uv_swap or u_neg or v_neg:
        x = x.copy()
        if uv_swap:
            u_vals = x[u_ch].copy()
            x[u_ch] = x[v_ch]
            x[v_ch] = u_vals
        if u_neg:
            x[u_ch] = -x[u_ch]
        if v_neg:
            x[v_ch] = -x[v_ch]

    return x, y


class WrfVitShardDataset(Dataset):
    """
    Loads NPZ shards of:
      X: (C,H,W) float32
      Y: (F,H,W) float32

    indices:        optional list of ints for train/test split.
    augment:        if True, applies a random physically-correct flip/rotation to
                    X and Y at load time. Only enable for the training split.
    u_wind_channel: X channel index for U10 (east-west wind). Default 3.
    v_wind_channel: X channel index for V10 (north-south wind). Default 4.
                    Standard channel layout: T2(0) Q2(1) PSFC(2) U10(3) V10(4)
                    HGT(5) T_k0(6) QVAPOR_k0(7) FIRE_AREA_t(8).
    """
    def __init__(self, shards_dir, normalize=None, target_transform=None,
                 indices=None, augment=False, u_wind_channel=3, v_wind_channel=4):
        all_files = sorted(glob.glob(f"{shards_dir}/*.npz"))
        if not all_files:
            raise FileNotFoundError(f"No .npz shards found in {shards_dir}")

        self.files = [all_files[i] for i in indices] if indices is not None else all_files
        self.normalize = normalize
        self.target_transform = target_transform
        self.augment = augment
        self.u_ch = u_wind_channel
        self.v_ch = v_wind_channel

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        X = d["X"].astype(np.float32)  # (C,H,W)
        Y = d["Y"].astype(np.float32)  # (F,H,W)

        if self.augment:
            op_idx = np.random.randint(len(_SPATIAL_FNS))
            X, Y = _apply_aug(X, Y, op_idx, self.u_ch, self.v_ch)
            X = np.ascontiguousarray(X)
            Y = np.ascontiguousarray(Y)

        if self.normalize is not None:
            mean = self.normalize["mean"][:, None, None]
            std  = self.normalize["std"][:, None, None]
            X = (X - mean) / (std + 1e-6)

        if self.target_transform is not None:
            Y = self.target_transform(Y)

        return torch.from_numpy(X), torch.from_numpy(Y)
