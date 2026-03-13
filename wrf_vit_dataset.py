import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# All valid joint augmentations for spatial fire data.
# Flips and 90-degree rotations are physically valid — fire spread is not
# rotationally biased in the grid frame, and wind/terrain channels transform
# consistently with the spatial flip/rotation applied to both X and Y.
_AUG_OPS = [
    lambda x, y: (x, y),                                           # identity
    lambda x, y: (np.flip(x, axis=2), np.flip(y, axis=2)),         # flip LR
    lambda x, y: (np.flip(x, axis=1), np.flip(y, axis=1)),         # flip UD
    lambda x, y: (np.rot90(x, 1, (1, 2)), np.rot90(y, 1, (1, 2))), # rot 90
    lambda x, y: (np.rot90(x, 2, (1, 2)), np.rot90(y, 2, (1, 2))), # rot 180
    lambda x, y: (np.rot90(x, 3, (1, 2)), np.rot90(y, 3, (1, 2))), # rot 270
    lambda x, y: (np.flip(np.rot90(x, 1, (1, 2)), axis=2),         # rot90 + flip LR
                  np.flip(np.rot90(y, 1, (1, 2)), axis=2)),
    lambda x, y: (np.flip(np.rot90(x, 1, (1, 2)), axis=1),         # rot90 + flip UD
                  np.flip(np.rot90(y, 1, (1, 2)), axis=1)),
]


class WrfVitShardDataset(Dataset):
    """
    Loads NPZ shards of:
      X: (C,H,W) float32
      Y: (F,H,W) float32

    indices:  optional list of ints to use a subset of files (train/test split).
    augment:  if True, applies random joint flip/rotation to X and Y at load time.
              Only enable for the training split — never for test/eval.
    """
    def __init__(self, shards_dir, normalize=None, target_transform=None,
                 indices=None, augment=False):
        all_files = sorted(glob.glob(f"{shards_dir}/*.npz"))
        if not all_files:
            raise FileNotFoundError(f"No .npz shards found in {shards_dir}")

        if indices is not None:
            self.files = [all_files[i] for i in indices]
        else:
            self.files = all_files

        self.normalize = normalize
        self.target_transform = target_transform
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        X = d["X"].astype(np.float32)  # (C,H,W)
        Y = d["Y"].astype(np.float32)  # (F,H,W)

        if self.augment:
            op = _AUG_OPS[np.random.randint(len(_AUG_OPS))]
            X, Y = op(X, Y)
            X = np.ascontiguousarray(X)
            Y = np.ascontiguousarray(Y)

        if self.normalize is not None:
            mean = self.normalize["mean"][:, None, None]
            std  = self.normalize["std"][:, None, None]
            X = (X - mean) / (std + 1e-6)

        if self.target_transform is not None:
            Y = self.target_transform(Y)

        return torch.from_numpy(X), torch.from_numpy(Y)
