"""
dataset.py
----------
PyTorch Dataset classes for GigaPath and classical H5 embeddings.

GigaPathDataset  → loads (N_tiles, 1536) bag, samples MAX_TILES during training
ClassicalDataset → loads (404,) mean-pooled vector

Usage:
  from data.dataset import GigaPathDataset, ClassicalDataset
"""

import csv
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

BASE_DIR    = os.environ.get("LUNG_WSI_DATA", os.path.expanduser("~/research_data"))
GP_H5       = os.path.join(BASE_DIR, "embeddings", "gigapath_embeddings.h5")
CL_H5       = os.path.join(BASE_DIR, "embeddings", "classical_embeddings.h5")
MAX_TILES   = 500   # training-time subsample cap


def load_split_csv(path):
    """Return list of dicts from a split CSV."""
    with open(path) as f:
        return list(csv.DictReader(f))


class GigaPathDataset(Dataset):
    """
    Loads GigaPath tile embeddings from H5.
    Training:  randomly subsamples up to MAX_TILES tiles per slide.
    Inference: uses all tiles.
    """

    def __init__(self, split_csv, train=True, max_tiles=MAX_TILES):
        import h5py
        self.rows      = load_split_csv(split_csv)
        self.train     = train
        self.max_tiles = max_tiles
        self.h5_path   = GP_H5

        # Validate all file_ids exist in H5
        with h5py.File(self.h5_path, "r") as h:
            available = set(h.keys())
        self.rows = [r for r in self.rows if r["file_id"] in available]

        print(f"GigaPathDataset: {len(self.rows)} slides "
              f"({'train' if train else 'eval'}, max_tiles={max_tiles})")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import h5py
        row   = self.rows[idx]
        fid   = row["file_id"]
        label = int(row["label_int"])

        with h5py.File(self.h5_path, "r") as h:
            feats = h[fid]["features"][:]   # (N_tiles, 1536)

        # Subsample during training
        if self.train and len(feats) > self.max_tiles:
            idxs  = np.random.choice(len(feats), self.max_tiles, replace=False)
            idxs  = np.sort(idxs)
            feats = feats[idxs]

        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_file_id(self, idx):
        return self.rows[idx]["file_id"]


class ClassicalDataset(Dataset):
    """
    Loads classical (HOG+LBP+GLCM) mean-pooled embeddings from H5.
    Each sample is a single (404,) vector — no tiling/subsampling needed.
    """

    def __init__(self, split_csv):
        import h5py
        self.rows    = load_split_csv(split_csv)
        self.h5_path = CL_H5

        with h5py.File(self.h5_path, "r") as h:
            available = set(h.keys())
        self.rows = [r for r in self.rows if r["file_id"] in available]

        print(f"ClassicalDataset: {len(self.rows)} slides")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import h5py
        row   = self.rows[idx]
        fid   = row["file_id"]
        label = int(row["label_int"])

        with h5py.File(self.h5_path, "r") as h:
            feats = h[fid]["features"][:]   # (N_tiles, 404) → mean pool
            feats = feats.mean(axis=0)      # (404,)

        return torch.tensor(feats, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def collate_bags(batch):
    """
    Custom collate for variable-length bags (GigaPath).
    Returns:
      bags   : list of (N_i, 1536) tensors  — NOT padded, ABMIL handles variable N
      labels : (B,) tensor
    """
    bags   = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return bags, labels


def get_gigapath_loaders(train_csv, val_csv, test_csv,
                         batch_size=1, num_workers=4, max_tiles=MAX_TILES):
    """
    Returns train/val/test DataLoaders for GigaPath embeddings.
    batch_size=1 is standard for ABMIL (variable bag sizes).
    Increase to >1 only if using padding or fixed tile counts.
    """
    from torch.utils.data import DataLoader

    train_ds = GigaPathDataset(train_csv, train=True,  max_tiles=max_tiles)
    val_ds   = GigaPathDataset(val_csv,   train=False, max_tiles=max_tiles)
    test_ds  = GigaPathDataset(test_csv,  train=False, max_tiles=max_tiles)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_bags,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_bags,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_bags,
                              pin_memory=True)

    return train_loader, val_loader, test_loader


def get_classical_loaders(train_csv, val_csv, test_csv,
                          batch_size=32, num_workers=4):
    from torch.utils.data import DataLoader

    train_ds = ClassicalDataset(train_csv)
    val_ds   = ClassicalDataset(val_csv)
    test_ds  = ClassicalDataset(test_csv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
