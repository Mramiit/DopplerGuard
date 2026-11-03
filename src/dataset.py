"""
PyTorch Dataset and DataLoaders for Drone Radar Classification.
Supports Trajectory-Aware Splitting and Leave-One-Drone-Out (LODO) Evaluation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path

try:
    from src import config
except ImportError:
    import config


class RadarDataset(Dataset):
    """
    PyTorch Dataset for 2-channel Range-Doppler + Micro-Doppler tensors.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        drone_models: np.ndarray,
        recording_ids: np.ndarray,
        augment: bool = False
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.drone_models = drone_models
        self.recording_ids = recording_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # 1. Subtle random noise injection (simulating atmospheric noise / low SNR)
            if torch.rand(1).item() < 0.4:
                noise = torch.randn_like(x) * 0.02
                x = torch.clamp(x + noise, 0.0, 1.0)

            # 2. Random cyclic roll along horizontal axis (simulating azimuth phase shift)
            if torch.rand(1).item() < 0.4:
                shift = torch.randint(-4, 5, (1,)).item()
                x = torch.roll(x, shifts=shift, dims=2)

            # 3. Random intensity jitter
            if torch.rand(1).item() < 0.3:
                scale = torch.empty(1).uniform_(0.9, 1.1).item()
                x = torch.clamp(x * scale, 0.0, 1.0)

        return x, y


def load_dataset_splits(
    npz_path: Optional[Path] = None,
    mode: str = "lodo",             # 'lodo' (Leave-One-Drone-Out) or 'standard'
    holdout_drone: str = config.LODO_HOLDOUT_DRONE,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Splits the dataset and creates PyTorch DataLoaders.

    Modes:
    - 'lodo': Leave-One-Drone-Out (Moat Benchmark).
              Excludes holdout_drone (e.g. D6) entirely from training,
              and evaluates zero-shot classification on it.
    - 'standard': Stratified recording-disjoint split across all classes.
    """
    if npz_path is None:
        npz_path = config.PROCESSED_DIR / "radar_balanced_dataset.npz"

    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y_coarse']
    drone_models = data['drone_models']
    recording_ids = data['recording_ids']

    rng = np.random.default_rng(seed)

    if mode.lower() == "lodo":
        print(f"[Dataset] Configured LODO Mode: Holding out unseen drone '{holdout_drone}' for Zero-Shot evaluation.")
        is_holdout_drone = (drone_models == holdout_drone)

        # Train and Val are chosen exclusively from non-holdout recordings
        non_holdout_indices = np.where(~is_holdout_drone)[0]
        unique_recs = np.unique(recording_ids[non_holdout_indices])
        rng.shuffle(unique_recs)

        n_val = max(1, int(len(unique_recs) * 0.15))
        n_test = max(1, int(len(unique_recs) * 0.15))

        val_recs = set(unique_recs[:n_val])
        test_recs = set(unique_recs[n_val:n_val + n_test])
        train_recs = set(unique_recs[n_val + n_test:])

        train_indices = [i for i in non_holdout_indices if recording_ids[i] in train_recs]
        val_indices = [i for i in non_holdout_indices if recording_ids[i] in val_recs]
        
        # Test set contains the unseen holdout drone (D6) PLUS test recordings of Birds, Humans, and other targets
        holdout_drone_indices = list(np.where(is_holdout_drone)[0])
        test_indices = holdout_drone_indices + [i for i in non_holdout_indices if recording_ids[i] in test_recs]

    else:
        # Standard trajectory-aware split
        unique_recs = np.unique(recording_ids)
        rng.shuffle(unique_recs)
        n_total = len(unique_recs)
        n_test = max(1, int(n_total * 0.15))
        n_val = max(1, int(n_total * 0.15))

        test_recs = set(unique_recs[:n_test])
        val_recs = set(unique_recs[n_test:n_test + n_val])
        train_recs = set(unique_recs[n_test + n_val:])

        train_indices = [i for i in range(len(y)) if recording_ids[i] in train_recs]
        val_indices = [i for i in range(len(y)) if recording_ids[i] in val_recs]
        test_indices = [i for i in range(len(y)) if recording_ids[i] in test_recs]

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    train_ds = RadarDataset(X[train_indices], y[train_indices], drone_models[train_indices], recording_ids[train_indices], augment=True)
    val_ds = RadarDataset(X[val_indices], y[val_indices], drone_models[val_indices], recording_ids[val_indices], augment=False)
    test_ds = RadarDataset(X[test_indices], y[test_indices], drone_models[test_indices], recording_ids[test_indices], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)

    metadata = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "mode": mode,
        "holdout_drone": holdout_drone if mode == "lodo" else None,
        "num_classes": len(config.COARSE_CLASSES)
    }

    print(f"[Dataset] Split sizes: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}")
    return train_loader, val_loader, test_loader, metadata
