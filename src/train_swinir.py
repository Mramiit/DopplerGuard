"""
Optimized Training Pipeline for Radar-SwinIR.
Uses Vectorized Batch PSNR, Progress Logging, and Residual Physics Loss.
"""

import sys
import time
import json
import math
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from src import config
    from src.swinir_model import get_radar_swinir
except ImportError:
    import config
    from swinir_model import get_radar_swinir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RadarPairDataset(Dataset):
    """
    Dataset of paired (degraded_input, clean_target) radar feature maps.
    """
    def __init__(self, X_degraded: np.ndarray, X_clean: np.ndarray):
        self.X_deg = torch.tensor(X_degraded, dtype=torch.float32)
        self.X_clean = torch.tensor(X_clean, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_deg)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_deg[idx], self.X_clean[idx]


class RadarPhysicsLoss(nn.Module):
    """
    Combines Charbonnier reconstruction loss with Doppler Harmonic Gradient loss.
    """
    def __init__(self, eps: float = 1e-3, harmonic_weight: float = 0.2):
        super().__init__()
        self.eps = eps
        self.harmonic_weight = harmonic_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Charbonnier Reconstruction Loss
        diff = pred - target
        loss_charbonnier = torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))

        # 2. Doppler Frequency Gradient Loss (Vertical blade flash sharpness)
        pred_grad_v = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_grad_v = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        loss_harmonic = F.l1_loss(pred_grad_v, target_grad_v)

        return loss_charbonnier + self.harmonic_weight * loss_harmonic


def calculate_batch_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Vectorized computation of mean PSNR across a batch of tensors.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20.0 * torch.log10(max_val / torch.sqrt(mse))
    return float(torch.mean(psnr).item())


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Single sample or batch PSNR helper."""
    return calculate_batch_psnr(pred, target, max_val)


def run_swinir_training(
    paired_npz_path: Optional[Path] = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 5e-4,
    max_train_pairs: int = 2000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if paired_npz_path is None:
        paired_npz_path = config.PROCESSED_DIR / "radar_sr_paired_dataset.npz"

    if not paired_npz_path.exists():
        raise FileNotFoundError(f"Paired dataset not found at {paired_npz_path}. Run range_degrader.py first!")

    print(f"\n{'='*65}\n  TRAINING RADAR-SWINIR (SUPER-RESOLUTION & DENOISING)\n  DEVICE: {device} | DATASET: {paired_npz_path.name}\n{'='*65}", flush=True)

    data = np.load(paired_npz_path, allow_pickle=True)
    X_deg = data["X_degraded"]
    X_clean = data["X_clean"]

    # Select representative balanced subset for fast CPU training
    if max_train_pairs and max_train_pairs < len(X_deg):
        indices = np.random.RandomState(42).permutation(len(X_deg))[:max_train_pairs]
        X_deg = X_deg[indices]
        X_clean = X_clean[indices]

    n_total = len(X_deg)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_ds = RadarPairDataset(X_deg[train_idx], X_clean[train_idx])
    val_ds = RadarPairDataset(X_deg[val_idx], X_clean[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Radar-SwinIR] Train pairs: {len(train_ds):,} | Val pairs: {len(val_ds):,} | Batches/epoch: {len(train_loader)}", flush=True)

    model = get_radar_swinir(device=str(device))
    criterion = RadarPhysicsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    save_model_path = config.MODEL_SAVE_DIR / "best_radar_swinir.pth"
    best_val_psnr = -1.0
    best_epoch = 0

    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "input_psnr": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        t_ep = time.time()

        # Training
        model.train()
        running_loss = 0.0
        for b_idx, (x_in, y_gt) in enumerate(train_loader):
            x_in, y_gt = x_in.to(device), y_gt.to(device)

            optimizer.zero_grad()
            restored = model(x_in)
            loss = criterion(restored, y_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * x_in.size(0)

            if (b_idx + 1) % 15 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch:02d} | Batch {b_idx + 1:02d}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        train_loss = running_loss / len(train_ds)

        # Validation (Vectorized batch PSNR)
        model.eval()
        val_running_loss = 0.0
        val_psnr_list = []
        in_psnr_list = []

        with torch.no_grad():
            for x_in, y_gt in val_loader:
                x_in, y_gt = x_in.to(device), y_gt.to(device)
                restored = model(x_in)
                loss = criterion(restored, y_gt)
                val_running_loss += loss.item() * x_in.size(0)

                val_psnr_list.append(calculate_batch_psnr(restored, y_gt))
                in_psnr_list.append(calculate_batch_psnr(x_in, y_gt))

        val_loss = val_running_loss / len(val_ds)
        mean_val_psnr = float(np.mean(val_psnr_list))
        mean_in_psnr = float(np.mean(in_psnr_list))
        psnr_gain = mean_val_psnr - mean_in_psnr

        scheduler.step()
        dur = time.time() - t_ep

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(mean_val_psnr)
        history["input_psnr"].append(mean_in_psnr)

        print(f"\n[*] Epoch {epoch:02d}/{epochs:02d} [{dur:.1f}s] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Input PSNR: {mean_in_psnr:.2f} dB -> Restored PSNR: {mean_val_psnr:.2f} dB (+{psnr_gain:.2f} dB)", flush=True)

        if mean_val_psnr > best_val_psnr:
            best_val_psnr = mean_val_psnr
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": mean_val_psnr,
                "input_psnr": mean_in_psnr
            }, save_model_path)
            print(f"  --> [Saved Best SwinIR] Restored PSNR: {mean_val_psnr:.2f} dB -> {save_model_path.name}\n", flush=True)

    print(f"\n[SwinIR Training Complete] Best Restored PSNR: {best_val_psnr:.2f} dB at Epoch {best_epoch} in {time.time()-t0:.1f}s", flush=True)

    # Save training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history["train_loss"], 'b-', label='Train Loss')
    plt.plot(range(1, epochs + 1), history["val_loss"], 'r--', label='Val Loss')
    plt.title('Radar-SwinIR Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history["input_psnr"], 'k:', label='Degraded Input PSNR')
    plt.plot(range(1, epochs + 1), history["val_psnr"], 'g-', label='Restored PSNR (SwinIR)', lw=2)
    plt.title('Denoising PSNR Improvement')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = config.REPORTS_DIR / "swinir_training_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[SwinIR] Saved training curve to: {curve_path}", flush=True)

    return save_model_path


if __name__ == "__main__":
    run_swinir_training(epochs=3, batch_size=32, lr=5e-4, max_train_pairs=2000)
