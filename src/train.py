"""
Training Script for AeroRadarNet Drone Radar Classifier.
Includes Class-Weighted Loss, Early Stopping, Checkpointing, and Metric Logging.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src import config
    from src.dataset import load_dataset_splits
    from src.model import get_model
except ImportError:
    import config
    from dataset import load_dataset_splits
    from model import get_model


def calculate_class_weights(train_loader, num_classes=3, device="cpu"):
    """
    Computes inverse class frequencies to balance cross-entropy loss.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, y in train_loader:
        for cls_idx in range(num_classes):
            counts[cls_idx] += (y == cls_idx).sum().item()

    total = counts.sum()
    weights = total / (num_classes * torch.clamp(counts, min=1.0))
    weights = weights / weights.mean()
    print(f"[Training] Class counts in train set: {counts.tolist()}")
    print(f"[Training] Computed class weights: {weights.tolist()}")
    return weights.to(device)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Compute Macro F1
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_pred_total = defaultdict(int)
    for p, t in zip(all_preds, all_targets):
        if p == t:
            class_correct[t] += 1
        class_total[t] += 1
        class_pred_total[p] += 1

    f1s = []
    for c in range(config.INPUT_CHANNELS + 1):  # 3 classes: 0, 1, 2
        precision = class_correct[c] / max(1, class_pred_total[c])
        recall = class_correct[c] / max(1, class_total[c])
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    return epoch_loss, epoch_acc, macro_f1


def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss Curve
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', lw=2)
    ax1.plot(epochs, history['val_loss'], 'r--', label='Val Loss', lw=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (CrossEntropy)')
    ax1.set_title('Training & Validation Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy / F1 Curve
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', lw=2)
    ax2.plot(epochs, history['val_acc'], 'r--', label='Val Acc', lw=2)
    ax2.plot(epochs, history['val_f1'], 'g-.', label='Val Macro F1', lw=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Training Accuracy & Validation F1')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Training] Saved training curves plot to: {save_path}")


def run_training(
    mode: str = "lodo",
    holdout_drone: str = config.LODO_HOLDOUT_DRONE,
    epochs: int = 12,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\n  STARTING RADAR CNN TRAINING ON DEVICE: {device}\n  MODE: {mode.upper()} | HOLDOUT DRONE: {holdout_drone}\n{'='*65}")

    # Load Splits
    train_loader, val_loader, test_loader, meta = load_dataset_splits(
        mode=mode,
        holdout_drone=holdout_drone,
        batch_size=batch_size
    )

    # Model & Optimization
    model = get_model(num_classes=meta["num_classes"], device=device)
    class_weights = calculate_class_weights(train_loader, num_classes=meta["num_classes"], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    save_model_path = config.MODEL_SAVE_DIR / f"best_aeroradarnet_{mode}.pth"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        dur = time.time() - t_epoch_start
        print(f"Epoch {epoch:02d}/{epochs:02d} [{dur:.1f}s] | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "meta": meta
            }, save_model_path)
            print(f"  --> [Saved Best Model] Val F1: {val_f1*100:.2f}% -> {save_model_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"[Training] Early stopping triggered at epoch {epoch} (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
                break

    total_time = time.time() - t_start
    print(f"\n[Training Complete] Total time: {total_time/60:.2f} mins. Best Val F1: {best_val_f1*100:.2f}% at epoch {best_epoch}")

    # Save history and curves
    history_path = config.REPORTS_DIR / f"training_history_{mode}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_path = config.REPORTS_DIR / f"training_curves_{mode}.png"
    plot_training_curves(history, plot_path)

    return save_model_path, test_loader, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AeroRadarNet")
    parser.add_argument("--mode", type=str, default="lodo", choices=["lodo", "standard"])
    parser.add_argument("--holdout", type=str, default="D6")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_training(
        mode=args.mode,
        holdout_drone=args.holdout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
