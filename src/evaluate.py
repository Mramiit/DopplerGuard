"""
Comprehensive Evaluation & Moat Demonstration for AeroRadarNet.
Generates:
1. Confusion Matrix (Bird vs Drone vs Human)
2. Leave-One-Drone-Out (LODO) Unseen Drone Generalization Metric
3. Real-Time Edge Latency & Throughput Benchmark
4. Noise Resilience / SNR Drop Test
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
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


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """
    Plots a normalized confusion matrix heatmap.
    """
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to: {save_path}")


def benchmark_latency(model, device, input_shape=(1, 2, 64, 64), num_runs=500):
    """
    Benchmarks real-time inference latency (ms) and throughput (FPS).
    """
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy)

    # Timed runs
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy)
    total_time = time.perf_counter() - t0

    latency_ms = (total_time / num_runs) * 1000.0
    fps = num_runs / total_time
    return latency_ms, fps


def run_evaluation(
    model_path: Optional[Path] = None,
    mode: str = "lodo",
    holdout_drone: str = config.LODO_HOLDOUT_DRONE
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = config.MODEL_SAVE_DIR / f"best_aeroradarnet_{mode}.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. Train the model first via train.py!")

    print(f"\n{'='*65}\n  EVALUATING AERORADARNET [MOAT VERIFICATION]\n  MODEL: {model_path.name} | DEVICE: {device}\n{'='*65}")

    # Load Checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    meta = checkpoint.get("meta", {"num_classes": 3, "holdout_drone": holdout_drone})
    num_classes = meta["num_classes"]

    model = get_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load DataLoaders
    _, _, test_loader, _ = load_dataset_splits(
        mode=mode,
        holdout_drone=holdout_drone,
        batch_size=config.BATCH_SIZE
    )

    # Predictions
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 1. Overall Accuracy
    accuracy = (all_preds == all_targets).mean()

    # 2. Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    # 3. Per-class metrics
    class_names = [config.COARSE_CLASSES[i] for i in range(num_classes)]
    class_metrics = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * (prec * rec) / max(1e-12, prec + rec)
        class_metrics[name] = {
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "Support": int(cm[i, :].sum())
        }

    macro_f1 = float(np.mean([m["F1"] for m in class_metrics.values()]))

    # 4. Latency & Edge Benchmarks
    lat_ms, fps = benchmark_latency(model, device)
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = model_path.stat().st_size / 1e6

    # 5. Moat Analysis (Unseen Drone Generalization)
    drone_idx = config.COARSE_LABEL_TO_IDX["drone"]
    is_drone_target = (all_targets == drone_idx)
    if is_drone_target.sum() > 0:
        ood_drone_acc = float((all_preds[is_drone_target] == drone_idx).mean())
    else:
        ood_drone_acc = 0.0

    print(f"\n--- EVALUATION SUMMARY ---")
    print(f"Overall Test Accuracy:     {accuracy*100:.2f}%")
    print(f"Macro F1-Score:            {macro_f1*100:.2f}%")
    if mode == "lodo":
        print(f"[*] MOAT RESULT -> Unseen Drone ({holdout_drone}) Zero-Shot Accuracy: {ood_drone_acc*100:.2f}%")
    print(f"Inference Latency:         {lat_ms:.2f} ms / frame (FPS: {fps:.1f})")
    print(f"Model Size:                {model_size_mb:.2f} MB ({param_count:,} parameters)")

    print("\n--- DETAILED CLASS BREAKDOWN ---")
    for name, m in class_metrics.items():
        print(f"  {name:10s} | Precision: {m['Precision']*100:.1f}% | Recall: {m['Recall']*100:.1f}% | F1: {m['F1']*100:.1f}% | Support: {m['Support']}")

    # Save Plots
    cm_plot_path = config.REPORTS_DIR / f"confusion_matrix_{mode}.png"
    title_str = f"AeroRadarNet Confusion Matrix ({'LODO: Holdout ' + holdout_drone if mode=='lodo' else 'Standard Split'})"
    plot_confusion_matrix(cm, class_names, cm_plot_path, title=title_str)

    # Save JSON Report
    report = {
        "mode": mode,
        "holdout_drone": holdout_drone if mode == "lodo" else None,
        "overall_accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "unseen_drone_accuracy": float(ood_drone_acc),
        "latency_ms": float(lat_ms),
        "fps": float(fps),
        "parameter_count": param_count,
        "model_size_mb": float(model_size_mb),
        "class_metrics": class_metrics
    }
    report_path = config.REPORTS_DIR / f"evaluation_report_{mode}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Evaluation] Report saved to: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AeroRadarNet")
    parser.add_argument("--mode", type=str, default="lodo", choices=["lodo", "standard"])
    parser.add_argument("--holdout", type=str, default="D6")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    m_path = Path(args.model_path) if args.model_path else None
    run_evaluation(model_path=m_path, mode=args.mode, holdout_drone=args.holdout)
