"""
End-to-End Range Recovery Benchmark: Raw Low-SNR vs Radar-SwinIR Restored.
Evaluates AeroRadarNet classification accuracy across distance with and without SwinIR.
"""

import sys
import time
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional
import numpy as np
import torch

try:
    from src import config
    from src.model import AeroRadarNet
    from src.swinir_model import get_radar_swinir
    from src.range_degrader import simulate_long_range_degradation, distance_to_snr
    from src.dataset import load_dataset_splits
except ImportError:
    import config
    from model import AeroRadarNet
    from swinir_model import get_radar_swinir
    from range_degrader import simulate_long_range_degradation, distance_to_snr
    from dataset import load_dataset_splits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_pipeline_range_benchmark(
    distances: Optional[List[float]] = None,
    samples_per_distance: int = 350,
    save_plot: bool = True
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if distances is None:
        distances = [70.0, 120.0, 200.0, 350.0, 500.0, 750.0, 1000.0]

    # 1. Load AeroRadarNet Classifier
    classifier_path = config.MODEL_SAVE_DIR / "best_aeroradarnet_lodo.pth"
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier model not found at {classifier_path}")

    classifier = AeroRadarNet(in_channels=2, num_classes=3).to(device)
    clf_ckpt = torch.load(classifier_path, map_location=device, weights_only=False)
    classifier.load_state_dict(clf_ckpt["model_state_dict"])
    classifier.eval()

    # 2. Load Radar-SwinIR Restorer
    swinir_path = config.MODEL_SAVE_DIR / "best_radar_swinir.pth"
    if not swinir_path.exists():
        raise FileNotFoundError(f"SwinIR model not found at {swinir_path}")

    swinir = get_radar_swinir(device=str(device))
    swin_ckpt = torch.load(swinir_path, map_location=device, weights_only=False)
    swinir.load_state_dict(swin_ckpt["model_state_dict"])
    swinir.eval()

    # 3. Load LODO Test Set (Representative Unseen Drone D6 + Birds + Humans)
    _, _, test_loader, _ = load_dataset_splits(mode="lodo", holdout_drone="D6", batch_size=128)
    all_x = []
    all_y = []
    for x, y in test_loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
    X_test_all = np.concatenate(all_x, axis=0)
    y_test_all = np.concatenate(all_y, axis=0)

    # Subsample stratified 400 test samples for fast benchmark
    np.random.seed(42)
    sample_indices = []
    per_class = {0: 100, 1: 250, 2: 50}
    for c_id, count in per_class.items():
        c_idxs = np.where(y_test_all == c_id)[0]
        chosen = np.random.choice(c_idxs, size=min(count, len(c_idxs)), replace=False)
        sample_indices.extend(chosen)

    sample_indices = np.array(sample_indices)
    X_clean = X_test_all[sample_indices]
    y_true = y_test_all[sample_indices]

    print(f"\n{'='*70}\n  END-TO-END RADAR RANGE RECOVERY BENCHMARK\n  Evaluating {len(distances)} distance steps | Test Samples: {len(X_clean)} (Birds: 100, Unseen D6 Drones: 250, Humans: 50)\n{'='*70}", flush=True)

    results = []

    for dist in distances:
        snr_db = distance_to_snr(dist)
        downsample_factor = 2 if dist > 150.0 else 1

        # Degrade all samples for current distance
        X_deg_list = [simulate_long_range_degradation(x, target_snr_db=snr_db, downsample_factor=downsample_factor) for x in X_clean]
        X_deg = np.array(X_deg_list, dtype=np.float32)

        # A. Unrestored Baseline Accuracy
        correct_raw = 0
        batch_size = 32
        num_batches = int(np.ceil(len(X_deg) / batch_size))

        with torch.no_grad():
            for b in range(num_batches):
                xb = torch.from_numpy(X_deg[b*batch_size:(b+1)*batch_size]).float().to(device)
                yb = torch.from_numpy(y_true[b*batch_size:(b+1)*batch_size]).long().to(device)
                preds = classifier(xb).argmax(dim=1)
                correct_raw += (preds == yb).sum().item()

        acc_raw = (correct_raw / len(y_true)) * 100.0

        # B. SwinIR Restored Accuracy (Degraded -> SwinIR -> Classifier)
        correct_restored = 0
        with torch.no_grad():
            for b in range(num_batches):
                xb = torch.from_numpy(X_deg[b*batch_size:(b+1)*batch_size]).float().to(device)
                yb = torch.from_numpy(y_true[b*batch_size:(b+1)*batch_size]).long().to(device)

                # Pass through SwinIR first
                x_restored = swinir(xb)

                # Pass restored features to AeroRadarNet
                preds_restored = classifier(x_restored).argmax(dim=1)
                correct_restored += (preds_restored == yb).sum().item()

        acc_restored = (correct_restored / len(y_true)) * 100.0
        acc_boost = acc_restored - acc_raw
        snr_drop = snr_db - 15.0  # drop relative to 70m baseline

        print(f"  Range: {dist:5.0f} m (SNR: {snr_db:+5.1f} dB) | "
              f"Raw Acc: {acc_raw:5.1f}% -> SwinIR Restored: {acc_restored:5.1f}% "
              f"({acc_boost:+5.1f}%)", flush=True)

        results.append({
            "distance_m": dist,
            "snr_db": round(snr_db, 2),
            "snr_drop_db": round(snr_drop, 2),
            "raw_accuracy": round(acc_raw, 2),
            "restored_accuracy": round(acc_restored, 2),
            "accuracy_gain": round(acc_boost, 2)
        })

    # Save Results
    json_path = config.REPORTS_DIR / "range_recovery_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot Comparison Curves
    if save_plot:
        dist_vals = [r["distance_m"] for r in results]
        raw_accs = [r["raw_accuracy"] for r in results]
        res_accs = [r["restored_accuracy"] for r in results]

        plt.figure(figsize=(9, 5.5))
        plt.plot(dist_vals, raw_accs, 'r--o', lw=2.2, label='Standard Radar (Unassisted Baseline)')
        plt.plot(dist_vals, res_accs, 'g-s', lw=2.5, label='DopplerGuard (Radar-SwinIR + AeroRadarNet)')

        plt.axhline(80.0, color='gray', linestyle=':', label='Operational Threshold (80%)')

        plt.title('Radar Range Extension: Classification Accuracy vs Distance\n(Physics-based R⁴ Power Falloff & Micro-Doppler Denoising)',
                  fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Target Detection Distance (meters)', fontsize=11)
        plt.ylabel('Classification Accuracy (%)', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(15, 105)
        plt.legend(fontsize=10, loc='lower left')

        # Highlight effective operational distance gain
        for d, r, s in zip(dist_vals, raw_accs, res_accs):
            if s > r + 10:
                plt.annotate(f'+{s-r:.1f}%',
                             (d, s),
                             textcoords="offset points",
                             xytext=(0, 10),
                             ha='center',
                             fontsize=9,
                             color='darkgreen',
                             fontweight='bold')

        plt.tight_layout()
        plot_path = config.REPORTS_DIR / "range_recovery_swinir_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[Plot Saved] Range extension benchmark plot: {plot_path}", flush=True)

    return {"results": results}


if __name__ == "__main__":
    run_pipeline_range_benchmark()
