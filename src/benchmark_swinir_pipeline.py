"""
Comprehensive Step 3 Pipeline Benchmark: Radar-SwinIR Range Recovery Evaluation.
Quantifies Drone Detection Sensitivity (Recall), False Alarm Resistance, and Overall
Accuracy across Operational Radar Ranges (70m to 500m) under R^4 physical propagation loss.
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


def run_comprehensive_range_benchmark(
    distances: Optional[List[float]] = None,
    num_drones: int = 200,
    num_birds: int = 150,
    num_humans: int = 50,
    save_plot: bool = True
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if distances is None:
        distances = [70.0, 90.0, 110.0, 130.0, 150.0, 180.0, 200.0, 250.0, 350.0, 500.0]

    # 1. Load AeroRadarNet Classifier (Trained with LODO)
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

    # 3. Load Zero-Shot LODO Held-Out Test Set (Unseen Drone D6)
    _, _, test_loader, _ = load_dataset_splits(mode="lodo", holdout_drone="D6", batch_size=128)
    all_x, all_y = [], []
    for x, y in test_loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
    X_test_all = np.concatenate(all_x, axis=0)
    y_test_all = np.concatenate(all_y, axis=0)

    # Stratified cohort selection
    np.random.seed(42)
    drone_idxs = np.where(y_test_all == 1)[0]
    bird_idxs = np.where(y_test_all == 0)[0]
    human_idxs = np.where(y_test_all == 2)[0]

    sel_drone = np.random.choice(drone_idxs, size=min(num_drones, len(drone_idxs)), replace=False)
    sel_bird = np.random.choice(bird_idxs, size=min(num_birds, len(bird_idxs)), replace=False)
    sel_human = np.random.choice(human_idxs, size=min(num_humans, len(human_idxs)), replace=False)

    sample_indices = np.concatenate([sel_drone, sel_bird, sel_human])
    np.random.shuffle(sample_indices)

    X_clean = X_test_all[sample_indices]
    y_true = y_test_all[sample_indices]
    is_drone = (y_true == 1)
    is_bird = (y_true == 0)

    print(f"\n{'='*75}")
    print(f"  STEP 3: END-TO-END RADAR RANGE RECOVERY BENCHMARK")
    print(f"  Test Cohort: {len(X_clean)} samples (Unseen D6 Drones: {len(sel_drone)}, Birds: {len(sel_bird)}, Humans: {len(sel_human)})")
    print(f"  Evaluating {len(distances)} distance steps from {min(distances):.0f}m to {max(distances):.0f}m")
    print(f"{'='*75}\n", flush=True)

    results = []

    for dist in distances:
        snr_db = distance_to_snr(dist)
        downsample_factor = 2 if dist > 140.0 else 1

        # Simulate physics-based degradation (R^4 power falloff + thermal noise)
        X_deg_list = [simulate_long_range_degradation(x, target_snr_db=snr_db, downsample_factor=downsample_factor) for x in X_clean]
        X_deg = np.array(X_deg_list, dtype=np.float32)

        # Batch inference
        batch_size = 32
        num_batches = int(np.ceil(len(X_deg) / batch_size))

        raw_preds_list = []
        restored_preds_list = []

        with torch.no_grad():
            for b in range(num_batches):
                xb = torch.from_numpy(X_deg[b*batch_size:(b+1)*batch_size]).float().to(device)

                # 1. Unassisted standard radar
                logits_raw = classifier(xb)
                raw_preds_list.append(logits_raw.argmax(dim=1).cpu().numpy())

                # 2. DopplerGuard (Radar-SwinIR super-resolution + classifier)
                x_restored = swinir(xb)
                logits_restored = classifier(x_restored)
                restored_preds_list.append(logits_restored.argmax(dim=1).cpu().numpy())

        preds_raw = np.concatenate(raw_preds_list)
        preds_restored = np.concatenate(restored_preds_list)

        # Metric Calculations
        overall_acc_raw = float((preds_raw == y_true).mean() * 100.0)
        overall_acc_res = float((preds_restored == y_true).mean() * 100.0)

        drone_recall_raw = float((preds_raw[is_drone] == 1).mean() * 100.0)
        drone_recall_res = float((preds_restored[is_drone] == 1).mean() * 100.0)
        drone_gain = drone_recall_res - drone_recall_raw

        bird_spec_raw = float((preds_raw[is_bird] == 0).mean() * 100.0)
        bird_spec_res = float((preds_restored[is_bird] == 0).mean() * 100.0)

        snr_drop = snr_db - 15.0

        print(f"  Range: {dist:4.0f}m (SNR: {snr_db:+5.1f}dB) | "
              f"Drone Recall: Raw {drone_recall_raw:5.1f}% -> SwinIR {drone_recall_res:5.1f}% ({drone_gain:+5.1f}%) | "
              f"Overall Acc: {overall_acc_raw:5.1f}% -> {overall_acc_res:5.1f}%", flush=True)

        results.append({
            "distance_m": dist,
            "snr_db": round(snr_db, 2),
            "snr_drop_db": round(snr_drop, 2),
            "raw_accuracy": round(overall_acc_raw, 2),
            "restored_accuracy": round(overall_acc_res, 2),
            "raw_drone_recall": round(drone_recall_raw, 2),
            "restored_drone_recall": round(drone_recall_res, 2),
            "drone_recall_gain": round(drone_gain, 2),
            "raw_bird_specificity": round(bird_spec_raw, 2),
            "restored_bird_specificity": round(bird_spec_res, 2)
        })

    # Save JSON report
    json_path = config.REPORTS_DIR / "range_recovery_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved Report] Results written to: {json_path}")

    # Plot 4-Panel Publication-Quality Benchmark
    if save_plot:
        dist_vals = [r["distance_m"] for r in results]
        raw_recalls = [r["raw_drone_recall"] for r in results]
        res_recalls = [r["restored_drone_recall"] for r in results]
        raw_accs = [r["raw_accuracy"] for r in results]
        res_accs = [r["restored_accuracy"] for r in results]
        snr_vals = [r["snr_db"] for r in results]
        gains = [r["drone_recall_gain"] for r in results]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Drone Recall / Threat Intercept vs Distance
        ax0 = axes[0, 0]
        ax0.plot(dist_vals, raw_recalls, 'r--o', lw=2.2, label='Standard 77GHz Radar (Raw Echo)')
        ax0.plot(dist_vals, res_recalls, 'g-s', lw=2.5, label='DopplerGuard (Radar-SwinIR + AeroRadarNet)')
        ax0.axhline(80.0, color='gray', linestyle=':', label='Security Intercept Threshold (80%)')
        ax0.set_title('A. Drone Intercept Sensitivity (Recall on Unseen D6 Drone)', fontsize=11, fontweight='bold')
        ax0.set_xlabel('Target Distance (meters)', fontsize=10)
        ax0.set_ylabel('Drone Detection Rate (%)', fontsize=10)
        ax0.grid(True, linestyle='--', alpha=0.4)
        ax0.set_ylim(-5, 105)
        ax0.legend(loc='lower left', fontsize=9)

        for d, r, s in zip(dist_vals, raw_recalls, res_recalls):
            if s > r + 10:
                ax0.annotate(f'+{s-r:.1f}%', (d, s), textcoords="offset points", xytext=(0, 8),
                             ha='center', fontsize=8, color='darkgreen', fontweight='bold')

        # Panel 2: Overall Surveillance Classification Accuracy
        ax1 = axes[0, 1]
        ax1.plot(dist_vals, raw_accs, 'm--o', lw=2.2, label='Standard Radar Accuracy')
        ax1.plot(dist_vals, res_accs, 'b-d', lw=2.5, label='DopplerGuard Restored Accuracy')
        ax1.axhline(80.0, color='gray', linestyle=':', label='Acceptable Airspace Accuracy (80%)')
        ax1.set_title('B. Overall 3-Class Airspace Classification Accuracy', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Target Distance (meters)', fontsize=10)
        ax1.set_ylabel('Overall Accuracy (%)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.set_ylim(20, 105)
        ax1.legend(loc='lower left', fontsize=9)

        # Panel 3: Radar Range Equation Physics (R^4 Power Falloff)
        ax2 = axes[1, 0]
        ax2.plot(dist_vals, snr_vals, 'purple', marker='^', lw=2.0)
        ax2.axhline(0.0, color='red', linestyle='--', label='0 dB SNR (Signal Equals Noise Floor)')
        ax2.set_title('C. Radar Signal-to-Noise Ratio Falloff: ΔSNR = -40·log₁₀(R/R₀)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Target Distance (meters)', fontsize=10)
        ax2.set_ylabel('Expected Radar SNR (dB)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend(loc='lower left', fontsize=9)

        # Panel 4: Drone Recall Recovery Gain Breakdown (Bar Chart)
        ax3 = axes[1, 1]
        colors = ['forestgreen' if g >= 0 else 'firebrick' for g in gains]
        bars = ax3.bar([str(int(d)) + 'm' for d in dist_vals], gains, color=colors, alpha=0.8, edgecolor='black', width=0.6)
        ax3.axhline(0.0, color='black', lw=1.0)
        ax3.set_title('D. Micro-Doppler Harmonic Recovery Gain (Δ Recall %)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Distance Step', fontsize=10)
        ax3.set_ylabel('Detection Gain (% Points)', fontsize=10)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.4)

        for bar, g in zip(bars, gains):
            yval = bar.get_height()
            offset = 1.0 if yval >= 0 else -3.0
            ax3.text(bar.get_x() + bar.get_width() / 2, yval + offset, f"{g:+.1f}%",
                     ha='center', va='bottom' if yval >= 0 else 'top', fontsize=8, fontweight='bold')

        plt.suptitle('DopplerGuard: End-to-End Radar Range Recovery & Threat Detection Benchmark\n'
                     'Overcoming R⁴ Radar Power Falloff via Physics-Guided Swin Transformer Super-Resolution',
                     fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()

        plot_path = config.REPORTS_DIR / "range_recovery_swinir_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Saved Plot] Benchmark visualization written to: {plot_path}", flush=True)

    return {"results": results}


if __name__ == "__main__":
    run_comprehensive_range_benchmark()
