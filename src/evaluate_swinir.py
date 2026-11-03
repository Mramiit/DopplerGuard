"""
Evaluation and Visualization Script for Radar-SwinIR.
Compares degraded low-SNR inputs against SwinIR-restored outputs and clean ground truth.
Saves before-and-after visual inspection heatmaps.
"""

import sys
import time
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch

try:
    from src import config
    from src.swinir_model import get_radar_swinir
    from src.train_swinir import calculate_psnr, calculate_batch_psnr
except ImportError:
    import config
    from swinir_model import get_radar_swinir
    from train_swinir import calculate_psnr, calculate_batch_psnr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate_radar_swinir(
    model_path: Optional[Path] = None,
    paired_npz_path: Optional[Path] = None,
    num_test_samples: int = 1000
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = config.MODEL_SAVE_DIR / "best_radar_swinir.pth"
    if paired_npz_path is None:
        paired_npz_path = config.PROCESSED_DIR / "radar_sr_paired_dataset.npz"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained SwinIR model not found at {model_path}!")

    print(f"\n{'='*65}\n  EVALUATING RADAR-SWINIR RESTORATION\n  MODEL: {model_path.name} | SAMPLES: {num_test_samples}\n{'='*65}")

    data = np.load(paired_npz_path, allow_pickle=True)
    X_deg = data["X_degraded"][:num_test_samples]
    X_clean = data["X_clean"][:num_test_samples]
    y_coarse = data["y_coarse"][:num_test_samples]

    model = get_radar_swinir(device=str(device))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    in_psnrs = []
    out_psnrs = []
    restored_samples = []

    batch_size = 32
    num_batches = int(np.ceil(len(X_deg) / batch_size))

    with torch.no_grad():
        for b in range(num_batches):
            x_b = torch.from_numpy(X_deg[b*batch_size:(b+1)*batch_size]).float().to(device)
            target_b = torch.from_numpy(X_clean[b*batch_size:(b+1)*batch_size]).float().to(device)

            out_b = model(x_b)

            out_psnrs.append(calculate_batch_psnr(out_b, target_b))
            in_psnrs.append(calculate_batch_psnr(x_b, target_b))

            if b == 0:
                restored_samples.append(out_b.cpu().numpy())

    restored_all = restored_samples[0]
    mean_in_psnr = float(np.mean(in_psnrs))
    mean_out_psnr = float(np.mean(out_psnrs))
    psnr_gain = mean_out_psnr - mean_in_psnr

    print(f"\n--- RADAR-SWINIR RESTORATION RESULTS ---")
    print(f"Degraded Input PSNR:       {mean_in_psnr:.2f} dB")
    print(f"SwinIR Restored PSNR:      {mean_out_psnr:.2f} dB")
    print(f"Average SNR Quality Gain:  +{psnr_gain:.2f} dB")

    # Plot Visual Before-and-After Heatmaps for 3 Representative Samples
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    sample_indices = [3, 12, 25]
    class_names = [config.COARSE_CLASSES[int(y_coarse[idx])] for idx in sample_indices]

    for row, (idx, cls_name) in enumerate(zip(sample_indices, class_names)):
        deg_map = X_deg[idx, 1]        # Micro-Doppler channel of degraded input
        res_map = restored_all[idx, 1] # Micro-Doppler channel of restored output
        clean_map = X_clean[idx, 1]    # Clean ground truth

        in_p = calculate_psnr(torch.tensor(X_deg[idx]), torch.tensor(X_clean[idx]))
        out_p = calculate_psnr(torch.tensor(restored_all[idx]), torch.tensor(X_clean[idx]))

        # Column 1: Degraded Input
        im0 = axes[row, 0].imshow(deg_map, aspect='auto', cmap='viridis', origin='lower')
        axes[row, 0].set_title(f"Input: Noisy/Low-Res ({cls_name})\nPSNR: {in_p:.1f} dB", fontsize=10, fontweight='bold')
        axes[row, 0].set_ylabel("Doppler Freq")

        # Column 2: SwinIR Restored
        im1 = axes[row, 1].imshow(res_map, aspect='auto', cmap='viridis', origin='lower')
        axes[row, 1].set_title(f"SwinIR Restored Output\nPSNR: {out_p:.1f} dB (+{out_p - in_p:.1f} dB)", fontsize=10, fontweight='bold', color='darkgreen')

        # Column 3: Ground Truth
        im2 = axes[row, 2].imshow(clean_map, aspect='auto', cmap='viridis', origin='lower')
        axes[row, 2].set_title(f"Clean Ground Truth Target\n(Pristine Echo)", fontsize=10, fontweight='bold')

        if row == 2:
            axes[row, 0].set_xlabel("Dwell Time Sweeps")
            axes[row, 1].set_xlabel("Dwell Time Sweeps")
            axes[row, 2].set_xlabel("Dwell Time Sweeps")

    plt.tight_layout()
    viz_path = config.REPORTS_DIR / "swinir_restoration_comparison.png"
    plt.savefig(viz_path, dpi=300)
    plt.close()
    print(f"[Visual] Saved Before-and-After comparison plot to: {viz_path}")

    # Save JSON report
    report = {
        "mean_input_psnr_db": mean_in_psnr,
        "mean_restored_psnr_db": mean_out_psnr,
        "psnr_gain_db": psnr_gain,
        "test_samples": num_test_samples
    }
    report_path = config.REPORTS_DIR / "swinir_evaluation_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved evaluation JSON to: {report_path}")

    return report


if __name__ == "__main__":
    evaluate_radar_swinir()
