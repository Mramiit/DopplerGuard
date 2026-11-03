"""
Radar Range & SNR Degradation Engine.
Simulates long-range propagation effects according to the Radar Range Equation (R^4 power drop),
thermal noise injection, and coarse spatial/frequency downsampling.
Creates paired (Low-Res Noisy Input, High-Res Clean Target) datasets for SwinIR training.
"""

import sys
import time
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

try:
    from src import config
except ImportError:
    import config

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src.model import get_model
    from src.dataset import load_dataset_splits
except ImportError:
    from model import get_model
    from dataset import load_dataset_splits


def simulate_long_range_degradation(
    clean_sample: np.ndarray,
    target_snr_db: float = 0.0,
    downsample_factor: int = 2
) -> np.ndarray:
    """
    Simulates long-range radar degradation on a 2-channel tensor [2, H, W]:
    1. Attenuates signal and injects additive Gaussian noise to achieve target_snr_db.
    2. Downsamples spatial/Doppler resolution by downsample_factor, then upsamples back
       to simulate low-resolution distant radar chirps.

    Args:
        clean_sample: np.ndarray of shape (2, H, W) in [0, 1] range.
        target_snr_db: Signal-to-Noise Ratio in dB. Lower = further distance / weaker echo.
        downsample_factor: Resolution degradation (e.g. 2x, 4x).

    Returns:
        np.ndarray of shape (2, H, W), degraded but normalized back to [0, 1].
    """
    degraded = clean_sample.copy()
    c, h, w = degraded.shape

    # 1. SNR Noise Injection
    # Linear SNR: SNR_lin = 10^(SNR_db / 10) = P_signal / P_noise
    snr_lin = 10.0 ** (target_snr_db / 10.0)
    sig_power = np.mean(degraded ** 2)
    noise_power = sig_power / max(1e-12, snr_lin)
    noise_sigma = np.sqrt(noise_power)

    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=degraded.shape)
    noisy = degraded + noise

    # 2. Resolution Downsampling & Re-upsampling (Simulating distant coarse sampling)
    if downsample_factor > 1:
        tensor = torch.from_numpy(noisy).float().unsqueeze(0)  # [1, 2, H, W]
        down_h = max(8, h // downsample_factor)
        down_w = max(8, w // downsample_factor)

        # Downsample (blur / loss of high-frequency rotor micro-harmonics)
        low_res = F.interpolate(tensor, size=(down_h, down_w), mode='area')
        # Upsample back to original grid for SwinIR input
        re_up = F.interpolate(low_res, size=(h, w), mode='bilinear', align_corners=False)
        noisy = re_up.squeeze(0).numpy()

    # Dynamic range clipping & per-channel normalization to [0, 1]
    for c_i in range(c):
        c_min, c_max = noisy[c_i].min(), noisy[c_i].max()
        noisy[c_i] = (noisy[c_i] - c_min) / (c_max - c_min + 1e-6)
    return noisy.astype(np.float32)


def distance_to_snr(
    range_meters: float,
    baseline_range_meters: float = 70.0,
    baseline_snr_db: float = 15.0
) -> float:
    """
    Computes expected SNR (dB) at a given distance using the Radar Range Equation:
    Delta SNR = -40 * log10( R / R_0 )
    """
    if range_meters <= baseline_range_meters:
        return baseline_snr_db
    delta_snr = -40.0 * np.log10(range_meters / baseline_range_meters)
    return baseline_snr_db + delta_snr


def generate_paired_sr_dataset(
    input_npz_path: Optional[Path] = None,
    output_npz_path: Optional[Path] = None,
    snr_range_db: Tuple[float, float] = (-10.0, 5.0),
    downsample_factor: int = 2
) -> Path:
    """
    Generates paired (Low-Res Noisy Input, High-Res Clean Ground Truth)
    radar feature maps to train SwinIR.
    """
    if input_npz_path is None:
        input_npz_path = config.PROCESSED_DIR / "radar_balanced_dataset.npz"
    if output_npz_path is None:
        output_npz_path = config.PROCESSED_DIR / "radar_sr_paired_dataset.npz"

    print(f"\n[SwinIR Prep] Generating Paired Super-Resolution Dataset...")
    print(f"  Source: {input_npz_path.name}")
    print(f"  Target SNR range: {snr_range_db[0]} dB to {snr_range_db[1]} dB")
    print(f"  Downsample factor: {downsample_factor}x")

    data = np.load(input_npz_path, allow_pickle=True)
    X_clean = data["X"]
    y_coarse = data["y_coarse"]
    drone_models = data["drone_models"]
    recording_ids = data["recording_ids"]

    num_samples = len(X_clean)
    X_degraded = np.empty_like(X_clean)

    t0 = time.time()
    for i in range(num_samples):
        # Randomize SNR across the target degradation range
        sample_snr = np.random.uniform(snr_range_db[0], snr_range_db[1])
        X_degraded[i] = simulate_long_range_degradation(
            X_clean[i],
            target_snr_db=sample_snr,
            downsample_factor=downsample_factor
        )

        if (i + 1) % 2500 == 0 or (i + 1) == num_samples:
            print(f"  Processed {i + 1}/{num_samples} samples ({time.time()-t0:.1f}s)")

    print(f"[SwinIR Prep] Saving paired dataset to: {output_npz_path}...")
    np.savez_compressed(
        output_npz_path,
        X_clean=X_clean,            # Ground truth for SwinIR target
        X_degraded=X_degraded,      # Degraded input for SwinIR
        y_coarse=y_coarse,
        drone_models=drone_models,
        recording_ids=recording_ids
    )
    print(f"[SwinIR Prep] Done! Saved {num_samples} pairs in {time.time()-t0:.2f}s ({output_npz_path.stat().st_size / 1e6:.1f} MB)")
    return output_npz_path


def benchmark_model_across_ranges(
    model_path: Optional[Path] = None,
    test_ranges: List[float] = [70, 120, 200, 350, 500, 750, 1000]
) -> Dict[str, Any]:
    """
    Evaluates current trained AeroRadarNet model against increasing simulated distances.
    Plots the baseline 'Accuracy vs Range' degradation curve.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = config.MODEL_SAVE_DIR / "best_aeroradarnet_lodo.pth"

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = get_model(num_classes=3, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load clean test set
    _, _, test_loader, meta = load_dataset_splits(mode="lodo", holdout_drone="D6", batch_size=128)

    # Collect all test tensors
    all_x = []
    all_y = []
    for x, y in test_loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
    X_test = np.concatenate(all_x, axis=0)
    y_test = np.concatenate(all_y, axis=0)

    print(f"\n{'='*65}")
    print(f"  RUNNING RADAR RANGE STRESS TEST: {len(X_test)} TEST SAMPLES")
    print(f"{'='*65}")

    results = {
        "ranges_m": test_ranges,
        "snr_db": [],
        "accuracy": [],
        "drone_accuracy": []
    }

    drone_idx = config.COARSE_LABEL_TO_IDX["drone"]
    is_drone = (y_test == drone_idx)

    for r_m in test_ranges:
        snr = distance_to_snr(r_m)
        downsample = 1 if r_m <= 120 else (2 if r_m <= 500 else 4)

        # Degrade test batch
        degraded_x = np.empty_like(X_test)
        for i in range(len(X_test)):
            degraded_x[i] = simulate_long_range_degradation(
                X_test[i],
                target_snr_db=snr,
                downsample_factor=downsample
            )

        # Predict
        tensor_x = torch.from_numpy(degraded_x).float().to(device)
        tensor_y = torch.from_numpy(y_test).long().to(device)

        with torch.no_grad():
            logits = model(tensor_x)
            preds = torch.argmax(logits, dim=1)
            acc = float((preds == tensor_y).float().mean().item())
            drone_acc = float((preds[is_drone] == drone_idx).float().mean().item())

        results["snr_db"].append(float(snr))
        results["accuracy"].append(acc)
        results["drone_accuracy"].append(drone_acc)

        print(f"Distance: {r_m:4.0f} m | Expected SNR: {snr:5.1f} dB | Downsample: {downsample}x | "
              f"Overall Acc: {acc*100:5.1f}% | Unseen Drone Acc: {drone_acc*100:5.1f}%")

    # Plot Degradation Curve
    plot_path = config.REPORTS_DIR / "range_degradation_benchmark.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Accuracy vs Distance
    ax1.plot(test_ranges, [a * 100 for a in results["accuracy"]], 'ro-', lw=2, label='Overall Accuracy')
    ax1.plot(test_ranges, [d * 100 for d in results["drone_accuracy"]], 'b^--', lw=2, label='Unseen Drone Accuracy')
    ax1.set_xlabel('Simulated Target Distance (meters)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Baseline Model Performance vs Range (R^4 Attenuation)', fontsize=12, fontweight='bold')
    ax1.axvline(120, color='gray', linestyle=':', label='Sensor Physical Limit (~120m)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    ax1.legend()

    # Plot 2: Accuracy vs SNR
    ax2.plot(results["snr_db"], [a * 100 for a in results["accuracy"]], 'go-', lw=2, label='Overall Accuracy')
    ax2.plot(results["snr_db"], [d * 100 for d in results["drone_accuracy"]], 'm^--', lw=2, label='Unseen Drone Accuracy')
    ax2.set_xlabel('Radar Signal-to-Noise Ratio (SNR in dB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Resilience Curve across SNR Drop', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()  # High SNR on left, low SNR on right
    ax2.set_ylim(0, 105)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[Benchmark] Saved Range Degradation Curve to: {plot_path}")

    # Save JSON Report
    json_path = config.REPORTS_DIR / "range_degradation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Benchmark] Saved results JSON to: {json_path}")

    return results


def visualize_range_degradation(sample_idx: int = 15):
    """
    Plots a visual comparison of a Clean Short-Range Radar Signature vs.
    Long-Range Degraded Signatures at 200m, 500m, and 1000m.
    """
    data_path = config.PROCESSED_DIR / "radar_balanced_dataset.npz"
    data = np.load(data_path, allow_pickle=True)
    clean = data["X"][sample_idx]
    cls_idx = data["y_coarse"][sample_idx]
    target_name = config.COARSE_CLASSES[int(cls_idx)]

    distances = [70, 250, 500, 1000]
    fig, axes = plt.subplots(len(distances), 2, figsize=(10, 3.2 * len(distances)))

    for i, d in enumerate(distances):
        snr = distance_to_snr(d)
        ds = 1 if d <= 120 else (2 if d <= 500 else 4)
        deg = simulate_long_range_degradation(clean, target_snr_db=snr, downsample_factor=ds)

        # Range-Doppler Map
        im0 = axes[i, 0].imshow(deg[0], aspect='auto', cmap='magma', origin='lower')
        axes[i, 0].set_title(f"Range-Doppler @ {d}m (SNR: {snr:.1f} dB)", fontsize=10, fontweight='bold')
        axes[i, 0].set_ylabel("Range Bins")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Micro-Doppler STFT
        im1 = axes[i, 1].imshow(deg[1], aspect='auto', cmap='viridis', origin='lower')
        axes[i, 1].set_title(f"Micro-Doppler @ {d}m ({target_name})", fontsize=10, fontweight='bold')
        axes[i, 1].set_ylabel("Frequency Bins")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    viz_path = config.REPORTS_DIR / "range_degradation_visual_comparison.png"
    plt.savefig(viz_path, dpi=300)
    plt.close()
    print(f"[Visual] Saved visual comparison plot to: {viz_path}")


if __name__ == "__main__":
    # 1. Run Range Benchmark to quantify accuracy drop
    benchmark_model_across_ranges()

    # 2. Generate Visual Comparison of Clean vs Degraded Spectrograms
    visualize_range_degradation(sample_idx=25)

    # 3. Generate Paired Dataset for SwinIR
    generate_paired_sr_dataset()
