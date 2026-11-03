"""
Radar Signal Preprocessing & Feature Extraction.
Transforms raw 77 GHz FMCW complex chirps into dual-channel
Range-Doppler and Micro-Doppler feature maps.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import stft
from pathlib import Path
from typing import Tuple, Dict, Any, List
import time

try:
    from src import config
except ImportError:
    import config


def raw_to_dual_channel(
    sample_vector: np.ndarray,
    target_size: Tuple[int, int] = config.FEATURE_SIZE
) -> np.ndarray:
    """
    Transforms a single raw complex sample vector (1280,) into a dual-channel 2D tensor:
    - Channel 0: 2D Range-Doppler Map (Range vs Velocity)
    - Channel 1: Micro-Doppler Spectrogram (Frequency vs Time cadence)

    Args:
        sample_vector: 1D complex array of length 1280 (5 range bins x 256 azimuth sweeps)
        target_size: Output 2D spatial dimensions (H, W), default (64, 64)

    Returns:
        np.ndarray of shape (2, H, W), float32, normalized to [0, 1]
    """
    # Reshape into (5 range bins, 256 azimuth pulses)
    matrix = sample_vector.reshape(config.RANGE_BINS, config.AZIMUTH_PULSES)

    # 1. Range-Doppler Map (FFT along azimuth/slow-time axis)
    rd = np.fft.fftshift(np.fft.fft(matrix, axis=1), axes=1)
    rd_db = 20.0 * np.log10(np.abs(rd) + 1e-6)
    rd_min, rd_max = rd_db.min(), rd_db.max()
    rd_norm = (rd_db - rd_min) / (rd_max - rd_min + 1e-6)

    # 2. Micro-Doppler STFT on the center target range bin (index 2)
    center_pulse_series = matrix[2, :]
    _, _, zxx = stft(
        center_pulse_series,
        nperseg=64,
        noverlap=56,
        return_onesided=False
    )
    stft_centered = np.fft.fftshift(zxx, axes=0)
    stft_db = 20.0 * np.log10(np.abs(stft_centered) + 1e-6)
    stft_min, stft_max = stft_db.min(), stft_db.max()
    stft_norm = (stft_db - stft_min) / (stft_max - stft_min + 1e-6)

    # Resize to standard CNN input dimension (H, W) using bilinear interpolation
    rd_tensor = torch.from_numpy(rd_norm).float().unsqueeze(0).unsqueeze(0)
    stft_tensor = torch.from_numpy(stft_norm).float().unsqueeze(0).unsqueeze(0)

    rd_resized = F.interpolate(rd_tensor, size=target_size, mode='bilinear', align_corners=False)
    stft_resized = F.interpolate(stft_tensor, size=target_size, mode='bilinear', align_corners=False)

    dual_channel = torch.cat([rd_resized, stft_resized], dim=1).squeeze(0).numpy()
    return dual_channel.astype(np.float32)


def parse_target_labels(raw_label_val: Any) -> Tuple[str, str, int]:
    """
    Maps raw recording labels to standardized fine and coarse classes.
    """
    if hasattr(raw_label_val, 'item'):
        lbl = str(raw_label_val.item())
    elif isinstance(raw_label_val, (list, np.ndarray)):
        lbl = str(raw_label_val[0])
    else:
        lbl = str(raw_label_val)
    lbl = lbl.strip()

    # Identify fine class and coarse class
    if lbl in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        fine = lbl
        coarse = 'drone'
        coarse_idx = 1
    elif 'human' in lbl.lower():
        fine = 'Human'
        coarse = 'human'
        coarse_idx = 2
    elif 'cr' in lbl.lower():
        fine = 'CR'
        coarse = 'clutter'
        coarse_idx = 3
    else:
        fine = 'Bird'
        coarse = 'bird'
        coarse_idx = 0

    return lbl, fine, coarse_idx


def build_preprocessed_dataset(
    max_samples_per_recording: int = 150,
    output_filename: str = "radar_balanced_dataset.npz"
) -> Path:
    """
    Builds a preprocessed feature dataset from the raw 1.55GB FMCW radar file.
    Uses balanced sampling across all 130 recordings to ensure equal representation
    and avoid drone-overfitting, while keeping disk size small (~150MB) and training ultra-fast.
    """
    raw_path = config.RAW_DATA_PATH
    out_path = config.PROCESSED_DIR / output_filename

    if out_path.exists():
        print(f"[Preprocessing] Preprocessed dataset already exists at: {out_path}")
        return out_path

    print(f"[Preprocessing] Loading raw dataset from: {raw_path}...")
    t0 = time.time()
    raw_data = np.load(raw_path, allow_pickle=True)
    num_recordings = raw_data.shape[0]
    print(f"[Preprocessing] Loaded {num_recordings} recordings in {time.time()-t0:.2f}s")

    features_list = []
    coarse_labels = []
    fine_labels = []
    recording_ids = []
    drone_models = []

    print("[Preprocessing] Extracting dual-channel Doppler features...")
    total_extracted = 0

    for r_idx in range(num_recordings):
        raw_lbl = raw_data[r_idx, 0]
        raw_name, fine_cls, coarse_idx = parse_target_labels(raw_lbl)

        # Skip corner reflectors (CR) for bird/drone/human classification
        if fine_cls == 'CR':
            continue

        mat = raw_data[r_idx, 1]  # shape: (1280, N_sweeps)
        n_sweeps = mat.shape[1]

        # Uniformly sample up to max_samples_per_recording sweeps
        step = max(1, n_sweeps // max_samples_per_recording)
        selected_sweeps = list(range(0, n_sweeps, step))[:max_samples_per_recording]

        for s_idx in selected_sweeps:
            vec = mat[:, s_idx]
            feat = raw_to_dual_channel(vec)

            features_list.append(feat)
            coarse_labels.append(coarse_idx)
            fine_labels.append(fine_cls)
            recording_ids.append(r_idx)
            drone_models.append(fine_cls if fine_cls.startswith('D') else 'None')
            total_extracted += 1

        if (r_idx + 1) % 25 == 0 or (r_idx + 1) == num_recordings:
            print(f"  Processed {r_idx + 1}/{num_recordings} recordings ({total_extracted} samples extracted)")

    X = np.stack(features_list, axis=0)  # Shape: (N, 2, 64, 64)
    y_coarse = np.array(coarse_labels, dtype=np.int64)
    y_fine = np.array(fine_labels)
    rec_ids = np.array(recording_ids, dtype=np.int32)
    drone_tags = np.array(drone_models)

    print(f"[Preprocessing] Saving {X.shape[0]} samples to {out_path}...")
    np.savez_compressed(
        out_path,
        X=X,
        y_coarse=y_coarse,
        y_fine=y_fine,
        recording_ids=rec_ids,
        drone_models=drone_tags
    )
    print(f"[Preprocessing] Completed in {time.time()-t0:.2f}s. File size: {out_path.stat().st_size / 1e6:.2f} MB")
    return out_path


if __name__ == "__main__":
    build_preprocessed_dataset()
