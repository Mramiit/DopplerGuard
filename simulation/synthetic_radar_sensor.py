"""
Synthetic 77 GHz FMCW Radar Sensor Simulation & Gazebo Bridge.
Generates realistic Range-Doppler micro-Doppler signatures from 3D drone kinematics
and runs them through the DopplerGuard (Radar-SwinIR + AeroRadarNet) inference engine.
"""

import math
import time
from typing import Dict, Tuple, Optional
import numpy as np
import torch

try:
    from src import config
    from src.model import AeroRadarNet
    from src.swinir_model import get_radar_swinir
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.model import AeroRadarNet
    from src.swinir_model import get_radar_swinir


class SyntheticRadarSensor:
    """
    Simulates a 77 GHz FMCW radar node positioned at (x, y, z).
    Calculates physical micro-Doppler signatures from target kinematics.
    """
    def __init__(self, radar_pos: Tuple[float, float, float] = (0.0, 0.0, 5.0)):
        self.radar_pos = np.array(radar_pos, dtype=np.float32)
        self.fc = config.RADAR_FC
        self.c = config.C
        self.wavelength = config.WAVELENGTH
        self.prf = config.PRF
        self.num_sweeps = config.AZIMUTH_PULSES

        # Load models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = AeroRadarNet(in_channels=2, num_classes=3).to(self.device)
        clf_ckpt = torch.load(config.MODEL_SAVE_DIR / "best_aeroradarnet_lodo.pth", map_location=self.device, weights_only=False)
        self.classifier.load_state_dict(clf_ckpt["model_state_dict"])
        self.classifier.eval()

        swinir_path = config.MODEL_SAVE_DIR / "best_radar_swinir.pth"
        if swinir_path.exists():
            self.swinir = get_radar_swinir(device=str(self.device))
            swin_ckpt = torch.load(swinir_path, map_location=self.device, weights_only=False)
            self.swinir.load_state_dict(swin_ckpt["model_state_dict"])
            self.swinir.eval()
            print("[Radar Sensor] Loaded SwinIR Range-Extension Model.")
        else:
            self.swinir = None
            print("[Radar Sensor] SwinIR model not found, running unassisted baseline.")

        # Pre-cache authentic radar target samples for realistic simulation
        self.dataset_path = config.PROCESSED_DIR / "radar_balanced_dataset.npz"
        self.cached_samples = {}
        if self.dataset_path.exists():
            data = np.load(self.dataset_path, allow_pickle=True)
            for c_name, c_idx in config.COARSE_LABEL_TO_IDX.items():
                idxs = np.where(data["y_coarse"] == c_idx)[0]
                if len(idxs) > 0:
                    self.cached_samples[c_name] = data["X"][idxs[:50]]

    def compute_kinematics(self, target_pos: np.ndarray, target_vel: np.ndarray) -> Tuple[float, float]:
        """
        Calculates slant range R (meters) and radial velocity Vr (m/s).
        """
        disp = target_pos - self.radar_pos
        range_m = float(np.linalg.norm(disp))
        range_m = max(range_m, 1.0)
        unit_vec = disp / range_m
        radial_vel = float(np.dot(target_vel, unit_vec))
        return range_m, radial_vel

    def get_radar_echo(
        self,
        range_m: float,
        radial_vel: float,
        target_type: str = "drone",
        sample_idx: int = 0
    ) -> np.ndarray:
        """
        Returns a physical dual-channel [2, 64, 64] radar tensor:
        Uses calibrated 77 GHz FMCW radar recordings degraded by physical R^4 power falloff
        and thermal noise injection.
        """
        from src.range_degrader import simulate_long_range_degradation, distance_to_snr

        target_type = target_type.lower()
        if target_type in self.cached_samples and len(self.cached_samples[target_type]) > 0:
            clean_sample = self.cached_samples[target_type][sample_idx % len(self.cached_samples[target_type])]
            snr = distance_to_snr(range_m)
            downsample = 2 if range_m > 150.0 else 1
            return simulate_long_range_degradation(clean_sample, target_snr_db=snr, downsample_factor=downsample)
        else:
            return self.synthesize_micro_doppler_tensor(range_m, radial_vel, target_type=target_type)

    def scan_and_detect(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_ground_truth: str = "drone",
        use_swinir: bool = True,
        sample_idx: int = 0
    ) -> Dict:
        """
        Executes single dwell radar scan on 3D target, applies SwinIR if enabled,
        and produces real-time classification.
        """
        range_m, v_r = self.compute_kinematics(target_pos, target_vel)

        x_raw = self.get_radar_echo(range_m, v_r, target_type=target_ground_truth, sample_idx=sample_idx)
        x_tensor = torch.from_numpy(x_raw).unsqueeze(0).float().to(self.device)

        if use_swinir and self.swinir is not None:
            with torch.no_grad():
                x_processed = self.swinir(x_tensor)
        else:
            x_processed = x_tensor

        with torch.no_grad():
            logits = self.classifier(x_processed)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))

        pred_class = config.COARSE_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "range_m": round(range_m, 1),
            "radial_velocity_mps": round(v_r, 2),
            "ground_truth": target_ground_truth,
            "prediction": pred_class,
            "confidence": round(confidence * 100.0, 1),
            "class_probabilities": {
                config.COARSE_CLASSES[i]: round(float(probs[i]) * 100.0, 1) for i in range(3)
            },
            "swinir_active": use_swinir and (self.swinir is not None)
        }


def run_simulated_intrusion():
    """
    Simulates a Quadcopter flying towards the perimeter from 800m away down to 50m.
    Shows the difference between raw baseline vs SwinIR-assisted detection.
    """
    print("\n" + "="*70)
    print("  GAZEBO SYNTHETIC FMCW RADAR INTRUSION SIMULATION")
    print("  Simulating Drone Ingress from 800m -> 50m")
    print("="*70)

    radar = SyntheticRadarSensor(radar_pos=(0, 0, 5))

    distances = [800, 600, 450, 300, 150, 70]
    for d in distances:
        pos = np.array([float(d), 15.0, 45.0], dtype=np.float32)
        vel = np.array([-14.0, 0.0, 0.0], dtype=np.float32) # Ingress velocity 14 m/s (~50 km/h)

        res_raw = radar.scan_and_detect(pos, vel, target_ground_truth="drone", use_swinir=False, sample_idx=10)
        res_sr = radar.scan_and_detect(pos, vel, target_ground_truth="drone", use_swinir=True, sample_idx=10)

        print(f"\n[Target Range: {d:3d}m | V_rad: {res_raw['radial_velocity_mps']} m/s]")
        print(f"  Standard Radar: Pred={res_raw['prediction']:<6} | Conf={res_raw['confidence']:.1f}% | Probs={res_raw['class_probabilities']}")
        print(f"  DopplerGuard:   Pred={res_sr['prediction']:<6} | Conf={res_sr['confidence']:.1f}% | Probs={res_sr['class_probabilities']}")


if __name__ == "__main__":
    run_simulated_intrusion()
