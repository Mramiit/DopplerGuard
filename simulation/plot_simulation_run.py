"""
Visual Tactical Radar Display for DopplerGuard Airspace Simulation.
Generates an overhead Plan Position Indicator (PPI) radar scope, range closure profile,
and threat intercept confidence telemetry curves.
"""

import sys
import json
from pathlib import Path
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_simulation_tactical_display(
    telemetry_path: Path = None,
    output_image_path: Path = None
):
    if telemetry_path is None:
        telemetry_path = config.REPORTS_DIR / "simulation_telemetry.json"
    if output_image_path is None:
        output_image_path = config.REPORTS_DIR / "gazebo_simulation_tactical_display.png"

    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry file not found at {telemetry_path}. Run run_simulation.py first!")

    with open(telemetry_path, "r") as f:
        data = json.load(f)

    scenario = data.get("scenario", "direct_ingress")
    records = data["telemetry"]

    times = [r["time_sec"] for r in records]
    ranges = [r["slant_range_m"] for r in records]
    v_rads = [r["radial_velocity_mps"] for r in records]
    x_coords = [r["position"][0] for r in records]
    y_coords = [r["position"][1] for r in records]

    raw_preds = [r["standard_radar"]["prediction"] for r in records]
    sr_preds = [r["dopplerguard"]["prediction"] for r in records]

    raw_drone_conf = [r["standard_radar"]["probabilities"].get("Drone", 0.0) for r in records]
    sr_drone_conf = [r["dopplerguard"]["probabilities"].get("Drone", 0.0) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # -------------------------------------------------------------
    # 1. Overhead Tactical PPI Radar Scope (Top-Left)
    # -------------------------------------------------------------
    ax_scope = axes[0, 0]
    ax_scope.set_facecolor('#0b101b') # Dark tactical background

    # Radar rings
    ring_500 = Circle((0, 0), 500, color='#e67e22', fill=False, linestyle='--', lw=1.5, label='500m Warning Perimeter')
    ring_200 = Circle((0, 0), 200, color='#2ecc71', fill=False, linestyle='-', lw=2.0, label='200m DopplerGuard Extended Intercept')
    ring_70 = Circle((0, 0), 70, color='#e74c3c', fill=False, linestyle=':', lw=1.5, label='70m Standard Radar Limit')

    ax_scope.add_patch(ring_500)
    ax_scope.add_patch(ring_200)
    ax_scope.add_patch(ring_70)

    # Radar station center icon
    ax_scope.plot(0, 0, marker='*', markersize=14, color='cyan', label='77 GHz Radar Mast [0,0]')

    # Plot drone flight trajectory
    ax_scope.plot(x_coords, y_coords, color='#7f8c8d', linestyle='-', alpha=0.6, lw=1.5)

    # Waypoints: Green if DopplerGuard detected Drone, Red if Standard Radar failed
    for i in range(len(records)):
        x, y = x_coords[i], y_coords[i]
        is_detected_sr = (sr_preds[i] == "Drone")
        is_detected_raw = (raw_preds[i] == "Drone")

        if is_detected_sr and not is_detected_raw:
            ax_scope.scatter(x, y, color='#2ecc71', s=60, edgecolors='black', zorder=5) # DopplerGuard exclusive catch
        elif is_detected_sr and is_detected_raw:
            ax_scope.scatter(x, y, color='#3498db', s=45, edgecolors='black', zorder=4) # Both detected
        else:
            ax_scope.scatter(x, y, color='#e74c3c', s=40, edgecolors='black', zorder=3) # Missed

    ax_scope.set_xlim(-550, 750)
    ax_scope.set_ylim(-350, 350)
    ax_scope.set_aspect('equal')
    ax_scope.set_title(f"A. 2D Tactical Airspace Radar Scope ({scenario.upper()})", fontsize=11, fontweight='bold', color='white')
    ax_scope.set_xlabel("East-West Distance (m)", fontsize=9, color='white')
    ax_scope.set_ylabel("North-South Distance (m)", fontsize=9, color='white')
    ax_scope.grid(True, linestyle=':', alpha=0.3, color='cyan')
    ax_scope.tick_params(colors='white')
    ax_scope.legend(loc='lower left', fontsize=8, facecolor='#1a252f', labelcolor='white')

    # -------------------------------------------------------------
    # 2. Slant Range vs Time (Top-Right)
    # -------------------------------------------------------------
    ax_range = axes[0, 1]
    ax_range.plot(times, ranges, 'navy', lw=2.5, marker='o', markersize=4, label='Target Slant Range')
    ax_range.axhline(500, color='#e67e22', linestyle='--', lw=1.5, label='500m Warning Boundary')
    ax_range.axhline(200, color='#2ecc71', linestyle='-', lw=2.0, label='200m DopplerGuard Edge')
    ax_range.axhline(70, color='#e74c3c', linestyle=':', lw=1.5, label='70m Standard Radar Baseline')

    ax_range.fill_between(times, 70, 200, color='#2ecc71', alpha=0.15, label='DopplerGuard Range Advantage')

    ax_range.set_title("B. Target Slant Range Closure vs. Security Boundaries", fontsize=11, fontweight='bold')
    ax_range.set_xlabel("Mission Elapsed Time (seconds)", fontsize=10)
    ax_range.set_ylabel("Slant Range to Radar (meters)", fontsize=10)
    ax_range.grid(True, linestyle='--', alpha=0.5)
    ax_range.set_ylim(0, max(ranges) + 50)
    ax_range.legend(loc='upper right', fontsize=8.5)

    # -------------------------------------------------------------
    # 3. Live Drone Threat Confidence Timeline (Bottom-Left)
    # -------------------------------------------------------------
    ax_conf = axes[1, 0]
    ax_conf.plot(times, raw_drone_conf, 'r--o', lw=2.0, label='Standard Radar Drone Conf (%)')
    ax_conf.plot(times, sr_drone_conf, 'g-s', lw=2.5, label='DopplerGuard (SwinIR) Drone Conf (%)')
    ax_conf.axhline(75.0, color='gray', linestyle=':', label='Security Intercept Alert (75%)')

    ax_conf.set_title("C. Threat Intercept Confidence (Drone Probability %)", fontsize=11, fontweight='bold')
    ax_conf.set_xlabel("Mission Elapsed Time (seconds)", fontsize=10)
    ax_conf.set_ylabel("Classification Confidence (%)", fontsize=10)
    ax_conf.grid(True, linestyle='--', alpha=0.5)
    ax_conf.set_ylim(-5, 105)
    ax_conf.legend(loc='lower right', fontsize=9)

    # -------------------------------------------------------------
    # 4. Target Radial Velocity (Bottom-Right)
    # -------------------------------------------------------------
    ax_vel = axes[1, 1]
    ax_vel.plot(times, v_rads, color='darkviolet', lw=2.2, marker='^', markersize=4, label='Radial Velocity (Vr)')
    ax_vel.axhline(0.0, color='black', lw=1.0)
    ax_vel.set_title("D. Doppler Radial Velocity Telemetry (Approaching: Vr < 0)", fontsize=11, fontweight='bold')
    ax_vel.set_xlabel("Mission Elapsed Time (seconds)", fontsize=10)
    ax_vel.set_ylabel("Doppler Radial Velocity (m/s)", fontsize=10)
    ax_vel.grid(True, linestyle='--', alpha=0.5)
    ax_vel.legend(loc='upper right', fontsize=9)

    plt.suptitle("DopplerGuard 3D Tactical Radar Airspace Simulation HUD\n"
                 "Real-Time Virtual Perimeter Defense: SwinIR Micro-Doppler Super-Resolution vs Standard Radar",
                 fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()

    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[HUD Display Saved] Tactical Scope image written to: {output_image_path}")


if __name__ == "__main__":
    plot_simulation_tactical_display()
