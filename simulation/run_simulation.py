"""
Step 4: Interactive Radar Simulation Runner for DopplerGuard.
Simulates tactical drone intrusion scenarios against a virtual 77 GHz radar station,
comparing Standard Unassisted Radar against DopplerGuard (Radar-SwinIR + AeroRadarNet).
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from simulation.synthetic_radar_sensor import SyntheticRadarSensor


def generate_trajectory(
    scenario: str = "direct_ingress",
    duration_sec: int = 30,
    dt: float = 1.0
) -> List[Dict]:
    """
    Generates 3D kinematic waypoints [x, y, z, vx, vy, vz, target_type] for the chosen scenario.
    """
    time_steps = np.arange(0, duration_sec + dt, dt)
    waypoints = []

    if scenario == "direct_ingress":
        # High-speed drone rushing straight from 700m away down to 50m
        start_x = 700.0
        end_x = 50.0
        speed_x = (end_x - start_x) / duration_sec # ~ -21.6 m/s (~78 km/h)

        for t in time_steps:
            pos = np.array([start_x + speed_x * t, 20.0 * np.sin(0.2 * t), 40.0 - 0.5 * t], dtype=np.float32)
            vel = np.array([speed_x, 4.0 * np.cos(0.2 * t), -0.5], dtype=np.float32)
            waypoints.append({
                "time_sec": round(float(t), 2),
                "pos": pos,
                "vel": vel,
                "target_type": "drone"
            })

    elif scenario == "perimeter_flank":
        # Drone stealthily tracing the 190m-210m perimeter boundary
        radius = 200.0
        angular_speed = 0.08 # rad/s (~16 m/s)

        for t in time_steps:
            angle = 0.2 + angular_speed * t
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 25.0], dtype=np.float32)
            vel = np.array([-radius * angular_speed * np.sin(angle), radius * angular_speed * np.cos(angle), 0.0], dtype=np.float32)
            waypoints.append({
                "time_sec": round(float(t), 2),
                "pos": pos,
                "vel": vel,
                "target_type": "drone"
            })

    elif scenario == "bird_vs_drone":
        # First 15 seconds: Bird crossing at 120m. Second 15 seconds: Drone hovering at 150m.
        for t in time_steps:
            if t <= 15.0:
                pos = np.array([120.0, -100.0 + 13.0 * t, 30.0], dtype=np.float32)
                vel = np.array([0.0, 13.0, 0.0], dtype=np.float32)
                tgt = "bird"
            else:
                pos = np.array([150.0, 15.0 + 1.0 * np.sin(t), 35.0], dtype=np.float32)
                vel = np.array([0.0, 1.0 * np.cos(t), 0.0], dtype=np.float32)
                tgt = "drone"
            waypoints.append({
                "time_sec": round(float(t), 2),
                "pos": pos,
                "vel": vel,
                "target_type": tgt
            })

    return waypoints


def run_airspace_simulation(
    scenario: str = "direct_ingress",
    duration_sec: int = 25,
    radar_height: float = 5.0
) -> Dict:
    print(f"\n{'='*75}")
    print(f"  DOPPLERGUARD STEP 4: 3D AIRSPACE INTRUSION RADAR SIMULATION")
    print(f"  Scenario: {scenario.upper()} | Duration: {duration_sec}s | Radar Station: [0, 0, {radar_height}]m")
    print(f"{'='*75}\n")

    radar = SyntheticRadarSensor(radar_pos=(0.0, 0.0, radar_height))
    waypoints = generate_trajectory(scenario=scenario, duration_sec=duration_sec, dt=1.0)

    telemetry_log = []

    print(f"{'TIME':>5} | {'RANGE':>7} | {'VRAD':>8} | {'TARGET':>6} | {'STANDARD RADAR':^22} | {'DOPPLERGUARD':^22} | {'PERIMETER'}")
    print(f"{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*22}-+-{'-'*22}-+-{'-'*11}")

    sample_counter = 0

    for wp in waypoints:
        t = wp["time_sec"]
        pos = wp["pos"]
        vel = wp["vel"]
        tgt = wp["target_type"]

        # Run unassisted baseline vs DopplerGuard (with SwinIR)
        sample_idx = 10 + (sample_counter % 30)
        sample_counter += 1

        res_raw = radar.scan_and_detect(pos, vel, target_ground_truth=tgt, use_swinir=False, sample_idx=sample_idx)
        res_sr = radar.scan_and_detect(pos, vel, target_ground_truth=tgt, use_swinir=True, sample_idx=sample_idx)

        range_m = res_raw["range_m"]
        v_rad = res_raw["radial_velocity_mps"]

        # Determine Perimeter Zone
        if range_m > 500.0:
            zone = "OUTER (Safe)"
        elif range_m > 200.0:
            zone = "WARNING ZONE"
        else:
            zone = "CRITICAL BREACH"

        raw_str = f"{res_raw['prediction']} ({res_raw['confidence']:.0f}%)"
        sr_str = f"{res_sr['prediction']} ({res_sr['confidence']:.0f}%)"

        print(f"{t:4.1f}s | {range_m:6.1f}m | {v_rad:+6.1f}m/s | {tgt:<6} | {raw_str:^22} | {sr_str:^22} | {zone}")

        telemetry_log.append({
            "time_sec": t,
            "position": [float(p) for p in pos],
            "velocity": [float(v) for v in vel],
            "slant_range_m": range_m,
            "radial_velocity_mps": v_rad,
            "ground_truth": tgt,
            "standard_radar": {
                "prediction": res_raw["prediction"],
                "confidence": res_raw["confidence"],
                "probabilities": res_raw["class_probabilities"]
            },
            "dopplerguard": {
                "prediction": res_sr["prediction"],
                "confidence": res_sr["confidence"],
                "probabilities": res_sr["class_probabilities"]
            },
            "perimeter_zone": zone
        })

    # Save Telemetry JSON
    telemetry_path = config.REPORTS_DIR / "simulation_telemetry.json"
    with open(telemetry_path, "w") as f:
        json.dump({
            "scenario": scenario,
            "duration_sec": duration_sec,
            "radar_coordinates": [0.0, 0.0, radar_height],
            "telemetry": telemetry_log
        }, f, indent=2)
    print(f"\n[Telemetry Saved] Exported flight records to: {telemetry_path}")

    return {"scenario": scenario, "telemetry": telemetry_log}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DopplerGuard Airspace Radar Simulation")
    parser.add_argument("--scenario", type=str, default="direct_ingress",
                        choices=["direct_ingress", "perimeter_flank", "bird_vs_drone"],
                        help="Tactical flight scenario to simulate")
    parser.add_argument("--duration", type=int, default=25, help="Simulation duration in seconds")
    args = parser.parse_args()

    run_airspace_simulation(scenario=args.scenario, duration_sec=args.duration)
