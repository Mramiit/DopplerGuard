# DopplerGuard Gazebo 3D Simulation Testbed

This directory provides the 3D physics-based simulation environment for DopplerGuard perimeter surveillance.

---

## 1. Overview
The simulation framework models a **77 GHz FMCW Radar Tower Station** guarding a sensitive perimeter (e.g., airport or critical infrastructure boundary) against unauthorized drone intrusions.

### Key Components:
1. **`gazebo_radar_world.sdf`**:
   - Gazebo simulation world containing a 5-meter radar surveillance mast at coordinates `[0, 0, 2.5]`, a 77 GHz antenna transceiver, and a 500-meter warning perimeter zone.
2. **`synthetic_radar_sensor.py`**:
   - Standalone FMCW radar sensor and micro-Doppler physics generator.
   - Calculates slant range $R(t)$ and radial velocity $v_r(t)$ relative to the radar mast.
   - Synthesizes real-time micro-Doppler spectrograms based on rotor RPM, blade count, and blade tip speed $v_{\text{tip}} = \omega \cdot r_{\text{prop}}$.
   - Injects physical $R^4$ power falloff ($\Delta\text{SNR} = -40\log_{10}(R/R_0)$) and thermal receiver noise.
   - Pipes the degraded signals into **Radar-SwinIR** (for super-resolution denoising) and **AeroRadarNet** (for 9 ms threat classification).

---

## 2. Running the Intrusion Simulation

To run the drone ingress test from 800m down to 70m:

```powershell
python simulation/synthetic_radar_sensor.py
```

### Expected Output:
- At distant ranges (>400m), standard radar models fail due to low SNR (predicting Bird or Human due to drowned blade flashes).
- **DopplerGuard (Radar-SwinIR + AeroRadarNet)** reconstructs blade harmonics and maintains high-confidence Drone classification all the way from long range!

---

## 3. Integrating with Gazebo & ROS 2 (Harmonic / Humble)

To connect this sensor bridge directly to a live PX4 / ArduPilot quadcopter inside Gazebo:

1. Launch Gazebo with the perimeter world:
   ```bash
   gz sim simulation/gazebo_radar_world.sdf
   ```
2. Spawn your drone model (e.g., `x500` quadcopter) at `[800, 15, 45]`.
3. In ROS 2, subscribe to the drone pose topic `/model/x500/odometry` or `/model/x500/pose`.
4. Forward the `pose.position` and `twist.linear` vectors into `SyntheticRadarSensor.scan_and_detect()`.
