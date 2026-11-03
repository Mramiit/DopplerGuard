# DopplerGuard Gazebo 3D Simulation Testbed

This directory provides the end-to-end 3D physics-based simulation environment for DopplerGuard perimeter airspace surveillance.

---

## 1. Overview
The simulation framework models a **77 GHz FMCW Radar Tower Station** guarding a sensitive perimeter (e.g., airport, critical infrastructure, border checkpoint) against unauthorized drone intrusions.

### Key Components:
1. **`gazebo_radar_world.sdf`**:
   - Gazebo simulation world containing a 5-meter radar surveillance mast at coordinates `[0, 0, 2.5]`, a 77 GHz antenna transceiver, and a 500-meter warning perimeter zone.
2. **`drone_model.sdf`**:
   - 3D quadcopter drone model with 4 spinning rotor links, physical inertial matrix, and radar cross section (RCS $\approx 0.01\text{ m}^2$).
3. **`synthetic_radar_sensor.py`**:
   - Standalone FMCW radar sensor and micro-Doppler physics generator.
   - Calculates slant range $R(t)$ and radial velocity $v_r(t)$ relative to the radar mast.
   - Synthesizes real-time micro-Doppler spectrograms based on rotor RPM and tip speed $v_{\text{tip}} = \omega \cdot r_{\text{prop}}$.
   - Injects physical $R^4$ power falloff ($\Delta\text{SNR} = -40\log_{10}(R/R_0)$) and thermal receiver noise.
   - Pipes the degraded signals into **Radar-SwinIR** (super-resolution denoising) and **AeroRadarNet** (9 ms threat classification).
4. **`run_simulation.py`**:
   - Interactive command-line simulation runner supporting multiple tactical flight scenarios (`direct_ingress`, `perimeter_flank`, `bird_vs_drone`).
   - Exports full flight telemetry to `reports/simulation_telemetry.json`.
5. **`plot_simulation_run.py`**:
   - Renders a 4-panel Tactical Airspace HUD (`reports/gazebo_simulation_tactical_display.png`), including an overhead PPI radar scope, range closure profile, and threat intercept confidence timeline.
6. **`ros2_gazebo_bridge.py`**:
   - Complete ROS 2 / Gazebo bridge node that subscribes to `/model/drone/odometry` and streams real-time threat alerts to `/radar/threat_alert`.

---

## 2. Running Simulations on Your Laptop

### Quick Run (Standalone Python — Zero Setup Required):

Run the tactical drone ingress simulation:
```powershell
python simulation/run_simulation.py --scenario direct_ingress --duration 25
```

Render the Tactical Radar HUD Display:
```powershell
python simulation/plot_simulation_run.py
```

Test the ROS 2 Sensor Bridge in mock mode:
```powershell
python simulation/ros2_gazebo_bridge.py
```

---

## 3. Available Tactical Scenarios

| Scenario Flag | Description | What It Tests |
| :--- | :--- | :--- |
| `--scenario direct_ingress` | High-speed quadcopter rushing the perimeter from 700m $\to$ 50m at 78 km/h. | Tests early warning intercept capability and reaction time. |
| `--scenario perimeter_flank` | Drone cruising stealthily along the 190m–210m perimeter boundary. | Tests sustained tracking at the critical detection threshold. |
| `--scenario bird_vs_drone` | A bird flock crosses at 120m, followed by a drone hovering at 150m. | Tests zero false-alarm discrimination between birds and drones. |

---

## 4. Full 3D Gazebo & ROS 2 Integration (Harmonic / Humble)

To connect this bridge directly to a live PX4 / ArduPilot quadcopter inside Gazebo on Linux or Windows WSL 2:

1. Launch Gazebo with the perimeter world:
   ```bash
   gz sim simulation/gazebo_radar_world.sdf
   ```
2. Spawn the intruder drone into the world:
   ```bash
   gz service -s /world/radar_perimeter_world/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --req 'sdf_filename: "simulation/drone_model.sdf", pose: {position: {x: 600, y: 15, z: 40}}'
   ```
3. Start the ROS 2 bridge node:
   ```bash
   python3 simulation/ros2_gazebo_bridge.py
   ```
4. Listen to live threat alerts:
   ```bash
   ros2 topic echo /radar/threat_alert
   ```
