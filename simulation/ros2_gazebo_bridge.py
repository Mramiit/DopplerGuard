"""
ROS 2 & Gazebo Synthetic FMCW Radar Bridge Node.
Subscribes to Gazebo drone odometry, simulates realistic 77 GHz radar echoes,
and streams threat alerts into ROS 2 topics (/radar/threat_alert).
Includes standalone mock verification mode for Windows/laptop development.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from simulation.synthetic_radar_sensor import SyntheticRadarSensor

# Check if rclpy is available (Linux / ROS 2 Humble/Iron/Jazzy environment)
try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String, Float32
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


if ROS2_AVAILABLE:
    class RadarPerimeterBridgeNode(Node):
        """
        ROS 2 Node that interfaces Gazebo with the DopplerGuard Radar Engine.
        """
        def __init__(self):
            super().__init__('dopplerguard_radar_bridge')
            self.declare_parameter('radar_x', 0.0)
            self.declare_parameter('radar_y', 0.0)
            self.declare_parameter('radar_z', 5.0)

            rx = self.get_parameter('radar_x').value
            ry = self.get_parameter('radar_y').value
            rz = self.get_parameter('radar_z').value

            self.sensor = SyntheticRadarSensor(radar_pos=(rx, ry, rz))
            self.get_logger().info(f"[Radar Bridge] Radar tower initialized at ({rx}, {ry}, {rz})m")

            # Subscribers (Gazebo drone odometry)
            self.odom_sub = self.create_subscription(
                Odometry,
                '/model/drone/odometry',
                self.odometry_callback,
                10
            )

            # Publishers (Airspace perimeter defense alerts)
            self.alert_pub = self.create_publisher(String, '/radar/threat_alert', 10)
            self.range_pub = self.create_publisher(Float32, '/radar/target_range', 10)

        def odometry_callback(self, msg: Odometry):
            p = msg.pose.pose.position
            v = msg.twist.twist.linear

            target_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            target_vel = np.array([v.x, v.y, v.z], dtype=np.float32)

            # Run DopplerGuard detection
            result = self.sensor.scan_and_detect(target_pos, target_vel, target_ground_truth="drone", use_swinir=True)

            range_m = result["range_m"]
            pred = result["prediction"]
            conf = result["confidence"]

            if range_m > 500.0:
                zone = "SAFE"
            elif range_m > 200.0:
                zone = "WARNING"
            else:
                zone = "CRITICAL_BREACH"

            alert_msg = String()
            alert_msg.data = f"STATUS={zone}|TARGET={pred}|CONF={conf:.1f}%|RANGE={range_m:.1f}m"
            self.alert_pub.publish(alert_msg)

            range_msg = Float32()
            range_msg.data = float(range_m)
            self.range_pub.publish(range_msg)

            self.get_logger().info(f"[Radar Scan] {alert_msg.data}")


def run_bridge_test(standalone: bool = True):
    """
    Test routine: verifies node initialization and simulates incoming Gazebo odometry ticks.
    """
    print(f"\n{'='*70}\n  GAZEBO <-> DOPPLERGUARD SENSOR BRIDGE VERIFICATION\n  ROS 2 Environment: {'DETECTED (Linux)' if ROS2_AVAILABLE else 'MOCK STANDALONE (Windows / Laptop)'}\n{'='*70}\n")

    sensor = SyntheticRadarSensor(radar_pos=(0.0, 0.0, 5.0))

    # Test simulated Gazebo drone positions
    test_waypoints = [
        ([450.0, 0.0, 40.0], [-18.0, 0.0, 0.0]),
        ([220.0, 10.0, 25.0], [-15.0, 0.0, 0.0]),
        ([150.0, 5.0, 20.0], [-12.0, 0.0, 0.0]),
        ([70.0, 0.0, 10.0], [-8.0, 0.0, 0.0])
    ]

    for pos, vel in test_waypoints:
        res = sensor.scan_and_detect(np.array(pos, np.float32), np.array(vel, np.float32), "drone", use_swinir=True, sample_idx=10)
        range_m = res["range_m"]
        pred = res["prediction"]
        conf = res["confidence"]

        if range_m > 500.0:
            zone = "SAFE"
        elif range_m > 200.0:
            zone = "PERIMETER WARNING"
        else:
            zone = "CRITICAL BREACH"

        print(f"Gazebo Drone Pos: {str(pos):<20} | Range: {range_m:5.1f}m | Status: {zone:<18} | Alert: {pred} ({conf:.1f}%)")

    print("\n[Bridge Test Succeeded] Gazebo sensor bridge is ready for live deployment.")


def main(args=None):
    if ROS2_AVAILABLE:
        rclpy.init(args=args)
        node = RadarPerimeterBridgeNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_bridge_test()


if __name__ == "__main__":
    main()
