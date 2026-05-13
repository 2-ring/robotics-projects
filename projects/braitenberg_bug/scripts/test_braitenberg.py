#!/usr/bin/env python3
"""
Q1 Test Script — Provided to students (DO NOT MODIFY).

Tests the Braitenberg controller by continuously moving the light source
along a smooth path. The robot must follow and stay near the light.

The light drifts slowly between random waypoints. The test checks that
the robot stays within a threshold distance of the light for most of
the test duration.

Prerequisites — these must already be running:
    ros2 launch q1 braitenberg_sim.launch.py
    ros2 run q1 braitenberg_controller.py       (your code!)

Run:
    ros2 run q1 test_braitenberg.py
"""

import json
import math
import random
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Float64MultiArray, String

# ── Test parameters ─────────────────────────────────────────────────
TEST_DURATION = 30.0  # total test time (seconds)
SUCCESS_RADIUS = 0.25  # robot must stay within this of the light (2D metres)
PASS_FRACTION = 0.7  # fraction of samples that must be within radius to pass
ARENA_RADIUS = 0.8  # light moves within this radius of origin
LIGHT_Z = 1.0  # light height above ground
LIGHT_SPEED = 0.3  # how fast the light drifts (m/s)
WAYPOINT_RADIUS = 0.1  # pick new waypoint when this close
INITIAL_WAIT = 5.0  # seconds to let robot reach first light position


class TestBraitenberg(Node):
    def __init__(self):
        super().__init__("test_braitenberg")

        # ── State ───────────────────────────────────────────────────
        self.robot_pos = None  # (x, y)
        self.light_pos = None  # (x, y)
        self.sensor_indices = {}
        self.metadata_ready = False

        # ── QoS ─────────────────────────────────────────────────────
        fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── Subscribe to MuJoCo sensor data (for robot & light pos) ─
        self.create_subscription(
            String, "/mujoco/sensor_metadata", self._metadata_cb, latched
        )
        self.create_subscription(
            Float64MultiArray, "/mujoco/sensor_data", self._sensor_cb, fast
        )

        # ── Publisher to move light ─────────────────────────────────
        self.light_cmd_pub = self.create_publisher(
            Point, "/q1/set_light_position", 10
        )

        self.get_logger().info("Test node ready.")

    # ── Callbacks ───────────────────────────────────────────────────

    def _metadata_cb(self, msg: String):
        try:
            meta = json.loads(msg.data)
            names = meta["sensor_names"]
            dims = meta["sensor_dims"]
            idx = 0
            for name, dim in zip(names, dims):
                self.sensor_indices[name] = (idx, dim)
                idx += dim
            self.metadata_ready = True
            self.get_logger().info(
                f"Sensor metadata received: {list(self.sensor_indices.keys())}"
            )
        except Exception as e:
            self.get_logger().error(f"Bad metadata: {e}")

    def _sensor_cb(self, msg: Float64MultiArray):
        if not self.metadata_ready:
            return
        data = msg.data

        # Read chassis (robot) position
        chassis = self._read_vec3(data, "chassis_pos")
        if chassis is not None:
            self.robot_pos = (chassis[0], chassis[1])

        # Read light source position
        light = self._read_vec3(data, "light_source_pos")
        if light is not None:
            self.light_pos = (light[0], light[1])

    def _read_vec3(self, data, name):
        if name not in self.sensor_indices:
            return None
        start, dim = self.sensor_indices[name]
        if dim != 3 or start + 3 > len(data):
            return None
        return np.array(data[start : start + 3])

    # ── Helpers ─────────────────────────────────────────────────────

    def _wait_for_data(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.robot_pos is not None:
                return True
        return False

    def _publish_light(self, x, y):
        msg = Point()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(LIGHT_Z)
        self.light_cmd_pub.publish(msg)

    def _distance_to_light(self):
        if self.robot_pos is None or self.light_pos is None:
            return float("inf")
        dx = self.robot_pos[0] - self.light_pos[0]
        dy = self.robot_pos[1] - self.light_pos[1]
        return math.sqrt(dx * dx + dy * dy)

    def _random_waypoint(self):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.2, ARENA_RADIUS)
        return r * math.cos(angle), r * math.sin(angle)

    # ── Main test ───────────────────────────────────────────────────

    def run_test(self):
        self.get_logger().info("Waiting for sensor data...")
        if not self._wait_for_data(timeout=10.0):
            self.get_logger().error(
                "No data received! Make sure the launch file and controller are running."
            )
            return False

        print()
        print("=" * 60)
        print("  Q1 BRAITENBERG VEHICLE TEST")
        print("=" * 60)
        print(f"  Duration:       {TEST_DURATION:.0f}s")
        print(f"  Success radius: {SUCCESS_RADIUS}m")
        print(f"  Pass threshold: {PASS_FRACTION * 100:.0f}% of samples within radius")
        print(f"  Light speed:    {LIGHT_SPEED} m/s")
        print("=" * 60)
        print()

        # Pick initial waypoint and move light there
        wx, wy = self._random_waypoint()
        lx, ly = wx, wy
        self._publish_light(lx, ly)

        # Let robot reach the first position
        print(f"  Moving light to ({lx:.2f}, {ly:.2f}) — waiting {INITIAL_WAIT:.0f}s for robot...")
        t0 = time.time()
        while time.time() - t0 < INITIAL_WAIT:
            rclpy.spin_once(self, timeout_sec=0.05)
            self._publish_light(lx, ly)

        # Pick first moving waypoint
        wx, wy = self._random_waypoint()

        # Now run the continuous test
        print(f"  Light is now moving continuously. Tracking for {TEST_DURATION:.0f}s...")
        print()

        total_samples = 0
        close_samples = 0
        max_dist = 0.0

        t_start = time.time()
        t_last_print = t_start
        last_time = t_start

        while time.time() - t_start < TEST_DURATION:
            now = time.time()
            elapsed_dt = now - last_time
            last_time = now

            # Move light toward current waypoint
            dx = wx - lx
            dy = wy - ly
            dist_to_wp = math.sqrt(dx * dx + dy * dy)

            if dist_to_wp < WAYPOINT_RADIUS:
                wx, wy = self._random_waypoint()
                dx = wx - lx
                dy = wy - ly
                dist_to_wp = math.sqrt(dx * dx + dy * dy)

            if dist_to_wp > 0:
                step = min(LIGHT_SPEED * elapsed_dt, dist_to_wp)
                lx += (dx / dist_to_wp) * step
                ly += (dy / dist_to_wp) * step

            self._publish_light(lx, ly)
            rclpy.spin_once(self, timeout_sec=0.05)

            # Sample distance
            d = self._distance_to_light()
            total_samples += 1
            if d <= SUCCESS_RADIUS:
                close_samples += 1
            max_dist = max(max_dist, d)

            # Print progress every 5 seconds
            if now - t_last_print >= 5.0:
                elapsed = now - t_start
                frac = close_samples / max(total_samples, 1)
                print(
                    f"  [{elapsed:5.1f}s] dist={d:.3f}m  "
                    f"tracking={frac * 100:.0f}%  "
                    f"light=({lx:.2f},{ly:.2f})",
                )
                t_last_print = now

        # Results
        fraction = close_samples / max(total_samples, 1)
        passed = fraction >= PASS_FRACTION

        print()
        print("=" * 60)
        print(
            f"  Samples within {SUCCESS_RADIUS}m: "
            f"{close_samples}/{total_samples} ({fraction * 100:.1f}%)",
        )
        print(f"  Max distance: {max_dist:.3f}m")
        print(f"  Required: {PASS_FRACTION * 100:.0f}%")
        print()
        if passed:
            print("  🎉 TEST PASSED!")
        else:
            print("  ❌ TEST FAILED — robot did not follow the light closely enough")
        print("=" * 60)
        print()

        return passed


# ════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = TestBraitenberg()
    try:
        success = node.run_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
