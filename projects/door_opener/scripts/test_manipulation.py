#!/usr/bin/env python3
"""Q2 Autograder — tests articulated object manipulation (door opening).

Test procedure:
  1. Waits for grasp controller to report "ready" (handle grasped).
  2. Times how long the student's planner takes to open the door.
  3. Checks the final door angle against the target.

Grading:
  - Door must reach within tolerance of target angle (read from q2_params.yaml).
  - Must complete within 120 seconds of grasp being ready.
  - Bonus: smooth motion (low peak angular velocity of the door).

This script is PROVIDED and should NOT be modified.
"""

import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from skills_utils import FAST_QoS

TIMEOUT_S = 120.0


class Q2Tester(Node):

    def __init__(self):
        super().__init__("q2_tester")

        self.declare_parameter("target_angle_rad", -1.3)
        self.declare_parameter("angle_tolerance_rad", 0.15)
        self.target_angle = self.get_parameter("target_angle_rad").value
        self.angle_tolerance = self.get_parameter("angle_tolerance_rad").value

        self.door_angle = 0.0
        self.door_velocity = 0.0
        self.grasp_status = "approaching"
        self.goal_reached = False

        self.peak_door_vel = 0.0
        self.angle_history = []

        self.create_subscription(
            Float64MultiArray, "/q2/door_angle", self._angle_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/door_velocity", self._vel_cb, FAST_QoS)
        self.create_subscription(
            String, "/q2/grasp_status", self._grasp_cb, FAST_QoS)
        self.create_subscription(
            String, "/q2/goal_reached", self._goal_cb, FAST_QoS)

    def _angle_cb(self, msg):
        if msg.data:
            self.door_angle = msg.data[0]

    def _vel_cb(self, msg):
        if msg.data:
            self.door_velocity = msg.data[0]
            self.peak_door_vel = max(self.peak_door_vel, abs(self.door_velocity))

    def _grasp_cb(self, msg):
        self.grasp_status = msg.data

    def _goal_cb(self, msg):
        self.goal_reached = (msg.data == "true")

    def run_tests(self):
        self.get_logger().info("=" * 50)
        self.get_logger().info("Q2 AUTOGRADE: Door Manipulation")
        self.get_logger().info("=" * 50)

        # Wait for simulation data
        self.get_logger().info("Waiting for simulation data...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.grasp_status != "approaching":
                break

        if self.grasp_status == "approaching":
            self.get_logger().warn("Grasp controller may not be running. Continuing to wait...")

        # Wait for grasp to be ready
        self.get_logger().info("Waiting for grasp controller to finish...")
        t0 = time.time()
        while self.grasp_status != "ready" and time.time() - t0 < 20.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.grasp_status != "ready":
            self.get_logger().error("Grasp controller did not reach 'ready' state!")
            return False

        self.get_logger().info(f"Grasp ready. Starting door angle: {self.door_angle:.3f} rad")
        self.get_logger().info(f"Target angle: {self.target_angle:.2f} rad (tolerance: ±{self.angle_tolerance:.2f})")
        self.get_logger().info("Waiting for student planner to open door...")

        # Monitor door opening
        t_start = time.time()
        success = False
        while time.time() - t_start < TIMEOUT_S:
            rclpy.spin_once(self, timeout_sec=0.1)
            self.angle_history.append((time.time() - t_start, self.door_angle))

            error = abs(self.door_angle - self.target_angle)
            if error < self.angle_tolerance:
                success = True
                break

        elapsed = time.time() - t_start

        # ── Results ──
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Final door angle:   {self.door_angle:.3f} rad")
        self.get_logger().info(f"Target angle:       {self.target_angle:.3f} rad")
        self.get_logger().info(f"Error:              {abs(self.door_angle - self.target_angle):.3f} rad")
        self.get_logger().info(f"Time elapsed:       {elapsed:.1f} s")
        self.get_logger().info(f"Peak door velocity: {self.peak_door_vel:.3f} rad/s")

        # Smoothness check
        smooth = self.peak_door_vel < 2.0
        self.get_logger().info(f"Smoothness:         {'PASS' if smooth else 'WARN (peak vel > 2.0 rad/s)'}")

        if success:
            self.get_logger().info(f"RESULT: PASS (door opened to target in {elapsed:.1f} s)")
        else:
            self.get_logger().info(f"RESULT: FAIL (door did not reach target within {TIMEOUT_S:.0f} s)")

        self.get_logger().info("=" * 50)
        return success


def main():
    rclpy.init()
    tester = Q2Tester()
    try:
        success = tester.run_tests()
    except KeyboardInterrupt:
        success = False
    finally:
        tester.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
