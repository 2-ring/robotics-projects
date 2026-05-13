#!/usr/bin/env python3
"""Braitenberg vehicle that drives toward a light source and stops once it's close.

Two distance sensors (left/right antennae) → differential-drive wheels. The
control law is a linear combination of:

  - average distance        → forward speed (clamped),
  - signed distance delta   → steering correction.

A small acceleration boost on the steering term reduces drift when ramping up,
and a low-pass filter on the wheel commands keeps the bug from twitching.

Topics:
    in   /q1/left_sensor_distance, /q1/right_sensor_distance   Float64 (m)
    out  /q1/left_wheel_velocity,  /q1/right_wheel_velocity    Float64 (rad/s)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class BraitenbergController(Node):
    def __init__(self):
        super().__init__("braitenberg_controller")

        # Control gains — chosen empirically on the bug_scene.
        self.speed_gain = 80
        self.steer_gain = 70
        self.alpha = 0.15            # low-pass coefficient on wheel commands
        self.left_wheel_radps = 0
        self.right_wheel_radps = 0
        self.last_drive_speed = 0

        self.left_distance_m = 0
        self.right_distance_m = 0

        self.create_subscription(
            Float64, "/q1/left_sensor_distance", self._left_sensor_cb, 10
        )
        self.create_subscription(
            Float64, "/q1/right_sensor_distance", self._right_sensor_cb, 10
        )

        self.left_wheel_pub = self.create_publisher(
            Float64, "/q1/left_wheel_velocity", 10
        )
        self.right_wheel_pub = self.create_publisher(
            Float64, "/q1/right_wheel_velocity", 10
        )

        self.create_timer(0.02, self._control_loop)  # 50 Hz

        self.get_logger().info(
            "Braitenberg controller started. Waiting for sensor data..."
        )

    def _left_sensor_cb(self, msg: Float64):
        self.left_distance_m = msg.data

    def _right_sensor_cb(self, msg: Float64):
        self.right_distance_m = msg.data

    def _control_loop(self):
        d_left_m = self.left_distance_m
        d_right_m = self.right_distance_m

        avg_dist = (d_left_m + d_right_m) / 2.0
        drive_speed = max(6.0, min(self.speed_gain * avg_dist, 15.0))

        steer = max(-5.0, min(self.steer_gain * (d_right_m - d_left_m), 5.0))

        # Boost steering while accelerating — the bug tends to overshoot
        # heading corrections once it's already at cruising speed.
        if self.last_drive_speed < drive_speed:
            steer *= 2

        left_wheel_radps = drive_speed - steer
        right_wheel_radps = drive_speed + steer

        # Stop once both antennae are essentially on the light.
        if d_left_m < 0.05 and d_right_m < 0.05:
            left_wheel_radps = 0.0
            right_wheel_radps = 0.0

        # Low-pass filter the wheel commands.
        self.left_wheel_radps = (
            self.alpha * left_wheel_radps + (1.0 - self.alpha) * self.left_wheel_radps
        )
        self.right_wheel_radps = (
            self.alpha * right_wheel_radps + (1.0 - self.alpha) * self.right_wheel_radps
        )
        self.last_drive_speed = drive_speed

        left_msg = Float64()
        left_msg.data = self.left_wheel_radps
        self.left_wheel_pub.publish(left_msg)

        right_msg = Float64()
        right_msg.data = self.right_wheel_radps
        self.right_wheel_pub.publish(right_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BraitenbergController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
