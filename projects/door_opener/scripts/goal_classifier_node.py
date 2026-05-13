#!/usr/bin/env python3
"""Q2 Goal Classifier Node — checks if the door has reached the target opening angle.

Subscribes to:
  /q2/door_angle   std_msgs/Float64MultiArray  (1: current hinge angle)

Publishes:
  /q2/goal_reached  std_msgs/String  ("true" | "false")
  /q2/goal_info     std_msgs/String  (JSON with angle, target, error, reached)

This node is PROVIDED and should NOT be modified.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from skills_utils import FAST_QoS


class GoalClassifierNode(Node):

    def __init__(self):
        super().__init__("goal_classifier_node")

        self.declare_parameter("check_rate_hz", 10.0)
        self.declare_parameter("target_angle_rad", -1.3)
        self.declare_parameter("angle_tolerance_rad", 0.15)

        self.target_angle = self.get_parameter("target_angle_rad").value
        self.tolerance = self.get_parameter("angle_tolerance_rad").value
        rate = self.get_parameter("check_rate_hz").value

        self.current_angle = 0.0

        self.create_subscription(
            Float64MultiArray, "/q2/door_angle", self._angle_cb, FAST_QoS)

        self.reached_pub = self.create_publisher(String, "/q2/goal_reached", FAST_QoS)
        self.info_pub = self.create_publisher(String, "/q2/goal_info", FAST_QoS)

        self.create_timer(1.0 / rate, self._check)
        self.get_logger().info(
            f"Goal classifier: target={self.target_angle:.2f} rad, "
            f"tolerance={self.tolerance:.2f} rad")

    def _angle_cb(self, msg: Float64MultiArray):
        if msg.data:
            self.current_angle = msg.data[0]

    def _check(self):
        error = abs(self.current_angle - self.target_angle)
        reached = error < self.tolerance

        reached_msg = String()
        reached_msg.data = "true" if reached else "false"
        self.reached_pub.publish(reached_msg)

        info = {
            "current_angle_rad": round(self.current_angle, 4),
            "target_angle_rad": self.target_angle,
            "error_rad": round(error, 4),
            "reached": reached,
        }
        info_msg = String()
        info_msg.data = json.dumps(info)
        self.info_pub.publish(info_msg)


def main():
    rclpy.init()
    node = GoalClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
