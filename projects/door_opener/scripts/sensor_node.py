#!/usr/bin/env python3
"""Q2 Sensor Node — bridges MuJoCo sensor data for door manipulation.

Subscribes to:
  /mujoco/sensor_data       std_msgs/Float64MultiArray
  /mujoco/sensor_metadata   std_msgs/String

Publishes:
  /q2/joint_positions       std_msgs/Float64MultiArray  (7 arm joints, rad)
  /q2/joint_velocities      std_msgs/Float64MultiArray  (7 arm joints, rad/s)
  /q2/ee_position           std_msgs/Float64MultiArray  (3: x, y, z)
  /q2/ee_orientation        std_msgs/Float64MultiArray  (4: qw, qx, qy, qz)
  /q2/door_angle            std_msgs/Float64MultiArray  (1: hinge angle rad)
  /q2/door_velocity         std_msgs/Float64MultiArray  (1: hinge angular vel)
  /q2/handle_position       std_msgs/Float64MultiArray  (3: x, y, z world frame)
  /q2/handle_orientation    std_msgs/Float64MultiArray  (4: qw, qx, qy, qz)

This node is PROVIDED and should NOT be modified.
"""

import json
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from skills_utils import FAST_QoS, LATCHED_QoS


JOINT_POS_SENSORS = [f"joint{i}_pos" for i in range(1, 8)]
JOINT_VEL_SENSORS = [f"joint{i}_vel" for i in range(1, 8)]


class Q2SensorNode(Node):

    def __init__(self):
        super().__init__("q2_sensor_node")

        self.declare_parameter("publish_rate_hz", 100.0)
        self.sensor_indices: dict[str, tuple[int, int]] = {}
        self.metadata_ready = False
        self.latest_data = None

        self.create_subscription(
            String, "/mujoco/sensor_metadata", self._metadata_cb, LATCHED_QoS)
        self.create_subscription(
            Float64MultiArray, "/mujoco/sensor_data", self._data_cb, FAST_QoS)

        self.joint_pos_pub = self.create_publisher(
            Float64MultiArray, "/q2/joint_positions", FAST_QoS)
        self.joint_vel_pub = self.create_publisher(
            Float64MultiArray, "/q2/joint_velocities", FAST_QoS)
        self.ee_pos_pub = self.create_publisher(
            Float64MultiArray, "/q2/ee_position", FAST_QoS)
        self.ee_quat_pub = self.create_publisher(
            Float64MultiArray, "/q2/ee_orientation", FAST_QoS)
        self.door_angle_pub = self.create_publisher(
            Float64MultiArray, "/q2/door_angle", FAST_QoS)
        self.door_vel_pub = self.create_publisher(
            Float64MultiArray, "/q2/door_velocity", FAST_QoS)
        self.handle_pos_pub = self.create_publisher(
            Float64MultiArray, "/q2/handle_position", FAST_QoS)
        self.handle_quat_pub = self.create_publisher(
            Float64MultiArray, "/q2/handle_orientation", FAST_QoS)

        rate = self.get_parameter("publish_rate_hz").value
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info("Q2 sensor node started.")

    def _metadata_cb(self, msg: String):
        try:
            meta = json.loads(msg.data)
            names = meta.get("sensor_names", [])
            dims = meta.get("sensor_dims", [])
            self.sensor_indices.clear()
            idx = 0
            for name, dim in zip(names, dims):
                self.sensor_indices[name] = (idx, int(dim))
                idx += int(dim)
            self.metadata_ready = True
            self.get_logger().info(f"Sensor metadata ready ({len(self.sensor_indices)} sensors).")
        except Exception as e:
            self.get_logger().error(f"Metadata parse error: {e}")

    def _data_cb(self, msg: Float64MultiArray):
        self.latest_data = msg.data

    def _read(self, name: str, dim: int) -> list[float] | None:
        if self.latest_data is None:
            return None
        info = self.sensor_indices.get(name)
        if info is None:
            return None
        start, d = info
        if d != dim or start + dim > len(self.latest_data):
            return None
        return [float(self.latest_data[start + i]) for i in range(dim)]

    def _publish_array(self, pub, values):
        if values is not None:
            msg = Float64MultiArray()
            msg.data = values
            pub.publish(msg)

    def _publish(self):
        if not self.metadata_ready or self.latest_data is None:
            return

        # Joint positions & velocities
        pos = []
        vel = []
        for s in JOINT_POS_SENSORS:
            v = self._read(s, 1)
            pos.append(v[0] if v else 0.0)
        for s in JOINT_VEL_SENSORS:
            v = self._read(s, 1)
            vel.append(v[0] if v else 0.0)
        self._publish_array(self.joint_pos_pub, pos)
        self._publish_array(self.joint_vel_pub, vel)

        # EE pose
        self._publish_array(self.ee_pos_pub, self._read("ee_pos", 3))
        self._publish_array(self.ee_quat_pub, self._read("ee_quat", 4))

        # Door state
        self._publish_array(self.door_angle_pub, self._read("door_hinge_pos", 1))
        self._publish_array(self.door_vel_pub, self._read("door_hinge_vel", 1))

        # Handle pose
        self._publish_array(self.handle_pos_pub, self._read("handle_pos", 3))
        self._publish_array(self.handle_quat_pub, self._read("handle_quat", 4))


def main():
    rclpy.init()
    node = Q2SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
