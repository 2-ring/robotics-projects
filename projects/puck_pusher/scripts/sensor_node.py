#!/usr/bin/env python3
"""Q3 Sensor Node — bridges MuJoCo sensor data for puck pushing LfD task.

Subscribes to:
  /mujoco/sensor_data       std_msgs/Float64MultiArray
  /mujoco/sensor_metadata   std_msgs/String

Publishes:
  /q3/joint_positions       std_msgs/Float64MultiArray  (7 arm joints)
  /q3/joint_velocities      std_msgs/Float64MultiArray  (7 arm joints)
  /q3/ee_position           std_msgs/Float64MultiArray  (3: x, y, z)
  /q3/ee_orientation        std_msgs/Float64MultiArray  (4: qw, qx, qy, qz)
  /q3/pusher_position       std_msgs/Float64MultiArray  (3: x, y, z)
  /q3/puck_position         std_msgs/Float64MultiArray  (3: x, y, z)
  /q3/puck_velocity         std_msgs/Float64MultiArray  (3: vx, vy, vz)
  /q3/goal_position         std_msgs/Float64MultiArray  (3: x, y, z)

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


class Q3SensorNode(Node):

    def __init__(self):
        super().__init__("q3_sensor_node")

        self.declare_parameter("publish_rate_hz", 100.0)
        self.sensor_indices: dict[str, tuple[int, int]] = {}
        self.metadata_ready = False
        self.latest_data = None

        self.create_subscription(
            String, "/mujoco/sensor_metadata", self._metadata_cb, LATCHED_QoS)
        self.create_subscription(
            Float64MultiArray, "/mujoco/sensor_data", self._data_cb, FAST_QoS)

        self.joint_pos_pub = self.create_publisher(
            Float64MultiArray, "/q3/joint_positions", FAST_QoS)
        self.joint_vel_pub = self.create_publisher(
            Float64MultiArray, "/q3/joint_velocities", FAST_QoS)
        self.ee_pos_pub = self.create_publisher(
            Float64MultiArray, "/q3/ee_position", FAST_QoS)
        self.ee_quat_pub = self.create_publisher(
            Float64MultiArray, "/q3/ee_orientation", FAST_QoS)
        self.pusher_pos_pub = self.create_publisher(
            Float64MultiArray, "/q3/pusher_position", FAST_QoS)
        self.puck_pos_pub = self.create_publisher(
            Float64MultiArray, "/q3/puck_position", FAST_QoS)
        self.puck_vel_pub = self.create_publisher(
            Float64MultiArray, "/q3/puck_velocity", FAST_QoS)
        self.goal_pos_pub = self.create_publisher(
            Float64MultiArray, "/q3/goal_position", FAST_QoS)

        rate = self.get_parameter("publish_rate_hz").value
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info("Q3 sensor node started.")

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

        # Pusher tip
        self._publish_array(self.pusher_pos_pub, self._read("pusher_tip_pos", 3))

        # Puck state
        self._publish_array(self.puck_pos_pub, self._read("puck_pos", 3))
        self._publish_array(self.puck_vel_pub, self._read("puck_vel", 3))

        # Goal position
        self._publish_array(self.goal_pos_pub, self._read("goal_pos", 3))


def main():
    rclpy.init()
    node = Q3SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
