#!/usr/bin/env python3
"""Q1 Sensor Node — bridges MuJoCo sensor data to ROS2 topics for semantic mapping.

Subscribes to:
  /mujoco/sensor_data       std_msgs/Float64MultiArray
  /mujoco/sensor_metadata   std_msgs/String (latched JSON)

Publishes:
  /q1/robot_pose            geometry_msgs/Pose2D  (x, y, theta from ground-truth)
  /q1/camera_pose           std_msgs/Float64MultiArray  [x, y, z, qw, qx, qy, qz]
  /q1/object_positions      std_msgs/String  (JSON: {name: [x, y, z], ...} ground truth)
  /q1/scan                  sensor_msgs/LaserScan  (61-beam 180° LiDAR, robot frame)

This node is PROVIDED and should NOT be modified.
"""

import json
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, String

from skills_utils import FAST_QoS, LATCHED_QoS, RELIABLE_QoS, quat_to_yaw_rad

_NUM_LASERS = 61
_LASER_NAMES = [f"tb_laser_{i:02d}" for i in range(_NUM_LASERS)]
_LASER_FOV_RAD = 1.2     # must match MuJoCo XML beam layout (~69°, ±0.6 rad)
_LASER_MAX_RANGE = 2.0   # treat "no hit" (-1) as this distance


# Object names must match the sensor names in the MuJoCo XML
OBJECT_SENSORS = {
    "mug":    "obj_mug_pos",
    "bottle": "obj_bottle_pos",
    "box":    "obj_box_pos",
    "bowl":   "obj_bowl_pos",
    "can":    "obj_can_pos",
}


class Q1SensorNode(Node):

    def __init__(self):
        super().__init__("q1_sensor_node")

        self.sensor_indices: dict[str, tuple[int, int]] = {}
        self.metadata_ready = False

        # Subscribers
        self.create_subscription(
            String, "/mujoco/sensor_metadata", self._metadata_cb, LATCHED_QoS)
        self.create_subscription(
            Float64MultiArray, "/mujoco/sensor_data", self._sensor_data_cb, FAST_QoS)

        # Publishers
        self.robot_pose_pub = self.create_publisher(Pose2D, "/q1/robot_pose", FAST_QoS)
        self.camera_pose_pub = self.create_publisher(
            Float64MultiArray, "/q1/camera_pose", FAST_QoS)
        self.object_positions_pub = self.create_publisher(
            String, "/q1/object_positions", FAST_QoS)
        self.scan_pub = self.create_publisher(LaserScan, "/q1/scan", RELIABLE_QoS)

        self.get_logger().info("Q1 sensor node started.")

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

    def _read_sensor(self, data, name: str, dim: int) -> np.ndarray | None:
        info = self.sensor_indices.get(name)
        if info is None:
            return None
        start, d = info
        if d != dim or start + dim > len(data):
            return None
        return np.array(data[start:start + dim], dtype=float)

    def _sensor_data_cb(self, msg: Float64MultiArray):
        if not self.metadata_ready:
            return

        data = msg.data

        # Robot pose (ground truth)
        pos = self._read_sensor(data, "tb_base_pos", 3)
        quat = self._read_sensor(data, "tb_base_quat", 4)
        if pos is not None and quat is not None:
            pose = Pose2D()
            pose.x = float(pos[0])
            pose.y = float(pos[1])
            pose.theta = quat_to_yaw_rad(quat)
            self.robot_pose_pub.publish(pose)

        # Camera pose
        cam_pos = self._read_sensor(data, "tb_camera_pos", 3)
        cam_quat = self._read_sensor(data, "tb_camera_quat", 4)
        if cam_pos is not None and cam_quat is not None:
            cam_msg = Float64MultiArray()
            cam_msg.data = list(cam_pos) + list(cam_quat)
            self.camera_pose_pub.publish(cam_msg)

        # Object positions (ground truth, for detector to add noise to)
        obj_dict = {}
        for obj_name, sensor_name in OBJECT_SENSORS.items():
            obj_pos = self._read_sensor(data, sensor_name, 3)
            if obj_pos is not None:
                obj_dict[obj_name] = [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])]

        if obj_dict:
            obj_msg = String()
            obj_msg.data = json.dumps(obj_dict)
            self.object_positions_pub.publish(obj_msg)

        # LiDAR scan — 61 rangefinders spanning ±0.6 rad (robot-local frame)
        ranges = []
        for name in _LASER_NAMES:
            val = self._read_sensor(data, name, 1)
            if val is not None:
                r = float(val[0])
                ranges.append(_LASER_MAX_RANGE if r < 0 else min(r, _LASER_MAX_RANGE))
            else:
                ranges.append(_LASER_MAX_RANGE)

        if len(ranges) == _NUM_LASERS:
            scan = LaserScan()
            scan.header.stamp = self.get_clock().now().to_msg()
            scan.header.frame_id = "tb_laser"
            scan.angle_min = -_LASER_FOV_RAD / 2.0
            scan.angle_max = _LASER_FOV_RAD / 2.0
            scan.angle_increment = _LASER_FOV_RAD / (_NUM_LASERS - 1)
            scan.range_min = 0.05
            scan.range_max = _LASER_MAX_RANGE
            scan.ranges = ranges
            self.scan_pub.publish(scan)


def main():
    rclpy.init()
    node = Q1SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
