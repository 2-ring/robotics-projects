#!/usr/bin/env python3
"""ROS 2 bridge node for a simulated TurtleBot robot with a LiDAR sensor.

Translates between the MuJoCo simulator node and ROS 2 topics/types.

Subscribes to:
  /mujoco/sensor_data       std_msgs/Float64MultiArray
  /mujoco/sensor_metadata   std_msgs/String (latched JSON)
  /cmd_vel                  geometry_msgs/Twist

Publishes:
  /mujoco/joint_controls    Float64MultiArray (left and right wheel velocities in rad/s)
  /laser_scans              sensor_msgs/LaserScan
  /gt/odometry              geometry_msgs/Pose2D

"""

import hashlib
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D, TransformStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import TransformBroadcaster

from skills_utils import FAST_QoS, LATCHED_QoS, quat_to_yaw_rad


class TurtleBotBridge(Node):
    """A bridge between the MuJoCo simulator and ROS 2 topics."""

    def __init__(self) -> None:
        """Initialize the TurtleBot bridge node."""
        super().__init__("turtlebot_bridge_node")

        self.declare_parameter("wheel_radius_m", 0.033)
        self.wheel_radius_m = float(self.get_parameter("wheel_radius_m").value)

        self.declare_parameter("wheel_base_m", 0.18)
        self.wheel_base_m = float(self.get_parameter("wheel_base_m").value)

        self.declare_parameter("command_timeout_s", 0.5)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)

        self.declare_parameter("control_rate_hz", 50.0)

        self.declare_parameter("laser_frame_id", "tb_laser")
        self.declare_parameter("laser_fov_rad", math.pi)
        self.declare_parameter("laser_sensor_prefix", "tb_laser_")
        self.declare_parameter("laser_min_range_m", 0.05)
        self.declare_parameter("laser_max_range_m", 5.0)
        self.laser_frame_id = str(self.get_parameter("laser_frame_id").value)
        self.laser_fov_rad = float(self.get_parameter("laser_fov_rad").value)
        self.laser_sensor_prefix = str(self.get_parameter("laser_sensor_prefix").value)
        self.laser_min_range_m = float(self.get_parameter("laser_min_range_m").value)
        self.laser_max_range_m = float(self.get_parameter("laser_max_range_m").value)

        self.declare_parameter("odom_pos_sensor", "tb_base_pos")
        self.declare_parameter("odom_quat_sensor", "tb_base_quat")
        self.odom_pos_sensor = str(self.get_parameter("odom_pos_sensor").value)
        self.odom_quat_sensor = str(self.get_parameter("odom_quat_sensor").value)

        self.declare_parameter("publish_tf", True)
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.map_frame_id = str(self.get_parameter("map_frame_id").value)
        self.odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)

        self.declare_parameter("laser_offset_x_m", 0.0)
        self.declare_parameter("laser_offset_y_m", 0.0)
        self.declare_parameter("laser_offset_z_m", 0.06)
        self.laser_offset_x_m = float(self.get_parameter("laser_offset_x_m").value)
        self.laser_offset_y_m = float(self.get_parameter("laser_offset_y_m").value)
        self.laser_offset_z_m = float(self.get_parameter("laser_offset_z_m").value)

        self.declare_parameter("publish_odometry", True)
        self.publish_odometry = bool(self.get_parameter("publish_odometry").value)

        self.declare_parameter("noise_seed_tag", "q1-noise")
        self.declare_parameter("lidar_noise_sigma0_m", 0.01)
        self.declare_parameter("lidar_noise_scale", 0.01)
        self.declare_parameter("lidar_dropout_prob", 0.02)
        self.lidar_noise_sigma0_m = float(
            self.get_parameter("lidar_noise_sigma0_m").value
        )
        self.lidar_noise_scale = float(self.get_parameter("lidar_noise_scale").value)
        self.lidar_dropout_prob = float(self.get_parameter("lidar_dropout_prob").value)

        noise_seed_tag = str(self.get_parameter("noise_seed_tag").value)
        noise_seed = int.from_bytes(
            hashlib.blake2b(noise_seed_tag.encode("utf-8"), digest_size=8).digest(),
            byteorder="little",
            signed=False,
        )
        self._noise_rng = np.random.Generator(np.random.PCG64(noise_seed))

        self.sensor_indices: dict[str, tuple[int, int]] = {}
        """A map from sensor names to their (initial index, size) in sensor data array."""

        self.metadata_ready = False
        self.laser_sensor_names: list[str] = []

        self.cmd_vx_mps = 0.0
        self.cmd_wz_radps = 0.0
        self.last_cmd_time = self.get_clock().now()

        self.create_subscription(
            String, "/mujoco/sensor_metadata", self._sensor_metadata_cb, LATCHED_QoS
        )
        self.create_subscription(
            Float64MultiArray, "/mujoco/sensor_data", self._sensor_data_cb, FAST_QoS
        )
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, FAST_QoS)

        self.scan_pub = self.create_publisher(LaserScan, "/laser_scans", LATCHED_QoS)
        self.odom_pub = self.create_publisher(Pose2D, "/gt/odometry", FAST_QoS)
        self.wheel_cmd_pub = self.create_publisher(
            Float64MultiArray, "/mujoco/joint_controls", FAST_QoS
        )

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.create_timer(1.0 / control_rate_hz, self._control_timer_cb)

    def _sensor_metadata_cb(self, msg: String) -> None:
        """Cache flattened sensor indices from the metadata payload."""
        try:
            meta = json.loads(msg.data)
            sensor_names = meta.get("sensor_names", [])
            sensor_dims = meta.get("sensor_dims", [])

            self.sensor_indices.clear()
            idx = 0
            for name, dim in zip(sensor_names, sensor_dims):
                self.sensor_indices[name] = (idx, int(dim))
                idx += int(dim)

            self.laser_sensor_names = sorted(
                [
                    name
                    for name in self.sensor_indices
                    if name.startswith(self.laser_sensor_prefix)
                ]
            )
            self.metadata_ready = True
            self.get_logger().info(
                f"Sensor metadata ready ({len(self.laser_sensor_names)} LiDAR beams)."
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to parse sensor metadata: {exc}")
            self.metadata_ready = False

    def _sensor_data_cb(self, msg: Float64MultiArray) -> None:
        """Process new sensor data from the MuJoCo simulator."""
        if not self.metadata_ready:
            return

        pos = self._read_sensor(msg.data, self.odom_pos_sensor, expected_dim=3)
        quat = self._read_sensor(msg.data, self.odom_quat_sensor, expected_dim=4)
        if pos is not None and quat is not None:
            pose = Pose2D()
            pose.x = float(pos[0])
            pose.y = float(pos[1])
            pose.theta = quat_to_yaw_rad(quat)

            if self.publish_odometry:
                self.odom_pub.publish(pose)

            if self.tf_broadcaster is not None:
                self._publish_tf_chain(pos=pos, quat=quat)

        if not self.laser_sensor_names:
            return

        ranges = []
        for sensor_name in self.laser_sensor_names:
            range_data = self._read_sensor(msg.data, sensor_name, expected_dim=1)
            if range_data is None:
                ranges.append(float("inf"))
                continue
            range_m = float(range_data[0])
            if range_m < self.laser_min_range_m or range_m > self.laser_max_range_m:
                ranges.append(float("inf"))
            else:
                ranges.append(self._apply_lidar_noise(range_m))

        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = self.laser_frame_id
        scan.range_min = self.laser_min_range_m
        scan.range_max = self.laser_max_range_m

        n = len(ranges)
        scan.angle_min = -0.5 * self.laser_fov_rad
        scan.angle_max = 0.5 * self.laser_fov_rad
        scan.angle_increment = self.laser_fov_rad / max(n - 1, 1)
        scan.ranges = ranges
        self.scan_pub.publish(scan)

    def _apply_lidar_noise(self, range_m: float) -> float:
        """Apply seeded range-dependent Gaussian noise and dropout.

        :param range_m: Raw LiDAR range (meters)
        :return: Noisy range in meters, or `inf` when dropped/out-of-bounds
        """
        if float(self._noise_rng.random()) < self.lidar_dropout_prob:
            return float("inf")

        sigma_m = self.lidar_noise_sigma0_m + self.lidar_noise_scale * max(range_m, 0.0)
        noisy_range_m = float(self._noise_rng.normal(loc=range_m, scale=sigma_m))
        if (
            noisy_range_m < self.laser_min_range_m
            or noisy_range_m > self.laser_max_range_m
        ):
            return float("inf")
        return noisy_range_m

    def _publish_tf_chain(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Publish map-to-odom, odom-to-base link, and base link-to-LiDAR transforms.

        :param pos: Robot odom-frame position as an array: [x, y, z]
        :param quat: Robot odom-frame orientation as a quaternion array: [w, x, y, z]
        """
        if self.tf_broadcaster is None:
            return

        stamp = self.get_clock().now().to_msg()

        map_to_odom_msg = TransformStamped()
        map_to_odom_msg.header.stamp = stamp
        map_to_odom_msg.header.frame_id = self.map_frame_id
        map_to_odom_msg.child_frame_id = self.odom_frame_id
        map_to_odom_msg.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(map_to_odom_msg)  # Map = Odom

        odom_to_base_msg = TransformStamped()
        odom_to_base_msg.header.stamp = stamp
        odom_to_base_msg.header.frame_id = self.odom_frame_id
        odom_to_base_msg.child_frame_id = self.base_frame_id
        odom_to_base_msg.transform.translation.x = float(pos[0])
        odom_to_base_msg.transform.translation.y = float(pos[1])
        odom_to_base_msg.transform.translation.z = float(pos[2])
        odom_to_base_msg.transform.rotation.w = float(quat[0])
        odom_to_base_msg.transform.rotation.x = float(quat[1])
        odom_to_base_msg.transform.rotation.y = float(quat[2])
        odom_to_base_msg.transform.rotation.z = float(quat[3])
        self.tf_broadcaster.sendTransform(odom_to_base_msg)

        base_to_laser_msg = TransformStamped()
        base_to_laser_msg.header.stamp = stamp
        base_to_laser_msg.header.frame_id = self.base_frame_id
        base_to_laser_msg.child_frame_id = self.laser_frame_id
        base_to_laser_msg.transform.translation.x = self.laser_offset_x_m
        base_to_laser_msg.transform.translation.y = self.laser_offset_y_m
        base_to_laser_msg.transform.translation.z = self.laser_offset_z_m
        base_to_laser_msg.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(base_to_laser_msg)

    def _cmd_vel_cb(self, msg: Twist) -> None:
        """Save the commanded body-frame velocities in member variables."""
        self.cmd_vx_mps = float(msg.linear.x)
        self.cmd_wz_radps = float(msg.angular.z)
        self.last_cmd_time = self.get_clock().now()

    def _control_timer_cb(self) -> None:
        """Callback to command the TurtleBot's wheel velocities."""
        now = self.get_clock().now()
        dt_s = (now - self.last_cmd_time).nanoseconds * 1e-9
        vx = self.cmd_vx_mps
        wz = self.cmd_wz_radps

        if dt_s > self.command_timeout_s:
            vx = 0.0
            wz = 0.0

        # Reference: Derived from Eq. 13.16 (pg. 726) of LaValle's "Planning Algorithms"
        left_radps = (vx - 0.5 * self.wheel_base_m * wz) / self.wheel_radius_m
        right_radps = (vx + 0.5 * self.wheel_base_m * wz) / self.wheel_radius_m

        wheel_cmd_msg = Float64MultiArray()
        wheel_cmd_msg.data = [left_radps, right_radps]
        self.wheel_cmd_pub.publish(wheel_cmd_msg)

    def _read_sensor(self, data, name: str, expected_dim: int) -> np.ndarray | None:
        """Read data for the named sensor from a combined sensor data array.

        :param data: Array of combined sensor data from MuJoCo
        :param name: Name of the sensor whose data is read
        :param expected_dim: Expected dimension of the read data
        :return: NumPy array containing the sensor data
        """
        sensor_info = self.sensor_indices.get(name)
        if sensor_info is None:
            return None
        start, dim = sensor_info
        if dim != expected_dim or start + dim > len(data):
            return None
        return np.array(data[start : start + dim], dtype=float)


def main() -> None:
    """Intialize and spin the TurtleBot bridge node."""
    rclpy.init()
    bridge_node = TurtleBotBridge()

    try:
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
