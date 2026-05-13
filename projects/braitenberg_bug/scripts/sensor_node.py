#!/usr/bin/env python3
"""
Q1 Sensor Node — Provided to students (DO NOT MODIFY).

Bridges between the generic MuJoCo sim and the student's Braitenberg controller.

Published topics (for the student to subscribe to):
  /q1/left_sensor_distance    std_msgs/Float64   — distance from left antenna to light
  /q1/right_sensor_distance   std_msgs/Float64   — distance from right antenna to light

Subscribed topics (for the student to publish to):
  /q1/left_wheel_velocity     std_msgs/Float64   — desired left wheel speed  (rad/s)
  /q1/right_wheel_velocity    std_msgs/Float64   — desired right wheel speed (rad/s)

Also:
  /q1/set_light_position      — move the light source (used by test script)
"""

import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float64, Float64MultiArray, String
from geometry_msgs.msg import Point


class SensorNode(Node):

    def __init__(self):
        super().__init__('sensor_node')

        # ── State ───────────────────────────────────────────────────
        self.sensor_indices = {}     # sensor_name → (flat_index, dim)
        self.metadata_ready = False
        self.left_wheel_cmd  = 0.0
        self.right_wheel_cmd = 0.0

        # ── QoS ─────────────────────────────────────────────────────
        fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ── Subscribe to raw sim data ───────────────────────────────
        self.create_subscription(
            String, '/mujoco/sensor_metadata', self._metadata_cb, latched)
        self.create_subscription(
            Float64MultiArray, '/mujoco/sensor_data', self._sensor_cb, fast)

        # ── Subscribe to student's wheel commands ───────────────────
        self.create_subscription(
            Float64, '/q1/left_wheel_velocity', self._left_wheel_cb, 10)
        self.create_subscription(
            Float64, '/q1/right_wheel_velocity', self._right_wheel_cb, 10)

        # ── Publish processed sensor data for students ──────────────
        self.left_dist_pub   = self.create_publisher(Float64, '/q1/left_sensor_distance', 10)
        self.right_dist_pub  = self.create_publisher(Float64, '/q1/right_sensor_distance', 10)

        # ── Publish wheel commands to sim ───────────────────────────
        self.ctrl_pub = self.create_publisher(
            Float64MultiArray, '/mujoco/joint_controls', fast)

        # ── Publish mocap position to sim (for moving light) ────────
        self.mocap_pub = self.create_publisher(
            Float64MultiArray, '/mujoco/mocap_pos', fast)

        # ── Light repositioning: test script publishes new position ──
        self.create_subscription(
            Point, '/q1/set_light_position', self._set_light_cb, 10)

        # ── Forward wheel commands at 100 Hz ────────────────────────
        self.create_timer(0.01, self._forward_wheel_commands)

        self.get_logger().info('Sensor node ready.')
        self.get_logger().info('  Publishing:   /q1/left_sensor_distance, /q1/right_sensor_distance')
        self.get_logger().info('  Subscribing:  /q1/left_wheel_velocity,  /q1/right_wheel_velocity')

    # ================================================================
    #  Metadata
    # ================================================================

    def _metadata_cb(self, msg: String):
        try:
            meta = json.loads(msg.data)
            names = meta['sensor_names']
            dims  = meta['sensor_dims']
            idx = 0
            for name, dim in zip(names, dims):
                self.sensor_indices[name] = (idx, dim)
                idx += dim
            self.metadata_ready = True
            self.get_logger().info(f'Sensor metadata received: {list(self.sensor_indices.keys())}')
        except Exception as e:
            self.get_logger().error(f'Bad metadata: {e}')

    # ================================================================
    #  Sensor data → compute distances → publish
    # ================================================================

    def _sensor_cb(self, msg: Float64MultiArray):
        if not self.metadata_ready:
            return
        data = msg.data

        # Read 3D positions from the flat sensor array
        left_pos  = self._read_vec3(data, 'light_left_pos')
        right_pos = self._read_vec3(data, 'light_right_pos')
        light_pos = self._read_vec3(data, 'light_source_pos')

        if left_pos is None or right_pos is None or light_pos is None:
            return

        # Compute 2D distance (xy plane) from each sensor to the light's ground position
        # This represents distance to the light pool on the floor
        left_dist  = float(np.linalg.norm(left_pos[:2] - light_pos[:2]))
        right_dist = float(np.linalg.norm(right_pos[:2] - light_pos[:2]))

        # Publish distances
        ld = Float64(); ld.data = left_dist
        rd = Float64(); rd.data = right_dist
        self.left_dist_pub.publish(ld)
        self.right_dist_pub.publish(rd)

    def _read_vec3(self, data, name):
        if name not in self.sensor_indices:
            return None
        start, dim = self.sensor_indices[name]
        if dim != 3 or start + 3 > len(data):
            return None
        return np.array(data[start:start+3])

    # ================================================================
    #  Student wheel commands
    # ================================================================

    def _left_wheel_cb(self, msg: Float64):
        self.left_wheel_cmd = msg.data

    def _right_wheel_cb(self, msg: Float64):
        self.right_wheel_cmd = msg.data

    def _forward_wheel_commands(self):
        """Forward student's wheel velocity commands to the MuJoCo sim."""
        msg = Float64MultiArray()
        msg.data = [self.left_wheel_cmd, self.right_wheel_cmd]
        self.ctrl_pub.publish(msg)

    # ================================================================
    #  Light repositioning (used by test script)
    # ================================================================

    def _set_light_cb(self, msg: Point):
        """Move the light source to a new position."""
        mocap_msg = Float64MultiArray()
        mocap_msg.data = [msg.x, msg.y, msg.z]
        self.mocap_pub.publish(mocap_msg)


# ════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
