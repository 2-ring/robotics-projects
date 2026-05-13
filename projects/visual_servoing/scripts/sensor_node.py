#!/usr/bin/env python3
"""
Q3 Sensor / Bridge Node

Bridges between the generic MuJoCo sim node and the Q3 visual servoing controller.

This node:
  1. Reads raw sensor data from /mujoco/sensor_data → publishes clean joint topics
  2. Reads the red ball's TRUE world position from the sim (mocap body)
     and publishes it as if detected by the overhead camera, expressed in
     the CAMERA frame.  (No actual rendering needed — we simulate a
     "perfect" vision pipeline.)
  3. Publishes the camera intrinsics & extrinsics so students can transform.
  4. Forwards student joint velocity commands to the sim.

Subscribes (from sim):
    /mujoco/sensor_data         Float64MultiArray
    /mujoco/sensor_metadata     String  (latched JSON)

Publishes (for student):
    /q3/joint_positions         Float64MultiArray  (7 joint angles, rad)
    /q3/joint_velocities        Float64MultiArray  (7 joint velocities, rad/s)
    /q3/ee_position             Float64MultiArray  (3: x, y, z in robot/world frame)
    /q3/ee_orientation          Float64MultiArray  (4: w, x, y, z quaternion in world frame)
    /q3/ball_position_camera    Float64MultiArray  (3: x, y, z in CAMERA frame)
    /q3/camera_extrinsics       Float64MultiArray  (16: 4×4 camera-to-world transform, row-major)

Subscribes (from student):
    /q3/joint_velocity_command  Float64MultiArray  (7 desired joint velocities, rad/s)

Subscribes (from test):
    /q3/set_ball_position       Float64MultiArray  (3: x, y, z — move the ball)

Forwards:
    /q3/joint_velocity_command  →  /mujoco/joint_controls
    /q3/set_ball_position       →  /mujoco/mocap_pos

Camera frame convention (overhead camera looking down):
    Camera is at world position (0.3, 0.0, 1.5), pointing straight down (-Z).
    Camera X → world -Y  (right in image)
    Camera Y → world -X  (down in image — but we negate for intuition)
    Camera Z → world -Z  (into the scene)

    The camera-to-world transform is published so students can invert it.
"""

import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, String


NUM_JOINTS = 7

# ── Camera extrinsics (matches visual_servo_scene.xml) ──────────────
# Camera body is at world pos (0.5, 0.0, 1.8)
# Camera quat in XML: "0 0.707107 0.707107 0" → 180° rotation about Y axis
# This means the camera looks straight down (-Z world).
#
# MuJoCo camera convention: camera looks along -Z_cam, X_cam is right, Y_cam is down.
# With the quat "0 0.707107 0.707107 0":
#   Camera X_cam → World +Y
#   Camera Y_cam → World +X
#   Camera Z_cam → World +Z  (so -Z_cam → -Z_world = looking down)
#
# Camera-to-world rotation matrix:
#   R_cam_to_world = [[0, 0, 0],   — we compute from quat below
#                      [...],
#                      [...]]

CAM_POS_WORLD = np.array([0.5, 0.0, 1.8])

# Quaternion from XML: w=0, x=0.707107, y=0.707107, z=0
# MuJoCo quat format: (w, x, y, z)
CAM_QUAT = np.array([0.0, 0.707107, 0.707107, 0.0])


def quat_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def build_cam_to_world():
    """Build 4×4 camera-to-world homogeneous transform."""
    R = quat_to_rotation_matrix(CAM_QUAT)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = CAM_POS_WORLD
    return T


CAM_TO_WORLD = build_cam_to_world()
WORLD_TO_CAM = np.linalg.inv(CAM_TO_WORLD)


class Q3SensorNode(Node):

    def __init__(self):
        super().__init__('q3_sensor_node')

        fast = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ── From sim ────────────────────────────────────────────────
        self.create_subscription(
            Float64MultiArray, '/mujoco/sensor_data', self._sensor_cb, fast)
        self.create_subscription(
            String, '/mujoco/sensor_metadata', self._meta_cb, latched)

        # ── To student ──────────────────────────────────────────────
        self.joint_pos_pub       = self.create_publisher(Float64MultiArray, '/q3/joint_positions',       fast)
        self.joint_vel_pub       = self.create_publisher(Float64MultiArray, '/q3/joint_velocities',      fast)
        self.ee_pos_pub          = self.create_publisher(Float64MultiArray, '/q3/ee_position',           fast)
        self.ee_quat_pub         = self.create_publisher(Float64MultiArray, '/q3/ee_orientation',        fast)
        self.ball_cam_pub        = self.create_publisher(Float64MultiArray, '/q3/ball_position_camera',  fast)
        self.cam_extrinsics_pub  = self.create_publisher(Float64MultiArray, '/q3/camera_extrinsics',     latched)

        # ── From student ────────────────────────────────────────────
        self.create_subscription(
            Float64MultiArray, '/q3/joint_velocity_command', self._vel_cmd_cb, reliable)

        # ── From test script ────────────────────────────────────────
        self.create_subscription(
            Float64MultiArray, '/q3/set_ball_position', self._set_ball_cb, reliable)

        # ── To sim ──────────────────────────────────────────────────
        self.ctrl_pub  = self.create_publisher(Float64MultiArray, '/mujoco/joint_controls', fast)
        self.mocap_pub = self.create_publisher(Float64MultiArray, '/mujoco/mocap_pos',      fast)

        # Sensor metadata
        self.sensor_names   = []
        self.sensor_dims    = []
        self.sensor_offsets = {}

        # Ball position (world frame, from mocap)
        self.ball_pos_world = np.array([0.5, 0.0, 0.445])

        # Publish camera extrinsics (latched — sent once)
        self._publish_camera_extrinsics()

        self.get_logger().info('Q3 sensor node started.')

    def _publish_camera_extrinsics(self):
        """Publish the 4×4 camera-to-world transform (row-major) so students can use it."""
        msg = Float64MultiArray()
        msg.data = CAM_TO_WORLD.flatten().tolist()
        self.cam_extrinsics_pub.publish(msg)
        self.get_logger().info(
            f'Camera extrinsics published. Cam pos: {CAM_POS_WORLD}')

    def _meta_cb(self, msg):
        """Parse sensor metadata to know offsets into the flat sensor vector."""
        meta = json.loads(msg.data)
        self.sensor_names = meta['sensor_names']
        self.sensor_dims  = meta['sensor_dims']
        offset = 0
        for name, dim in zip(self.sensor_names, self.sensor_dims):
            self.sensor_offsets[name] = offset
            offset += dim
        self.get_logger().info(f'Sensor metadata: {self.sensor_names}')

    def _sensor_cb(self, msg):
        """Unpack raw sensor data and republish as clean topics."""
        if not self.sensor_offsets:
            return

        data = msg.data

        # Joint positions
        joint_pos = Float64MultiArray()
        joint_pos.data = [
            data[self.sensor_offsets[f'joint{i+1}_pos']]
            for i in range(NUM_JOINTS)
        ]
        self.joint_pos_pub.publish(joint_pos)

        # Joint velocities
        joint_vel = Float64MultiArray()
        joint_vel.data = [
            data[self.sensor_offsets[f'joint{i+1}_vel']]
            for i in range(NUM_JOINTS)
        ]
        self.joint_vel_pub.publish(joint_vel)

        # End-effector position (world frame)
        ee_pos = Float64MultiArray()
        off = self.sensor_offsets['ee_pos']
        ee_pos.data = [data[off], data[off+1], data[off+2]]
        self.ee_pos_pub.publish(ee_pos)

        # End-effector orientation (quaternion w,x,y,z in world frame)
        ee_quat = Float64MultiArray()
        off_q = self.sensor_offsets['ee_quat']
        ee_quat.data = [data[off_q], data[off_q+1], data[off_q+2], data[off_q+3]]
        self.ee_quat_pub.publish(ee_quat)

        # Ball position in camera frame
        # (Simulates a "perfect" overhead camera detecting the red ball)
        ball_world_h = np.array([*self.ball_pos_world, 1.0])
        ball_cam_h = WORLD_TO_CAM @ ball_world_h
        ball_cam = ball_cam_h[:3]

        ball_cam_msg = Float64MultiArray()
        ball_cam_msg.data = ball_cam.tolist()
        self.ball_cam_pub.publish(ball_cam_msg)

    def _vel_cmd_cb(self, msg):
        """Forward student velocity commands → sim controls."""
        ctrl = Float64MultiArray()
        ctrl.data = list(msg.data[:NUM_JOINTS])
        self.ctrl_pub.publish(ctrl)

    def _set_ball_cb(self, msg):
        """Move the red ball mocap body and remember position."""
        if len(msg.data) >= 3:
            self.ball_pos_world = np.array([msg.data[0], msg.data[1], msg.data[2]])
            mocap = Float64MultiArray()
            mocap.data = self.ball_pos_world.tolist()
            self.mocap_pub.publish(mocap)


def main(args=None):
    rclpy.init(args=args)
    node = Q3SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
