#!/usr/bin/env python3
"""
Q2 Sensor / Bridge Node

Bridges between the generic MuJoCo sim node and the Q2 PID controller.

Subscribes (from sim):
    /mujoco/sensor_data         Float64MultiArray  (raw sensor vector)

Publishes (for student):
    /q2/joint_positions         Float64MultiArray  (7 joint angles, rad)
    /q2/joint_velocities        Float64MultiArray  (7 joint velocities, rad/s)
    /q2/ee_position             Float64MultiArray  (3: x, y, z of end-effector)
    /q2/target_position         Float64MultiArray  (3: x, y, z of target mocap body)

Subscribes (from student):
    /q2/joint_torques           Float64MultiArray  (7 torques, N·m)
    /q2/set_target_position     Float64MultiArray  (3: x, y, z — move the target)

Forwards:
    /q2/joint_torques   →  /mujoco/joint_controls
    /q2/set_target_position  →  /mujoco/mocap_pos

Sensor layout in pid_scene.xml (16 sensors):
    0-6:   joint1_pos .. joint7_pos   (7 × 1D)
    7-13:  joint1_vel .. joint7_vel   (7 × 1D)
    14:    ee_pos                     (1 × 3D)
    15:    ee_quat                    (1 × 4D)
    → total flat vector length = 7 + 7 + 3 + 4 = 21
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, String


NUM_JOINTS = 7


class Q2SensorNode(Node):

    def __init__(self):
        super().__init__('q2_sensor_node')

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
        self.joint_pos_pub    = self.create_publisher(Float64MultiArray, '/q2/joint_positions',  fast)
        self.joint_vel_pub    = self.create_publisher(Float64MultiArray, '/q2/joint_velocities', fast)
        self.ee_pos_pub       = self.create_publisher(Float64MultiArray, '/q2/ee_position',      fast)
        self.target_pos_pub   = self.create_publisher(Float64MultiArray, '/q2/target_position',  fast)

        # ── From student ────────────────────────────────────────────
        self.create_subscription(
            Float64MultiArray, '/q2/joint_torques', self._torque_cb, reliable)
        self.create_subscription(
            Float64MultiArray, '/q2/set_target_position', self._set_target_cb, reliable)

        # ── To sim ──────────────────────────────────────────────────
        self.ctrl_pub  = self.create_publisher(Float64MultiArray, '/mujoco/joint_controls', fast)
        self.mocap_pub = self.create_publisher(Float64MultiArray, '/mujoco/mocap_pos',      fast)

        # Sensor metadata (parsed from JSON)
        self.sensor_names = []
        self.sensor_dims  = []
        self.sensor_offsets = {}  # name → offset in flat vector

        # Store target position from mocap (initially from scene)
        self.target_pos = [0.4, 0.0, 0.5]

        self.get_logger().info('Q2 sensor node started.')

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
            return  # metadata not yet received

        data = msg.data

        # Joint positions (7 × 1D sensors: joint1_pos .. joint7_pos)
        joint_pos = Float64MultiArray()
        joint_pos.data = [
            data[self.sensor_offsets[f'joint{i+1}_pos']]
            for i in range(NUM_JOINTS)
        ]
        self.joint_pos_pub.publish(joint_pos)

        # Joint velocities (7 × 1D sensors: joint1_vel .. joint7_vel)
        joint_vel = Float64MultiArray()
        joint_vel.data = [
            data[self.sensor_offsets[f'joint{i+1}_vel']]
            for i in range(NUM_JOINTS)
        ]
        self.joint_vel_pub.publish(joint_vel)

        # End-effector position (3D sensor: ee_pos)
        ee_pos = Float64MultiArray()
        off = self.sensor_offsets['ee_pos']
        ee_pos.data = [data[off], data[off+1], data[off+2]]
        self.ee_pos_pub.publish(ee_pos)

        # Target position (from stored mocap position)
        target = Float64MultiArray()
        target.data = list(self.target_pos)
        self.target_pos_pub.publish(target)

    def _torque_cb(self, msg):
        """Forward student torques → sim controls."""
        ctrl = Float64MultiArray()
        ctrl.data = list(msg.data[:NUM_JOINTS])
        self.ctrl_pub.publish(ctrl)

    def _set_target_cb(self, msg):
        """Move the target mocap body and remember position."""
        if len(msg.data) >= 3:
            self.target_pos = [msg.data[0], msg.data[1], msg.data[2]]
            mocap = Float64MultiArray()
            mocap.data = list(self.target_pos)
            self.mocap_pub.publish(mocap)


def main(args=None):
    rclpy.init(args=args)
    node = Q2SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
