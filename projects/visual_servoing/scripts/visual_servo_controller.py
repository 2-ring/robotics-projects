#!/usr/bin/env python3
"""Jacobian-based visual servoing: a Panda arm tracks a red ball on a table.

An overhead camera publishes the ball position in the camera frame plus the
4x4 camera-to-world extrinsics. The controller:

    1. Transforms the ball into the world frame via homogeneous coordinates.
    2. Builds a 6-D twist:
         - linear:  P-control on (target_xy, fixed_z) → end-effector position
         - angular: orientation error around a downward-pointing gripper goal,
                    extracted from the skew-symmetric part of R_desired @ R_eeᵀ.
    3. Inverts the 6x7 manipulator Jacobian with damped least-squares:
                 dq = Jᵀ (J Jᵀ + λ² I)⁻¹ v_desired
    4. Clamps and publishes joint velocity commands at 100 Hz.

Topics:
    in   /q3/joint_positions, /q3/joint_velocities, /q3/ee_position,
         /q3/ee_orientation, /q3/ball_position_camera, /q3/camera_extrinsics
    out  /q3/joint_velocity_command   Float64MultiArray (7,)
"""

import os

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

NUM_JOINTS = 7


def quat_to_rotation_matrix(q):
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class VisualServoController(Node):
    def __init__(self):
        super().__init__("visual_servo_controller")

        q3_pkg = get_package_share_directory("visual_servoing")
        model_path = os.path.join(q3_pkg, "models", "visual_servo_scene.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.gain = 8.0       # P-gain for XY tracking
        self.fixed_z = 0.55   # cruising height above the table (m)
        self.z_gain = 5.0     # P-gain for Z hold
        self.ori_gain = 3.0   # P-gain for orientation correction
        self.damping = 0.01   # DLS regularizer (λ²)
        self.max_vel = 2.0    # per-joint velocity clamp (rad/s)

        # Desired gripper pose: pointing straight down at the table.
        self.R_desired = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )

        self.joint_positions = None
        self.joint_velocities = np.zeros(NUM_JOINTS)
        self.position_w_ee = np.zeros(3)
        self.orientation_w_ee = None
        self.position_c_b = None
        self.transform_w_c = None

        fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            Float64MultiArray, "/q3/joint_positions", self._joint_positions_cb, fast
        )
        self.create_subscription(
            Float64MultiArray, "/q3/joint_velocities", self._joint_velocities_cb, fast
        )
        self.create_subscription(
            Float64MultiArray, "/q3/ee_position", self._ee_position_cb, fast
        )
        self.create_subscription(
            Float64MultiArray, "/q3/ee_orientation", self._ee_orientation_cb, fast
        )
        self.create_subscription(
            Float64MultiArray, "/q3/ball_position_camera", self._ball_position_cb, fast
        )
        # Extrinsics are static and published once — use a latched topic so
        # late subscribers still receive the most recent value.
        self.create_subscription(
            Float64MultiArray, "/q3/camera_extrinsics", self._camera_extrinsics_cb, latched
        )

        self.vel_pub = self.create_publisher(
            Float64MultiArray, "/q3/joint_velocity_command", 10
        )

        self.dt = 0.01
        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info("Visual servo controller started. Waiting for data...")

    def _joint_positions_cb(self, msg):
        self.joint_positions = np.array(msg.data[:NUM_JOINTS])

    def _joint_velocities_cb(self, msg):
        self.joint_velocities = np.array(msg.data[:NUM_JOINTS])

    def _ee_position_cb(self, msg):
        self.position_w_ee = np.array(msg.data[:3])

    def _ee_orientation_cb(self, msg):
        self.orientation_w_ee = np.array(msg.data[:4])

    def _ball_position_cb(self, msg):
        self.position_c_b = np.array(msg.data[:3])

    def _camera_extrinsics_cb(self, msg):
        self.transform_w_c = np.array(msg.data).reshape(4, 4)

    def _compute_jacobian(self):
        """Return the 6x7 manipulator Jacobian at the current configuration.

        Top three rows are the positional Jacobian (v_ee = Jp @ dq); bottom
        three are the rotational Jacobian (ω_ee = Jr @ dq).
        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:NUM_JOINTS] = self.joint_positions.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        ee_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center"
        )
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, ee_site_id)

        return np.vstack([jacp[:, :NUM_JOINTS], jacr[:, :NUM_JOINTS]])  # 6x7

    def _control_loop(self):
        if (
            self.joint_positions is None
            or self.position_c_b is None
            or self.transform_w_c is None
            or self.orientation_w_ee is None
        ):
            return

        # Camera frame → world frame (homogeneous).
        h_position_c_b = np.append(self.position_c_b, 1)
        h_position_w_b = self.transform_w_c @ h_position_c_b
        position_w_b = h_position_w_b[:3]

        # Linear velocity: track the ball in XY, hold the cruising height.
        target_w_ee = np.array([position_w_b[0], position_w_b[1], self.fixed_z])
        error = target_w_ee - self.position_w_ee
        v_linear = np.array(
            [self.gain * error[0], self.gain * error[1], self.z_gain * error[2]]
        )

        # Angular velocity: rotation error from skew-symmetric part of R_err.
        R_current = quat_to_rotation_matrix(self.orientation_w_ee)
        R_err = self.R_desired @ R_current.T
        e_ori = 0.5 * np.array(
            [
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ]
        )
        v_angular = self.ori_gain * e_ori

        # Damped least-squares pseudo-inverse.
        v_desired = np.concatenate([v_linear, v_angular])
        J = self._compute_jacobian()
        A = J @ J.T + self.damping * np.identity(6)
        dq = J.T @ np.linalg.solve(A, v_desired)
        dq = np.clip(dq, -self.max_vel, self.max_vel)

        msg = Float64MultiArray()
        msg.data = dq.tolist()
        self.vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
