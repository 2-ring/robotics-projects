#!/usr/bin/env python3
"""Joint-space PID controller for a 7-DoF Panda arm tracking Cartesian targets.

Each control tick:

    1. Damped least-squares IK maps the Cartesian target → joint-space goal.
    2. A per-joint PID computes proportional, integral, and derivative terms.
    3. Gravity + Coriolis bias from MuJoCo's `mj_forward` is added on top.
    4. The resulting torques are published to /q2/joint_torques.

The integral term is anti-windup clamped to keep the arm from creeping after
large transients. The IK only re-solves when the target changes, since the
per-joint PID is what does the moving.

Topics:
    in   /q2/joint_positions, /q2/joint_velocities, /q2/ee_position    Float64MultiArray
         /q2/target_position                                            Float64MultiArray (3,)
    out  /q2/joint_torques                                              Float64MultiArray (7,)
"""

import os

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray

NUM_JOINTS = 7


class PIDController(Node):
    def __init__(self):
        super().__init__("pid_controller")

        q2_pkg = get_package_share_directory("arm_pid")
        model_path = os.path.join(q2_pkg, "models", "panda", "pid_scene.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Per-joint gains: larger joints (1-4) carry the bulk of the load and
        # need higher P/D; wrist joints stay soft to avoid jitter.
        self.Kp = np.array([60.0, 60.0, 55.0, 55.0, 30.0, 20.0, 10.0])
        self.Ki = np.array([0.5,  0.5,  0.5,  0.5,  0.3,  0.2,  0.1])
        self.Kd = np.array([12.0, 12.0, 10.0, 10.0, 4.0,  3.0,  1.5])

        self.ik_step_size = 0.5
        self.ik_iterations = 50
        self.ik_tolerance = 0.001

        self.joint_positions = None
        self.joint_velocities = np.zeros(NUM_JOINTS)
        self.ee_position = np.zeros(3)
        self.target_position = np.array([0.4, 0.0, 0.5])

        self.integral_error = np.zeros(NUM_JOINTS)
        self.prev_error = np.zeros(NUM_JOINTS)
        self.target_joints = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

        self.ik_converged = False
        self.sensor_ready = False

        fast = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Float64MultiArray, "/q2/joint_positions", self._joint_pos_cb, fast
        )
        self.create_subscription(
            Float64MultiArray, "/q2/joint_velocities", self._joint_vel_cb, fast
        )
        self.create_subscription(
            Float64MultiArray, "/q2/ee_position", self._ee_pos_cb, fast
        )
        self.create_subscription(
            Float64MultiArray, "/q2/target_position", self._target_cb, fast
        )

        self.torque_pub = self.create_publisher(
            Float64MultiArray, "/q2/joint_torques", 10
        )

        self.dt = 0.002
        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info("PID controller started. Waiting for sensor data...")

    def _joint_pos_cb(self, msg):
        self.joint_positions = np.array(msg.data[:NUM_JOINTS])
        if not self.sensor_ready:
            self.sensor_ready = True
            self.target_joints = self.joint_positions.copy()
            self.get_logger().info("Sensor data received. Controller active.")

    def _joint_vel_cb(self, msg):
        self.joint_velocities = np.array(msg.data[:NUM_JOINTS])

    def _ee_pos_cb(self, msg):
        self.ee_position = np.array(msg.data[:3])

    def _target_cb(self, msg):
        new_target = np.array(msg.data[:3])
        if not np.allclose(new_target, self.target_position, atol=0.005):
            self.target_position = new_target
            self.ik_converged = False

    def _compute_ik(self, target_pos):
        """Damped least-squares IK on the positional Jacobian.

        Iterates qpos += step * J^T (J J^T + lam^2 I)^{-1} (target - ee_pos)
        until ||error|| < ik_tolerance or ik_iterations is exhausted. The
        damping (lam=0.1) keeps the update stable near singularities.
        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:NUM_JOINTS] = self.joint_positions.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        ee_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center"
        )
        jacp = np.zeros((3, self.mj_model.nv))
        residual = np.inf

        for _ in range(self.ik_iterations):
            mujoco.mj_forward(self.mj_model, self.mj_data)
            ee_pos = self.mj_data.site_xpos[ee_site_id].copy()
            error = target_pos - ee_pos
            residual = np.linalg.norm(error)
            if residual < self.ik_tolerance:
                break
            mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, None, ee_site_id)
            J = jacp[:, :NUM_JOINTS]
            lam = 0.1
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), error)
            dq *= self.ik_step_size
            self.mj_data.qpos[:NUM_JOINTS] += dq

        converged = residual < self.ik_tolerance * 10  # within 1cm
        return self.mj_data.qpos[:NUM_JOINTS].copy(), converged, residual

    def _gravity_compensation(self):
        """Return gravity + Coriolis/centrifugal bias torques (qfrc_bias)."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:NUM_JOINTS] = self.joint_positions.copy()
        self.mj_data.qvel[:NUM_JOINTS] = self.joint_velocities.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.mj_data.qfrc_bias[:NUM_JOINTS].copy()

    def _control_loop(self):
        if not self.sensor_ready:
            return

        q = self.joint_positions
        qd = self.joint_velocities

        gravity = self._gravity_compensation()

        if not self.ik_converged:
            q_ik, converged, residual = self._compute_ik(self.target_position)
            if converged:
                self.target_joints = q_ik
                self.ik_converged = True
                self.integral_error[:] = 0.0
                self.get_logger().info(
                    f"IK solved (residual={residual:.4f}m). "
                    f"Target joints: {np.round(self.target_joints, 3)}"
                )
            else:
                # IK failed — hold position with gravity comp only.
                msg = Float64MultiArray()
                msg.data = gravity.tolist()
                self.torque_pub.publish(msg)
                return

        q_desired = self.target_joints

        error = q_desired - q
        P = self.Kp * error
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -0.5, 0.5)
        I = self.Ki * self.integral_error
        D = self.Kd * (-qd)
        gravity = self._gravity_compensation()
        torque = P + I + D + gravity

        msg = Float64MultiArray()
        msg.data = torque.tolist()
        self.torque_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
