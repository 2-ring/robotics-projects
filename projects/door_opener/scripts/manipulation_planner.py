#!/usr/bin/env python3
"""Articulated-object planner: a Panda arm opens a hinged door.

After the grasp controller seats the gripper on the handle, this node:

  1. Computes a sequence of handle waypoints that trace the arc of the door
     opening between the current and target hinge angles (Z-axis rotation).
  2. For each waypoint, derives the required hand orientation by composing
     the closed-door grasp quaternion with the door's current rotation, so
     the gripper rotates with the handle.
  3. Solves IK (damped least-squares, position+orientation) to get a joint
     target, and blends toward it (alpha=0.3) so the door's inertia doesn't
     yank the gripper off the handle.
  4. Re-plans periodically — the end-effector drifts off the arc as the door
     swings, so a finer trajectory from the current angle keeps tracking tight.

Door kinematics (hinge position/axis, handle offset, target angle) live as
module constants.
"""

import math
import os
import numpy as np

import mujoco

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseArray, Pose

from skills_utils import FAST_QoS
from door_opener.ik import solve_panda_ik, quat_to_rot, TCP_OFFSET_LOCAL


# Door geometry (from the MuJoCo scene).
HINGE_POSITION = np.array([0.35, 0.35, 0.42])
HINGE_AXIS = np.array([0.0, 0.0, 1.0])
HANDLE_OFFSET_FROM_HINGE = np.array([0.225, -0.085, 0.35])
TARGET_ANGLE_RAD = -1.3  # ≈ -75°

# Closed-door grasp quaternion (gripper Z into the door). Must match the
# grasp controller; rotates with the door angle in compute_grasp_quat_at_angle.
GRASP_ORIENTATION_WXYZ = (0.0, 0.0, 0.92388, 0.38268)

# Pregrasp offset at door_angle = 0 (-Y is "in front of the handle"); rotated
# about Z by the current door angle when asking the grasper to re-grasp a
# partially-open door.
BASE_PREGRASP_OFFSET = np.array([0.0, -0.08, 0.0])


def _quat_mul(q1, q2):
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


class ManipulationPlanner(Node):

    def __init__(self):
        super().__init__("manipulation_planner")

        self.joint_positions = None
        self.joint_velocities = None
        self.ee_position = None
        self.ee_orientation = None
        self.door_angle = 0.0
        self.handle_position = None
        self.handle_orientation = None
        self.grasp_status = "approaching"
        self.goal_reached = False

        self._trajectory_angles = []
        self.since_last_waypoint = 0
        self.trajectory = []
        self.next_pose = None
        self.waypoint_index = 0

        self.mj_model = None
        self.mj_data = None
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg = get_package_share_directory("door_opener")
            model_path = os.path.join(pkg, "models", "door_manipulation_scene.xml")
            self.mj_model = mujoco.MjModel.from_xml_path(model_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            self.get_logger().info("MuJoCo model loaded for IK.")
        except Exception as e:
            self.get_logger().warn(f"Could not load MuJoCo model for IK: {e}")

        self.create_subscription(
            Float64MultiArray, "/q2/joint_positions", self._jp_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/joint_velocities", self._jv_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/ee_position", self._ee_pos_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/ee_orientation", self._ee_quat_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/door_angle", self._door_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/handle_position", self._handle_pos_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/handle_orientation", self._handle_quat_cb, FAST_QoS)
        self.create_subscription(
            String, "/q2/grasp_status", self._grasp_cb, FAST_QoS)
        self.create_subscription(
            String, "/q2/goal_reached", self._goal_cb, FAST_QoS)

        self.target_pub = self.create_publisher(
            Float64MultiArray, "/panda/position_targets", FAST_QoS)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, "/panda/gripper_command", FAST_QoS)
        self.plan_pub = self.create_publisher(
            PoseArray, "/q2/planned_waypoints", 10)
        self.regrasp_pub = self.create_publisher(
            Float64MultiArray, "/q2/regrasp", 10)

        self.create_timer(0.02, self._control_loop)  # 50 Hz

        self.get_logger().info("Manipulation planner started. Waiting for grasp...")

    def _jp_cb(self, msg):
        self.joint_positions = np.array(msg.data)

    def _jv_cb(self, msg):
        self.joint_velocities = np.array(msg.data)

    def _ee_pos_cb(self, msg):
        self.ee_position = np.array(msg.data)

    def _ee_quat_cb(self, msg):
        self.ee_orientation = np.array(msg.data)

    def _door_cb(self, msg):
        if msg.data:
            self.door_angle = msg.data[0]

    def _handle_pos_cb(self, msg):
        self.handle_position = np.array(msg.data)

    def _handle_quat_cb(self, msg):
        self.handle_orientation = np.array(msg.data)

    def _grasp_cb(self, msg):
        self.grasp_status = msg.data

    def _goal_cb(self, msg):
        self.goal_reached = (msg.data == "true")

    def compute_handle_position_at_angle(self, angle_rad: float) -> np.ndarray:
        """World-frame handle position at a given hinge angle.

        Rotates HANDLE_OFFSET_FROM_HINGE about the Z axis and translates by
        HINGE_POSITION.
        """
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad),  math.cos(angle_rad), 0],
            [0,                    0,                   1],
        ])
        rotated_offset = rotation_matrix @ HANDLE_OFFSET_FROM_HINGE
        return HINGE_POSITION + rotated_offset

    def plan_door_trajectory(self, current_angle: float, target_angle: float,
                              num_waypoints: int = 20) -> list[np.ndarray]:
        """Linear interpolation of door angles → handle positions along the arc."""
        total_difference = target_angle - current_angle
        angle_increment = total_difference / num_waypoints
        self._trajectory_angles = [
            current_angle + index * angle_increment for index in range(1, num_waypoints + 1)
        ]
        return [self.compute_handle_position_at_angle(angle) for angle in self._trajectory_angles]

    def compute_grasp_quat_at_angle(self, angle_rad: float) -> tuple:
        """Grasp orientation at a given door angle.

        Composes a Z-axis rotation (by `angle_rad`) with the closed-door grasp
        quaternion so the gripper rotates together with the door.
        """
        q_rotation = (math.cos(angle_rad / 2), 0.0, 0.0, math.sin(angle_rad / 2))
        return _quat_mul(q_rotation, GRASP_ORIENTATION_WXYZ)

    def compute_joint_targets(
        self,
        desired_ee_pos: np.ndarray,
        target_quat_wxyz: tuple = GRASP_ORIENTATION_WXYZ,
    ) -> np.ndarray | None:
        """Thin wrapper around the shared damped-LS Panda IK solver."""
        if self.joint_positions is None:
            return None
        return solve_panda_ik(
            self.mj_model, self.mj_data, self.joint_positions,
            desired_ee_pos, target_quat_wxyz=target_quat_wxyz,
        )

    def _publish_plan(self, waypoints):
        """Publish the trajectory waypoints as a PoseArray for RViz visualization."""
        msg = PoseArray()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        for i, pos in enumerate(waypoints):
            p = Pose()
            p.position.x = float(pos[0])
            p.position.y = float(pos[1])
            p.position.z = float(pos[2])
            angle = (self._trajectory_angles[i]
                     if hasattr(self, '_trajectory_angles') else 0.0)
            qw, qx, qy, qz = self.compute_grasp_quat_at_angle(angle)
            p.orientation.w = qw
            p.orientation.x = qx
            p.orientation.y = qy
            p.orientation.z = qz
            msg.poses.append(p)
        self.plan_pub.publish(msg)

    def _calculate_next_pose(self):
        self.next_pose = self.compute_joint_targets(
            self.position_trajectory[self.waypoint_index],
            self._trajectory_quants[self.waypoint_index],
        )

    def _control_loop(self):
        """50 Hz: re-plan when stale, advance through waypoints, emit blended targets."""
        if self.joint_positions is None:
            return

        if self.grasp_status != "ready":
            return

        if self.goal_reached:
            return

        WAYPOINT_THRESHOLD = 0.08
        REPLAN_INTERVAL = 100

        # Replan from the current door angle on first entry or when stale —
        # the EE drifts off the arc as the door swings.
        if self.next_pose is None or self.since_last_waypoint > REPLAN_INTERVAL:
            self.position_trajectory = self.plan_door_trajectory(
                self.door_angle, TARGET_ANGLE_RAD, 40
            )
            self._trajectory_quants = [self.compute_grasp_quat_at_angle(pos)
                                       for pos in self._trajectory_angles]
            self.waypoint_index = 0
            self._calculate_next_pose()

        self.since_last_waypoint += 1
        # TCP (between fingertips) lives at hand_pos + R_hand @ TCP_OFFSET_LOCAL.
        tcp_offset_global = quat_to_rot(*self.ee_orientation) @ TCP_OFFSET_LOCAL
        tcp_position = self.ee_position + tcp_offset_global
        waypoint_distance = np.linalg.norm(tcp_position - self.position_trajectory[self.waypoint_index])

        # Hold each waypoint for at least 10 ticks to cap speed and let the
        # door's inertia settle before commanding the next target.
        if waypoint_distance < WAYPOINT_THRESHOLD and self.since_last_waypoint > 10:
            self.since_last_waypoint = 0
            if self.waypoint_index < len(self.position_trajectory) - 1:
                self.waypoint_index += 1
            self._calculate_next_pose()

        if self.next_pose is not None:
            # Blend toward the IK target so the door's inertia doesn't pry
            # the gripper off the handle on aggressive joint deltas.
            alpha = 0.3
            blended = self.joint_positions + alpha * (self.next_pose - self.joint_positions)
            self.target_pub.publish(Float64MultiArray(data=blended.tolist()))


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
