#!/usr/bin/env python3
"""Q2 Grasp Controller — moves the Panda to a pre-grasp pose and grasps the door handle.

This controller:
  1. Waits for handle position from sensor.
  2. Uses IK to move to a pre-grasp pose (offset from handle along approach direction).
  3. Uses IK to move to the grasp position (at the handle).
  4. Closes the gripper.
  5. Publishes /q2/grasp_status = "ready" when the robot is grasping the handle.

The student's manipulation_planner should wait for grasp_status == "ready"
before beginning to plan and execute the door opening trajectory.

Subscribes:
  /q2/joint_positions    std_msgs/Float64MultiArray
  /q2/handle_position    std_msgs/Float64MultiArray
  /q2/ee_position        std_msgs/Float64MultiArray

Publishes:
  /panda/position_targets   std_msgs/Float64MultiArray  (7 joint angles)
  /panda/gripper_command    std_msgs/Float64MultiArray  (gripper width)
  /q2/grasp_status          std_msgs/String  ("approaching" | "grasping" | "ready")

This node is PROVIDED and should NOT be modified.
"""

import time
import os
import numpy as np

import mujoco

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import PoseArray, Pose

from skills_utils import FAST_QoS
from door_opener.ik import solve_panda_ik


# Initial grasp orientation at door_angle = 0 (gripper Z axis points along -Y,
# into the door surface). This is the orientation the controller uses on the
# very first grasp; on re-grasp the planner sends a new orientation on the
# /q2/regrasp topic that reflects the current door angle.
INITIAL_GRASP_ORIENTATION_WXYZ = (0.0, 0.0, 0.92388, 0.38268)

# Initial approach offset at door_angle = 0: come from in front of the handle
# (along -Y in world frame). Like the orientation, this is replaced on re-grasp
# by a value supplied by the planner.
INITIAL_PREGRASP_OFFSET = np.array([0.0, -0.08, 0.0])


class GraspController(Node):

    def __init__(self):
        super().__init__("grasp_controller")

        self.joint_positions = None
        self.handle_position = None
        self.ee_position = None
        self.phase = "waiting"
        self.phase_start_time = time.time()

        self.pregrasp_joints = None
        self.grasp_joints = None

        # Current approach offset and grasp orientation. These are replaced
        # whenever a /q2/regrasp message arrives carrying new values.
        self.pregrasp_offset = np.array(INITIAL_PREGRASP_OFFSET, dtype=float)
        self.current_grasp_quat = tuple(INITIAL_GRASP_ORIENTATION_WXYZ)

        # Load MuJoCo model for IK
        self.mj_model = None
        self.mj_data = None
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg = get_package_share_directory("door_opener")
            model_path = os.path.join(pkg, "models", "door_manipulation_scene.xml")
            self.mj_model = mujoco.MjModel.from_xml_path(model_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            self.get_logger().info("MuJoCo model loaded for grasp IK.")
        except Exception as e:
            self.get_logger().error(f"Could not load MuJoCo model: {e}")

        self.create_subscription(
            Float64MultiArray, "/q2/joint_positions", self._jp_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/handle_position", self._hp_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/ee_position", self._ee_cb, FAST_QoS)
        self.create_subscription(
            Float64MultiArray, "/q2/regrasp", self._regrasp_cb, 10)

        self.target_pub = self.create_publisher(
            Float64MultiArray, "/panda/position_targets", FAST_QoS)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, "/panda/gripper_command", FAST_QoS)
        self.status_pub = self.create_publisher(
            String, "/q2/grasp_status", FAST_QoS)
        self.plan_pub = self.create_publisher(
            PoseArray, "/q2/planned_waypoints", 10)

        self.create_timer(0.02, self._control_loop)  # 50 Hz
        self.get_logger().info("Grasp controller started. Waiting for handle position...")

    def _jp_cb(self, msg):
        self.joint_positions = np.array(msg.data)

    def _hp_cb(self, msg):
        self.handle_position = np.array(msg.data)

    def _ee_cb(self, msg):
        self.ee_position = np.array(msg.data)

    def _regrasp_cb(self, msg):
        """Re-grasp request from the planner.

        Payload: Float64MultiArray with 7 floats
            [offset_x, offset_y, offset_z, qw, qx, qy, qz]
        containing the pregrasp approach offset (relative to the handle, in
        world frame) and the grasp orientation to use for this re-grasp.
        """
        if len(msg.data) >= 7:
            self.pregrasp_offset = np.array(msg.data[0:3], dtype=float)
            self.current_grasp_quat = (
                float(msg.data[3]),
                float(msg.data[4]),
                float(msg.data[5]),
                float(msg.data[6]),
            )
            self.get_logger().info(
                f"Re-grasp triggered — offset={self.pregrasp_offset.tolist()}, "
                f"quat={self.current_grasp_quat}")
        else:
            self.get_logger().warn(
                f"Re-grasp message has {len(msg.data)} floats, expected 7. "
                "Keeping previous offset/orientation.")

        self.phase = "waiting"
        self.phase_start_time = time.time()
        self.pregrasp_joints = None
        self.grasp_joints = None

    def _send_joints(self, joints):
        msg = Float64MultiArray()
        msg.data = joints.tolist()
        self.target_pub.publish(msg)

    def _send_gripper(self, width_normalized):
        msg = Float64MultiArray()
        msg.data = [width_normalized]
        self.gripper_pub.publish(msg)

    def _compute_ik(self, target_pos, target_quat_wxyz=None):
        """Delegate to the shared Panda IK solver in :mod:`q2.ik`."""
        return solve_panda_ik(
            self.mj_model, self.mj_data, self.joint_positions,
            target_pos, target_quat_wxyz=target_quat_wxyz,
        )

    def _publish_plan(self, positions, quat_wxyz):
        """Publish a PoseArray of planned waypoints with the given grasp orientation."""
        msg = PoseArray()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        for pos in positions:
            p = Pose()
            p.position.x = float(pos[0])
            p.position.y = float(pos[1])
            p.position.z = float(pos[2])
            p.orientation.w = float(quat_wxyz[0])
            p.orientation.x = float(quat_wxyz[1])
            p.orientation.y = float(quat_wxyz[2])
            p.orientation.z = float(quat_wxyz[3])
            msg.poses.append(p)
        self.plan_pub.publish(msg)

    def _control_loop(self):
        # Publish status
        status_msg = String()
        status_msg.data = self.phase if self.phase != "waiting" else "approaching"
        self.status_pub.publish(status_msg)

        if self.joint_positions is None or self.handle_position is None:
            return

        elapsed = time.time() - self.phase_start_time

        if self.phase == "waiting":
            # Use the offset and orientation currently held by the controller.
            # On the first grasp these are the initial values; on re-grasp
            # they were replaced by whatever the planner sent on /q2/regrasp.
            pregrasp_pos = self.handle_position + self.pregrasp_offset
            self.pregrasp_joints = self._compute_ik(
                pregrasp_pos, target_quat_wxyz=self.current_grasp_quat)

            if self.pregrasp_joints is not None:
                # Publish the planned grasp waypoints for visualization
                self._publish_plan(
                    [pregrasp_pos, self.handle_position], self.current_grasp_quat)
                self.phase = "approaching"
                self.phase_start_time = time.time()
                self.get_logger().info(
                    f"Handle at {self.handle_position}, approaching pre-grasp...")
            else:
                self.get_logger().warn("IK failed for pre-grasp pose", throttle_duration_sec=2.0)

        elif self.phase == "approaching":
            # Open gripper and move to pre-grasp
            self._send_gripper(1.0)
            self._send_joints(self.pregrasp_joints)
            time.sleep(1.0)  # Small delay to ensure the robot starts moving
            if elapsed > 4.0:
                # Now compute grasp IK from current handle position (orientation
                # still adapted to the current door angle)
                self.grasp_joints = self._compute_ik(
                    self.handle_position, target_quat_wxyz=self.current_grasp_quat)
                if self.grasp_joints is not None:
                    self.phase = "grasping"
                    self.phase_start_time = time.time()
                    self.get_logger().info("Phase: grasping (moving to handle)")
                else:
                    self.get_logger().warn("IK failed for grasp pose")

        elif self.phase == "grasping":
            # Move to handle
            self._send_joints(self.grasp_joints)
            if elapsed > 3.0:
                self._send_gripper(0.0)
            if elapsed > 5.0:
                self.phase = "ready"
                time.sleep(1.0)
                self.get_logger().info("Phase: ready (handle grasped)")

        elif self.phase == "ready":
            self._send_gripper(0.0)


def main():
    rclpy.init()
    node = GraspController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
